"""Turn an uploaded file into plain text: Word (DOCX), PDF (two text readers plus OCR for sparse pages) and screenshots (OCR of the main column)."""

import logging
import re
import zipfile
from pathlib import Path

from defusedxml import ElementTree
from PIL import Image, ImageOps

from agents_service.documents.matching import canonical
from agents_service.documents.text import tidy_text

PDF_MAX_PAGES = 30
# Set by the worker from AGENTS_TESSERACT_CMD (needed on Windows).
TESSERACT_CMD = ''


class ExtractionError(Exception):
    """A safe, actionable message that may be shown to the candidate."""


def public_message(code):
    from agents_service.core.errors import MESSAGES

    return MESSAGES.get(code, MESSAGES['invalid_document'])


def main_column_text(data, width):
    """Text of the main column only, from tesseract word boxes.

    A desktop LinkedIn page has a right sidebar ("People also viewed").
    Tesseract reads it first and joins its lines to the profile's own lines
    ("Hansa Vahini v Jane Doe · 2nd"). When a clear vertical gap separates a
    right-hand column, its words are dropped. Returns None when there is no
    such gap (phone screenshots, documents), so the caller keeps plain OCR.
    """
    words = [
        (
            data['left'][i],
            data['top'][i],
            data['width'][i],
            data['height'][i],
            (data['block_num'][i], data['par_num'][i], data['line_num'][i]),
            data['text'][i].strip(),
        )
        for i in range(len(data['text']))
        if data['text'][i].strip()
    ]
    if len(words) < 30:
        return None
    best = None
    for x in range(int(width * 0.55), int(width * 0.85), 2):
        if any(left <= x <= left + w for left, _, w, _, _, _ in words):
            continue
        right = sum(1 for left, *_ in words if left > x)
        if right >= 8 and right <= len(words) * 0.5:
            best = x
            break
    if best is None:
        return None
    lines = {}
    for left, top, w, h, key, text in words:
        if left + w / 2 < best:
            line = lines.setdefault(key, {'top': top, 'height': h, 'words': []})
            line['top'] = min(line['top'], top)
            line['height'] = max(line['height'], h)
            line['words'].append((left, text))
    ordered = sorted(lines.values(), key=lambda line: line['top'])
    heights = sorted(line['height'] for line in ordered) or [10]
    gap = heights[len(heights) // 2] * 1.8
    out, previous = [], None
    for line in ordered:
        if previous is not None and line['top'] - previous > gap:
            out.append('')
        out.append(' '.join(text for _, text in sorted(line['words'])))
        previous = line['top']
    return '\n'.join(out)


SMALL_IMAGE = 2000


def ocr(image, columns=False):
    import pytesseract

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    image = ImageOps.exif_transpose(image).convert('RGB')
    image.thumbnail((3000, 3000))
    if max(image.size) < SMALL_IMAGE:
        # A screenshot of a part of a page is small: OCR loses digits ("Aug 202") and
        # characters there. Twice the size reads it fully (tested on LinkedIn screenshots).
        factor = min(2.0, SMALL_IMAGE / max(image.size))
        image = image.resize((round(image.width * factor), round(image.height * factor)), Image.LANCZOS)
    try:
        if columns:
            data = pytesseract.image_to_data(
                image, config='--psm 3', timeout=45, output_type=pytesseract.Output.DICT
            )
            main = main_column_text(data, image.width)
            if main and len(main.strip()) >= 30:
                return main
        text = pytesseract.image_to_string(image, config='--psm 3', timeout=45)
        if len(text.strip()) < 30:
            text = pytesseract.image_to_string(image, config='--psm 11', timeout=45)
        return text
    except pytesseract.TesseractNotFoundError as exc:
        raise ExtractionError('Image reading is not configured. Please contact support.') from exc


MC_FALLBACK = '{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback'


def _own(element, ns, tag):
    """Descendants with `tag`, not entering text boxes or compatibility
    fallbacks (Word stores each text box twice: the shape and a fallback)."""
    for child in element:
        if child.tag in (MC_FALLBACK, ns + 'txbxContent'):
            continue
        if child.tag == tag:
            yield child
        yield from _own(child, ns, tag)


def docx_paragraphs(root, ns):
    """Every paragraph once, in reading order: a paragraph, then the text-box
    paragraphs anchored in it. Fallback copies are skipped."""

    def walk(element):
        for child in element:
            if child.tag == MC_FALLBACK:
                continue
            if child.tag == ns + 'p':
                yield child
                for box in child.iter(ns + 'txbxContent'):
                    if box in skipped:
                        continue
                    skipped.add(box)
                    yield from walk(box)
            else:
                yield from walk(child)

    fallback_boxes = {box for fb in root.iter(MC_FALLBACK) for box in fb.iter(ns + 'txbxContent')}
    skipped = set(fallback_boxes)
    yield from walk(root)


def docx_paragraph_text(paragraph, ns):
    """Paragraph text with Word line breaks and tabs kept as separators."""
    # Only run children: w:tab inside paragraph properties is a tab stop, not text.
    parts = []
    for run in _own(paragraph, ns, ns + 'r'):
        for node in run:
            if node.tag == ns + 't' and node.text:
                parts.append(node.text)
            elif node.tag in (ns + 'br', ns + 'cr'):
                parts.append('\n')
            elif node.tag == ns + 'tab':
                parts.append('\t')
    return ''.join(parts)


# A lone letter before a lower-case word: "D eveloped", "Classi f ication".
# "a" and "I" are real words, so they do not count.
SPLIT_WORD = re.compile(r'(?<![\w.@/])[b-hj-zB-HJ-Z] (?=[a-z])|(?<=[a-z]) [b-hj-z](?= [a-z])')


def split_words(text):
    """How many words a PDF text layer split into pieces."""
    return len(SPLIT_WORD.findall(text))


def letter_per_line(text):
    """True when most lines hold one character: a text layer that broke every letter
    onto its own line ("P\\nI\\nY\\nU..."), which nothing downstream can read."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(lines) >= 20 and sum(len(line) == 1 for line in lines) > 0.6 * len(lines)


# Text a PDF stores as UTF-8 read as Windows-1252: "â€“" for "–", "ðŸ“§" for an emoji.
MOJIBAKE = re.compile(r'(?:â€|Ã[\x80-\xbf]|ðŸ|Â[\xa0-\xbf])')


def repair_mojibake(text):
    """Undo UTF-8 read as Windows-1252, line by line, only where it gives clean text."""
    if not MOJIBAKE.search(text):
        return text
    fixed = []
    for line in text.split('\n'):
        if MOJIBAKE.search(line):
            try:
                line = line.encode('cp1252', errors='strict').decode('utf-8', errors='strict')
            except UnicodeError:
                try:
                    repaired = line.encode('cp1252', errors='ignore').decode('utf-8', errors='ignore')
                    line = repaired if repaired.strip() else line
                except UnicodeError:
                    pass
        fixed.append(line)
    return '\n'.join(fixed)


# A heading set with a space between its letters: "C O N T A C T", "E D U C AT I O N".
SPACED_HEADING = re.compile(r'(?m)^(?:[A-Z&]{1,2} ){3,}[A-Z&]{1,2}$')

# Private-use symbols some CV templates use as bullets (Wingdings and icon fonts).
PRIVATE_USE = re.compile('[\ue000-\uf8ff]')


DASHES = '-‐‑‒–—'


def restore_dashes(text, other):
    """Put back dashes one text layer dropped and the other kept.

    PDFium can read "2023–2024" as "2023 2024" for some fonts. The two layers
    are aligned without spaces; only dash characters are inserted, each where
    the text has a space or no gap. Nothing else is taken from `other`.
    """
    from difflib import SequenceMatcher

    positions = [i for i, char in enumerate(text) if not char.isspace()]
    mine = ''.join(text[i] for i in positions)
    theirs = ''.join(char for char in other if not char.isspace())
    if not mine or not theirs or len(theirs) > 60000:
        return text
    inserts = {}
    for tag, a1, _a2, b1, b2 in SequenceMatcher(None, mine, theirs, autojunk=False).get_opcodes():
        if tag == 'insert' and 0 < a1 < len(mine) and set(theirs[b1:b2]) <= set(DASHES):
            inserts[positions[a1]] = theirs[b1:b2]
    if not inserts:
        return text
    out, last = [], 0
    for at in sorted(inserts):
        gap_start = positions[positions.index(at) - 1] + 1
        gap = text[gap_start:at]
        out.append(text[last:gap_start])
        # "2023 2024" -> "2023–2024", "ROC AUC" -> "ROC-AUC"; an em dash keeps its spaces.
        spaced = gap if gap and '—' in inserts[at] else ''
        out.extend([spaced, inserts[at], spaced])
        last = at
    out.append(text[last:])
    return ''.join(out)


def layout_text(textpage):
    """The page's text in reading order by position, or '' to keep the stored order.

    Most PDFs store their text in reading order, two-column CVs included, and
    that order is kept. Some CV templates store every heading first and every
    date last; then the stored order jumps back up the page into the same area
    it already covered. Only such a page is re-read by position: pieces at the
    same height form one line (a date stays with its entry), lines go top to
    bottom.
    """
    runs = _runs(textpage)
    if not runs or not _stored_order_scrambled(runs):
        return ''
    return '\n'.join(line for column in _columns(runs) for line in _lines(column))


def _columns(runs):
    """The runs split into side-by-side columns in reading order, or [runs] when there are none.

    Read row by row across the whole page, a sidebar's lines are mixed into the
    main text ("Akola, Maharashtra   Projects"), so each column is read on its own.
    A split needs a clear gap that no run crosses, a fair share of the text on
    each side, and sides that do not pair up row by row (a title with its date
    at the right edge is one entry, not two columns). The column whose text
    starts higher on the page comes first.
    """
    total = sum(len(run[4]) for run in runs)
    best = None
    for split in sorted({run[0] for run in runs}):
        left = [run for run in runs if run[2] <= split]
        right = [run for run in runs if run[0] >= split]
        if len(left) + len(right) < len(runs) or min(len(left), len(right)) < 3:
            continue
        if min(sum(len(run[4]) for run in side) for side in (left, right)) < 0.2 * total:
            continue
        gap = split - max(run[2] for run in left)
        if best is None or gap > best[0]:
            best = (gap, left, right)
    if best is None or best[0] < MIN_COLUMN_GAP:
        return [runs]
    _, left, right = best
    if _paired_rows(left, right) > 0.5 * min(len(left), len(right)):
        return [runs]
    return sorted((left, right), key=lambda side: -max(run[3] for run in side))


MIN_COLUMN_GAP = 6


def _paired_rows(left, right):
    """How many runs of `right` sit on the same row as a run of `left`."""
    count = 0
    for run in right:
        middle, height = (run[1] + run[3]) / 2, max(run[3] - run[1], 1)
        if any(abs((other[1] + other[3]) / 2 - middle) <= 0.5 * height for other in left):
            count += 1
    return count


def _runs(textpage):
    """Pieces of one line each, in stored order, built from their own characters.

    Characters are taken by index, so a piece never takes letters from the line
    above or below, and nothing at the baseline (a decimal point) is lost.
    """
    runs, start, box = [], None, None

    def close(end):
        if start is not None and end > start:
            text = textpage.get_text_range(start, end - start).replace('\r', ' ').replace('\n', ' ').strip()
            if text:
                runs.append((*box, text))

    for index in range(textpage.count_chars()):
        char = textpage.get_text_range(index, 1)
        if char in ('\r', '\n'):
            close(index)
            start, box = None, None
            continue
        left, bottom, right, top = textpage.get_charbox(index)
        if char.isspace() or right <= left:
            if start is None:
                start, box = index, None
            continue
        if box is not None:
            height = max(box[3] - box[1], top - bottom, 1)
            new_line = left < box[2] - height or bottom > box[3] or top < box[1]
            if new_line:
                close(index)
                start, box = index, None
        if start is None:
            start = index
        box = (
            (left, bottom, right, top)
            if box is None
            else (min(box[0], left), min(box[1], bottom), max(box[2], right), max(box[3], top))
        )
    close(textpage.count_chars())
    return runs


def _stored_order_scrambled(runs):
    """True when the stored order returns to page areas it has already passed.

    The stored order is split where it jumps back up the page. In a two-column
    CV the pieces are the columns: side by side, never inside each other. In a
    scrambled template a later piece wraps around an earlier one in the same
    column (headings stored first sit between the body lines stored later).
    """
    pieces, current = [], [runs[0]]
    for run in runs[1:]:
        previous = current[-1]
        if run[3] > previous[3] + 1.5 * max(previous[3] - previous[1], 1):  # back up the page
            pieces.append(current)
            current = []
        current.append(run)
    pieces.append(current)
    inside = 0
    for i, first in enumerate(pieces):
        for second in pieces[i + 1 :]:
            inside += _wrapped(first, second) + _wrapped(second, first)
    return inside >= 2


def _wrapped(inner, outer):
    """How many runs of `inner` sit between two runs of `outer` in the same column."""
    count = 0
    for run in inner:
        middle = (run[1] + run[3]) / 2
        same_column = [o for o in outer if _overlap(run[0], run[2], o[0], o[2]) > 0.5]
        if any((o[1] + o[3]) / 2 > middle for o in same_column) and any(
            (o[1] + o[3]) / 2 < middle for o in same_column
        ):
            count += 1
    return count


def _overlap(start_a, end_a, start_b, end_b):
    """How much two ranges overlap, as a share of the shorter one."""
    shorter = min(end_a - start_a, end_b - start_b)
    if shorter <= 0:
        return 0.0
    return max(0.0, min(end_a, end_b) - max(start_a, start_b)) / shorter


def _lines(runs):
    rows = []
    for run in sorted(runs, key=lambda r: -(r[1] + r[3]) / 2):
        middle, height = (run[1] + run[3]) / 2, max(run[3] - run[1], 1)
        row = rows[-1] if rows else None
        if row and abs(row['middle'] - middle) <= 0.5 * max(height, row['height']):
            row['runs'].append(run)
        else:
            rows.append({'middle': middle, 'height': height, 'runs': [run]})
    lines = []
    for row in rows:
        parts, last_right = [], None
        for left, _bottom, right, _top, text in sorted(row['runs'], key=lambda r: r[0]):
            gap = '' if last_right is None else (' ' if left - last_right < 15 else '   ')
            parts.append(gap + text)
            last_right = right
        lines.append(''.join(parts))
    return lines


def best_pdf_text(path, pypdf_pages):
    """Per page, the cleaner of two text layers.

    pypdf can split words ("D eveloped") and join lines ("NAMEB.E.") for some
    fonts. PDFium keeps the visual lines, in reading order by position (see
    layout_text). The page uses whichever has fewer split words, and PDFium
    unless it lost text.
    """
    try:
        import pypdfium2 as pdfium

        with pdfium.PdfDocument(str(path)) as pdf:
            pdfium_pages = []
            for i in range(len(pdf)):
                page = pdf[i]
                textpage = page.get_textpage()
                try:
                    stream = textpage.get_text_bounded().replace('\r\n', '\n').replace('\r', '\n')
                    try:
                        # A rotated page keeps its stored order: its positions are not upright.
                        placed = '' if page.get_rotation() else layout_text(textpage)
                    except Exception:  # noqa: BLE001 - the stored order is used instead
                        placed = ''
                    # Re-read by position only when it is scrambled and nothing is lost or added.
                    same = abs(len(re.sub(r'\s', '', placed)) - len(re.sub(r'\s', '', stream)))
                    keep = placed and same <= 0.02 * len(re.sub(r'\s', '', stream))
                    pdfium_pages.append(placed if keep else stream)
                finally:
                    textpage.close()
                    page.close()
    except Exception as exc:  # noqa: BLE001 - the pypdf text is used instead
        logging.getLogger(__name__).info('PDFium text skipped: exception_type=%s', type(exc).__name__)
        return pypdf_pages
    chosen = []

    def size(text):
        return len(re.sub(r'\s', '', text))

    for first, second in zip(pypdf_pages, pdfium_pages, strict=False):
        lost_text = size(second) < 0.8 * size(first)
        if letter_per_line(first) and not letter_per_line(second) and not lost_text:
            chosen.append(second)  # pypdf broke every letter onto its own line
            continue
        chosen.append(
            first if lost_text or split_words(second) > split_words(first) else restore_dashes(second, first)
        )
    return chosen + pypdf_pages[len(chosen) :]


WORD_NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


REL_NS = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'


MAX_DOCX_TEXT_BYTES = 10 * 1024 * 1024


MIN_TEXT_CHARS = 30


MAX_TEXT_CHARS = 36000


def _docx_parts(archive, root):
    """Word parts to read: referenced headers first, then the body, then footers."""
    import posixpath

    parts, headers = ['word/document.xml'], []
    referenced = {
        node.get(REL_NS + 'id')
        for kind in ('headerReference', 'footerReference')
        for node in root.iter(WORD_NS + kind)
    }
    if not referenced or 'word/_rels/document.xml.rels' not in archive.namelist():
        return parts
    for rel in ElementTree.fromstring(archive.read('word/_rels/document.xml.rels')):
        kind = rel.get('Type', '')
        if rel.get('Id') not in referenced or rel.get('TargetMode') == 'External':
            continue
        if not kind.endswith(('/header', '/footer')):
            continue
        target = rel.get('Target', '')
        part = posixpath.normpath(
            target.lstrip('/') if target.startswith('/') else posixpath.join('word', target)
        )
        if not part.startswith('word/') or part in parts or part in headers or part not in archive.namelist():
            continue
        # Headers go first: a CV may put the owner's name only there.
        (headers if kind.endswith('/header') else parts).append(part)
    return headers + parts


def _docx_text(path):
    # Resolve only local header/footer parts referenced by the document.
    from agents_service.documents.safety import deep_check_docx

    with zipfile.ZipFile(path) as archive:
        problem = deep_check_docx(archive)
        if problem:
            raise ExtractionError(public_message(problem))
        root = ElementTree.fromstring(archive.read('word/document.xml'))
        chunks, total = [], 0
        for part in _docx_parts(archive, root):
            total += archive.getinfo(part).file_size
            if total > MAX_DOCX_TEXT_BYTES:
                raise ExtractionError('The document text exceeds the processing limit.')
            part_root = ElementTree.fromstring(archive.read(part))
            chunk = '\n'.join(docx_paragraph_text(p, WORD_NS) for p in docx_paragraphs(part_root, WORD_NS))
            if chunk and chunk not in chunks:
                chunks.append(chunk)
        return '\n'.join(chunks)


OCR_WORD = re.compile(r'[0-9a-z]{2,}')
# An OCR line whose words the text layer mostly has already is a noisy copy of it
# ("+ Java" for "• Java", "Al-Powered" for "AI-Powered"), not new text.
OCR_KNOWN_SHARE = 0.6


def _compact(text):
    return re.sub(r'[^0-9a-z]', '', canonical(text))


def _with_ocr_lines(text, image_text):
    """Selectable text plus the OCR lines it does not already contain.

    A page with pictures is read by OCR as well, so text inside the pictures is
    not lost. OCR also reads the page's own text again, with small errors; such a
    line is left out when the text layer already holds most of its words.
    """
    seen = {canonical(line) for line in text.splitlines() if line.strip()}
    compact_text = _compact(text)
    known_words = set(OCR_WORD.findall(canonical(text)))
    extra = []
    for line in image_text.splitlines():
        key = canonical(line)
        if not key or key in seen:
            continue
        compact = _compact(line)
        # Too little to be text ("0", "e"), unless a short name ("Go", "C++") or a
        # number such as a year range or a postcode ("2024-2028", "444203").
        if (
            len(re.findall(r'[^\W\d_]', key)) < 2
            and not re.search(r'[+#]', key)
            and len(re.findall(r'\d', key)) < 4
        ):
            continue
        # A long line found whole in the text layer is a copy; a short word inside a
        # longer one ("java" in "javascript") is not.
        if len(compact) >= 12 and compact in compact_text:
            continue
        words = re.findall(r'[0-9a-z]{2,}', key)
        if known_words and words and sum(w in known_words for w in words) / len(words) >= OCR_KNOWN_SHARE:
            continue
        extra.append(line)
        seen.add(key)
    return text + '\n' + '\n'.join(extra) if extra else text


def _ocr_page(pdf, index, text):
    page = pdf[index]
    try:
        width, height = page.get_size()
        bitmap = page.render(scale=min(2, 3000 / max(width, height)))
        try:
            return _with_ocr_lines(text, ocr(bitmap.to_pil()))
        finally:
            bitmap.close()
    finally:
        page.close()


def _pdf_text(path):
    from pypdf import PdfReader

    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ExtractionError(public_message('invalid_document'))
    # The declared count is read first: a huge page tree is refused before it is walked.
    # The real count is still checked below, because the declared one can be wrong.
    try:
        declared = int(reader.trailer['/Root']['/Pages'].get('/Count', 0))
    except Exception:  # noqa: BLE001 - a broken catalogue is checked by the full count below
        declared = 0
    if declared > PDF_MAX_PAGES or len(reader.pages) > PDF_MAX_PAGES:
        raise ExtractionError(f'Upload a PDF with no more than {PDF_MAX_PAGES} pages.')
    pages = best_pdf_text(path, [p.extract_text() or '' for p in reader.pages])
    # Read each sparse page, including scanned pages inside a mixed PDF.
    needs_ocr = [
        len(text.strip()) < 60 or bool(page.images) for text, page in zip(pages, reader.pages, strict=False)
    ]
    if any(needs_ocr):
        import pypdfium2 as pdfium

        with pdfium.PdfDocument(str(path)) as pdf:
            for i, text in enumerate(pages):
                if needs_ocr[i]:
                    pages[i] = _ocr_page(pdf, i, text)
    return '\n\n'.join(pages)


def read_document(path, clean=None):
    """Plain text of a DOCX, PDF or screenshot, cleaned and within size limits.

    `clean(text, screenshot)` tidies the text; agents can pass their own (the
    LinkedIn agent also removes browser and menu lines from screenshots).
    """
    path = Path(path)
    screenshot = path.suffix not in ('.docx', '.pdf')
    if path.suffix == '.docx':
        text = _docx_text(path)
    elif path.suffix == '.pdf':
        text = PRIVATE_USE.sub('•', repair_mojibake(_pdf_text(path)))
    if not screenshot:
        text = SPACED_HEADING.sub(lambda m: m.group().replace(' ', ''), text)
    else:
        # Screenshots: keep the page's main column, not its sidebar.
        with Image.open(path) as image:
            text = ocr(image, columns=True)
    text = (clean or tidy_text)(text, screenshot)
    if len(text) < MIN_TEXT_CHARS:
        raise ExtractionError('No readable text was found. Upload a clearer document or screenshot.')
    if len(text) > MAX_TEXT_CHARS:
        raise ExtractionError(
            'The extracted text exceeds 36,000 characters. Please upload a shorter document.'
        )
    return text
