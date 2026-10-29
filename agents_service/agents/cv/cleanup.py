"""Tidy a CV's checked facts before they become records.

Every fact here has already passed the evidence check. These rules only remove
repeats and plain mistakes of the small model, or split a list it returned as
one line; every value that remains is still a copy of the document's text.
"""

import re

from agents_service.agents.cv.source_education import EDUCATION_HEADING, OTHER_HEADING, is_section_heading
from agents_service.agents.cv.source_lists import LIST_END
from agents_service.documents.matching import canonical

ENTRY_SECTIONS = (
    'education',
    'work_experience_entries',
    'projects',
    'certifications',
    'awards',
    'publications',
    'volunteering',
)
# "Backend: Django, Node.js" / "Technical Skills: • Programming Languages: Python ,Numpy"
SKILL_LABEL = re.compile(r'^[^\W\d_][\w &/+().-]{0,40}:\s*')
SKILL_SEPARATORS = ',;|•●▪'
MAX_SKILL_CHARS = 40


def _key(text):
    return canonical(text or '').strip(' .,;:-–—')


def _entry_key(entry):
    return (
        _key(entry.get('title')),
        _key(entry.get('organization')),
        entry.get('start_date'),
        entry.get('end_date'),
    )


def _richness(entry):
    return len(entry.get('description') or '') + len(entry.get('evidence') or '')


def merge_repeats(entries):
    """One entry per title, organisation and dates; the one with more detail is kept.

    The model and the document reading can both find a job: once with its bullets,
    once without them.
    """
    kept = []
    for entry in entries:
        key = _entry_key(entry)
        twin = next((n for n, other in enumerate(kept) if _entry_key(other) == key), None)
        if twin is None:
            kept.append(entry)
        elif _richness(entry) > _richness(kept[twin]):
            kept[twin] = entry
    return kept


def _first_line(entry):
    return _key((entry.get('evidence') or '').split('\n', 1)[0])


def drop_projects_filed_as_jobs(jobs, projects):
    """A job that is the same entry as a project (same title, or the same first line of
    the document) is the project, filed twice. It stays a project."""
    project_titles = {_key(p.get('title')) for p in projects}
    project_lines = {_first_line(p) for p in projects}
    return [
        job
        for job in jobs
        if not (
            job.get('organization') is None
            and (_key(job.get('title')) in project_titles or _first_line(job) in project_lines)
        )
    ]


def _holds_headings(text):
    """True when the text runs over two or more section headings: a block of the CV,
    not a summary."""
    lines = [line.strip() for line in (text or '').split('\n') if line.strip()]
    headings = [
        line
        for line in lines
        if OTHER_HEADING.match(line) or EDUCATION_HEADING.fullmatch(line) or is_section_heading(line)
    ]
    return len(headings) >= 2


# Fewer words is a name or a job title the model filed as the summary ("Daniel Example").
MIN_SUMMARY_WORDS = 4


def _not_a_summary(text):
    return _holds_headings(text) or len(str(text or '').split()) < MIN_SUMMARY_WORDS


def clean_summary(value):
    """None when the 'summary' is a block of the CV holding other sections, or too short to be one."""
    return None if isinstance(value, str) and _not_a_summary(value) else value


def clean_summary_variants(entries):
    """The summary alternatives whose own text is a summary (the quote may be longer)."""
    return [entry for entry in entries if not _not_a_summary(entry.get('value'))]


def _split_outside_brackets(text):
    """Parts at , ; | and bullets, but not inside brackets: "Scraping (BeautifulSoup, Scrapy)" stays whole."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char in '([':
            depth += 1
        elif char in ')]':
            depth = max(0, depth - 1)
        elif char in SKILL_SEPARATORS and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return [part.strip() for part in parts]


def split_skills(skills):
    """Skills one per item: "Backend: Django, Node.js" -> "Django", "Node.js".

    Each item is a part of the line, so it is still a copy of the document's text.
    """
    result, seen = [], set()
    for skill in skills:
        text = skill.strip()
        parts = [text]
        pieces = _split_outside_brackets(text)
        if SKILL_LABEL.match(text) or len(pieces) > 1:
            parts = []
            for piece in pieces:
                while SKILL_LABEL.match(piece):
                    piece = SKILL_LABEL.sub('', piece, count=1)
                piece = piece.strip(' .')
                if piece and len(piece) <= MAX_SKILL_CHARS and re.search(r'[^\W\d_]', piece):
                    parts.append(piece)
            if not parts:
                parts = [text]
        for part in parts:
            key = canonical(part)
            if key not in seen:
                seen.add(key)
                result.append(part)
    return result


LIST_SECTIONS = ('certifications', 'awards')
# A title whose next line was put in the description: "…dedication to" + "sports."
END_MARK = re.compile(r'[.!?:;)\]]$')
AWARD_WORDS = re.compile(r'award|prize|winner|won|secured|position|rank|medal|first|1st|2nd|3rd', re.I)
# "(2024–25 SGPA: 7.82)": a year of study next to a grade, not the end of the degree.
STUDY_YEAR = re.compile(r'(?<!\d)(\d{4})\s*[–-]\s*\d{2}(?!\d)')
GRADE_WORDS = re.compile(r'gpa|semester|\bsem\b|percent|%', re.I)
LINK_LABELS = {
    'linkedin',
    'github',
    'gitlab',
    'portfolio',
    'website',
    'leetcode',
    'blog',
    'twitter',
    'x',
    'resume',
}


def join_wrapped_titles(entries):
    """A title cut at a line break gets its last words back from the description."""
    for entry in entries:
        title, description = entry.get('title') or '', entry.get('description') or ''
        if (
            title
            and description
            and not END_MARK.search(title.strip())
            and '\n' not in description.strip()
            and re.match(r'^[a-z]', description.strip())
            and len(description.split()) <= 8
            and re.search(
                re.escape(title.split()[-1]) + r'\s*\n\s*' + re.escape(description.split()[0]),
                entry.get('evidence') or '',
            )
        ):
            entry['title'] = f'{title.strip()} {description.strip().rstrip(".")}'
            entry['description'] = None
    return entries


def _contains(a, b):
    """True when one text holds the other (compared as canonical text)."""
    a, b = _key(a), _key(b)
    return bool(a and b) and (a in b or b in a)


def drop_cross_repeats(data):
    """One item in one section: an award listed again as a certificate stays an award;
    a volunteer role listed again as a job stays volunteering."""
    awards = data.get('awards') or []
    certificates = data.get('certifications') or []
    if awards and certificates:
        keep_certs = []
        for cert in certificates:
            twin = next(
                (a for a in awards if _contains(a.get('evidence'), cert.get('evidence'))
                 or (len(_key(a.get('title'))) >= 10 and _key(cert.get('title')).startswith(_key(a.get('title'))))),
                None,
            )  # fmt: skip
            if twin is None or not AWARD_WORDS.search(cert.get('title') or ''):
                keep_certs.append(cert)
        data['certifications'] = keep_certs
    volunteering = data.get('volunteering') or []
    if volunteering and data.get('work_experience_entries'):
        data['work_experience_entries'] = [
            job
            for job in data['work_experience_entries']
            if not any(
                _first_line(job) and canonical(v.get('evidence') or '').startswith(_first_line(job))
                for v in volunteering
            )
        ]
    return data


def drop_study_years(entries):
    """A degree date read from "2024–25 SGPA" is the year of study, not the degree's dates."""
    for entry in entries:
        lines = (entry.get('evidence') or '').split('\n')
        for field in ('start_date', 'end_date'):
            year = str(entry.get(field) or '')
            if not re.fullmatch(r'\d{4}', year):
                continue
            uses = [line for line in lines if re.search(rf'\b{year}\b', line)]
            if uses and all(
                GRADE_WORDS.search(line) and any(m.group(1) == year for m in STUDY_YEAR.finditer(line))
                for line in uses
            ):
                entry[field] = None
    return entries


def drop_title_technologies(entries):
    """Certificates and awards: "technologies" that only repeat the title or issuer go."""
    for entry in entries:
        named = _key(entry.get('title')) + ' ' + _key(entry.get('organization'))
        entry['technologies'] = [t for t in entry.get('technologies') or [] if _key(t) not in named]
    return entries


def drop_bare_links(links):
    """ "LinkedIn" alone is a label, not a link."""
    return [
        link
        for link in links
        if _key(link if isinstance(link, str) else link.get('value')) not in LINK_LABELS
    ]


def tidy(data):
    """Apply the rules above to checked CV facts in place; returns data."""
    for section in ENTRY_SECTIONS:
        if isinstance(data.get(section), list):
            data[section] = merge_repeats(data[section])
    for section in LIST_SECTIONS:
        if isinstance(data.get(section), list):
            join_wrapped_titles(data[section])
            drop_title_technologies(data[section])
    if isinstance(data.get('education'), list):
        drop_study_years(data['education'])
    if isinstance(data.get('links'), list):
        data['links'] = drop_bare_links(data['links'])
    drop_cross_repeats(data)
    if isinstance(data.get('work_experience_entries'), list) and isinstance(data.get('projects'), list):
        data['work_experience_entries'] = drop_projects_filed_as_jobs(
            data['work_experience_entries'], data['projects']
        )
    if 'summary' in data:
        data['summary'] = clean_summary(data['summary'])
    if isinstance(data.get('skills'), list):
        data['skills'] = split_skills(data['skills'])
    return data


# --- contact details, graduation year and institution ------------------------------------

# An email in the text, without OCR junk stuck to its front ("_name@example.com").
EMAIL = re.compile(r'[A-Za-z0-9][A-Za-z0-9.+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+')
TOP_CHARS = 1500
# Icons read as text in front of a phone number or a place ("& 8010784202", "9 Mehkar").
PHONE_START = re.compile(r'^[^\d+(]+')
ICON_TOKEN = re.compile(r'^(?:[^\w\s(]{1,2}|\d)\s+(?=[^\W\d_])')
STREET_WORDS = re.compile(
    r'\b(?:road|rd|street|st|lane|nagar|colony|sector|flat|house|plot|no|ward|block)\b', re.I
)
GRADUATION_CUE = re.compile(r'graduat|expected|batch|class of|passing|pass(?:ed)? out|completion', re.I)
DEGREE_LEVEL = re.compile(
    r'bachelor|master|b\.?\s?(?:e|eng|tech|sc|s|com|a|ca|ba)\b|m\.?\s?(?:e|eng|tech|sc|s|ba|ca)\b|diploma|ph\.?d',
    re.I,
)
# "CGPA: 3.9/4.0", "GPA 8.6 out of 10": a grade written with its scale.
GPA_WITH_SCALE = re.compile(
    r'\b(?:c?gpa|grade point average)\b\s*[:\-–]?\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:/|out of)\s*(\d{1,3}(?:\.\d{1,2})?)(?![\d.])',
    re.I,
)


SPLIT_EMAIL = re.compile(r'([A-Za-z0-9][A-Za-z0-9.+-]*@[A-Za-z0-9.-]*)\n([A-Za-z0-9.-]*\.[A-Za-z]{2,})\b')


def email_from_document(source):
    """The first email near the top of the CV, as written there; None when there is none.

    An email cut over two lines ("name@g" / "mail.com") is returned joined, with the two
    lines as its evidence.
    """
    top = source[:TOP_CHARS]
    found = EMAIL.search(top)
    if found:
        return found.group()
    split = SPLIT_EMAIL.search(top)
    if split and EMAIL.fullmatch(split.group(1) + split.group(2)):
        return {'value': split.group(1) + split.group(2), 'evidence': split.group()}
    return None


def join_split_name(owners, source):
    """An owner name printed over two lines ("ABHISHEK SHRIKRISHNA" / "MORKHADE") is one name.

    Only a one-word second line (the surname) is joined; two names on two lines stay two."""
    if not owners or len(owners) < 2:
        return owners
    keys = {canonical(row['value']) for row in owners}
    lines = [line.strip() for line in source.split('\n')[:15]]
    for first, second in zip(lines, lines[1:], strict=False):
        if (
            canonical(first) in keys
            and canonical(second) in keys
            and canonical(first) != canonical(second)
            and len(second.split()) == 1
            and len(first.split()) <= 3
            and canonical(second) not in canonical(first).split()
        ):
            others = [
                row for row in owners if canonical(row['value']) not in {canonical(first), canonical(second)}
            ]
            return [{'value': f'{first} {second}', 'evidence': f'{first}\n{second}'}, *others]
    return owners


SKILLS_HEADING = re.compile(
    r'^[^\w]*(?:(?:technical|key|core|professional|it)\s+)?skills?(?:\s*(?:&|and)\s*[a-z]+)?\s*(?::\s*(.*))?$',
    re.I,
)
MAX_SKILL_WORDS = 5
# "Soft Skills" is a heading inside the list; "C Programming CERTIFICATION" mixes in the next column.
SUB_HEADING = re.compile(r'\bskills?\s*$', re.I)
CAPS_HEADING = re.compile(
    r'\b(?:CERTIFICATIONS?|LANGUAGES?|PROJECTS?|EDUCATION|EXPERIENCE|ACHIEVEMENTS?|INTERESTS?|HOBBIES)\b'
)
NOT_A_SKILL = re.compile(r'scanned with|page \d', re.I)


def _cut_open_bracket(piece):
    """ "Canva (Design & Presentation Tool" -> "Canva": an unclosed bracket is cut off."""
    if piece.count('(') > piece.count(')'):
        piece = piece[: piece.rfind('(')]
    return piece.strip(' .')


def skills_from_document(source):
    """Skills under the CV's skills heading, one per item, each a part of a line."""
    lines = source.split('\n')
    start = next((n for n, line in enumerate(lines) if SKILLS_HEADING.match(line.strip())), None)
    if start is None:
        return []
    first = SKILLS_HEADING.match(lines[start].strip()).group(1)
    body = ([first] if first else []) + lines[start + 1 : start + 26]
    items = []
    for line in body:
        text = line.strip()
        if (
            text
            and not SKILLS_HEADING.match(text)
            and (
                OTHER_HEADING.match(text)
                or EDUCATION_HEADING.fullmatch(text)
                or is_section_heading(text)
                or LIST_END.match(text)
            )
        ):
            break
        text = re.sub(r'^[^\w(]+', '', text)
        if not text or NOT_A_SKILL.search(text):
            continue
        pieces = _split_outside_brackets(text)
        if not SKILL_LABEL.match(text) and len(pieces) == 1 and len(text.split()) > MAX_SKILL_WORDS:
            continue  # a sentence, not a list
        for piece in pieces:
            while SKILL_LABEL.match(piece):
                piece = SKILL_LABEL.sub('', piece, count=1)
            piece = _cut_open_bracket(re.sub(r'^[^\w(]+', '', piece))
            if (
                piece
                and not SUB_HEADING.search(piece)
                and not CAPS_HEADING.search(piece)
                and len(piece) <= MAX_SKILL_CHARS
                and len(piece.split()) <= MAX_SKILL_WORDS
                and re.search(r'[^\W\d_]', piece)
            ):
                items.append(piece)
    return items


def _bare_key(text):
    return _key(re.sub(r'\s*\(.*?\)\s*', ' ', text))


def add_document_skills(skills, source, languages=()):
    """The model's skills, then any skill under the skills heading that it missed.

    A spoken language already found ("Languages: English | Korean" right after the skills
    list) stays a language, not a skill.
    """
    result = list(skills or [])
    keys = {_key(s) for s in result} | {_bare_key(s) for s in result} | {_key(s) for s in languages or []}
    model_keys = [_key(s) for s in result]
    for item in skills_from_document(source):
        if any(_key(item) in key for key in model_keys):
            continue  # a part of a skill the model already has ("BASICS OF JAVA AND")
        if _key(item) not in keys and _bare_key(item) not in keys:
            keys |= {_key(item), _bare_key(item)}
            result.append(item)
    return result


# "B.E. Information Technology" / "Bachelor of Engineering in Information Technology" -> the major.
DEGREE_PREFIX = re.compile(
    r'^(?:(?:bachelor|master)(?:\s+of)?\s+[a-z]+(?:\s+[a-z]+)?\s+(?:in|[-–—,])\s+|'
    r'(?:b|m)\.?\s?(?:e|tech|sc|s)\.?(?![a-z])\s*(?:in\s+|[-–—,(]\s*)?)',
    re.I,
)


def major_only(value):
    if not isinstance(value, str):
        return value
    rest = DEGREE_PREFIX.sub('', value.strip(), count=1).strip(' ()')
    return rest if len(rest) >= 3 and rest != value.strip() else value


def location_from_school_line(location, source):
    """True when the place is only found on a line naming a school (the college's town)."""
    from agents_service.agents.cv.source_education import names_institution

    key = canonical(location)
    lines = [line for line in source.split('\n') if key and key in canonical(line)]
    return bool(lines) and all(names_institution(line) for line in lines)


def clean_phone(value):
    if not isinstance(value, str):
        return value
    cleaned = PHONE_START.sub('', value).strip()
    return cleaned if re.search(r'\d{6}', re.sub(r'\D', '', cleaned)) else value


def clean_location(value):
    if not isinstance(value, str) or STREET_WORDS.search(value):
        return value
    cleaned = ICON_TOKEN.sub('', value.strip(), count=1)
    return cleaned or value


def gpa_from_document(source):
    """(gpa, scale) when the CV writes one grade with its scale; None when there is none
    or more than one, so the model's choice is never replaced by a guess."""
    found = {}
    for match in GPA_WITH_SCALE.finditer(source):
        gpa, scale = float(match.group(1)), match.group(2)
        if gpa <= float(scale):
            found.setdefault((gpa, scale), match.group())
    if len(found) != 1:
        return None
    (((gpa, scale), quote),) = found.items()
    return {'value': gpa, 'evidence': quote}, {'value': scale, 'evidence': quote}


def _graduation_year_ok(year, education, source):
    """A graduation year is kept when the CV ties it to finishing a degree: the end year of
    a degree entry, or a line that says so ("Expected graduation 2027", "Batch of 2026")."""
    text = str(year)
    for entry in education or []:
        if DEGREE_LEVEL.search(entry.get('title') or '') and text in str(entry.get('end_date') or ''):
            return True
    return any(GRADUATION_CUE.search(line) for line in source.split('\n') if re.search(rf'\b{text}\b', line))


def _institution_ok(value, education):
    """An institution names a school, or is the school of an education entry."""
    from agents_service.agents.cv.source_education import names_institution

    organizations = {_key(entry.get('organization')) for entry in education or []}
    return bool(value) and (names_institution(str(value)) or _key(str(value)) in organizations)


def tidy_scalars(data, variants, source):
    """Contact details, graduation year and institution, checked against the rules above.

    data holds single values; variants holds the alternatives still to choose from
    ({field: [{'value', 'evidence'}, ...]}). Both are changed in place.
    """
    if not data.get('email') and 'email' not in variants:
        data['email'] = email_from_document(source)
    data['phone'] = clean_phone(data.get('phone'))
    data['location'] = clean_location(data.get('location'))
    if isinstance(data.get('location'), str) and location_from_school_line(data['location'], source):
        data['location'] = None
    if isinstance(data.get('skills'), list):
        data['skills'] = add_document_skills(data['skills'], source, data.get('languages'))
    if data.get('gpa') is None and 'gpa' not in variants:
        grade = gpa_from_document(source)
        if grade:
            data['gpa'] = grade[0]
            if data.get('gpa_scale') is None and 'gpa_scale' not in variants:
                data['gpa_scale'] = grade[1]
    data['major'] = major_only(data.get('major'))
    if 'major' in variants:
        merged = {}
        for entry in variants['major']:
            entry = {**entry, 'value': major_only(entry['value'])}
            key = canonical(entry['value'])
            if key not in merged or len(entry['evidence']) < len(merged[key]['evidence']):
                merged[key] = entry
        variants['major'] = list(merged.values())
        if len(variants['major']) < 2:
            remaining = variants.pop('major')
            if remaining and not data.get('major'):
                data['major'] = remaining[0]['value']
    checks = {
        'graduation_year': lambda value: _graduation_year_ok(value, data.get('education'), source),
        'institution': lambda value: _institution_ok(value, data.get('education')),
    }
    for field, ok in checks.items():
        if data.get(field) is not None and not ok(data[field]):
            data[field] = None
        if field in variants:
            variants[field] = [entry for entry in variants[field] if ok(entry['value'])]
            if len(variants[field]) < 2:
                # One left is the value, with nothing to choose between; none left is no value.
                remaining = variants.pop(field)
                if remaining and data.get(field) is None:
                    data[field] = remaining[0]['value']
    return data, variants
