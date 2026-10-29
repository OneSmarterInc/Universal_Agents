"""Read common CV job and project layouts without a model.

Used only when the local model returns no records or invalid output for a
chunk. Every value is copied from the source; evidence is an exact source span.
An unclear layout returns None so the caller keeps the model's result.
"""

import re

from agents_service.agents.cv.schemas import Fact
from agents_service.agents.cv.source_education import (
    EDUCATION_HEADING,
    OTHER_HEADING,
    date_candidates,
    is_section_heading,
)

WORK_HEADING = re.compile(
    r'^\s*(?:(?:work|professional|employment|relevant|internship)\s+)?(?:experience|history|employment)\s*:?\s*$|'
    r'^\s*internships?(?:\s*(?:&|and)\s*(?:work\s+)?experience)?\s*:?\s*$',
    re.I,
)
PROJECT_HEADING = re.compile(
    r'^\s*(?:(?:key|selected|academic|personal|featured|major|mini|technical|college|software)\s*)?projects?\s*:?\s*$',
    re.I,
)
# A line that starts with a bullet is a detail of an entry, never the entry's title.
BULLET = re.compile(r'^\s*[•●▪◦○■□➢➤►▶✓✔\-–*·]\s*')
# "Technologies: React, Node" or "Tech Stack: ..." under a project title.
DETAIL_LABEL = re.compile(
    r'^\s*(?:tech(?:nolog(?:y|ies))?|tech stack|stack|tools?|role|links?|github)\s*[:|]', re.I
)
# One section for both: "WORK EXPERIENCE / PROJECTS", "Experience & Projects".
COMBINED_HEADING = re.compile(
    r'^\s*(?:(?:work|professional|relevant)\s+)?(?:experience|employment)\s*(?:/|&|and)\s*projects?\s*:?\s*$|'
    r'^\s*projects?\s*(?:/|&|and)\s*(?:(?:work|professional)\s+)?experience\s*:?\s*$',
    re.I,
)
SEPARATOR = re.compile(r'\s+[|·•–—]\s+|\s*\|\s*')
AT = re.compile(r'^(.+?)\s+(?:at|@)\s+(.+)$')
MAX_DESCRIPTION_LINES = 12


def _lines(source):
    lines, offsets, position = source.split('\n'), [], 0
    for line in lines:
        offsets.append(position)
        position += len(line) + 1
    return lines, offsets


def _wraps_bullet(lines, index):
    """The line goes on from an unfinished bullet above it ("… a National level" / "Project Expo")."""
    previous = lines[index - 1].strip() if index else ''
    line = lines[index].strip()
    return bool(
        BULLET.match(previous)
        and not previous.endswith(('.', '!', '?', ')', ':'))
        and line
        and not line.isupper()
        and not BULLET.match(line)
    )


def _region(lines, heading):
    """Line indexes under the wanted heading; lines before any heading are kept too
    (a chunk can start in the middle of the section)."""
    region, state = [], None
    for index, line in enumerate(lines):
        if heading.match(line) or COMBINED_HEADING.match(line):
            state = 'in'
            continue
        # A named heading always ends the section. A line that only looks like one
        # ("Project Expo") is the wrapped end of an unfinished bullet above it.
        if (
            OTHER_HEADING.match(line)
            or EDUCATION_HEADING.match(line)
            or WORK_HEADING.match(line)
            or PROJECT_HEADING.match(line)
            or (is_section_heading(line) and not _wraps_bullet(lines, index))
        ):
            state = 'out'
            continue
        if state != 'out':
            region.append((index, state == 'in'))
    return region


def _dates_in(source, start, end):
    return [m for m in date_candidates(source) if start <= m.start() and m.end() <= end]


def _only_date(line):
    """A line holding a date or a date range and nothing else."""
    rest = line
    for found in date_candidates(line):
        rest = rest.replace(found.group(), ' ')
    return bool(date_candidates(line)) and not re.search(r'[^\W\d_]', rest)


def _starts_with_date(line):
    stripped = line.strip()
    return bool(stripped) and any(m.start() == 0 for m in date_candidates(stripped))


def _short_label(line):
    """A title or employer line: short, no sentence ending, no field separators."""
    text = line.strip()
    return 0 < len(text) <= 80 and not text.endswith(('.', ';', ':')) and not SEPARATOR.search(text)


def _text_parts(line):
    """Non-empty separator-delimited parts of a line, without their dates
    ("Acme Labs Feb 2025" -> "Acme Labs"); a part that is only a date is left out."""
    parts = []
    for part in (p.strip() for p in SEPARATOR.split(line.strip())):
        dates = date_candidates(part)
        if dates:
            part = part[: dates[0].start()].rstrip(' -–—:,(') if dates[0].start() else ''
        if part and re.search(r'[^\W\d_]', part):
            parts.append(part)
    return parts


def _title_employer_header(lines, i, allowed, parts):
    """A work entry written as title line, employer line, then a date line."""
    return (
        i + 1 in allowed
        and i + 2 in allowed
        and _starts_with_date(lines[i + 2])
        and _short_label(lines[i])
        and _short_label(lines[i + 1])
        and not date_candidates(lines[i + 1])
        and len(parts) == 1
        and len(_text_parts(lines[i + 1])) == 1
    )


def _title_above_employer(lines, i, allowed, parts, consumed):
    """A work entry written as a title line, then "Employer (dates)": "Firmware Engineer" /
    "Demo Devices (Aug 2019 to Present)". The title line starts the entry: it follows the
    heading, a blank line or the previous entry's header."""
    title = lines[i - 1].strip() if i - 1 in allowed else ''
    return (
        len(parts) == 1
        and i - 1 > consumed
        and _short_label(title)
        and not date_candidates(title)
        and not BULLET.match(title)
        and not DETAIL_LABEL.match(title)
        and not title[0].islower()
        and (i - 2 not in allowed or not lines[i - 2].strip() or i - 2 == consumed)
    )


def _next_is_bullet(lines, i, allowed):
    return i + 1 in allowed and bool(BULLET.match(lines[i + 1]))


def _undated_project_title(lines, i, allowed, inside):
    """A project written as a title line, then its bullets, with no date anywhere:
    "Student Portal" / "Technologies: HTML, MySQL" / "• Built ...". The title follows
    the section heading, a blank line or a finished bullet; a bullet comes within 3 lines."""
    line = lines[i].strip()
    if (
        i not in inside
        or not line
        or BULLET.match(line)
        or DETAIL_LABEL.match(line)
        or date_candidates(line)
        or line[0].islower()
        or line.endswith(('.', ',', ';', ':'))
        or len(line) > 120
    ):
        return False
    previous = lines[i - 1].strip() if i - 1 in allowed else ''
    # After the heading, a blank line or a finished sentence (a bullet, or its wrapped end).
    after_entry = not previous or previous.endswith(('.', '!', '?', ')'))
    if not after_entry:
        return False
    for j in range(i + 1, i + 4):
        if j not in allowed or not lines[j].strip():
            return False
        if BULLET.match(lines[j]):
            return True
    return False


def _headers(lines, ids, kind, inside=frozenset()):
    """Entry headers as (first line, last line, text parts)."""
    allowed = set(ids)
    headers, consumed = [], -1
    for i in ids:
        line = lines[i]
        if i <= consumed or not line.strip() or _starts_with_date(line) or BULLET.match(line):
            continue
        parts = _text_parts(line)
        if not parts:
            continue
        previous = lines[i - 1].strip() if i - 1 in allowed else ''
        # A line after an unfinished bullet is that bullet's wrapped text, unless
        # bullets of its own follow it ("Volunteer — Management Team, NAVONMESH 2026").
        continuation = line.strip()[0].islower() or bool(
            BULLET.match(previous)
            and not previous.endswith(('.', '!', '?', ')'))
            and not _next_is_bullet(lines, i, allowed)
        )
        if (
            headers
            and headers[-1][1] == i - 1
            and not continuation
            and not DETAIL_LABEL.match(line)
            and _next_is_bullet(lines, i, allowed)
        ):
            # A title that wraps onto a second line ("... AI/ML" / "Internship, 2026").
            first, _last, title_parts = headers[-1]
            headers[-1] = (first, i, title_parts)
            consumed = i
            continue
        if continuation:
            continue
        if (
            date_candidates(line)
            and kind == 'work'
            and _title_above_employer(lines, i, allowed, parts, consumed)
        ):
            headers.append((i - 1, i, [lines[i - 1].strip(), *parts]))
            consumed = i
            continue
        if date_candidates(line):
            headers.append((i, i, parts))
        elif i + 1 in allowed and _starts_with_date(lines[i + 1]) and not line.rstrip().endswith('.'):
            # A sentence ending in "." is the last description line, not a title.
            headers.append((i, i + 1, parts))
        elif i - 1 in allowed and _only_date(lines[i - 1]) and _short_label(line) and i - 1 > consumed:
            # The date on its own line just above the title ("Dec 2024" / "Assistant").
            headers.append((i - 1, i, parts))
        elif kind == 'work' and _title_employer_header(lines, i, allowed, parts):
            headers.append((i, i + 2, parts + _text_parts(lines[i + 1])[:1]))
        elif _undated_project_title(lines, i, allowed, inside):
            # A project, or a role such as "Member, Google Developers Student Clubs",
            # written as a title and its bullets with no date.
            headers.append((i, i, parts))
        if headers and headers[-1][0] == i:
            consumed = headers[-1][1]
    return headers


def _overlapping(headers):
    return any(b <= a_end for (_a, a_end, _), (b, _, _) in zip(headers, headers[1:], strict=False))


def _description_ids(ids, lines, last, following):
    """The unbroken lines right after a header, up to the next header or blank line."""
    description_ids = []
    for i in ids[ids.index(last) + 1 :]:
        previous = description_ids[-1] if description_ids else last
        if i == following or not lines[i].strip() or i != previous + 1:
            break
        description_ids.append(i)
    return description_ids


def _title_and_organization(kind, parts, dated=False):
    if kind != 'work':
        return parts[0], None
    if len(parts) >= 2:
        return parts[0], parts[1]
    match = AT.match(parts[0])
    if match:
        return match.groups()
    # "Software Engineer, Example Corp, Jan 2022": one comma between title and employer on a
    # dated line. More commas ("Engineer, Platform, Example Corp") or no date
    # ("Member, Example Club" is a role) stay in the title.
    title, comma, organization = parts[0].partition(', ')
    if dated and comma and ',' not in title + organization and title.strip() and organization.strip():
        return title, organization
    return parts[0], None


def _record(source, lines, offsets, header, description_ids, kind):
    """One record from a header and its description lines, or None if unclear."""
    first, last, parts = header
    if len(description_ids) > MAX_DESCRIPTION_LINES:
        return None
    dates = _dates_in(source, offsets[first], offsets[last] + len(lines[last]))
    if len(dates) > 2:
        return None
    title, organization = _title_and_organization(kind, parts, bool(dates))
    end_line = description_ids[-1] if description_ids else last
    return Fact(
        title=title.strip(),
        organization=organization and organization.strip(),
        start_date=dates[0].group() if dates else None,
        end_date=dates[1].group() if len(dates) == 2 else None,
        description='\n'.join(lines[i].strip() for i in description_ids) or None,
        evidence=source[offsets[first] : offsets[end_line] + len(lines[end_line])].strip(),
    ).model_dump()


def layout_entries(source, kind):
    """Job ('work') or project records from a common CV layout.

    Returns [] when the chunk has none, or None when the layout is unclear.
    """
    if kind not in ('work', 'projects'):
        raise ValueError(kind)
    heading = WORK_HEADING if kind == 'work' else PROJECT_HEADING
    lines, offsets = _lines(source)
    region = _region(lines, heading)
    headed = any(inside for _, inside in region)
    ids = [i for i, _ in region]
    inside = frozenset(i for i, within in region if within)
    combined = any(COMBINED_HEADING.match(line) for line in lines)
    # Undated titles only under a plain Projects heading; a combined section needs dates.
    headers = _headers(lines, ids, kind, frozenset() if combined else inside)
    if not headers:
        # Unheaded chunks without a clear entry hold no records of this kind.
        return None if headed and any(lines[i].strip() for i in ids) else []
    if _overlapping(headers):
        return None
    records, unclear = [], False
    for number, header in enumerate(headers):
        if combined and (len(header[2]) >= 2) != (kind == 'work'):
            # One section for jobs and projects: a "Title – Organisation" entry is a
            # job, an entry with a title only is a project. Each entry still ends
            # where the next one starts, whichever kind that is.
            continue
        following = headers[number + 1][0] if number + 1 < len(headers) else None
        description_ids = _description_ids(ids, lines, header[1], following)
        record = _record(source, lines, offsets, header, description_ids, kind)
        if record is None:
            unclear = True
        else:
            records.append(record)
    # An unclear entry is left to the model; the clear ones are still read.
    return None if unclear and not records else records
