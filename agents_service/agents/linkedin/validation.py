"""Checks on LinkedIn facts: tidy the OCR text and keep only facts the screenshot supports."""

import re
from datetime import UTC, datetime
from typing import TypedDict

from agents_service.agents.linkedin.literal import literal_observation
from agents_service.agents.linkedin.sections import HEADINGS, STOP_HEADINGS, heading_of, sections
from agents_service.agents.linkedin.upload_sections import context_source, restrict_fields
from agents_service.documents.matching import canonical, evidence_supported, value_supported


class ExtractionState(TypedDict, total=False):
    profile_id: int
    run_id: int
    documents: list[dict]
    cleaned_documents: list[dict]
    observations: list[dict]
    validated_observations: list[dict]
    merged_profile: dict
    successful_images: int


# Words of a browser bookmarks bar and LinkedIn's top menu, as whole words.
CHROME_WORDS = {
    'gmail',
    'youtube',
    'maps',
    'news',
    'translate',
    'all',
    'bookmarks',
    'bookm',
    'search',
    'home',
    'my',
    'network',
    'jobs',
    'messaging',
    'notifications',
    'try',
    'premium',
    'for',
    'business',
    'me',
    'work',
    'learning',
}


def _window_chrome(line: str) -> bool:
    """Browser address bar, bookmarks bar or LinkedIn's top menu: a short line
    made mostly of those words (at least two of them)."""
    value = line.casefold()
    if re.match(r'^\W*(?:<\s*>\s*)?(?:localhost[:/]|relaunch to update)', value):
        return True
    words = re.findall(r'[a-z]+', value)
    known = [word for word in words if word in CHROME_WORDS]
    # OCR garbles some menu words, so a long menu line needs only half known.
    ratio = len(known) / len(words) if words else 0
    # Two known words count only when they are the whole line ("News Translate");
    # a headline like "Business Network Manager" stays.
    return len(line) <= 120 and (
        (len(known) >= 2 and ratio == 1)
        or (len(known) >= 3 and ratio >= 0.6)
        or (len(known) >= 4 and ratio >= 0.5)
    )


STATUS_BAR = re.compile(r'^(?:[01]?\d|2[0-3])[:.]?[0-5]\d\b(?P<rest>.*)$')


def _status_bar(line: str) -> bool:
    """A phone's status bar read by OCR as the first line: the time, then icons
    ("754 OOS ie" for 7:54). No word of four letters or more follows the time."""
    found = STATUS_BAR.match(line)
    return bool(found) and len(line) <= 30 and not re.search(r'[^\W\d_]{4,}', found.group('rest'))


def clean_text(text: str, screenshot: bool = False) -> str:
    """Tidy OCR or document text. Screenshot text also loses browser and
    LinkedIn menu lines; document text keeps every line (a CV may well say
    'Notifications' or list a gmail address)."""
    text = text.replace('\x00', ' ')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines, first = [], True
    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            lines.append('')
            continue
        if screenshot and first and _status_bar(clean):
            first = False
            continue
        first = False
        if screenshot and (UI_NOISE_LINE.fullmatch(clean) or _window_chrome(clean)):
            continue
        if screenshot:
            clean = OCR_ROMAN.sub(lambda m: m.group(1) + m.group(2).replace('l', 'I'), clean)
        lines.append(clean)
    return '\n'.join(lines).strip()


# OCR reads the Roman numeral III as "Ill" ("Level Ill Python Certification").
OCR_ROMAN = re.compile(r'\b((?:Level|Part|Grade|Phase|Tier|Stage|Class)\s+)((?=[Il]*l)[Il]{2,3})\b')


UI_NOISE_LINE = re.compile(
    r'(?:premium subscribers.*|try premium.*|enhance profile|.*\bworks here|'
    r'https?://(?:www\.)?linkedin\.com/jobs/\S*|logout|sign in|notifications?|'
    # The owner's own view of a profile: buttons, analytics and app boxes.
    r'\W*add (?:connected apps|custom button|section).*|\W*connected apps\b.*|analytics|'
    r'\W*private to you|.*\b(?:profile views?|post impressions?|search appearances?)\b.*|'
    r'\W*create\s*a\s*post\W*|\W*show all\b.*|\W*open to\W*)',
    re.I,
)

FIELD_NOISE = re.compile(
    r'(?:try premium|premium subscribers|enhance profile|works here|linkedin\.com/jobs/|'
    r'^email issue$)',
    re.I,
)

CERTIFICATE_LANGUAGE = re.compile(
    r'\b(certificat(?:e|ion)|credential|issued by|course completion|academy graduate)\b',
    re.I,
)

COURSE_TITLE = re.compile(
    r'\b(?:academy graduate|training\s*:|getting started with|introduction to|'
    r'certificate (?:in|of)|certification|certified\b)',
    re.I,
)

SKILL_UI = re.compile(
    r'^(?:home|my network|messaging|notifications?|jobs|search|follow|'
    r'tools & technologies|promoted\W*|[a-z]\s+(?:follow|search))$|'
    r'\bfollow\b|\bget the latest jobs\b|\balso follow\b|'
    r'linkedin|©|\bin\s*\(\d+\)|^\W*\+?\d+\s+skills?\W*$',
    re.I,
)

ACADEMIC_DEGREE = re.compile(
    r"\b(degree|diploma|bachelor(?:'s)?|master(?:'s)?|associate(?:'s)?|doctorate|"
    r'ph\.?\s*d\.?|b\.?\s*(?:a|s|sc|tech|e)\.?|m\.?\s*(?:a|s|sc|tech|e|ba)\.?)\b',
    re.I,
)

POST_CONTEXT = re.compile(r'\b(posts?|activity|published|reposted|comments?)\b', re.I)

PROJECT_CONTEXT = re.compile(
    r'\b(projects?|portfolio|repository|github|demo|built|developed|application|app)\b',
    re.I,
)

DATE_VALUE = re.compile(
    r'^(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+)?'
    r'(?:19|20)\d{2}$|^present$',
    re.I,
)

EDUCATION_LANGUAGE = re.compile(
    r'\b(education|university|college|school|institute|academy|degree|diploma|'
    r"bachelor(?:'s)?|master(?:'s)?|ph\.?\s*d\.?|b\.?\s*(?:a|s|sc|tech|e)\.?|"
    r'm\.?\s*(?:a|s|sc|tech|e|ba)\.?)\b',
    re.I,
)


EDUCATION_INSTITUTION = re.compile(r'\b(university|college|school|academy)\b', re.I)

EMPLOYMENT_LANGUAGE = re.compile(
    r'\b(experience|employment|employed|employee|company|engineer|developer|'
    r'manager|director|consultant|intern|full[ -]?time|part[ -]?time)\b',
    re.I,
)

# Browser and LinkedIn chrome that OCR reads from a full-window screenshot.
UI_NAMES = {
    'gmail',
    'youtube',
    'maps',
    'news',
    'translate',
    'google',
    'chrome',
    'bookmarks',
    'all bookmarks',
    'linkedin',
    'home',
    'my network',
    'jobs',
    'messaging',
    'notifications',
    'me',
    'for business',
    'search',
    'try premium',
    'premium',
    'learning',
    'post',
    'start a post',
    'open to',
    'more',
    'contact info',
    'relaunch to update',
    'update',
    'profile',
    'people',
    'following',
    'followers',
}
# What LinkedIn prints on the owner's name line after the name.
NAME_LINE_TAIL = re.compile(
    r'(?:\s*[(\[]?\s*(?:he/him|she/her|they/them|he/they|she/they)\s*[)\]]?'
    r'|\s*[·•-]?\s*(?:1st|2nd|3rd\+?)'
    r'|\s+[^\w\s]+|\s+\d|\s*[^\w\s]+$)+$',
    re.I,
)


def _owner_name_supported(value, source):
    """A person name standing alone on its own source line.

    Rejects menu, bookmark and button words that a small model can return
    when a screenshot includes the browser window.
    """
    name = canonical(value)
    if (
        not name
        or name in UI_NAMES
        or len(name) > 70
        or not re.search(r'[^\W\d_]', name)
        or re.search(r'\d|@|https?:|www\.|\||linkedin', name)
    ):
        return False
    # OCR can join the header with a side column on one line, so the name
    # must start the line rather than fill it.
    start = re.compile(re.escape(name) + r'(?!\w)')
    for line in source.splitlines():
        text = NAME_LINE_TAIL.sub('', canonical(re.sub(r'^[^\w]+', '', line))).strip()
        if start.match(text):
            return True
    return False


def _all_values_supported(item: dict) -> bool:
    evidence = item.get('evidence', '')
    for key, value in item.items():
        if key == 'evidence' or not value:
            continue
        values = value if isinstance(value, list) else [value]
        if any(not value_supported(part, evidence) for part in values):
            return False
    return True


def _education_supported(item: dict, source: str, certifications: list[dict] | None = None) -> bool:
    evidence = item.get('evidence', '')
    institution = item.get('institution')
    if not institution or not evidence_supported(evidence, source):
        return False
    fields = ('institution', 'degree', 'field_of_study', 'start_date', 'end_date')
    if any(not value_supported(item.get(field), evidence) for field in fields):
        return False
    has_explicit_section = _in_named_section(evidence, source, 'education') or bool(
        re.match(r'\s*Education\b', evidence, re.I)
    )
    has_academic_detail = bool(item.get('degree') and ACADEMIC_DEGREE.search(item['degree']))
    has_school_date_relationship = bool(
        (item.get('start_date') or item.get('end_date')) and EDUCATION_INSTITUTION.search(evidence)
    )
    if EMPLOYMENT_LANGUAGE.search(evidence) and not (has_explicit_section or has_academic_detail):
        return False
    if not has_explicit_section and CERTIFICATE_LANGUAGE.search(evidence):
        return False
    for certificate in certifications or []:
        cert_evidence = certificate.get('evidence', '')
        cert_name = certificate.get('name', '')
        cert_issuer = certificate.get('issuer', '')
        if (
            not has_explicit_section
            and not has_academic_detail
            and evidence
            and cert_evidence
            and (
                _normalize_identity(evidence) in _normalize_identity(cert_evidence)
                or _normalize_identity(cert_evidence) in _normalize_identity(evidence)
            )
        ):
            return False
        institution = _normalize_identity(item.get('institution'))
        if (
            not has_explicit_section
            and not has_academic_detail
            and institution
            and institution in {_normalize_identity(cert_name), _normalize_identity(cert_issuer)}
        ):
            return False
    # The profile card shows the school on its own line, without a heading, degree or dates.
    profile_card_school = bool(
        EDUCATION_INSTITUTION.search(institution or '')
        and not (item.get('degree') or item.get('start_date') or item.get('end_date'))
        and not EMPLOYMENT_LANGUAGE.search(evidence)
        and not CERTIFICATE_LANGUAGE.search(evidence)
        and not _works_there(institution, source)
        and _on_profile_card(item.get('institution') or evidence, source)
    )
    if profile_card_school:
        return True
    return bool(EDUCATION_LANGUAGE.search(evidence)) and (
        has_explicit_section or has_academic_detail or has_school_date_relationship
    )


def _is_page_title(value):
    heading = heading_of(str(value or ''))
    return heading in HEADINGS or heading in STOP_HEADINGS


CARD_MARKER = re.compile(r'^\s*[\d,.]+\+?\s*(?:connections|followers)\b', re.I)


def _on_profile_card(evidence, source):
    """True when the evidence is in the profile card: before any section heading and
    before the card's "N connections" line, which a profile card always shows."""
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if heading_of(line) in HEADINGS or heading_of(line) in STOP_HEADINGS or CARD_MARKER.search(line):
            return False
        if evidence_supported(evidence, line):
            return any(CARD_MARKER.search(later) for later in lines[index + 1 :])
    return False


def _works_there(institution, source):
    """True when the screenshot names the institution as an employer ("Lecturer at X")."""
    name = re.escape(institution.strip())
    return bool(re.search(rf'(?:\bat|@|·)\s*{name}\b', source, re.I))


def _clean_field(value: str | None) -> str:
    return re.sub(r'\s+', ' ', str(value or '')).strip()


def _in_named_section(evidence: str, source: str, heading: str) -> bool:

    for fields, block in sections(source):
        if heading_of(block.splitlines()[0]) in HEADINGS:
            expected = HEADINGS.get(heading, (heading,))
            if (
                any(field in fields for field in expected if field != 'skills')
                or heading == 'skills'
                and fields == ('skills',)
            ):
                if evidence_supported(evidence, block):
                    return True
    return False


def _skill_supported(item: dict, source: str) -> bool:
    name = _clean_field(item.get('name'))
    evidence = item.get('evidence', '')
    # A skills heading or pipe anywhere in the screenshot is not evidence for
    # every word in that screenshot. Require a local skills line/section, or
    # an exact pipe-delimited skill in the professional headline.
    contextual = any(
        value_supported(name, line)
        and not re.search(r'tutorial|download|email\.|https?://|www\.', line, re.I)
        and (
            re.search(r'\b(skills?|expertise|proficient|using)\b', line, re.I)
            or _in_named_section(line, source, 'skills')
        )
        for line in source.splitlines()
    )
    if not contextual:
        contextual = any(
            canonical(skill.get('name')) == canonical(name)
            for skill in literal_observation(source)['skills']
            if '|' in skill.get('evidence', '')
            and re.search(
                r'\b(?:intern|engineer|developer|manager|consultant)\b.*(?:@| at )',
                skill['evidence'],
                re.I,
            )
        )
    return bool(
        name
        and len(name) <= 60
        and 'http' not in name.casefold()
        and re.search(r'[a-zA-Z]', name)
        and not SKILL_UI.search(name)
        and contextual
        and '|' not in name
        and '@' not in name
        and not FIELD_NOISE.search(name)
        and not re.search(r'tutorial|\.{2,}|…', name, re.I)
        and evidence_supported(evidence, source)
        and value_supported(name, evidence)
    )


def _dates_sane(item: dict) -> bool:
    years = []
    for field in ('start_date', 'end_date'):
        value = _clean_field(item.get(field))
        if not value:
            continue
        if not DATE_VALUE.fullmatch(value):
            return False
        match = re.search(r'(?:19|20)\d{2}', value)
        if match:
            years.append((field, int(match.group())))
    by_field = dict(years)
    current_year = datetime.now(UTC).year
    if any(year > current_year + 1 for _, year in years):
        return False
    return not (
        by_field.get('start_date')
        and by_field.get('end_date')
        and by_field['start_date'] > by_field['end_date']
    )


def _anchor_in_section(item: dict, source: str, field: str) -> bool:
    """Use the entry's identity, not unrelated words elsewhere in a screenshot."""

    title = item.get('title') or item.get('company')
    return bool(title) and any(
        heading_of(block.splitlines()[0]) in HEADINGS
        and field in fields
        and value_supported(title, block)
        and value_supported(item.get('company'), block)
        for fields, block in sections(source)
    )


def _certificate_role(item: dict, source: str) -> bool:
    return bool(COURSE_TITLE.search(item.get('title') or '')) or (
        _anchor_in_section(item, source, 'certifications')
        and not _anchor_in_section(item, source, 'experiences')
    )


def _experience_supported(item: dict, source: str) -> bool:
    title = _clean_field(item.get('title'))
    company = _clean_field(item.get('company'))
    evidence = item.get('evidence', '')
    explicit_job = _anchor_in_section(item, source, 'experiences')
    headline_job = bool(
        title
        and company
        and re.search(
            re.escape(canonical(title)) + r'\s*(?:@|at\s+)' + re.escape(canonical(company)),
            canonical(evidence),
        )
    )
    return bool(
        (title or company)
        and len(title) <= 255
        and len(company) <= 255
        and (title and company or _in_named_section(evidence, source, 'experience'))
        and not any(char in title + company for char in ('|', '@'))
        and not FIELD_NOISE.search(title + ' ' + company)
        and not _certificate_role(item, source)
        and (explicit_job or headline_job)
        and evidence_supported(evidence, source)
        and _all_values_supported(item)
    )


def _project_supported(item: dict, source: str) -> bool:
    name = _clean_field(item.get('name'))
    evidence = item.get('evidence', '')
    url = _clean_field(item.get('url'))
    return bool(
        name
        and not FIELD_NOISE.search(name + ' ' + (item.get('description') or ''))
        and 'linkedin.com/jobs' not in url.casefold()
        and evidence_supported(evidence, source)
        and _all_values_supported(item)
        and (PROJECT_CONTEXT.search(evidence) or _in_named_section(evidence, source, 'projects'))
    )


def _post_supported(item: dict, source: str) -> bool:
    text = _clean_field(item.get('text'))
    evidence = item.get('evidence', '')
    return bool(
        text
        and not FIELD_NOISE.search(text)
        and evidence_supported(evidence, source)
        and _all_values_supported(item)
        and (
            POST_CONTEXT.search(evidence)
            or _in_named_section(evidence, source, 'posts')
            or _in_named_section(evidence, source, 'activity')
        )
    )


def _generic_item_supported(item: dict, source: str) -> bool:
    evidence = item.get('evidence', '')
    return bool(
        item.get('name')
        and not FIELD_NOISE.search(item.get('name', ''))
        and not HEADING_COUNT.match(item.get('name', ''))
        and (
            CERTIFICATE_LANGUAGE.search(evidence)
            or COURSE_TITLE.search(item.get('name', ''))
            or _in_named_section(evidence, source, 'certifications')
            or _in_named_section(evidence, source, 'licenses & certifications')
        )
        and evidence_supported(evidence, source)
        and _all_values_supported(item)
    )


def _sanitize_item(item, source):
    """Drop unsupported fields individually; retain supported parts of the entry."""
    item = dict(item)
    evidence = item.get('evidence') or ''
    name = item.get('name')
    if name and evidence and not value_supported(name, evidence):
        # The model named a skill that is a whole line of the screenshot but quoted
        # another line: the line itself is the quote.
        line = next(
            (ln for ln in source.splitlines() if canonical(re.sub(r'^[^\w(]+', '', ln)) == canonical(name)),
            None,
        )
        if line:
            evidence = line
    if not evidence:
        anchor = next(
            (
                item.get(k)
                for k in ('institution', 'title', 'name', 'text', 'value', 'company')
                if item.get(k)
            ),
            None,
        )
        # Recover only a local source block around an exact anchor, never facts
        # elsewhere in the document that happen to share a date/organization.
        if anchor:
            for block in re.split(r'\n\s*\n', source):
                if len(block) <= 1200 and value_supported(anchor, block):
                    evidence = block
                    break
    if not evidence_supported(evidence, source):
        return None
    evidence = _widened(evidence, item, source)
    item['evidence'] = evidence
    for key, value in list(item.items()):
        if key == 'evidence':
            continue
        if isinstance(value, list):
            item[key] = [v for v in value if value_supported(v, evidence)]
        elif value and not value_supported(value, evidence):
            item[key] = None
    return item


WIDEN_LINES = 3


def _widened(evidence, item, source):
    """The quote and up to three lines right under it, when the entry's other values are there.

    A small model often quotes only an entry's first line ("Example University") while
    it read the degree and the years from the lines below. The quote grows line by line,
    never past a blank line (the next entry), and only as far as those values need.
    """
    missing = [
        value
        for key, value in item.items()
        if key != 'evidence' and isinstance(value, str) and value and not value_supported(value, evidence)
    ]
    quote = evidence.strip()
    start = source.find(quote) if quote else -1
    if not missing or start < 0:
        return evidence
    end = start + len(quote)
    for _ in range(WIDEN_LINES):
        if not source.startswith('\n', end):
            break
        line_end = source.find('\n', end + 1)
        line_end = len(source) if line_end < 0 else line_end
        if not source[end + 1 : line_end].strip():
            break
        end = line_end
        widened = source[start:end]
        if all(value_supported(value, widened) for value in missing):
            return widened
    return evidence


def validate_data(data, source, section='AUTO'):
    result = validate_node(
        {'observations': [{'image_id': 0, 'source_text': source, 'data': data, 'section': section}]}
    )
    return result['validated_observations'][0]['data']


def _sanitized_fields(obs, source):
    """The observation's allowed fields, each item cleaned against the OCR text."""
    data = {}
    for key, value in restrict_fields(obs['data'], obs.get('section', 'AUTO')).items():
        if isinstance(value, list):
            data[key] = [clean for item in value if (clean := _sanitize_item(item, source))]
        elif isinstance(value, dict):
            data[key] = _sanitize_item(value, source)
    return data


def _tidy_experiences(data, source):
    for item in data.get('experiences', []):
        # A course returned in the wrong model field belongs only in
        # certifications. Never turn its dates into employment dates.
        if _certificate_role(item, source) and item.get('title'):
            data.setdefault('certifications', []).append(
                {
                    'name': item['title'],
                    'issuer': item.get('company'),
                    'evidence': item['evidence'],
                }
            )
        title = item.get('title') or ''
        company = item.get('company') or ''
        if company:
            title = re.sub(r'\s+at\s+' + re.escape(company) + r'$', '', title, flags=re.I)
        # A single OCR icon before a known role is not a different job.
        title = re.sub(r'^[^\w]*[a-z]\s+(?=(?:Intern|Engineer|Developer)\b)', '', title, flags=re.I)
        item['title'] = title or None
        if not _dates_sane(item):
            item['start_date'] = item['end_date'] = None


def _tidy_education(data):
    for item in data.get('education', []):
        for field in ('start_date', 'end_date'):
            if item.get(field) and not DATE_VALUE.fullmatch(item[field]):
                item[field] = None
        degree = item.get('degree') or ''
        if ',' in degree:
            qualification, _, subject = degree.partition(',')
            if (
                not item.get('field_of_study')
                or subject.strip().casefold() == item['field_of_study'].casefold()
            ):
                item['degree'] = qualification.strip()
                item['field_of_study'] = subject.strip() or None


LOCATION_NOISE = r'\b(?:intern|engineer|company)\b|@'
# A place is words of letters ("Pune, Maharashtra, India", "Greater Boston Area") with
# a capitalised word of three letters or more; no digits or symbols.
PLACE = re.compile(r"(?=.*\b[A-Z][^\W\d_]{2,})[^\W\d_][^\W\d_ .,'’()&/-]*(?:[ .,'’()&/-]+[^\W\d_]+)*\.?")

# A section heading with its count ("Skills (12)", "Licenses & certifications (3)").
HEADING_COUNT = re.compile(r'^\W*[^\W\d_][^\d()]{0,40}\(\s*\d+\s*\)\W*$')


def _location_supported(item, source):
    if re.search(LOCATION_NOISE, item.get('evidence', ''), re.I):
        return False
    if not PLACE.fullmatch(_clean_field(item.get('value'))):
        return False  # "t 32 = ep": icons of the top menu read by OCR
    supporting_lines = [line for line in source.splitlines() if value_supported(item.get('value'), line)]
    return any(not re.search(LOCATION_NOISE + r'|\|', line, re.I) for line in supporting_lines)


def _scalar(field, item, source):
    """name, headline, about or location if the OCR text supports it, else None."""
    if not item:
        return None
    if field == 'name' and item.get('value'):
        # OCR reads LinkedIn's verified badge after the name as "v",
        # "y" or a symbol. Real initials are capitals and stay.
        item = {
            **item,
            'value': re.sub(r'(?<=\w)\s+(?:[vy]|[^\w\s])$', '', item['value'].strip()),
        }
    evidence = item.get('evidence', '')
    if field == 'about' and not (
        _in_named_section(evidence, source, 'about') or _in_named_section(evidence, source, 'summary')
    ):
        return None
    if field == 'name' and not _owner_name_supported(item.get('value'), source):
        return None
    if field == 'location' and not _location_supported(item, source):
        return None
    if field == 'headline' and _is_page_title(item.get('value')):
        # "top skills", "About", "Experience": a page or section title, never a headline.
        return None
    if (
        item.get('value')
        and value_supported(item.get('value'), evidence)
        and evidence_supported(evidence, source)
    ):
        return item
    return None


def _supported_fields(data, source):
    out = {field: _scalar(field, data.get(field), source) for field in SCALARS}
    out['skills'] = [item for item in data.get('skills', []) if _skill_supported(item, source)]
    out['experiences'] = [item for item in data.get('experiences', []) if _experience_supported(item, source)]
    out['education'] = []
    out['projects'] = [item for item in data.get('projects', []) if _project_supported(item, source)]
    out['posts'] = [item for item in data.get('posts', []) if _post_supported(item, source)]
    out['certifications'] = [
        item for item in data.get('certifications', []) if _generic_item_supported(item, source)
    ]
    for item in data.get('education', []):
        if _education_supported(item, source, out['certifications']):
            study = item.get('field_of_study')
            if study and not (item.get('degree') or item.get('start_date') or item.get('end_date')):
                # The profile card shows the school only: "Engineering" taken from
                # "College of Engineering" is not a field of study.
                if value_supported(study, item.get('institution') or ''):
                    item = {**item, 'field_of_study': None}
            out['education'].append(item)
    return out


SCALARS = ('name', 'headline', 'about', 'location')


def validate_node(state: ExtractionState):
    """Keep only observation fields that the OCR text of the same image supports."""
    validated = []
    for obs in state.get('observations', []):
        section = obs.get('section', 'AUTO')
        data = _sanitized_fields(obs, obs['source_text'])
        # Only category checks get this context. Evidence above is always
        # checked against the actual OCR, never against an invented heading.
        source = context_source(obs['source_text'], section)
        _tidy_experiences(data, source)
        _tidy_education(data)
        out = _supported_fields(data, source)
        validated.append({'image_id': obs['image_id'], 'data': restrict_fields(out, section)})
    return {'validated_observations': validated}


def _normalize_identity(value):
    # Ignore OCR separators, but do not conflate meaningful names such as
    # C++ and C# projects or employers.
    return re.sub(r'(?:[^\w+#/&]|_)+', ' ', canonical(value)).strip()


def _degree_identity(value):
    value = _normalize_identity(value)
    # Redundant abbreviation suffixes do not make a separate qualification.
    # Only known exact pairs are collapsed; different degrees/subjects remain.
    aliases = {
        'bachelor of engineering': ('be', 'b e'),
        'bachelor of science': ('bsc', 'b sc', 'bs', 'b s'),
        'master of science': ('msc', 'm sc', 'ms', 'm s'),
        'bachelor of technology': ('btech', 'b tech'),
        'master of technology': ('mtech', 'm tech'),
    }
    for full, abbreviations in aliases.items():
        if value in {full, *(full + ' ' + suffix for suffix in abbreviations)}:
            return full
    return value


def _records_match(left, right, identity_fields):
    # A shared date alone is never a person's school/job/project identity.
    anchors = [key for key in identity_fields if key not in {'start_date', 'end_date'}]
    if not any(left.get(key) and right.get(key) for key in anchors):
        return False
    compared = False
    for field in identity_fields:
        a, b = _normalize_identity(left.get(field)), _normalize_identity(right.get(field))
        if field == 'degree':
            a, b = _degree_identity(left.get(field)), _degree_identity(right.get(field))
        if a and b:
            compared = True
            if field in {'start_date', 'end_date', 'issue_date'}:
                # A year-only crop and a month/year crop can describe the same
                # date. Two different known months/years must remain distinct.
                if re.fullmatch(r'(?:19|20)\d{2}', a) and b.endswith(a):
                    continue
                if re.fullmatch(r'(?:19|20)\d{2}', b) and a.endswith(b):
                    continue
            if a != b:
                return False
    return compared


def _merge_records(items, identity_fields):
    out = []
    for item in items:
        existing = next((row for row in out if _records_match(row, item, identity_fields)), None)
        if existing is None:
            out.append(dict(item))
            continue
        for key, value in item.items():
            if not value:
                continue
            if isinstance(value, list):
                existing[key] = list(dict.fromkeys([*(existing.get(key) or []), *value]))
            elif not existing.get(key) or (
                key in {'description', 'text'} and len(str(value)) > len(str(existing[key]))
            ):
                existing[key] = value
            elif key in {'start_date', 'end_date', 'issue_date'} and len(str(value)) > len(
                str(existing[key])
            ):
                existing[key] = value
    return out
