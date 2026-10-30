"""Keep only facts the document supports: every value must appear in its quotation, and every quotation in the document text."""

import re

from agents_service.agents.cv.owner import supported_owner
from agents_service.agents.cv.schemas import ResumeFacts
from agents_service.agents.cv.source_education import DEGREE, INSTITUTION
from agents_service.ai.providers import AIError
from agents_service.documents.matching import canonical, evidence_supported, value_supported

COLLECTIONS = (
    'education',
    'work_experience_entries',
    'projects',
    'publications',
    'certifications',
    'awards',
    'volunteering',
    'other_information',
)


REVIEW_SCALARS = (
    'email',
    'phone',
    'location',
    'summary',
    'institution',
    'major',
    'graduation_year',
    'gpa',
    'gpa_scale',
)


def supported_variants(variants, source):
    """Validate internal local-merge alternatives before emitting review records."""
    if not isinstance(variants, dict) or any(key not in REVIEW_SCALARS for key in variants):
        raise AIError('format')
    result = {}
    for field, entries in variants.items():
        if not isinstance(entries, list):
            raise AIError('format')
        kept = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {'value', 'evidence'}:
                raise AIError('format')
            value = getattr(ResumeFacts.model_validate({field: entry['value']}), field)
            quote = entry['evidence']
            if (
                value is not None
                and isinstance(quote, str)
                and quote.strip()
                and evidence_supported(quote, source)
                and value_supported(str(value), quote)
            ):
                kept.append({'value': value, 'evidence': quote})
        if kept:
            result[field] = kept
    return result


def supported_owners(entries, source):

    if not isinstance(entries, list) or len(entries) > 32:
        raise AIError('format')
    kept = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {'value', 'evidence'}:
            raise AIError('format')
        if supported_owner(entry['value'], entry['evidence'], source):
            kept.append(entry)
    return kept


def value_in_source(value, source):
    """Numbers match as numbers: a GPA of 3.8 is supported by "3.80", 4 by "4.0"."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        for match in re.finditer(r'(?<![\w.])\d+(?:\.\d+)?(?!\w|\.\d)', source):
            try:
                if abs(float(match.group()) - float(value)) < 1e-9:
                    return True
            except ValueError:
                continue
        return False
    return value_supported(str(value), source)


EMAIL = re.compile(r'^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$')
PROGRAMMING = re.compile(
    r'^\s*(?:python|java|javascript|typescript|c|c\+\+|c#|c/c\+\+|go|golang|rust|ruby|php|kotlin|swift|dart|'
    r'r|sql|mysql|html5?|css3?|scala|perl|matlab|bash|shell|solidity|julia)\b',
    re.I,
)
# Words then a bare number: "English 40" is a typing speed or a mark; "English C1",
# "French B2" and "English 90%" are kept.
BARE_SCORE = re.compile(r'^[^\W\d_][^\d]*?[\s:-]+\d{1,3}$')


def _is_link(value):
    text = str(value or '').strip()
    if EMAIL.match(text) or text.lower().startswith('mailto:'):
        return False
    if len(re.sub(r'\D', '', text)) >= 7 and not re.search(r'[a-z]{2,}', text, re.I):
        return False  # a phone number
    return True


def _words(text):
    return re.findall(r'[^\W_]+', canonical(text))


def _source_copy(key, value, source):
    """The CV's own wording of a summary the model copied with its own punctuation
    ("NumPy. Actively" for "NumPy Actively"): the same words in the same order, taken
    from the CV. None when the words are not there in that order."""
    if key != 'summary' or not isinstance(value, str):
        return None
    words = _words(value)
    if len(words) < 8:
        return None
    found = re.search(r'(?<![^\W_])' + r'[\W_]+'.join(map(re.escape, words)) + r'(?![^\W_])', source, re.I)
    return found.group() if found else None


def supported_resume(data, source):
    """Retain source-supported fields only, without truncating record collections."""
    warnings = list(data.get('warnings') or [])
    for key, value in list(data.items()):
        if key == 'warnings':
            continue
        if key == '_owner_candidates':
            data[key] = supported_owners(value, source)
            continue
        if key == '_scalar_variants':
            data[key] = supported_variants(value, source)
            continue
        if key in COLLECTIONS:
            kept = []
            for original in value:
                item = dict(original)
                quote = item.get('evidence', '')
                if not quote or not evidence_supported(quote, source):
                    warnings.append(f'An unsupported {key.replace("_", " ")} record was excluded.')
                    continue
                for field, val in list(item.items()):
                    if field == 'evidence' or not val:
                        continue
                    if isinstance(val, list):
                        item[field] = [v for v in val if value_supported(v, quote)]
                    elif not value_supported(str(val), quote):
                        item[field] = None
                        warnings.append(f'An unsupported {field.replace("_", " ")} was excluded.')
                if item.get('title'):
                    kept.append(item)
            data[key] = kept
        elif isinstance(value, list):
            data[key] = [v for v in value if value_supported(str(v), source)]
        elif value is not None and not value_in_source(value, source):
            copy = _source_copy(key, value, source)
            data[key] = copy
            if copy is None:
                warnings.append(f'An unsupported {key.replace("_", " ")} was excluded.')
    # An education entry names a qualification or a school. "(IT) 8" with two years
    # is the model joining fragments, even when each fragment is in the quote.
    data['education'] = [
        item
        for item in data.get('education') or []
        if DEGREE.search(item.get('title') or '')
        or INSTITUTION.search(item.get('title') or '')
        or INSTITUTION.search(item.get('organization') or '')
    ]
    # Links are web addresses or profile handles, never an email or a phone number.
    data['links'] = [link for link in data.get('links') or [] if _is_link(link)]
    # Languages are the languages a person speaks; programming languages are skills.
    # A name and a bare number ("English 40" under "Typing:") is a score, not a language.
    skills = {canonical(skill) for skill in data.get('skills') or []}
    data['languages'] = [
        language
        for language in data.get('languages') or []
        if canonical(language) not in skills
        and not PROGRAMMING.search(language)
        and not BARE_SCORE.match(language.strip())
    ]
    # Education belongs in education: a degree under other_information is the model
    # filing it in the wrong place, and the education reader has it already.
    data['other_information'] = [
        item for item in data.get('other_information') or [] if not DEGREE.search(item.get('title') or '')
    ]
    # A GPA scale must be written as a scale ("8.9/10", "out of 10", "10-point scale").
    # A model asked for a scale the CV does not give tends to repeat the GPA instead.
    if data.get('gpa_scale') is not None and not stated_scale(data['gpa_scale'], source):
        data['gpa_scale'] = None
        warnings.append('An unsupported gpa scale was excluded.')
    variants = data.get('_scalar_variants') or {}
    if 'gpa_scale' in variants:
        variants['gpa_scale'] = [v for v in variants['gpa_scale'] if stated_scale(v['value'], v['evidence'])]
        if not variants['gpa_scale']:
            del variants['gpa_scale']
    data['warnings'] = list(dict.fromkeys(warnings))
    return data


def stated_scale(scale, text):
    """True when the text writes this number as a grading scale."""
    text_scale = str(scale).strip()
    try:
        value = float(text_scale)
    except ValueError:
        return False
    # 4, 4.0 and 4.00 are the same scale.
    number = rf'{int(value)}(?:\.0+)?' if value.is_integer() else re.escape(text_scale)
    words = {4: 'four', 5: 'five', 7: 'seven', 10: 'ten', 20: 'twenty', 100: 'hundred'}
    if value.is_integer() and int(value) in words:
        number = rf'(?:{number}|{words[int(value)]})'
    after = r'(?![\d.])'
    patterns = (
        rf'/\s*{number}{after}',  # 8.9/10
        rf'\bout\s+of\s+{number}{after}',  # out of 10, out of ten
        rf'\bscale\s*(?:of|:)?\s*{number}{after}',  # scale of 10, Scale: 10
        rf'(?<![\d.]){number}\s*-?\s*(?:point\s+)?(?:scale|grading)',  # 10-point scale, (4.0 scale)
        rf'\d[.,]\d+\s+(?:on|of)\s+(?:a\s+)?{number}{after}',  # 3.6 on 4, 3.7 of 4.0, 9.1 on a 10
    )
    return any(re.search(pattern, text, re.I) for pattern in patterns)
