"""LinkedIn screenshots of the owner's own profile, as OCR reads them (made-up people):
an entry quoted by its first line only, menu icons read as text, a section heading
with its count, and the Roman numeral III read as "Ill"."""

from agents_service.agents.linkedin.validation import clean_text, validate_data

EDUCATION = (
    'Education\n\n'
    '@ Example State University, Amravati\n'
    "Bachelor's of computer applications , BCA\n"
    '2017 - 2020\n\n'
    '@ Express.js, Front-End Development and +7 skills\n\n'
    'SampleCoach\n'
    'Jun 2023 - Sep 2023'
)


def school(**fields):
    return {
        'institution': None,
        'degree': None,
        'field_of_study': None,
        'start_date': None,
        'end_date': None,
        **fields,
    }


def test_an_entry_quoted_by_its_first_line_keeps_the_details_below_it():
    # Seen with qwen3:1.7b-q8_0: every value right, the quote only the school's line.
    item = school(
        institution='Example State University',
        degree="Bachelor's of computer applications",
        field_of_study='BCA',
        start_date='2017',
        end_date='2020',
        evidence='@ Example State University, Amravati',
    )
    [kept] = validate_data({'education': [item]}, EDUCATION)['education']
    assert (kept['degree'], kept['field_of_study'], kept['start_date'], kept['end_date']) == (
        "Bachelor's of computer applications",
        'BCA',
        '2017',
        '2020',
    )


def test_the_quote_never_grows_into_the_next_entry():
    # SampleCoach's dates are after a blank line: they are not the university's.
    item = school(
        institution='Example State University',
        start_date='Jun 2023',
        end_date='Sep 2023',
        evidence='@ Example State University, Amravati',
    )
    kept = validate_data({'education': [item]}, EDUCATION)['education']
    assert all(not entry['start_date'] and not entry['end_date'] for entry in kept)


HEADER = 't 32 = ep\nAsha Example\nProduct Designer\nPune, Maharashtra, India\n\nAbout\nI design clinic apps.'


def test_menu_icons_are_not_a_location():
    junk = validate_data({'location': {'value': 't 32 = ep', 'evidence': 't 32 = ep'}}, HEADER)
    assert junk['location'] is None
    place = 'Pune, Maharashtra, India'
    kept = validate_data({'location': {'value': place, 'evidence': place}}, HEADER)
    assert kept['location']['value'] == place


CERTIFICATES = (
    'Licenses & certifications (3)\n\n'
    'Level III Python Certification\nExample Robotics\nIssued Jan 2024\n\n'
    'Ckille (12)'
)


def test_a_section_heading_with_its_count_is_not_a_certificate():
    certificates = [
        {
            'name': 'Level III Python Certification',
            'evidence': 'Level III Python Certification\nExample Robotics',
        },
        {'name': 'Ckille (12)', 'evidence': 'Ckille (12)'},
    ]
    kept = validate_data({'certifications': certificates}, CERTIFICATES)['certifications']
    assert [c['name'] for c in kept] == ['Level III Python Certification']


def test_ocr_reads_roman_three_as_ill():
    text = clean_text(
        'Level Ill Python Certification\nPart Il exam\nLevel 1 course\nFelt Ill once', screenshot=True
    )
    assert text.splitlines() == [
        'Level III Python Certification',
        'Part II exam',
        'Level 1 course',
        'Felt Ill once',
    ]
    # Documents keep their own words.
    assert clean_text('Level Ill Python Certification') == 'Level Ill Python Certification'
