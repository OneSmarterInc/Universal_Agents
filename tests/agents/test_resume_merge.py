"""Multi-chunk CV regressions with synthetic responses, not live model quality."""

import copy
import re
import tempfile
from pathlib import Path
from unittest import TestCase as SimpleTestCase
from unittest.mock import patch

from agents_service.agents.cv.evidence import supported_resume, supported_variants
from agents_service.agents.cv.local_reader import extract_local_resume
from agents_service.ai.providers import AIError, Completion
from tests.agents.kormic_adapter import extract


class ResumeMergeTests(SimpleTestCase):
    def merge(self, field, values, quotes=None):
        quotes = quotes or [str(value) for value in values]
        with (
            patch('agents_service.agents.cv.local_reader.extract_owner', return_value=[]),
            patch('agents_service.agents.cv.local_reader.GROUPS', [(field,)]),
            patch('agents_service.agents.cv.local_reader.chunks', return_value=quotes),
            patch('agents_service.agents.cv.local_reader.provider') as factory,
        ):
            factory.return_value.__enter__.return_value.generate.side_effect = [
                Completion(data={field: value}) for value in values
            ]
            return extract_local_resume('\n'.join(quotes), {})

    def test_multiple_universities_are_retained_not_arbitrarily_selected(self):
        result = self.merge('institution', ['Example University', 'Sample College', 'Example University'])
        self.assertIsNone(result['institution'])
        self.assertEqual(
            [e['value'] for e in result['_scalar_variants']['institution']],
            ['Example University', 'Sample College'],
        )

    def test_each_non_identity_scalar_can_have_reviewable_alternatives(self):
        cases = {
            'email': ['one@example.com', 'two@example.com'],
            'phone': ['1234567890', '2345678901'],
            'location': ['London', 'Paris'],
            'major': ['Physics', 'Chemistry'],
            'graduation_year': [2020, 2024],
            'gpa': [3.2, 3.8],
            'gpa_scale': ['4.0', '5.0'],
            'summary': ['Python developer.', 'Python developer'],
        }
        # A scale must be written as a scale in the CV (see test_a_gpa_scale_needs_scale_wording).
        quotes = {'gpa_scale': ['GPA 3.2/4.0', 'GPA 4.1/5.0']}
        for field, values in cases.items():
            with self.subTest(field=field):
                result = self.merge(field, values, quotes.get(field))
                self.assertIsNone(result[field])
                self.assertEqual([e['value'] for e in result['_scalar_variants'][field]], values)

    def test_invented_scalar_is_not_promoted_to_alternative(self):
        result = self.merge(
            'institution',
            ['Example University', 'Invented College'],
            ['Example University', 'Nothing relevant here'],
        )
        self.assertEqual(result['institution'], 'Example University')
        self.assertNotIn('_scalar_variants', result)

    def test_pipeline_emits_alternatives_with_field_labels_and_all_evidence(self):
        quotes = [
            'Education Example University',
            'Education Sample College',
            'Postgraduate Example University',
        ]
        result = self.merge(
            'institution', ['Example University', 'Sample College', 'Example University'], quotes
        )
        result['name'] = 'Test Candidate'
        source = 'Test Candidate\nEducation\n' + '\n'.join(quotes)
        result['_owner_candidates'] = [{'value': 'Test Candidate', 'evidence': 'Test Candidate'}]
        checked = supported_resume(copy.deepcopy(result), source)
        self.assertEqual(checked['_scalar_variants'], result['_scalar_variants'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cv.txt'
            path.write_text(source)
            with patch('tests.agents.kormic_adapter.read_document', return_value=source):
                output = extract([path], 'cv', {'full_name': 'Test Candidate'}, provider=lambda *_: result)
        self.assertEqual(output['status'], 'ready')
        records = [r for r in output['records'] if r['section'] == 'institution']
        self.assertEqual([r['value'] for r in records], ['Example University', 'Sample College'])
        self.assertEqual([s['evidence'] for s in records[0]['sources']], [quotes[0], quotes[2]])
        self.assertTrue(any('Multiple institution' in w for w in output['warnings']))
        self.assertFalse(any(r['section'] == '_scalar_variants' for r in output['records']))

    def test_untrusted_alternatives_cannot_bypass_evidence_or_name_checks(self):
        with self.assertRaises(AIError):
            supported_variants(
                {'name': [{'value': 'Other Person', 'evidence': 'Other Person'}]}, 'Other Person'
            )
        self.assertEqual(
            supported_variants(
                {
                    'institution': [
                        {'value': 'Invented College', 'evidence': 'Example University'},
                        {'value': 'Example University', 'evidence': 'Invented quotation'},
                    ]
                },
                'Example University',
            ),
            {},
        )

    def test_failed_name_merge_has_safe_diagnostic_and_no_partial_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cv.txt'
            path.write_text('Test Candidate')
            with patch('tests.agents.kormic_adapter.read_document', return_value='Test Candidate'):

                def provider(*_):
                    raise AIError('quality', field='conflicting CV name')

                output = extract([path], 'cv', {'full_name': 'Test Candidate'}, provider=provider)
        self.assertEqual(output['status'], 'failed')
        self.assertEqual(output['records'], [])
        self.assertEqual(
            output['files'][0]['diagnostic'],
            {
                'stage': 'provider',
                'exception_type': 'AIError',
                'reason': 'quality',
                'field': 'name',
            },
        )

    def test_diagnostic_does_not_echo_arbitrary_error_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cv.txt'
            path.write_text('Test Candidate')
            with patch('tests.agents.kormic_adapter.read_document', return_value='Test Candidate'):

                def provider(*_):
                    raise AIError('quality', field='private CV contents')

                output = extract([path], 'cv', {'full_name': 'Test Candidate'}, provider=provider)
        self.assertNotIn('private CV contents', str(output))


class SectionGateTests(SimpleTestCase):
    def test_absent_heading_families_are_skipped_and_others_kept(self):
        from agents_service.agents.cv.local_reader import section_may_exist

        chunk = 'Work Experience\nAnalyst | Example Co\n2020 - 2022'
        for field in ('publications', 'awards', 'volunteering', 'certifications', 'projects'):
            self.assertFalse(section_may_exist((field,), chunk))
        for fields in (
            ('education',),
            ('work_experience_entries',),
            ('other_information',),
            ('skills', 'languages'),
        ):
            self.assertTrue(section_may_exist(fields, chunk))
        self.assertTrue(section_may_exist(('projects',), 'Projects\nDashboard'))
        self.assertTrue(section_may_exist(('certifications',), 'Licenses\nRN licence'))


DEMO_CV = """Meerapriya
Clinical Data Analyst
Columbus, Ohio, United States | meerapriya@example.com | +1 202-555-0147
Professional Summary
Clinical data analyst with four years of experience.
Work Experience
Clinical Data Analyst | Northbridge Health Analytics
July 2024 - Present | Columbus, Ohio, United States
Built SQL reports for appointment volumes.
Created Power BI dashboards.
Junior Data Analyst | Cedar Vale Research Services
July 2022 - June 2024 | Dayton, Ohio, United States
Cleaned survey and service data.
Education
Bachelor of Science in Health Informatics
Riverstone College of Technology | August 2018 - May 2022
Skills
Python, SQL
Project
Clinic Appointment Dashboard | January 2024 - March 2024
Developed a dashboard using synthetic appointment data."""


def empty_reply(**kwargs):
    data = {}
    for key, spec in kwargs['schema'].get('properties', {}).items():
        types = {spec.get('type')} | {o.get('type') for o in spec.get('anyOf', [])}
        data[key] = [] if 'array' in types else None
    return Completion(data=data)


class LayoutRecordTests(SimpleTestCase):
    def test_empty_model_output_still_finds_jobs_projects_and_education(self):
        with (
            patch(
                'agents_service.agents.cv.local_reader.extract_owner',
                return_value=[{'value': 'Meerapriya', 'evidence': 'Meerapriya'}],
            ),
            patch('agents_service.agents.cv.local_reader.provider') as factory,
        ):
            factory.return_value.__enter__.return_value.generate.side_effect = empty_reply
            result = extract_local_resume(DEMO_CV, {})
        jobs = [
            (j['title'], j['organization'], j['start_date'], j['end_date'])
            for j in result['work_experience_entries']
        ]
        self.assertEqual(
            jobs,
            [
                ('Clinical Data Analyst', 'Northbridge Health Analytics', 'July 2024', 'Present'),
                ('Junior Data Analyst', 'Cedar Vale Research Services', 'July 2022', 'June 2024'),
            ],
        )
        self.assertIn('Created Power BI dashboards.', result['work_experience_entries'][0]['description'])
        self.assertNotIn('Cleaned', result['work_experience_entries'][0]['description'])
        self.assertEqual([p['title'] for p in result['projects']], ['Clinic Appointment Dashboard'])
        self.assertEqual(
            [e['organization'] for e in result['education']], ['Riverstone College of Technology']
        )
        self.assertTrue(
            all(
                r['evidence'] in DEMO_CV
                for key in ('work_experience_entries', 'projects', 'education')
                for r in result[key]
            )
        )

    def education_read_from(self, title, organization, education=None):
        """The education read when the model selects these words (first, last) of the degree line."""
        cv = 'CURRICULUM VITAE\nDaniel Okafor\nMechanical Engineer\nEducation\n' + (
            education or 'B.Eng. Mechanical Engineering, University of Lagos, 2012 - 2016'
        )

        def reply(**kwargs):
            if 'education' not in kwargs['schema'].get('properties', {}):
                return empty_reply(**kwargs)
            # The last token with each word: the degree line's.
            index = {
                word: int(number)
                for number, word in re.findall(r'^(\d+): (\S+)$', kwargs['messages'][-1]['content'], re.M)
            }

            def span(words):
                return {'first': index[words[0]], 'last': index[words[1]]} if words else None

            selection = {'title': span(title), 'organization': span(organization), 'start_date': None}
            return Completion(data={'education': [{**selection, 'end_date': None}]})

        with (
            patch(
                'agents_service.agents.cv.local_reader.extract_owner',
                return_value=[{'value': 'Daniel Okafor', 'evidence': 'Daniel Okafor'}],
            ),
            patch('agents_service.agents.cv.local_reader.provider') as factory,
        ):
            factory.return_value.__enter__.return_value.generate.side_effect = reply
            result = extract_local_resume(cv, {})
        return [(e['title'], e['organization']) for e in result['education']]

    def test_a_degree_the_checks_reject_falls_back_to_the_layout(self):
        # The degree's words but no university: the evidence check drops that record.
        self.assertEqual(
            self.education_read_from(('Mechanical', 'Engineering'), None),
            [('B.Eng. Mechanical Engineering', 'University of Lagos')],
        )

    def test_a_degree_cut_in_the_wrong_place_is_read_from_the_layout(self):
        # Seen with qwen3:1.7b-q8_0: "Engineering, University" and "of Lagos".
        self.assertEqual(
            self.education_read_from(('Engineering', 'University'), ('of', 'Lagos')),
            [('B.Eng. Mechanical Engineering', 'University of Lagos')],
        )

    def test_the_school_the_model_left_out_comes_from_the_layout(self):
        # "SGBAU" names no school word, so only the degree / school / dates layout finds it.
        self.assertEqual(
            self.education_read_from(
                ('Bachelor', 'Applications'),
                None,
                'Bachelor of Computer Applications\nSGBAU, Amravati, Maharashtra\nMay 2014 - May 2017',
            ),
            [('Bachelor of Computer Applications', 'SGBAU')],
        )

    def test_a_clean_model_reading_of_the_degree_is_kept(self):
        self.assertEqual(
            self.education_read_from(('Mechanical', 'Engineering'), ('University', 'Lagos')),
            [('Mechanical Engineering', 'University of Lagos')],
        )

    def test_layout_reader_cases(self):
        from agents_service.agents.cv.source_records import layout_entries

        two_line = layout_entries(
            'Experience\nSoftware Engineer\nSample Systems\n2021 - 2023\nBuilt APIs.\n\n'
            'Analyst at Example Co\nJan 2019 - Dec 2020',
            'work',
        )
        self.assertEqual(
            [(r['title'], r['organization']) for r in two_line],
            [('Software Engineer', 'Sample Systems'), ('Analyst', 'Example Co')],
        )
        self.assertEqual(layout_entries('Skills\nPython, SQL', 'work'), [])
        self.assertIsNone(layout_entries('Work Experience\nVarious freelance work', 'work'))
        self.assertIsNone(layout_entries('Experience\nEngineer | X | 2019 2020 2021', 'work'))


class NumericSupportTests(SimpleTestCase):
    def test_gpa_matches_as_a_number(self):
        from agents_service.agents.cv.evidence import value_in_source

        self.assertTrue(value_in_source(3.8, 'GPA: 3.80 / 4.00'))
        self.assertTrue(value_in_source(4.0, 'GPA 4.0'))
        self.assertTrue(value_in_source(4, 'Scale 4'))
        self.assertFalse(value_in_source(3.9, 'GPA: 3.80'))
        self.assertFalse(value_in_source(3.8, 'Room 13.8'))  # part of a larger number is not support
        self.assertTrue(value_in_source(3.8, 'GPA 3.8.'))  # end of a sentence


class GpaScaleTests(SimpleTestCase):
    def test_a_gpa_scale_needs_scale_wording(self):
        from agents_service.agents.cv.evidence import stated_scale

        assert not stated_scale('8.91', 'Current CGPA: 8.91')  # the GPA repeated is not a scale
        assert not stated_scale('10', 'Built 10 projects')
        for scale, text in [('10', 'CGPA 8.91/10'), ('10', '8.9 out of 10'), ('4.0', 'GPA: 3.8 / 4'),
                            ('4', 'GPA 3.8/4.0'), ('10', 'on a 10-point scale'), ('10', 'Scale: 10')]:  # fmt: skip
            assert stated_scale(scale, text), (scale, text)

    def test_a_repeated_gpa_is_dropped_as_a_scale(self):
        data = {'gpa': 8.91, 'gpa_scale': '8.91', 'warnings': []}
        result = supported_resume(data, 'Current CGPA: 8.91')
        assert result['gpa'] == 8.91 and result['gpa_scale'] is None
        kept = supported_resume({'gpa': 8.91, 'gpa_scale': '10', 'warnings': []}, 'CGPA: 8.91/10')
        assert kept['gpa_scale'] == '10'
