"""
Unit tests for PatientContextExtractor (task 5.1).

Covers:
- Demographics extraction: name (text / given+family fallback), DOB, gender, MRN
- Condition formatting: code.text → code.coding[0].display → "Unknown condition"
- Medication formatting: drug name + dosage; dosage omitted when absent
- Allergy formatting: substance, criticality, reaction; absent fields omitted
- Empty resource lists render as "None" section values
- extract() returns a non-empty string for any valid PatientResources
"""

import pytest

from src.context_extractor import PatientContextExtractor
from src.models import PatientResources


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def minimal_patient(**overrides) -> dict:
    """Return a minimal Patient resource dict, with optional field overrides."""
    base = {
        "resourceType": "Patient",
        "id": "patient-001",
        "gender": "male",
        "birthDate": "1980-06-15",
        "name": [{"text": "John Doe"}],
        "identifier": [
            {
                "type": {"coding": [{"code": "MR"}]},
                "value": "MRN12345",
            }
        ],
    }
    base.update(overrides)
    return base


def make_resources(**kwargs) -> PatientResources:
    """Build a PatientResources with a minimal patient and empty lists by default."""
    defaults = dict(
        patient=minimal_patient(),
        conditions=[],
        medications=[],
        allergies=[],
        observations=[],
        encounters=[],
        care_plans=[],
    )
    defaults.update(kwargs)
    return PatientResources(**defaults)


extractor = PatientContextExtractor()


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------

class TestDemographics:
    def test_name_from_text_field(self):
        resources = make_resources(patient=minimal_patient())
        output = extractor.extract(resources)
        assert "Name: John Doe" in output

    def test_name_from_given_and_family_when_text_absent(self):
        patient = minimal_patient()
        patient["name"] = [{"given": ["Jane"], "family": "Smith"}]
        resources = make_resources(patient=patient)
        output = extractor.extract(resources)
        assert "Name: Jane Smith" in output

    def test_name_given_only(self):
        patient = minimal_patient()
        patient["name"] = [{"given": ["Alice"]}]
        resources = make_resources(patient=patient)
        output = extractor.extract(resources)
        assert "Name: Alice" in output

    def test_name_family_only(self):
        patient = minimal_patient()
        patient["name"] = [{"family": "Jones"}]
        resources = make_resources(patient=patient)
        output = extractor.extract(resources)
        assert "Name: Jones" in output

    def test_name_unknown_when_no_name_field(self):
        patient = minimal_patient()
        patient.pop("name")
        resources = make_resources(patient=patient)
        output = extractor.extract(resources)
        assert "Name: Unknown" in output

    def test_birth_date_present(self):
        resources = make_resources()
        output = extractor.extract(resources)
        assert "DOB: 1980-06-15" in output

    def test_birth_date_missing_shows_unknown(self):
        patient = minimal_patient()
        patient.pop("birthDate")
        resources = make_resources(patient=patient)
        output = extractor.extract(resources)
        assert "DOB: Unknown" in output

    def test_gender_present(self):
        resources = make_resources()
        output = extractor.extract(resources)
        assert "Gender: male" in output

    def test_gender_missing_shows_unknown(self):
        patient = minimal_patient()
        patient.pop("gender")
        resources = make_resources(patient=patient)
        output = extractor.extract(resources)
        assert "Gender: Unknown" in output

    def test_mrn_extracted_from_mr_identifier(self):
        resources = make_resources()
        output = extractor.extract(resources)
        assert "MRN: MRN12345" in output

    def test_mrn_unknown_when_no_matching_identifier(self):
        patient = minimal_patient()
        patient["identifier"] = [
            {
                "type": {"coding": [{"code": "SS"}]},
                "value": "999-99-9999",
            }
        ]
        resources = make_resources(patient=patient)
        output = extractor.extract(resources)
        assert "MRN: Unknown" in output

    def test_mrn_unknown_when_no_identifier_field(self):
        patient = minimal_patient()
        patient.pop("identifier")
        resources = make_resources(patient=patient)
        output = extractor.extract(resources)
        assert "MRN: Unknown" in output


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

class TestConditions:
    def test_condition_using_code_text(self):
        condition = {
            "resourceType": "Condition",
            "code": {
                "text": "Type 2 Diabetes",
                "coding": [{"display": "Diabetes mellitus type 2"}],
            },
        }
        resources = make_resources(conditions=[condition])
        output = extractor.extract(resources)
        assert "Type 2 Diabetes" in output

    def test_condition_fallback_to_coding_display(self):
        condition = {
            "resourceType": "Condition",
            "code": {
                "coding": [{"display": "Hypertension"}],
            },
        }
        resources = make_resources(conditions=[condition])
        output = extractor.extract(resources)
        assert "Hypertension" in output

    def test_condition_fallback_to_unknown(self):
        condition = {"resourceType": "Condition", "code": {}}
        resources = make_resources(conditions=[condition])
        output = extractor.extract(resources)
        assert "Unknown condition" in output

    def test_empty_conditions_renders_none(self):
        resources = make_resources(conditions=[])
        output = extractor.extract(resources)
        assert "=== Active Conditions ===" in output
        # The word "None" should appear after the section header
        section_start = output.index("=== Active Conditions ===")
        section_excerpt = output[section_start:section_start + 80]
        assert "None" in section_excerpt

    def test_multiple_conditions_all_rendered(self):
        conditions = [
            {"resourceType": "Condition", "code": {"text": "Asthma"}},
            {"resourceType": "Condition", "code": {"text": "Hypertension"}},
        ]
        resources = make_resources(conditions=conditions)
        output = extractor.extract(resources)
        assert "Asthma" in output
        assert "Hypertension" in output


# ---------------------------------------------------------------------------
# Medications
# ---------------------------------------------------------------------------

class TestMedications:
    def test_medication_with_drug_name_and_dosage(self):
        med = {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"text": "Metformin"},
            "dosageInstruction": [{"text": "500mg twice daily"}],
        }
        resources = make_resources(medications=[med])
        output = extractor.extract(resources)
        assert "Metformin" in output
        assert "500mg twice daily" in output

    def test_medication_without_dosage_omits_dosage_field(self):
        med = {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"text": "Lisinopril"},
        }
        resources = make_resources(medications=[med])
        output = extractor.extract(resources)
        assert "Lisinopril" in output
        # No colon-separated dosage should follow the drug name on that line
        for line in output.splitlines():
            if "Lisinopril" in line:
                assert ":" not in line.split("Lisinopril")[1]

    def test_medication_with_empty_dosage_instruction_list(self):
        med = {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"text": "Aspirin"},
            "dosageInstruction": [],
        }
        resources = make_resources(medications=[med])
        output = extractor.extract(resources)
        assert "Aspirin" in output

    def test_empty_medications_renders_none(self):
        resources = make_resources(medications=[])
        output = extractor.extract(resources)
        section_start = output.index("=== Active Medications ===")
        section_excerpt = output[section_start:section_start + 80]
        assert "None" in section_excerpt


# ---------------------------------------------------------------------------
# Allergies
# ---------------------------------------------------------------------------

class TestAllergies:
    def test_allergy_with_all_fields(self):
        allergy = {
            "resourceType": "AllergyIntolerance",
            "code": {"text": "Penicillin"},
            "criticality": "high",
            "reaction": [
                {"manifestation": [{"text": "Anaphylaxis"}]}
            ],
        }
        resources = make_resources(allergies=[allergy])
        output = extractor.extract(resources)
        assert "Penicillin" in output
        assert "high" in output
        assert "Anaphylaxis" in output

    def test_allergy_substance_only(self):
        allergy = {
            "resourceType": "AllergyIntolerance",
            "code": {"text": "Sulfa drugs"},
        }
        resources = make_resources(allergies=[allergy])
        output = extractor.extract(resources)
        assert "Sulfa drugs" in output
        # criticality and reaction should not appear
        assert "criticality" not in output
        assert "reaction" not in output

    def test_allergy_missing_reaction_text_omitted(self):
        allergy = {
            "resourceType": "AllergyIntolerance",
            "code": {"text": "Latex"},
            "criticality": "low",
            "reaction": [{"manifestation": [{}]}],  # no 'text' key
        }
        resources = make_resources(allergies=[allergy])
        output = extractor.extract(resources)
        assert "Latex" in output
        assert "low" in output
        # "reaction:" label must not appear since text is absent
        assert "reaction:" not in output

    def test_allergy_missing_criticality_omitted(self):
        allergy = {
            "resourceType": "AllergyIntolerance",
            "code": {"text": "Aspirin"},
            "reaction": [{"manifestation": [{"text": "Hives"}]}],
        }
        resources = make_resources(allergies=[allergy])
        output = extractor.extract(resources)
        assert "Aspirin" in output
        assert "Hives" in output
        assert "criticality" not in output

    def test_empty_allergies_renders_none(self):
        resources = make_resources(allergies=[])
        output = extractor.extract(resources)
        section_start = output.index("=== Allergies ===")
        section_excerpt = output[section_start:section_start + 60]
        assert "None" in section_excerpt


# ---------------------------------------------------------------------------
# General contract
# ---------------------------------------------------------------------------

class TestExtractContract:
    def test_returns_non_empty_string(self):
        resources = make_resources()
        output = extractor.extract(resources)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_all_section_headers_present(self):
        resources = make_resources()
        output = extractor.extract(resources)
        for header in [
            "=== Patient Demographics ===",
            "=== Active Conditions ===",
            "=== Active Medications ===",
            "=== Allergies ===",
            "=== Recent Observations ===",
            "=== Recent Encounters ===",
            "=== Care Plan ===",
        ]:
            assert header in output, f"Missing section header: {header}"

    def test_does_not_mutate_input(self):
        """extract() must not modify the PatientResources input."""
        import copy
        conditions = [{"resourceType": "Condition", "code": {"text": "Asthma"}}]
        medications = [{"resourceType": "MedicationRequest",
                        "medicationCodeableConcept": {"text": "Albuterol"}}]
        resources = make_resources(conditions=conditions, medications=medications)

        original = copy.deepcopy(resources)
        extractor.extract(resources)

        assert resources.patient == original.patient
        assert resources.conditions == original.conditions
        assert resources.medications == original.medications
        assert resources.allergies == original.allergies
        assert resources.observations == original.observations
        assert resources.encounters == original.encounters
        assert resources.care_plans == original.care_plans
