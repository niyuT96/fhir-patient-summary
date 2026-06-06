"""
Unit tests for PatientContextExtractor (task 5.1).

Covers:
- Demographics extraction: name (text / given+family fallback), DOB, gender, MRN
- Condition formatting: code.text -> code.coding[0].display -> "Unknown condition"
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


# ---------------------------------------------------------------------------
# Observations (task 5.2)
# ---------------------------------------------------------------------------

class TestObservations:
    def test_observation_formatted_correctly(self):
        obs = {
            "resourceType": "Observation",
            "code": {"text": "Heart Rate"},
            "valueQuantity": {"value": 72, "unit": "bpm"},
            "effectiveDateTime": "2024-03-15T10:00:00Z",
        }
        resources = make_resources(observations=[obs])
        output = extractor.extract(resources)
        assert "Heart Rate: 72 bpm (2024-03-15)" in output

    def test_observation_without_unit(self):
        obs = {
            "resourceType": "Observation",
            "code": {"text": "Pain Score"},
            "valueQuantity": {"value": 5},
            "effectiveDateTime": "2024-03-15T10:00:00Z",
        }
        resources = make_resources(observations=[obs])
        output = extractor.extract(resources)
        assert "Pain Score: 5" in output
        assert "(2024-03-15)" in output

    def test_observation_ordered_newest_first(self):
        obs1 = {
            "resourceType": "Observation",
            "code": {"text": "Glucose"},
            "valueQuantity": {"value": 100, "unit": "mg/dL"},
            "effectiveDateTime": "2024-01-01T00:00:00Z",
        }
        obs2 = {
            "resourceType": "Observation",
            "code": {"text": "Glucose"},
            "valueQuantity": {"value": 200, "unit": "mg/dL"},
            "effectiveDateTime": "2024-06-01T00:00:00Z",
        }
        resources = make_resources(observations=[obs1, obs2])
        output = extractor.extract(resources)
        pos_200 = output.index("200")
        pos_100 = output.index("100")
        assert pos_200 < pos_100, "Newer observation (200) should appear before older (100)"

    def test_only_10_most_recent_observations(self):
        obs_list = [
            {
                "resourceType": "Observation",
                "code": {"text": f"Test{i:02d}"},
                "valueQuantity": {"value": i, "unit": "u"},
                "effectiveDateTime": f"2024-{(i % 12) + 1:02d}-01T00:00:00Z",
            }
            for i in range(15)
        ]
        resources = make_resources(observations=obs_list)
        output = extractor.extract(resources)
        # At most 10 observation lines should appear
        obs_section_start = output.index("=== Recent Observations ===")
        obs_section_end = output.index("=== Recent Encounters ===")
        obs_section = output[obs_section_start:obs_section_end]
        obs_lines = [l for l in obs_section.splitlines() if l.startswith("- ")]
        assert len(obs_lines) <= 10

    def test_empty_observations_renders_none(self):
        resources = make_resources(observations=[])
        output = extractor.extract(resources)
        section_start = output.index("=== Recent Observations ===")
        section_excerpt = output[section_start:section_start + 80]
        assert "None" in section_excerpt

    def test_observation_without_value_skipped(self):
        obs_no_value = {
            "resourceType": "Observation",
            "code": {"text": "BloodPressure"},
            # no valueQuantity
            "effectiveDateTime": "2024-03-15T10:00:00Z",
        }
        obs_with_value = {
            "resourceType": "Observation",
            "code": {"text": "HeartRate"},
            "valueQuantity": {"value": 80, "unit": "bpm"},
            "effectiveDateTime": "2024-03-15T09:00:00Z",
        }
        resources = make_resources(observations=[obs_no_value, obs_with_value])
        output = extractor.extract(resources)
        # BloodPressure has no value so it should not show up
        assert "BloodPressure" not in output
        assert "HeartRate: 80 bpm" in output

    def test_input_observations_list_not_mutated(self):
        """The original observations list order must not be changed."""
        obs_old = {
            "resourceType": "Observation",
            "code": {"text": "Old"},
            "valueQuantity": {"value": 1, "unit": "u"},
            "effectiveDateTime": "2023-01-01T00:00:00Z",
        }
        obs_new = {
            "resourceType": "Observation",
            "code": {"text": "New"},
            "valueQuantity": {"value": 2, "unit": "u"},
            "effectiveDateTime": "2024-01-01T00:00:00Z",
        }
        resources = make_resources(observations=[obs_old, obs_new])
        original_order = list(resources.observations)
        extractor.extract(resources)
        assert resources.observations == original_order


# ---------------------------------------------------------------------------
# Encounters (task 5.2)
# ---------------------------------------------------------------------------

class TestEncounters:
    def test_encounter_formatted_with_all_fields(self):
        enc = {
            "resourceType": "Encounter",
            "type": [{"text": "Emergency Visit"}],
            "period": {"start": "2024-03-10T08:00:00Z"},
            "reasonCode": [{"text": "Chest pain"}],
        }
        resources = make_resources(encounters=[enc])
        output = extractor.extract(resources)
        assert "Emergency Visit" in output
        assert "2024-03-10" in output
        assert "Chest pain" in output

    def test_encounter_ordered_newest_first(self):
        enc1 = {
            "resourceType": "Encounter",
            "type": [{"text": "Visit A"}],
            "period": {"start": "2023-01-01T00:00:00Z"},
        }
        enc2 = {
            "resourceType": "Encounter",
            "type": [{"text": "Visit B"}],
            "period": {"start": "2024-06-01T00:00:00Z"},
        }
        resources = make_resources(encounters=[enc1, enc2])
        output = extractor.extract(resources)
        pos_b = output.index("Visit B")
        pos_a = output.index("Visit A")
        assert pos_b < pos_a, "Newer encounter (Visit B) should appear before older (Visit A)"

    def test_only_3_most_recent_encounters(self):
        enc_list = [
            {
                "resourceType": "Encounter",
                "type": [{"text": f"Visit{i}"}],
                "period": {"start": f"2024-{i:02d}-01T00:00:00Z"},
            }
            for i in range(1, 6)
        ]
        resources = make_resources(encounters=enc_list)
        output = extractor.extract(resources)
        enc_section_start = output.index("=== Recent Encounters ===")
        enc_section_end = output.index("=== Care Plan ===")
        enc_section = output[enc_section_start:enc_section_end]
        enc_lines = [l for l in enc_section.splitlines() if l.startswith("- ")]
        assert len(enc_lines) <= 3

    def test_encounter_absent_fields_omitted(self):
        enc = {
            "resourceType": "Encounter",
            "type": [{"text": "Office Visit"}],
            # no period, no reasonCode
        }
        resources = make_resources(encounters=[enc])
        output = extractor.extract(resources)
        assert "Office Visit" in output
        # No date or reason should appear
        assert "reason:" not in output

    def test_empty_encounters_renders_none(self):
        resources = make_resources(encounters=[])
        output = extractor.extract(resources)
        section_start = output.index("=== Recent Encounters ===")
        section_excerpt = output[section_start:section_start + 80]
        assert "None" in section_excerpt

    def test_input_encounters_list_not_mutated(self):
        enc_old = {
            "resourceType": "Encounter",
            "type": [{"text": "Old Visit"}],
            "period": {"start": "2022-01-01T00:00:00Z"},
        }
        enc_new = {
            "resourceType": "Encounter",
            "type": [{"text": "New Visit"}],
            "period": {"start": "2024-01-01T00:00:00Z"},
        }
        resources = make_resources(encounters=[enc_old, enc_new])
        original_order = list(resources.encounters)
        extractor.extract(resources)
        assert resources.encounters == original_order


# ---------------------------------------------------------------------------
# Care Plan (task 5.2)
# ---------------------------------------------------------------------------

class TestCarePlan:
    def test_care_plan_activity_description_rendered(self):
        plan = {
            "resourceType": "CarePlan",
            "status": "active",
            "activity": [
                {"detail": {"description": "Walk 30 minutes daily"}},
                {"detail": {"description": "Monitor blood pressure weekly"}},
            ],
        }
        resources = make_resources(care_plans=[plan])
        output = extractor.extract(resources)
        assert "Walk 30 minutes daily" in output
        assert "Monitor blood pressure weekly" in output

    def test_care_plan_no_activities_renders_none(self):
        plan = {
            "resourceType": "CarePlan",
            "status": "active",
            "activity": [],
        }
        resources = make_resources(care_plans=[plan])
        output = extractor.extract(resources)
        section_start = output.index("=== Care Plan ===")
        section_excerpt = output[section_start:section_start + 80]
        assert "None" in section_excerpt

    def test_empty_care_plans_renders_none(self):
        resources = make_resources(care_plans=[])
        output = extractor.extract(resources)
        section_start = output.index("=== Care Plan ===")
        section_excerpt = output[section_start:section_start + 80]
        assert "None" in section_excerpt

    def test_care_plan_activity_without_description_skipped(self):
        plan = {
            "resourceType": "CarePlan",
            "status": "active",
            "activity": [
                {"detail": {}},  # no description
                {"detail": {"description": "Take medication as prescribed"}},
            ],
        }
        resources = make_resources(care_plans=[plan])
        output = extractor.extract(resources)
        assert "Take medication as prescribed" in output


# ---------------------------------------------------------------------------
# Token budget enforcement (task 5.2)
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def test_output_within_3000_tokens_for_large_input(self):
        """Token count must never exceed 3,000 even with many resources."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        # Generate many observations, encounters, conditions, medications, allergies
        many_obs = [
            {
                "resourceType": "Observation",
                "code": {"text": f"Lab test number {i} with a fairly long name to inflate tokens"},
                "valueQuantity": {"value": i * 1.23456, "unit": "units/mL"},
                "effectiveDateTime": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}T10:00:00Z",
            }
            for i in range(50)
        ]
        many_enc = [
            {
                "resourceType": "Encounter",
                "type": [{"text": f"Encounter type {i} detailed description here"}],
                "period": {"start": f"2024-{(i % 12) + 1:02d}-01T00:00:00Z"},
                "reasonCode": [{"text": f"Reason for visit {i} with extended clinical notes"}],
            }
            for i in range(20)
        ]
        many_cond = [
            {
                "resourceType": "Condition",
                "code": {"text": f"Active chronic diagnosis number {i} with ICD-10 coding"},
            }
            for i in range(30)
        ]
        many_meds = [
            {
                "resourceType": "MedicationRequest",
                "medicationCodeableConcept": {"text": f"Medication {i} 100mg extended release"},
                "dosageInstruction": [{"text": f"Take 2 tablets by mouth twice daily with food {i}"}],
            }
            for i in range(30)
        ]
        many_allerg = [
            {
                "resourceType": "AllergyIntolerance",
                "code": {"text": f"Allergen substance {i}"},
                "criticality": "high",
                "reaction": [{"manifestation": [{"text": f"Severe reaction type {i}"}]}],
            }
            for i in range(20)
        ]
        many_plans = [
            {
                "resourceType": "CarePlan",
                "status": "active",
                "activity": [
                    {"detail": {"description": f"Care activity {j} for plan {i}: detailed instructions"}}
                    for j in range(10)
                ],
            }
            for i in range(5)
        ]

        resources = make_resources(
            observations=many_obs,
            encounters=many_enc,
            conditions=many_cond,
            medications=many_meds,
            allergies=many_allerg,
            care_plans=many_plans,
        )
        output = extractor.extract(resources)
        token_count = len(enc.encode(output))
        assert token_count <= 3000, (
            f"Token count {token_count} exceeds 3,000 budget"
        )

    def test_demographics_always_present_after_truncation(self):
        """Demographics section must survive even extreme truncation."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        # Fill with many long conditions to force truncation
        many_cond = [
            {
                "resourceType": "Condition",
                "code": {"text": "A" * 200},  # very long condition name
            }
            for _ in range(100)
        ]
        resources = make_resources(conditions=many_cond)
        output = extractor.extract(resources)
        assert "=== Patient Demographics ===" in output
        assert "Name: John Doe" in output
        token_count = len(enc.encode(output))
        assert token_count <= 3000

    def test_truncation_removes_careplan_activities_first(self):
        """When over budget, CarePlan activities should be removed before encounters."""
        # Build a context that is just barely over budget only when care plan is added
        # Use moderate data that fits in budget without care plan activities
        care_plan_with_many_activities = {
            "resourceType": "CarePlan",
            "status": "active",
            "activity": [
                {"detail": {"description": "B" * 300}}
                for _ in range(30)
            ],
        }
        resources = make_resources(
            encounters=[
                {
                    "resourceType": "Encounter",
                    "type": [{"text": "Important Encounter"}],
                    "period": {"start": "2024-01-01T00:00:00Z"},
                }
            ],
            care_plans=[care_plan_with_many_activities],
        )
        output = extractor.extract(resources)
        # The encounter should survive even if care plan activities are truncated
        assert "Important Encounter" in output

    def test_normal_sized_input_within_budget(self):
        """A typical patient record should comfortably fit within 3,000 tokens."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        resources = make_resources(
            conditions=[{"resourceType": "Condition", "code": {"text": "Hypertension"}}],
            medications=[{
                "resourceType": "MedicationRequest",
                "medicationCodeableConcept": {"text": "Lisinopril"},
                "dosageInstruction": [{"text": "10mg once daily"}],
            }],
            allergies=[{
                "resourceType": "AllergyIntolerance",
                "code": {"text": "Penicillin"},
                "criticality": "high",
            }],
            observations=[{
                "resourceType": "Observation",
                "code": {"text": "Blood Pressure"},
                "valueQuantity": {"value": 130, "unit": "mmHg"},
                "effectiveDateTime": "2024-03-01T00:00:00Z",
            }],
        )
        output = extractor.extract(resources)
        token_count = len(enc.encode(output))
        assert token_count <= 3000, (
            f"Token count {token_count} exceeds 3,000 budget for normal input"
        )
