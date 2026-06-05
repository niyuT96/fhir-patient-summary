"""
Unit tests for _fetch_all_fhir_resources() (task 7.2).

Covers Requirements 7.1–7.5:
- 7.1  Non-Patient FHIRClientError/FHIRUnavailableError → warn, set to [], continue
- 7.2  Patient list empty → SummaryResult with "Patient {id} not found"
- 7.3  Patient fetch raises error → SummaryResult with "Failed to fetch Patient {id}: …"
- 7.4  Failed types default to []; successfully fetched types retain values
- 7.5  Previously fetched results preserved across failures
"""

from unittest.mock import MagicMock, call, patch

import pytest

from src.agent import _fetch_all_fhir_resources
from src.exceptions import FHIRClientError, FHIRUnavailableError
from src.models import PatientResources, SummaryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATIENT = {"resourceType": "Patient", "id": "p1", "name": [{"text": "Jane Doe"}]}
_CONDITION = {"resourceType": "Condition", "id": "c1"}
_MEDICATION = {"resourceType": "MedicationRequest", "id": "m1"}
_ALLERGY = {"resourceType": "AllergyIntolerance", "id": "a1"}
_OBSERVATION = {"resourceType": "Observation", "id": "o1"}
_ENCOUNTER = {"resourceType": "Encounter", "id": "e1"}
_CARE_PLAN = {"resourceType": "CarePlan", "id": "cp1"}


def _make_client_returns(responses: dict[str, list[dict]]) -> MagicMock:
    """Build a mock FHIRClient whose get_resource() returns values keyed by resource_type."""

    def _side_effect(resource_type: str, patient_id: str, params: dict | None = None):
        return responses[resource_type]

    client = MagicMock()
    client.get_resource.side_effect = _side_effect
    return client


def _make_client_all_happy(patient_id: str = "p1") -> MagicMock:
    """Return a client that successfully returns one resource per type."""
    return _make_client_returns({
        "Patient":            [_PATIENT],
        "Condition":          [_CONDITION],
        "MedicationRequest":  [_MEDICATION],
        "AllergyIntolerance": [_ALLERGY],
        "Observation":        [_OBSERVATION],
        "Encounter":          [_ENCOUNTER],
        "CarePlan":           [_CARE_PLAN],
    })


# ---------------------------------------------------------------------------
# Happy-path: successful fetch assembles PatientResources correctly
# ---------------------------------------------------------------------------

class TestSuccessfulFetch:
    def test_returns_patient_resources_on_success(self):
        client = _make_client_all_happy()
        result = _fetch_all_fhir_resources(client, "p1")
        assert isinstance(result, PatientResources)

    def test_patient_field_is_first_patient_resource(self):
        client = _make_client_all_happy()
        result = _fetch_all_fhir_resources(client, "p1")
        assert isinstance(result, PatientResources)
        assert result.patient == _PATIENT

    def test_all_seven_resource_types_are_fetched(self):
        client = _make_client_all_happy()
        _fetch_all_fhir_resources(client, "p1")
        # Exactly 7 calls, one per resource type
        assert client.get_resource.call_count == 7

    def test_resource_types_fetched_in_correct_order(self):
        client = _make_client_all_happy()
        _fetch_all_fhir_resources(client, "p1")
        expected_order = [
            "Patient", "Condition", "MedicationRequest",
            "AllergyIntolerance", "Observation", "Encounter", "CarePlan",
        ]
        actual_order = [c.args[0] for c in client.get_resource.call_args_list]
        assert actual_order == expected_order

    def test_conditions_populated(self):
        client = _make_client_all_happy()
        result = _fetch_all_fhir_resources(client, "p1")
        assert isinstance(result, PatientResources)
        assert result.conditions == [_CONDITION]

    def test_medications_populated(self):
        client = _make_client_all_happy()
        result = _fetch_all_fhir_resources(client, "p1")
        assert isinstance(result, PatientResources)
        assert result.medications == [_MEDICATION]

    def test_allergies_populated(self):
        client = _make_client_all_happy()
        result = _fetch_all_fhir_resources(client, "p1")
        assert isinstance(result, PatientResources)
        assert result.allergies == [_ALLERGY]

    def test_observations_populated(self):
        client = _make_client_all_happy()
        result = _fetch_all_fhir_resources(client, "p1")
        assert isinstance(result, PatientResources)
        assert result.observations == [_OBSERVATION]

    def test_encounters_populated(self):
        client = _make_client_all_happy()
        result = _fetch_all_fhir_resources(client, "p1")
        assert isinstance(result, PatientResources)
        assert result.encounters == [_ENCOUNTER]

    def test_care_plans_populated(self):
        client = _make_client_all_happy()
        result = _fetch_all_fhir_resources(client, "p1")
        assert isinstance(result, PatientResources)
        assert result.care_plans == [_CARE_PLAN]


# ---------------------------------------------------------------------------
# Query parameter contract
# ---------------------------------------------------------------------------

class TestQueryParameters:
    def test_patient_fetch_uses_id_param(self):
        client = _make_client_all_happy()
        _fetch_all_fhir_resources(client, "patient-abc")
        patient_call = client.get_resource.call_args_list[0]
        assert patient_call.args[0] == "Patient"
        params = patient_call.args[2]
        assert params.get("_id") == "patient-abc"

    def test_condition_fetch_uses_patient_param_and_clinical_status(self):
        client = _make_client_all_happy()
        _fetch_all_fhir_resources(client, "patient-abc")
        cond_call = client.get_resource.call_args_list[1]
        params = cond_call.args[2]
        assert params.get("patient") == "patient-abc"
        assert params.get("clinical-status") == "active"

    def test_medication_request_fetch_uses_status_active(self):
        client = _make_client_all_happy()
        _fetch_all_fhir_resources(client, "patient-abc")
        med_call = client.get_resource.call_args_list[2]
        params = med_call.args[2]
        assert params.get("status") == "active"
        assert params.get("patient") == "patient-abc"

    def test_observation_fetch_uses_sort_and_count(self):
        client = _make_client_all_happy()
        _fetch_all_fhir_resources(client, "patient-abc")
        obs_call = client.get_resource.call_args_list[4]
        params = obs_call.args[2]
        assert params.get("_sort") == "-date"
        assert params.get("_count") == "20"

    def test_encounter_fetch_uses_sort_and_count_5(self):
        client = _make_client_all_happy()
        _fetch_all_fhir_resources(client, "patient-abc")
        enc_call = client.get_resource.call_args_list[5]
        params = enc_call.args[2]
        assert params.get("_sort") == "-date"
        assert params.get("_count") == "5"

    def test_care_plan_fetch_uses_status_active(self):
        client = _make_client_all_happy()
        _fetch_all_fhir_resources(client, "patient-abc")
        cp_call = client.get_resource.call_args_list[6]
        params = cp_call.args[2]
        assert params.get("status") == "active"
        assert params.get("patient") == "patient-abc"


# ---------------------------------------------------------------------------
# Requirement 7.2 — Patient list empty → "Patient {id} not found"
# ---------------------------------------------------------------------------

class TestPatientNotFound:
    def test_returns_summary_result_when_patient_list_empty(self):
        client = _make_client_returns({
            "Patient": [],  # empty!
            "Condition":          [_CONDITION],
            "MedicationRequest":  [_MEDICATION],
            "AllergyIntolerance": [_ALLERGY],
            "Observation":        [_OBSERVATION],
            "Encounter":          [_ENCOUNTER],
            "CarePlan":           [_CARE_PLAN],
        })
        result = _fetch_all_fhir_resources(client, "unknown-patient")
        assert isinstance(result, SummaryResult)
        assert result.error == "Patient unknown-patient not found"

    def test_patient_not_found_sets_correct_patient_id(self):
        client = _make_client_returns({
            "Patient": [],
            "Condition":          [],
            "MedicationRequest":  [],
            "AllergyIntolerance": [],
            "Observation":        [],
            "Encounter":          [],
            "CarePlan":           [],
        })
        result = _fetch_all_fhir_resources(client, "xyz-999")
        assert isinstance(result, SummaryResult)
        assert "xyz-999" in result.error

    def test_patient_not_found_does_not_call_llm(self):
        """When Patient list is empty, subsequent resource fetches still run,
        but the function returns before any LLM call (no SummaryAgent involved)."""
        client = _make_client_returns({
            "Patient": [],
            "Condition":          [],
            "MedicationRequest":  [],
            "AllergyIntolerance": [],
            "Observation":        [],
            "Encounter":          [],
            "CarePlan":           [],
        })
        result = _fetch_all_fhir_resources(client, "p-missing")
        # Should return a SummaryResult with error, not PatientResources
        assert isinstance(result, SummaryResult)
        assert result.current_issues == ""
        assert result.recent_changes == ""
        assert result.risks_and_followup == ""


# ---------------------------------------------------------------------------
# Requirement 7.3 — Patient fetch raises → "Failed to fetch Patient …"
# ---------------------------------------------------------------------------

class TestPatientFetchError:
    def test_fhir_client_error_on_patient_returns_error_summary_result(self):
        client = MagicMock()
        client.get_resource.side_effect = FHIRClientError(404, "Not Found")
        result = _fetch_all_fhir_resources(client, "p1")
        assert isinstance(result, SummaryResult)
        assert result.error.startswith("Failed to fetch Patient p1:")

    def test_fhir_unavailable_error_on_patient_returns_error_summary_result(self):
        client = MagicMock()
        client.get_resource.side_effect = FHIRUnavailableError("Server down")
        result = _fetch_all_fhir_resources(client, "p1")
        assert isinstance(result, SummaryResult)
        assert result.error.startswith("Failed to fetch Patient p1:")

    def test_patient_error_contains_error_message(self):
        client = MagicMock()
        error_msg = "Connection refused"
        client.get_resource.side_effect = FHIRUnavailableError(error_msg)
        result = _fetch_all_fhir_resources(client, "p1")
        assert isinstance(result, SummaryResult)
        assert error_msg in result.error

    def test_patient_error_stops_remaining_fetches(self):
        """If Patient fetch fails, no further get_resource calls are made."""
        client = MagicMock()
        client.get_resource.side_effect = FHIRClientError(500, "Internal Server Error")
        _fetch_all_fhir_resources(client, "p1")
        # Only Patient was attempted
        assert client.get_resource.call_count == 1

    def test_patient_error_sets_empty_section_fields(self):
        client = MagicMock()
        client.get_resource.side_effect = FHIRClientError(503, "Unavailable")
        result = _fetch_all_fhir_resources(client, "p1")
        assert isinstance(result, SummaryResult)
        assert result.current_issues == ""
        assert result.recent_changes == ""
        assert result.risks_and_followup == ""

    def test_patient_error_sets_correct_patient_id_in_result(self):
        client = MagicMock()
        client.get_resource.side_effect = FHIRClientError(400, "Bad Request")
        result = _fetch_all_fhir_resources(client, "specific-patient-id")
        assert isinstance(result, SummaryResult)
        assert result.patient_id == "specific-patient-id"


# ---------------------------------------------------------------------------
# Requirement 7.1 — Non-Patient errors: warn, set to [], continue
# ---------------------------------------------------------------------------

class TestNonPatientFetchErrors:
    def _client_with_single_failure(
        self, failing_type: str, exc: Exception
    ) -> MagicMock:
        """Return a client that succeeds for all types except *failing_type*."""
        happy = {
            "Patient":            [_PATIENT],
            "Condition":          [_CONDITION],
            "MedicationRequest":  [_MEDICATION],
            "AllergyIntolerance": [_ALLERGY],
            "Observation":        [_OBSERVATION],
            "Encounter":          [_ENCOUNTER],
            "CarePlan":           [_CARE_PLAN],
        }

        def side_effect(resource_type, patient_id, params=None):
            if resource_type == failing_type:
                raise exc
            return happy[resource_type]

        client = MagicMock()
        client.get_resource.side_effect = side_effect
        return client

    @pytest.mark.parametrize("failing_type", [
        "Condition", "MedicationRequest", "AllergyIntolerance",
        "Observation", "Encounter", "CarePlan",
    ])
    def test_fhir_client_error_on_non_patient_returns_patient_resources(
        self, failing_type: str
    ):
        """Any non-Patient FHIRClientError still produces a PatientResources."""
        client = self._client_with_single_failure(
            failing_type, FHIRClientError(500, "Err")
        )
        result = _fetch_all_fhir_resources(client, "p1")
        assert isinstance(result, PatientResources)

    @pytest.mark.parametrize("failing_type", [
        "Condition", "MedicationRequest", "AllergyIntolerance",
        "Observation", "Encounter", "CarePlan",
    ])
    def test_fhir_unavailable_error_on_non_patient_returns_patient_resources(
        self, failing_type: str
    ):
        """Any non-Patient FHIRUnavailableError still produces a PatientResources."""
        client = self._client_with_single_failure(
            failing_type, FHIRUnavailableError("Down")
        )
        result = _fetch_all_fhir_resources(client, "p1")
        assert isinstance(result, PatientResources)

    @pytest.mark.parametrize("failing_type,field", [
        ("Condition",          "conditions"),
        ("MedicationRequest",  "medications"),
        ("AllergyIntolerance", "allergies"),
        ("Observation",        "observations"),
        ("Encounter",          "encounters"),
        ("CarePlan",           "care_plans"),
    ])
    def test_failed_non_patient_type_defaults_to_empty_list(
        self, failing_type: str, field: str
    ):
        """The field corresponding to the failing resource type must be []."""
        client = self._client_with_single_failure(
            failing_type, FHIRClientError(503, "Unavailable")
        )
        result = _fetch_all_fhir_resources(client, "p1")
        assert isinstance(result, PatientResources)
        assert getattr(result, field) == []

    @pytest.mark.parametrize("failing_type", [
        "Condition", "MedicationRequest", "AllergyIntolerance",
        "Observation", "Encounter", "CarePlan",
    ])
    def test_remaining_types_still_fetched_after_non_patient_error(
        self, failing_type: str
    ):
        """All 7 resource types must be attempted even when one non-Patient fails."""
        client = self._client_with_single_failure(
            failing_type, FHIRClientError(500, "Err")
        )
        _fetch_all_fhir_resources(client, "p1")
        assert client.get_resource.call_count == 7

    def test_warning_is_logged_for_non_patient_error(self):
        client = self._client_with_single_failure(
            "Condition", FHIRClientError(500, "Server Error")
        )
        with patch("src.agent.logger") as mock_logger:
            _fetch_all_fhir_resources(client, "p1")
            mock_logger.warning.assert_called_once()
            warn_args = mock_logger.warning.call_args
            # First arg is the format string; first %s is resource_type
            assert "Condition" in str(warn_args)


# ---------------------------------------------------------------------------
# Requirement 7.4 + 7.5 — Previously fetched values preserved
# ---------------------------------------------------------------------------

class TestPreservationOfPreviouslyFetchedResults:
    def test_successfully_fetched_conditions_preserved_when_later_type_fails(self):
        """Conditions fetched before Observations fails must remain in the result."""
        conditions = [{"resourceType": "Condition", "id": "c1"},
                      {"resourceType": "Condition", "id": "c2"}]

        def side_effect(resource_type, patient_id, params=None):
            data = {
                "Patient":            [_PATIENT],
                "Condition":          conditions,
                "MedicationRequest":  [_MEDICATION],
                "AllergyIntolerance": [_ALLERGY],
                "Encounter":          [_ENCOUNTER],
                "CarePlan":           [_CARE_PLAN],
            }
            if resource_type == "Observation":
                raise FHIRClientError(500, "Observation server error")
            return data[resource_type]

        client = MagicMock()
        client.get_resource.side_effect = side_effect
        result = _fetch_all_fhir_resources(client, "p1")

        assert isinstance(result, PatientResources)
        assert result.conditions == conditions          # preserved
        assert result.observations == []               # failed → empty

    def test_multiple_non_patient_failures_handled_independently(self):
        """Multiple non-Patient failures all default to [] independently."""
        def side_effect(resource_type, patient_id, params=None):
            if resource_type == "Patient":
                return [_PATIENT]
            if resource_type in ("Condition", "Encounter"):
                raise FHIRUnavailableError("Down")
            return {
                "MedicationRequest":  [_MEDICATION],
                "AllergyIntolerance": [_ALLERGY],
                "Observation":        [_OBSERVATION],
                "CarePlan":           [_CARE_PLAN],
            }[resource_type]

        client = MagicMock()
        client.get_resource.side_effect = side_effect
        result = _fetch_all_fhir_resources(client, "p1")

        assert isinstance(result, PatientResources)
        assert result.conditions == []
        assert result.encounters == []
        assert result.medications == [_MEDICATION]
        assert result.observations == [_OBSERVATION]

    def test_all_non_patient_types_fail_still_returns_patient_resources(self):
        """Even if all non-Patient types fail, PatientResources is returned."""
        def side_effect(resource_type, patient_id, params=None):
            if resource_type == "Patient":
                return [_PATIENT]
            raise FHIRClientError(500, "Err")

        client = MagicMock()
        client.get_resource.side_effect = side_effect
        result = _fetch_all_fhir_resources(client, "p1")

        assert isinstance(result, PatientResources)
        assert result.patient == _PATIENT
        assert result.conditions == []
        assert result.medications == []
        assert result.allergies == []
        assert result.observations == []
        assert result.encounters == []
        assert result.care_plans == []
