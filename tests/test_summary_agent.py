"""
Unit tests for SummaryAgent (tasks 7.1, 7.2, 7.4).

Covers:
- Role validation: unsupported roles return error SummaryResult without LLM call
- Data-source determination: fhir_server vs local_fallback
- FHIR fetch: graceful degradation for non-Patient errors
- FHIR fetch: Patient not found / Patient fetch error -> error SummaryResult
- generated_at is always set (ISO 8601 UTC)
- patient_id and role always match inputs
- LLM success -> populated sections, error=None
- LLM failure -> error set, sections empty
- No unhandled exception ever propagates
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

from src.agent import SummaryAgent, _fetch_all_fhir_resources, parse_sections
from src.context_extractor import PatientContextExtractor
from src.exceptions import FHIRClientError, FHIRUnavailableError
from src.models import PatientResources, SummaryResult

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

ISO_8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

MINIMAL_PATIENT = {
    "resourceType": "Patient",
    "id": "patient-001",
    "name": [{"text": "Jane Doe"}],
    "birthDate": "1970-01-01",
    "gender": "female",
}


def _make_fhir_client(
    *,
    available: bool = True,
    patient_resources: list[dict] | None = None,
    side_effects: dict | None = None,
) -> MagicMock:
    """Build a mock FHIRClient.

    Args:
        available:         Return value of is_available().
        patient_resources: List returned for Patient get_resource call.
        side_effects:      Mapping of resource_type -> exception to raise
                           instead of returning a list.
    """
    client = MagicMock()
    client.is_available.return_value = available

    if patient_resources is None:
        patient_resources = [MINIMAL_PATIENT]

    side_effects = side_effects or {}

    def _get_resource(resource_type, patient_id, params=None):
        if resource_type in side_effects:
            raise side_effects[resource_type]
        if resource_type == "Patient":
            return patient_resources
        return []

    client.get_resource.side_effect = _get_resource
    return client


def _make_extractor() -> MagicMock:
    extractor = MagicMock()
    extractor.extract.return_value = "Patient context string"
    return extractor


def _make_llm(content: str = "## Current Issues\nIssues\n## Recent Changes\nChanges\n## Risks and Follow-up\nRisks") -> MagicMock:
    llm = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    llm.chat.completions.create.return_value = MagicMock(choices=[choice])
    return llm


def _make_agent(*, available=True, patient_resources=None, side_effects=None, llm_content=None):
    fhir = _make_fhir_client(
        available=available,
        patient_resources=patient_resources,
        side_effects=side_effects,
    )
    extractor = _make_extractor()
    llm = _make_llm() if llm_content is None else _make_llm(content=llm_content)
    return SummaryAgent(fhir_client=fhir, extractor=extractor, llm_client=llm), fhir, extractor, llm


# ---------------------------------------------------------------------------
# Role validation (Req 4.3, 6.6, 6.7)
# ---------------------------------------------------------------------------

class TestRoleValidation:
    def test_unsupported_role_returns_error_without_llm_call(self):
        agent, _, _, llm = _make_agent()
        result = agent.generate_summary("patient-001", "Radiologist")
        assert result.error == "Unsupported role: Radiologist"
        llm.chat.completions.create.assert_not_called()

    def test_unsupported_role_sections_are_empty(self):
        agent, _, _, _ = _make_agent()
        result = agent.generate_summary("patient-001", "Nurse")
        assert result.current_issues == ""
        assert result.recent_changes == ""
        assert result.risks_and_followup == ""

    def test_unsupported_role_patient_id_preserved(self):
        agent, _, _, _ = _make_agent()
        result = agent.generate_summary("patient-42", "Unknown")
        assert result.patient_id == "patient-42"

    def test_unsupported_role_role_preserved(self):
        agent, _, _, _ = _make_agent()
        result = agent.generate_summary("patient-001", "Surgeon")
        assert result.role == "Surgeon"

    def test_ed_doctor_is_valid(self):
        agent, _, _, _ = _make_agent()
        result = agent.generate_summary("patient-001", "ED Doctor")
        assert result.error is None or "Unsupported role" not in (result.error or "")

    def test_care_manager_is_valid(self):
        agent, _, _, _ = _make_agent()
        result = agent.generate_summary("patient-001", "Care Manager")
        assert result.error is None or "Unsupported role" not in (result.error or "")


# ---------------------------------------------------------------------------
# Data-source determination (Req 2.3, 2.4)
# ---------------------------------------------------------------------------

class TestDataSource:
    def test_data_source_fhir_server_when_available(self):
        agent, _, _, _ = _make_agent(available=True)
        result = agent.generate_summary("patient-001", "ED Doctor")
        assert result.data_source == "fhir_server"

    def test_data_source_local_fallback_when_unavailable(self):
        fhir = MagicMock()
        fhir.is_available.return_value = False
        fhir._load_fallback_bundle.return_value = [MINIMAL_PATIENT]
        extractor = _make_extractor()
        llm = _make_llm()
        agent = SummaryAgent(fhir_client=fhir, extractor=extractor, llm_client=llm)
        result = agent.generate_summary("patient-001", "ED Doctor")
        assert result.data_source == "local_fallback"

    def test_fhir_available_calls_get_resource(self):
        agent, fhir, _, _ = _make_agent(available=True)
        agent.generate_summary("patient-001", "ED Doctor")
        fhir.get_resource.assert_called()

    def test_fhir_unavailable_calls_load_fallback(self):
        fhir = MagicMock()
        fhir.is_available.return_value = False
        fhir._load_fallback_bundle.return_value = [MINIMAL_PATIENT]
        extractor = _make_extractor()
        llm = _make_llm()
        agent = SummaryAgent(fhir_client=fhir, extractor=extractor, llm_client=llm)
        agent.generate_summary("patient-001", "ED Doctor")
        fhir._load_fallback_bundle.assert_called_once()

    def test_fallback_resources_are_filtered_to_selected_patient(self):
        selected_patient = {
            "resourceType": "Patient",
            "id": "p1",
            "name": [{"text": "Selected Patient"}],
        }
        other_patient = {
            "resourceType": "Patient",
            "id": "p2",
            "name": [{"text": "Other Patient"}],
        }
        selected_condition = {
            "resourceType": "Condition",
            "id": "c1",
            "subject": {"reference": "Patient/p1"},
            "code": {"text": "Selected condition"},
        }
        other_condition = {
            "resourceType": "Condition",
            "id": "c2",
            "subject": {"reference": "Patient/p2"},
            "code": {"text": "Other condition"},
        }

        fhir = MagicMock()
        fhir._load_fallback_bundle.return_value = [
            selected_patient,
            other_patient,
            selected_condition,
            other_condition,
        ]
        agent = SummaryAgent(
            fhir_client=fhir,
            extractor=_make_extractor(),
            llm_client=_make_llm(),
        )

        resources = agent._load_fallback_resources("p1")

        assert isinstance(resources, PatientResources)
        assert resources.patient["id"] == "p1"
        assert resources.conditions == [selected_condition]

    def test_fallback_resources_match_urn_uuid_patient_references(self):
        selected_patient = {
            "resourceType": "Patient",
            "id": "9e43a3bf-fb4f-4007-8a1f-d8e00e57d4e5",
            "name": [{"text": "Allen Runte"}],
        }
        selected_condition = {
            "resourceType": "Condition",
            "id": "condition-1",
            "subject": {"reference": "urn:uuid:9e43a3bf-fb4f-4007-8a1f-d8e00e57d4e5"},
            "code": {"text": "Hypertension"},
        }

        fhir = MagicMock()
        fhir._load_fallback_bundle.return_value = [selected_patient, selected_condition]
        agent = SummaryAgent(
            fhir_client=fhir,
            extractor=_make_extractor(),
            llm_client=_make_llm(),
        )

        resources = agent._load_fallback_resources("9e43a3bf-fb4f-4007-8a1f-d8e00e57d4e5")

        assert isinstance(resources, PatientResources)
        assert resources.conditions == [selected_condition]


# ---------------------------------------------------------------------------
# Patient not found / Patient fetch errors (Req 7.2, 7.3)
# ---------------------------------------------------------------------------

class TestPatientFetchErrors:
    def test_patient_not_found_returns_error(self):
        agent, _, _, llm = _make_agent(patient_resources=[])
        result = agent.generate_summary("patient-999", "ED Doctor")
        assert result.error is not None
        assert "not found" in result.error.lower() or "patient-999" in result.error
        llm.chat.completions.create.assert_not_called()

    def test_patient_fhir_client_error_returns_error(self):
        agent, _, _, llm = _make_agent(
            side_effects={"Patient": FHIRClientError(404, "Not Found")}
        )
        result = agent.generate_summary("patient-001", "ED Doctor")
        assert result.error is not None
        assert "patient-001" in result.error.lower() or "Failed" in result.error
        llm.chat.completions.create.assert_not_called()

    def test_patient_fhir_unavailable_error_returns_error(self):
        agent, _, _, llm = _make_agent(
            side_effects={"Patient": FHIRUnavailableError("Server unreachable")}
        )
        result = agent.generate_summary("patient-001", "Care Manager")
        assert result.error is not None
        llm.chat.completions.create.assert_not_called()

    def test_patient_not_found_sections_empty(self):
        agent, _, _, _ = _make_agent(patient_resources=[])
        result = agent.generate_summary("patient-999", "ED Doctor")
        assert result.current_issues == ""
        assert result.recent_changes == ""
        assert result.risks_and_followup == ""


# ---------------------------------------------------------------------------
# Graceful degradation for non-Patient fetch errors (Req 7.1, 7.4, 7.5)
# ---------------------------------------------------------------------------

class TestNonPatientFetchGracefulDegradation:
    def test_condition_error_does_not_stop_summary(self):
        agent, _, _, _ = _make_agent(
            side_effects={"Condition": FHIRClientError(500, "Internal Error")}
        )
        result = agent.generate_summary("patient-001", "ED Doctor")
        # Should still produce a result (no fatal error about Condition)
        assert result.patient_id == "patient-001"

    def test_observation_unavailable_does_not_stop_summary(self):
        agent, _, _, _ = _make_agent(
            side_effects={"Observation": FHIRUnavailableError("Timeout")}
        )
        result = agent.generate_summary("patient-001", "Care Manager")
        assert result.patient_id == "patient-001"

    def test_multiple_non_patient_errors_gracefully_handled(self):
        agent, _, _, _ = _make_agent(
            side_effects={
                "Condition": FHIRClientError(503, "Service Unavailable"),
                "MedicationRequest": FHIRUnavailableError("Timeout"),
                "Encounter": FHIRClientError(503, "Service Unavailable"),
            }
        )
        result = agent.generate_summary("patient-001", "ED Doctor")
        assert result.patient_id == "patient-001"
        # Should not have a fatal error from non-Patient types
        assert result.error is None or "not found" not in result.error


# ---------------------------------------------------------------------------
# _fetch_all_fhir_resources unit tests
# ---------------------------------------------------------------------------

class TestFetchAllFhirResources:
    def test_returns_patient_resources_on_success(self):
        fhir = _make_fhir_client()
        result = _fetch_all_fhir_resources(fhir, "patient-001")
        assert isinstance(result, PatientResources)
        assert result.patient == MINIMAL_PATIENT

    def test_non_patient_error_sets_field_to_empty_list(self):
        fhir = _make_fhir_client(
            side_effects={"Condition": FHIRClientError(500, "Error")}
        )
        result = _fetch_all_fhir_resources(fhir, "patient-001")
        assert isinstance(result, PatientResources)
        assert result.conditions == []

    def test_patient_error_returns_summary_result_sentinel(self):
        fhir = _make_fhir_client(
            side_effects={"Patient": FHIRUnavailableError("Timeout")}
        )
        result = _fetch_all_fhir_resources(fhir, "patient-001")
        assert isinstance(result, SummaryResult)
        assert result.error is not None

    def test_empty_patient_list_returns_error_sentinel(self):
        fhir = _make_fhir_client(patient_resources=[])
        result = _fetch_all_fhir_resources(fhir, "patient-001")
        assert isinstance(result, SummaryResult)
        assert "not found" in result.error.lower()

    def test_previously_fetched_resources_preserved_after_later_error(self):
        """Resources fetched before a later error must be retained (Req 7.5)."""
        conditions = [{"resourceType": "Condition", "code": {"text": "Hypertension"}}]

        def _get_resource(resource_type, patient_id, params=None):
            if resource_type == "Patient":
                return [MINIMAL_PATIENT]
            if resource_type == "Condition":
                return conditions
            if resource_type == "MedicationRequest":
                raise FHIRClientError(500, "Error")
            return []

        fhir = MagicMock()
        fhir.get_resource.side_effect = _get_resource
        result = _fetch_all_fhir_resources(fhir, "patient-001")
        assert isinstance(result, PatientResources)
        assert result.conditions == conditions
        assert result.medications == []


# ---------------------------------------------------------------------------
# generated_at / patient_id / role invariants (Req 6.4, 6.5, 6.6)
# ---------------------------------------------------------------------------

class TestResultInvariants:
    def test_generated_at_always_set_on_success(self):
        agent, _, _, _ = _make_agent()
        result = agent.generate_summary("patient-001", "ED Doctor")
        assert result.generated_at
        assert ISO_8601_RE.match(result.generated_at)

    def test_generated_at_set_on_invalid_role(self):
        agent, _, _, _ = _make_agent()
        result = agent.generate_summary("patient-001", "Unknown")
        assert ISO_8601_RE.match(result.generated_at)

    def test_generated_at_set_on_patient_not_found(self):
        agent, _, _, _ = _make_agent(patient_resources=[])
        result = agent.generate_summary("patient-999", "ED Doctor")
        assert ISO_8601_RE.match(result.generated_at)

    def test_patient_id_matches_input(self):
        agent, _, _, _ = _make_agent()
        result = agent.generate_summary("patient-XYZ", "ED Doctor")
        assert result.patient_id == "patient-XYZ"

    def test_role_matches_input(self):
        agent, _, _, _ = _make_agent()
        result = agent.generate_summary("patient-001", "Care Manager")
        assert result.role == "Care Manager"

    def test_never_raises_on_complete_fhir_and_llm_failure(self):
        """Req 6.1 - generate_summary must always return SummaryResult."""
        fhir = MagicMock()
        fhir.is_available.side_effect = RuntimeError("Unexpected crash")
        extractor = _make_extractor()
        llm = _make_llm()
        agent = SummaryAgent(fhir_client=fhir, extractor=extractor, llm_client=llm)
        # Must not raise
        result = agent.generate_summary("patient-001", "ED Doctor")
        assert isinstance(result, SummaryResult)
        assert result.error is not None


# ---------------------------------------------------------------------------
# LLM invocation results (Task 7.4 coverage)
# ---------------------------------------------------------------------------

class TestLLMInvocation:
    def test_successful_llm_call_populates_sections(self):
        agent, _, _, _ = _make_agent(
            llm_content=(
                "## Current Issues\n- Hypertension\n"
                "## Recent Changes\n- Started Lisinopril\n"
                "## Risks and Follow-up\n- BP recheck in 2 weeks"
            )
        )
        result = agent.generate_summary("patient-001", "ED Doctor")
        assert result.error is None
        assert "Hypertension" in result.current_issues
        assert "Lisinopril" in result.recent_changes
        assert "BP recheck" in result.risks_and_followup

    def test_llm_error_sets_error_field_and_empties_sections(self):
        fhir = _make_fhir_client()
        extractor = _make_extractor()
        llm = MagicMock()
        llm.chat.completions.create.side_effect = Exception("Rate limit exceeded")
        agent = SummaryAgent(fhir_client=fhir, extractor=extractor, llm_client=llm)
        result = agent.generate_summary("patient-001", "ED Doctor")
        assert result.error is not None
        assert "Rate limit" in result.error
        assert result.current_issues == ""
        assert result.recent_changes == ""
        assert result.risks_and_followup == ""

    def test_llm_called_with_correct_model_and_params(self):
        agent, _, _, llm = _make_agent()
        agent.generate_summary("patient-001", "ED Doctor")
        call_kwargs = llm.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs.kwargs["temperature"] == 0.3
        assert call_kwargs.kwargs["max_tokens"] == 800

    def test_llm_called_with_system_and_user_messages(self):
        agent, _, _, llm = _make_agent()
        agent.generate_summary("patient-001", "ED Doctor")
        messages = llm.chat.completions.create.call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_ed_doctor_prompt_used_for_ed_doctor_role(self):
        from src.agent import ED_DOCTOR_PROMPT
        agent, _, _, llm = _make_agent()
        agent.generate_summary("patient-001", "ED Doctor")
        system_msg = llm.chat.completions.create.call_args.kwargs["messages"][0]
        assert system_msg["content"] == ED_DOCTOR_PROMPT

    def test_care_manager_prompt_used_for_care_manager_role(self):
        from src.agent import CARE_MANAGER_PROMPT
        agent, _, _, llm = _make_agent()
        agent.generate_summary("patient-001", "Care Manager")
        system_msg = llm.chat.completions.create.call_args.kwargs["messages"][0]
        assert system_msg["content"] == CARE_MANAGER_PROMPT


class TestSectionBySectionStreaming:
    def test_streaming_yields_one_completed_section_at_a_time(self):
        fhir = _make_fhir_client()
        extractor = _make_extractor()
        llm = MagicMock()

        responses = []
        for content in [
            "## Current Issues\n- Hypertension",
            "## Recent Changes\n- A1c increased",
            "## Risks and Follow-up\n- Recheck BP",
        ]:
            choice = MagicMock()
            choice.message.content = content
            responses.append(MagicMock(choices=[choice]))

        llm.chat.completions.create.side_effect = responses
        agent = SummaryAgent(fhir_client=fhir, extractor=extractor, llm_client=llm)

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))

        assert len(chunks) == 4
        assert chunks[0][0] == ""
        assert chunks[0][1] is not None
        assert "## Current Issues" in chunks[1][0]
        assert "## Recent Changes" not in chunks[1][0]
        assert "## Recent Changes" in chunks[2][0]
        assert "## Risks and Follow-up" not in chunks[2][0]
        assert "## Risks and Follow-up" in chunks[3][0]
        assert chunks[1][1] is not None
        assert chunks[2][1] is not None
        assert chunks[3][1] is not None
        assert llm.chat.completions.create.call_count == 3

    def test_source_sections_are_compact(self):
        fhir = _make_fhir_client()
        extractor = PatientContextExtractor()
        llm = _make_llm()
        agent = SummaryAgent(fhir_client=fhir, extractor=extractor, llm_client=llm)
        resources = PatientResources(
            patient=MINIMAL_PATIENT,
            conditions=[
                {"code": {"text": "A"}},
                {"code": {"text": "B"}},
                {"code": {"text": "C"}},
                {"code": {"text": "D"}},
            ],
        )

        sections = agent._build_source_sections(resources)
        condition_section = sections[0]

        assert condition_section.label == "Active Conditions (4)"
        assert condition_section.items == ["A", "B", "C"]
        assert condition_section.hidden_items == ["D"]
