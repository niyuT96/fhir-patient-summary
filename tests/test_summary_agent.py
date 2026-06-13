"""
Unit tests for SummaryAgent streaming behavior.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from src.agent import (
    DEFAULT_OPENAI_MAX_TOKENS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_TEMPERATURE,
    STREAM_THROTTLE_SECONDS,
    SummaryAgent,
    _extract_stream_delta,
    _fetch_all_fhir_resources,
    _get_model_config,
    parse_sections,
)
from src.context_extractor import PatientContextExtractor
from src.exceptions import FHIRClientError, FHIRUnavailableError
from src.models import PatientResources, SummaryResult
from src.tools.prompt_loader import get_role_prompt

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


def _stream_chunk(content: str):
    chunk = MagicMock()
    choice = MagicMock()
    choice.delta.content = content
    chunk.choices = [choice]
    return chunk


def _make_llm_stream(*contents: str) -> MagicMock:
    llm = MagicMock()
    llm.chat.completions.create.return_value = [_stream_chunk(content) for content in contents]
    return llm


def _make_agent(
    *,
    available=True,
    patient_resources=None,
    side_effects=None,
    stream_contents: tuple[str, ...] = ("## Current Issues\n", "- Hypertension [S1]\n"),
):
    fhir = _make_fhir_client(
        available=available,
        patient_resources=patient_resources,
        side_effects=side_effects,
    )
    extractor = _make_extractor()
    llm = _make_llm_stream(*stream_contents)
    return SummaryAgent(fhir_client=fhir, extractor=extractor, llm_client=llm), fhir, extractor, llm


class TestModelConfig:
    def test_default_model_config_is_used_for_streaming_call(self, monkeypatch):
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_TEMPERATURE", raising=False)
        monkeypatch.delenv("OPENAI_MAX_TOKENS", raising=False)
        monkeypatch.delenv("STREAM_THROTTLE_SECONDS", raising=False)
        agent, _, _, llm = _make_agent()

        list(agent.generate_summary_stream("patient-001", "ED Doctor"))

        call_kwargs = llm.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == DEFAULT_OPENAI_MODEL
        assert call_kwargs["temperature"] == DEFAULT_OPENAI_TEMPERATURE
        assert call_kwargs["max_tokens"] == DEFAULT_OPENAI_MAX_TOKENS

    def test_environment_model_config_overrides_are_used_for_streaming_call(self, monkeypatch):
        monkeypatch.setenv("OPENAI_MODEL", "test-model")
        monkeypatch.setenv("OPENAI_TEMPERATURE", "0.7")
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "1234")
        monkeypatch.setenv("STREAM_THROTTLE_SECONDS", "0.05")
        agent, _, _, llm = _make_agent()

        list(agent.generate_summary_stream("patient-001", "ED Doctor"))

        call_kwargs = llm.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1234
        assert _get_model_config().stream_throttle_seconds == 0.05

    def test_invalid_environment_model_config_falls_back_to_defaults(self, monkeypatch):
        monkeypatch.setenv("OPENAI_MODEL", "   ")
        monkeypatch.setenv("OPENAI_TEMPERATURE", "not-a-float")
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "0")
        monkeypatch.setenv("STREAM_THROTTLE_SECONDS", "-0.1")

        config = _get_model_config()

        assert config.model == DEFAULT_OPENAI_MODEL
        assert config.temperature == DEFAULT_OPENAI_TEMPERATURE
        assert config.max_tokens == DEFAULT_OPENAI_MAX_TOKENS
        assert config.stream_throttle_seconds == STREAM_THROTTLE_SECONDS


class TestStreamingSummary:
    def test_unsupported_role_yields_error_without_fetch_or_llm_call(self):
        agent, fhir, extractor, llm = _make_agent()

        chunks = list(agent.generate_summary_stream("patient-001", "Radiologist"))

        assert chunks == [("**Error:** Unsupported role: Radiologist", [])]
        fhir.is_available.assert_not_called()
        extractor.extract.assert_not_called()
        llm.chat.completions.create.assert_not_called()

    def test_patient_not_found_yields_error_without_llm_call(self):
        agent, _, extractor, llm = _make_agent(patient_resources=[])

        chunks = list(agent.generate_summary_stream("patient-999", "ED Doctor"))

        assert len(chunks) == 1
        assert chunks[0][0].startswith("**Error:**")
        assert "not found" in chunks[0][0]
        extractor.extract.assert_not_called()
        llm.chat.completions.create.assert_not_called()

    def test_local_fallback_resources_are_filtered_to_selected_patient(self):
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
        fhir.is_available.return_value = False
        fhir._load_fallback_bundle.return_value = [
            selected_patient,
            other_patient,
            selected_condition,
            other_condition,
        ]
        extractor = _make_extractor()
        extractor._format_condition.return_value = "- Selected condition"
        llm = _make_llm_stream("## Current Issues\n- Selected condition")
        agent = SummaryAgent(fhir_client=fhir, extractor=extractor, llm_client=llm)

        chunks = list(agent.generate_summary_stream("p1", "ED Doctor"))

        resources = extractor.extract.call_args.args[0]
        assert isinstance(resources, PatientResources)
        assert resources.patient == selected_patient
        assert resources.conditions == [selected_condition]
        assert chunks[0][1][1].items[0].summary == "Selected condition"
        assert chunks[-1][0] == "## Current Issues\n- Selected condition"

    def test_local_fallback_resources_match_urn_uuid_patient_references(self):
        patient_id = "9e43a3bf-fb4f-4007-8a1f-d8e00e57d4e5"
        selected_patient = {
            "resourceType": "Patient",
            "id": patient_id,
            "name": [{"text": "Allen Runte"}],
        }
        selected_condition = {
            "resourceType": "Condition",
            "id": "condition-1",
            "subject": {"reference": f"urn:uuid:{patient_id}"},
            "code": {"text": "Hypertension"},
        }
        fhir = MagicMock()
        fhir.is_available.return_value = False
        fhir._load_fallback_bundle.return_value = [selected_patient, selected_condition]
        extractor = _make_extractor()
        llm = _make_llm_stream("## Current Issues\n- Hypertension")
        agent = SummaryAgent(fhir_client=fhir, extractor=extractor, llm_client=llm)

        list(agent.generate_summary_stream(patient_id, "ED Doctor"))

        resources = extractor.extract.call_args.args[0]
        assert isinstance(resources, PatientResources)
        assert resources.conditions == [selected_condition]

    def test_source_sections_are_yielded_before_llm_markdown(self):
        agent, _, _, _ = _make_agent(stream_contents=("A", "B"))

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))

        assert chunks[0][0] == ""
        assert chunks[0][1] is not None
        assert chunks[1][0] == "A"

    def test_single_llm_stream_call_accumulates_markdown(self):
        agent, _, _, llm = _make_agent(
            stream_contents=("## Current", " Issues\n", "- BP high [S1]")
        )

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))

        assert llm.chat.completions.create.call_count == 1
        call_kwargs = llm.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert "=== Citeable FHIR Source Index ===" in call_kwargs["messages"][1]["content"]
        assert chunks[-1][0] == "## Current Issues\n- BP high [S1]"

    def test_streaming_throttles_intermediate_yields_but_keeps_first_and_final(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv("STREAM_THROTTLE_SECONDS", "0.2")
        timestamps = iter([0.0, 0.0, 0.05, 0.19, 0.21])
        monkeypatch.setattr("src.agent.time.monotonic", lambda: next(timestamps))
        agent, _, _, _ = _make_agent(stream_contents=("A", "B", "C", "D"))

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))
        markdown_chunks = [chunk[0] for chunk in chunks[1:]]

        assert markdown_chunks[0] == "A"
        assert markdown_chunks[-1] == "ABCD"
        assert "AB" not in markdown_chunks
        assert "ABC" not in markdown_chunks
        assert len(markdown_chunks) < 4

    def test_streaming_call_uses_system_and_user_messages(self):
        agent, _, _, llm = _make_agent()

        list(agent.generate_summary_stream("patient-001", "Care Manager"))

        messages = llm.chat.completions.create.call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[0]["content"] == get_role_prompt("Care Manager")
        assert "FHIR patient context:" in messages[1]["content"]

    def test_llm_stream_exception_yields_error(self):
        fhir = _make_fhir_client()
        extractor = _make_extractor()
        llm = MagicMock()
        llm.chat.completions.create.side_effect = Exception("Rate limit exceeded")
        agent = SummaryAgent(fhir_client=fhir, extractor=extractor, llm_client=llm)

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))

        assert chunks[0][0] == ""
        assert chunks[-1] == ("**Error:** Rate limit exceeded", [])

    def test_no_generate_summary_non_streaming_api(self):
        agent, _, _, _ = _make_agent()

        assert not hasattr(agent, "generate_summary")


class TestStreamDeltaExtraction:
    def test_extracts_delta_content(self):
        assert _extract_stream_delta(_stream_chunk("hello")) == "hello"

    def test_extracts_delta_content_from_dict_chunk(self):
        chunk = {"choices": [{"delta": {"content": "hello"}}]}

        assert _extract_stream_delta(chunk) == "hello"

    @pytest.mark.parametrize("chunk", [MagicMock(choices=[]), object(), None])
    def test_missing_delta_content_returns_empty_string(self, chunk):
        assert _extract_stream_delta(chunk) == ""


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

    def test_fetch_plan_uses_broad_bounded_queries(self):
        calls = {}

        def _get_resource(resource_type, patient_id, params=None):
            calls[resource_type] = dict(params or {})
            if resource_type == "Patient":
                return [MINIMAL_PATIENT]
            return []

        fhir = MagicMock()
        fhir.get_resource.side_effect = _get_resource

        result = _fetch_all_fhir_resources(fhir, "patient-001")

        assert isinstance(result, PatientResources)
        assert calls["Condition"]["_count"] == "100"
        assert calls["MedicationRequest"]["_count"] == "150"
        assert calls["MedicationRequest"]["_sort"] == "-authoredon"
        assert calls["Observation"]["_count"] == "150"
        assert calls["Observation"]["_sort"] == "-date"
        assert calls["Encounter"]["_count"] == "75"
        assert calls["Encounter"]["_sort"] == "-date"
        assert calls["CarePlan"]["_count"] == "50"


class TestSourceSections:
    def test_source_sections_are_structured_and_unlimited(self):
        fhir = _make_fhir_client()
        extractor = PatientContextExtractor()
        llm = _make_llm_stream("unused")
        agent = SummaryAgent(fhir_client=fhir, extractor=extractor, llm_client=llm)
        resources = PatientResources(
            patient=MINIMAL_PATIENT,
            conditions=[
                {"resourceType": "Condition", "id": "c1", "code": {"text": "A"}},
                {"resourceType": "Condition", "id": "c2", "code": {"text": "B"}},
                {"resourceType": "Condition", "id": "c3", "code": {"text": "C"}},
                {"resourceType": "Condition", "id": "c4", "code": {"text": "D"}},
            ],
        )

        sections = agent._build_source_sections(resources)
        condition_section = sections[1]

        assert condition_section.label == "Conditions (4)"
        assert [item.summary for item in condition_section.items] == ["A", "B", "C", "D"]
        assert condition_section.items[0].source_id == "S2"
        assert condition_section.items[0].resource_type == "Condition"
        assert condition_section.items[0].evidence["code"] == "A"
        assert condition_section.hidden_items == []


class TestParseSectionsCompatibility:
    def test_parse_sections_still_available_for_validator_use(self):
        result = parse_sections(
            "## Current Issues\nA\n## Recent Changes\nB\n## Risks and Follow-up\nC"
        )

        assert result == {
            "current_issues": "A",
            "recent_changes": "B",
            "risks_and_followup": "C",
        }


class TestUtcNowIso:
    def test_error_sentinel_timestamp_is_iso_utc(self):
        fhir = _make_fhir_client(patient_resources=[])
        result = _fetch_all_fhir_resources(fhir, "missing")

        assert isinstance(result, SummaryResult)
        assert ISO_8601_RE.match(result.generated_at)
