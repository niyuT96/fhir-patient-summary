"""
Unit tests for SummaryAgent streaming behavior.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from src.agent import (
    CITATION_VALIDATION_WARNING,
    DEFAULT_OPENAI_MAX_TOKENS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_TEMPERATURE,
    SOURCE_CONTEXT_TRUNCATED_WARNING,
    STREAM_THROTTLE_SECONDS,
    SummaryAgent,
    _extract_stream_delta,
    _fetch_all_fhir_resources,
    _get_model_config,
    parse_sections,
)
from src.exceptions import FHIRClientError, FHIRUnavailableError
from src.models import PatientResources, SourceItem, SourceScopeInfo, SourceSection, SummaryResult
from src.tools.citation_validator import validate_citations
from src.tools.prompt_loader import get_role_prompt
from src.tools.source_items import SourceContextResult, build_source_context, build_source_context_result

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


def _source_item(
    source_id: str,
    resource_type: str,
    resource_id: str,
    summary: str,
) -> SourceItem:
    return SourceItem(
        source_id=source_id,
        label=f"{source_id} | {resource_type}/{resource_id} | {summary}",
        resource_type=resource_type,
        resource_id=resource_id,
        summary=summary,
        evidence={"resourceType": resource_type, "id": resource_id, "summary": summary},
        raw_resource={"resourceType": resource_type, "id": resource_id},
    )


def _make_agent(
    *,
    available=True,
    patient_resources=None,
    side_effects=None,
    stream_contents: tuple[str, ...] = ("## Current Issues\n", "- Jane Doe [S1]\n"),
):
    fhir = _make_fhir_client(
        available=available,
        patient_resources=patient_resources,
        side_effects=side_effects,
    )
    llm = _make_llm_stream(*stream_contents)
    return SummaryAgent(fhir_client=fhir, llm_client=llm), fhir, llm


class TestModelConfig:
    def test_default_model_config_is_used_for_streaming_call(self, monkeypatch):
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_TEMPERATURE", raising=False)
        monkeypatch.delenv("OPENAI_MAX_TOKENS", raising=False)
        monkeypatch.delenv("STREAM_THROTTLE_SECONDS", raising=False)
        agent, _, llm = _make_agent()

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
        agent, _, llm = _make_agent()

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
        agent, fhir, llm = _make_agent()

        chunks = list(agent.generate_summary_stream("patient-001", "Radiologist"))

        assert chunks == [("**Error:** Unsupported role: Radiologist", [], "")]
        fhir.is_available.assert_not_called()
        llm.chat.completions.create.assert_not_called()

    def test_patient_not_found_yields_error_without_llm_call(self):
        agent, _, llm = _make_agent(patient_resources=[])

        chunks = list(agent.generate_summary_stream("patient-999", "ED Doctor"))

        assert len(chunks) == 1
        assert chunks[0][0].startswith("**Error:**")
        assert "not found" in chunks[0][0]
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
        llm = _make_llm_stream("## Current Issues\n- Selected condition")
        agent = SummaryAgent(fhir_client=fhir, llm_client=llm)

        chunks = list(agent.generate_summary_stream("p1", "ED Doctor"))

        source_sections = chunks[0][1]
        assert source_sections[0].items[0].raw_resource == selected_patient
        assert source_sections[1].items[0].raw_resource == selected_condition
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
        llm = _make_llm_stream("## Current Issues\n- Hypertension")
        agent = SummaryAgent(fhir_client=fhir, llm_client=llm)

        chunks = list(agent.generate_summary_stream(patient_id, "ED Doctor"))

        assert chunks[0][1][1].items[0].raw_resource == selected_condition

    def test_source_sections_are_yielded_before_llm_markdown(self):
        agent, _, _ = _make_agent(stream_contents=("A", "B"))

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))

        assert chunks[0][0] == ""
        assert chunks[0][1] is not None
        assert chunks[1][0] == "A"

    def test_single_llm_stream_call_accumulates_markdown(self):
        agent, _, llm = _make_agent(
            stream_contents=("## Current", " Issues\n", "- Jane Doe [S1]")
        )

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))

        assert llm.chat.completions.create.call_count == 1
        call_kwargs = llm.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert "=== Source-Indexed FHIR Context ===" in call_kwargs["messages"][1]["content"]
        assert chunks[-1][0] == "## Current Issues\n- Jane Doe [S1]"

    def test_streaming_throttles_intermediate_yields_but_keeps_first_and_final(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv("STREAM_THROTTLE_SECONDS", "0.2")
        timestamps = iter([0.0, 0.0, 0.05, 0.19, 0.21])
        monkeypatch.setattr("src.agent.time.monotonic", lambda: next(timestamps))
        agent, _, _ = _make_agent(stream_contents=("A", "B", "C", "D"))

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))
        markdown_chunks = [chunk[0] for chunk in chunks[1:]]

        assert markdown_chunks[0] == "A"
        assert markdown_chunks[-1] == "ABCD"
        assert "AB" not in markdown_chunks
        assert "ABC" not in markdown_chunks
        assert len(markdown_chunks) < 4

    def test_streaming_call_uses_system_and_user_messages(self):
        agent, _, llm = _make_agent()

        list(agent.generate_summary_stream("patient-001", "Care Manager"))

        messages = llm.chat.completions.create.call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[0]["content"] == get_role_prompt("Care Manager")
        assert "FHIR source-indexed context:" in messages[1]["content"]

    def test_llm_stream_exception_yields_error(self):
        fhir = _make_fhir_client()
        llm = MagicMock()
        llm.chat.completions.create.side_effect = Exception("Rate limit exceeded")
        agent = SummaryAgent(fhir_client=fhir, llm_client=llm)

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))

        assert chunks[0][0] == ""
        assert chunks[-1] == ("**Error:** Rate limit exceeded", [], "")

    def test_no_generate_summary_non_streaming_api(self):
        agent, _, _ = _make_agent()

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
        agent, _, _ = _make_agent()
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
        assert condition_section.items[0].evidence["code.text"] == "A"
        assert condition_section.hidden_items == []

    def test_source_item_evidence_is_flattened_from_full_resource(self):
        agent, _, _ = _make_agent()
        resources = PatientResources(
            patient=MINIMAL_PATIENT,
            observations=[
                {
                    "resourceType": "Observation",
                    "id": "obs-1",
                    "code": {"text": "Blood pressure"},
                    "component": [
                        {
                            "code": {"text": "Systolic Blood Pressure"},
                            "valueQuantity": {"value": 120, "unit": "mmHg"},
                        }
                    ],
                    "extension": [{"url": "ignored"}],
                    "valueString": "",
                    "meta": {"versionId": "1"},
                }
            ],
        )

        item = agent._build_source_sections(resources)[4].items[0]

        assert item.evidence["resourceType"] == "Observation"
        assert item.evidence["id"] == "obs-1"
        assert item.evidence["code.text"] == "Blood pressure"
        assert item.evidence["component[0].code.text"] == "Systolic Blood Pressure"
        assert item.evidence["component[0].valueQuantity.value"] == 120
        assert item.evidence["meta.versionId"] == "1"
        assert "extension[0].url" not in item.evidence
        assert "valueString" not in item.evidence
        assert "code" not in item.evidence

    def test_source_context_includes_source_ids_and_evidence_paths(self):
        agent, _, _ = _make_agent()
        sections = agent._build_source_sections(
            PatientResources(
                patient=MINIMAL_PATIENT,
                conditions=[
                    {
                        "resourceType": "Condition",
                        "id": "c1",
                        "code": {"text": "Hypertension"},
                    }
                ],
            )
        )

        context = build_source_context(sections)

        assert context.startswith("=== Source-Indexed FHIR Context ===")
        assert "[S1] Patient/patient-001: Jane Doe; DOB: 1970-01-01; gender: female" in context
        assert "[S2] Condition/c1: Hypertension" in context
        assert "  code.text: Hypertension" in context

    def test_source_context_result_tracks_supplied_items_without_truncation(self):
        sections = [
            SourceSection(
                label="Conditions (2)",
                items=[
                    _source_item("S1", "Condition", "c1", "Hypertension"),
                    _source_item("S2", "Condition", "c2", "Diabetes"),
                ],
            )
        ]

        result = build_source_context_result(sections, max_chars=2000)

        assert result.truncated is False
        assert result.supplied_source_ids == {"S1", "S2"}
        assert [item.source_id for item in result.sections[0].items] == ["S1", "S2"]
        assert "[S1]" in result.text
        assert "[S2]" in result.text

    def test_source_context_result_excludes_items_not_written_to_context(self):
        sections = [
            SourceSection(
                label="Conditions (2)",
                items=[
                    _source_item("S1", "Condition", "c1", "Hypertension"),
                    _source_item("S2", "Condition", "c2", "Diabetes " + ("x" * 500)),
                ],
            )
        ]

        result = build_source_context_result(sections, max_chars=500)

        assert result.truncated is True
        assert result.supplied_source_ids == {"S1"}
        assert [item.source_id for item in result.sections[0].items] == ["S1"]
        assert "[S1]" in result.text
        assert "[S2]" not in result.text

    def test_source_context_result_does_not_mark_truncated_for_missing_blank_line_only(self):
        sections = [
            SourceSection(
                label="Conditions (1)",
                items=[_source_item("S1", "Condition", "c1", "Hypertension")],
            )
        ]
        full_result = build_source_context_result(sections, max_chars=2000)

        result = build_source_context_result(sections, max_chars=len(full_result.text) + 1)

        assert result.truncated is False
        assert result.supplied_source_ids == {"S1"}
        assert [item.source_id for item in result.sections[0].items] == ["S1"]

    def test_validator_rejects_source_id_not_in_supplied_sections(self):
        supplied_sections = [
            SourceSection(
                label="Conditions (1)",
                items=[_source_item("S1", "Condition", "c1", "Hypertension")],
            )
        ]

        validation = validate_citations("## Current Issues\nDiabetes noted [S2]", supplied_sections)

        assert validation.invalid_source_ids == {"S2"}

    def test_validator_returns_cited_source_ids_by_summary_line(self):
        supplied_sections = [
            SourceSection(
                label="Conditions (2)",
                items=[
                    _source_item("S1", "Condition", "c1", "Hypertension"),
                    _source_item("S2", "Condition", "c2", "Diabetes"),
                ],
            )
        ]

        validation = validate_citations(
            "## Current Issues\n- Hypertension and diabetes noted [S1, S2]",
            supplied_sections,
        )

        assert len(validation.cited_lines) == 1
        assert validation.cited_lines[0].text == "- Hypertension and diabetes noted [S1, S2]"
        assert validation.cited_lines[0].source_ids == {"S1", "S2"}

    def test_validator_flags_citation_that_does_not_support_line(self):
        supplied_sections = [
            SourceSection(
                label="Conditions (1)",
                items=[_source_item("S1", "Condition", "c1", "Hypertension")],
            )
        ]

        validation = validate_citations("## Current Issues\nDiabetes noted [S1]", supplied_sections)

        assert len(validation.unsupported_citations) == 1
        assert validation.unsupported_citations[0].line == "Diabetes noted [S1]"
        assert validation.unsupported_citations[0].source_ids == {"S1"}
        assert validation.has_errors is True

    def test_validator_does_not_flag_supported_citation(self):
        supplied_sections = [
            SourceSection(
                label="Conditions (1)",
                items=[_source_item("S1", "Condition", "c1", "Hypertension")],
            )
        ]

        validation = validate_citations(
            "## Current Issues\nHypertension noted [S1]",
            supplied_sections,
        )

        assert validation.unsupported_citations == []

    def test_validator_flags_patient_citation_that_does_not_support_line(self):
        supplied_sections = [
            SourceSection(
                label="Patient (1)",
                items=[_source_item("S1", "Patient", "patient-001", "Jane Doe")],
            )
        ]

        validation = validate_citations(
            "## Current Issues\nDiabetes noted [S1]",
            supplied_sections,
        )

        assert len(validation.unsupported_citations) == 1
        assert validation.unsupported_citations[0].source_ids == {"S1"}

    def test_validator_does_not_flag_when_one_of_multiple_sources_supports_line(self):
        supplied_sections = [
            SourceSection(
                label="Mixed (2)",
                items=[
                    _source_item("S1", "Patient", "patient-001", "Jane Doe"),
                    _source_item("S2", "Condition", "c1", "Hypertension"),
                ],
            )
        ]

        validation = validate_citations(
            "## Current Issues\nHypertension noted [S1, S2]",
            supplied_sections,
        )

        assert validation.unsupported_citations == []

    def test_agent_yields_and_prompts_with_supplied_sections(self, monkeypatch):
        supplied_sections = [
            SourceSection(
                label="Patient (1)",
                items=[_source_item("S1", "Patient", "patient-001", "Jane Doe")],
            )
        ]

        def _fake_context_result(_sections):
            return SourceContextResult(
                text="=== Source-Indexed FHIR Context ===\n[S1] Patient/patient-001: Jane Doe",
                sections=supplied_sections,
                supplied_source_ids={"S1"},
                truncated=True,
            )

        monkeypatch.setattr("src.agent.build_source_context_result", _fake_context_result)
        agent, _, llm = _make_agent(stream_contents=("## Current Issues\n", "Jane Doe [S1]"))

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))
        user_message = llm.chat.completions.create.call_args.kwargs["messages"][1]["content"]

        assert chunks[0][1] == supplied_sections
        assert chunks[0][2] == SOURCE_CONTEXT_TRUNCATED_WARNING
        assert "[S1]" in user_message
        assert "[S2]" not in user_message

    def test_agent_retrieves_sources_before_building_supplied_context(self, monkeypatch):
        monkeypatch.setenv("VECTOR_SEARCH_MAX_ITEMS", "2")
        all_sections = [
            SourceSection(
                label="Patient (1)",
                items=[_source_item("S1", "Patient", "patient-001", "Jane Doe")],
            ),
            SourceSection(
                label="Conditions (2)",
                items=[
                    _source_item("S2", "Condition", "c1", "Hypertension"),
                    _source_item("S3", "Condition", "c2", "Diabetes"),
                ],
            ),
        ]
        monkeypatch.setattr(
            "src.agent.build_patient_scope_retrieval_query",
            lambda _role: "hypertension",
        )
        agent, _, llm = _make_agent(
            stream_contents=("## Current Issues\nHypertension noted [S2]",)
        )
        monkeypatch.setattr(agent, "_build_source_sections", lambda _resources: all_sections)

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))
        user_message = llm.chat.completions.create.call_args.kwargs["messages"][1]["content"]

        assert [item.source_id for section in chunks[0][1] for item in section.items] == [
            "S1",
            "S2",
        ]
        assert "[S2]" in user_message
        assert "[S3]" not in user_message
        assert isinstance(chunks[0][5], SourceScopeInfo)
        assert chunks[0][5].retrieved_source_ids == {"S1", "S2"}
        assert chunks[0][5].supplied_source_ids == {"S1", "S2"}

    def test_agent_final_chunk_includes_validation_and_cited_scope(self):
        supplied_sections = [
            SourceSection(
                label="Conditions (1)",
                items=[_source_item("S1", "Condition", "c1", "Hypertension")],
            )
        ]
        agent, _, _ = _make_agent(
            stream_contents=("## Current Issues\nHypertension noted [S1]",)
        )
        agent._build_source_sections = lambda _resources: supplied_sections

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))

        assert len(chunks[-1]) == 6
        assert chunks[-1][4].has_errors is False
        assert chunks[-1][5].cited_source_ids == {"S1"}

    def test_citation_repair_prompt_uses_only_supplied_source_index(self, monkeypatch):
        supplied_sections = [
            SourceSection(
                label="Conditions (1)",
                items=[_source_item("S1", "Condition", "c1", "Hypertension")],
            )
        ]

        def _fake_context_result(_sections):
            return SourceContextResult(
                text="=== Source-Indexed FHIR Context ===\n[S1] Condition/c1: Hypertension",
                sections=supplied_sections,
                supplied_source_ids={"S1"},
                truncated=True,
            )

        llm = MagicMock()
        llm.chat.completions.create.side_effect = [
            [_stream_chunk("## Current Issues\nDiabetes noted [S2]")],
            {
                "choices": [
                    {
                        "message": {
                            "content": "## Current Issues\nHypertension noted [S1]"
                        }
                    }
                ]
            },
        ]
        monkeypatch.setattr("src.agent.build_source_context_result", _fake_context_result)
        agent = SummaryAgent(fhir_client=_make_fhir_client(), llm_client=llm)

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))
        repair_user_prompt = llm.chat.completions.create.call_args_list[1].kwargs[
            "messages"
        ][1]["content"]
        valid_source_index = repair_user_prompt.split("Valid source index:\n", 1)[1]

        assert chunks[-1][0] == "## Current Issues\nHypertension noted [S1]"
        assert "[S1]" in valid_source_index
        assert "[S2]" not in valid_source_index

    def test_citation_repair_prompt_includes_cited_source_evidence_by_line(self, monkeypatch):
        supplied_sections = [
            SourceSection(
                label="Conditions (1)",
                items=[_source_item("S1", "Condition", "c1", "Hypertension")],
            )
        ]

        def _fake_context_result(_sections):
            return SourceContextResult(
                text="=== Source-Indexed FHIR Context ===\n[S1] Condition/c1: Hypertension",
                sections=supplied_sections,
                supplied_source_ids={"S1"},
                truncated=False,
            )

        llm = MagicMock()
        llm.chat.completions.create.side_effect = [
            [_stream_chunk("## Current Issues\nDiabetes noted without citation\nHypertension [S1]")],
            {
                "choices": [
                    {
                        "message": {
                            "content": "## Current Issues\nHypertension noted [S1]"
                        }
                    }
                ]
            },
        ]
        monkeypatch.setattr("src.agent.build_source_context_result", _fake_context_result)
        agent = SummaryAgent(fhir_client=_make_fhir_client(), llm_client=llm)

        list(agent.generate_summary_stream("patient-001", "ED Doctor"))

        repair_user_prompt = llm.chat.completions.create.call_args_list[1].kwargs[
            "messages"
        ][1]["content"]
        assert "Cited source evidence by summary line:" in repair_user_prompt
        assert "- Summary line: Hypertension [S1]" in repair_user_prompt
        assert "[S1] Condition/c1: Hypertension" in repair_user_prompt
        assert "summary: Hypertension" in repair_user_prompt

    def test_unsupported_citation_triggers_repair(self, monkeypatch):
        supplied_sections = [
            SourceSection(
                label="Conditions (1)",
                items=[_source_item("S1", "Condition", "c1", "Hypertension")],
            )
        ]

        def _fake_context_result(_sections):
            return SourceContextResult(
                text="=== Source-Indexed FHIR Context ===\n[S1] Condition/c1: Hypertension",
                sections=supplied_sections,
                supplied_source_ids={"S1"},
                truncated=False,
            )

        llm = MagicMock()
        llm.chat.completions.create.side_effect = [
            [_stream_chunk("## Current Issues\nDiabetes noted [S1]")],
            {
                "choices": [
                    {
                        "message": {
                            "content": "## Current Issues\nHypertension noted [S1]"
                        }
                    }
                ]
            },
        ]
        monkeypatch.setattr("src.agent.build_source_context_result", _fake_context_result)
        agent = SummaryAgent(fhir_client=_make_fhir_client(), llm_client=llm)

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))
        repair_user_prompt = llm.chat.completions.create.call_args_list[1].kwargs[
            "messages"
        ][1]["content"]

        assert "may not support the stated fact" in repair_user_prompt
        assert chunks[-1][0] == "## Current Issues\nHypertension noted [S1]"
        assert chunks[-1][3] == ""

    def test_repair_residual_errors_return_citation_warning(self, monkeypatch):
        supplied_sections = [
            SourceSection(
                label="Conditions (1)",
                items=[_source_item("S1", "Condition", "c1", "Hypertension")],
            )
        ]

        def _fake_context_result(_sections):
            return SourceContextResult(
                text="=== Source-Indexed FHIR Context ===\n[S1] Condition/c1: Hypertension",
                sections=supplied_sections,
                supplied_source_ids={"S1"},
                truncated=False,
            )

        llm = MagicMock()
        llm.chat.completions.create.side_effect = [
            [_stream_chunk("## Current Issues\nDiabetes noted [S1]")],
            {
                "choices": [
                    {
                        "message": {
                            "content": "## Current Issues\nDiabetes still noted [S1]"
                        }
                    }
                ]
            },
        ]
        monkeypatch.setattr("src.agent.build_source_context_result", _fake_context_result)
        agent = SummaryAgent(fhir_client=_make_fhir_client(), llm_client=llm)

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))

        assert chunks[-1][0] == "## Current Issues\nDiabetes still noted [S1]"
        assert chunks[-1][3] == CITATION_VALIDATION_WARNING

    def test_repair_residual_invalid_source_ids_return_citation_warning(self, monkeypatch):
        supplied_sections = [
            SourceSection(
                label="Conditions (1)",
                items=[_source_item("S1", "Condition", "c1", "Hypertension")],
            )
        ]

        def _fake_context_result(_sections):
            return SourceContextResult(
                text="=== Source-Indexed FHIR Context ===\n[S1] Condition/c1: Hypertension",
                sections=supplied_sections,
                supplied_source_ids={"S1"},
                truncated=False,
            )

        llm = MagicMock()
        llm.chat.completions.create.side_effect = [
            [_stream_chunk("## Current Issues\nHypertension noted [S2]")],
            {
                "choices": [
                    {
                        "message": {
                            "content": "## Current Issues\nHypertension still noted [S2]"
                        }
                    }
                ]
            },
        ]
        monkeypatch.setattr("src.agent.build_source_context_result", _fake_context_result)
        agent = SummaryAgent(fhir_client=_make_fhir_client(), llm_client=llm)

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))

        assert chunks[-1][0] == "## Current Issues\nHypertension still noted [S2]"
        assert chunks[-1][3] == CITATION_VALIDATION_WARNING

    def test_repair_residual_uncited_lines_return_citation_warning(self, monkeypatch):
        supplied_sections = [
            SourceSection(
                label="Conditions (1)",
                items=[_source_item("S1", "Condition", "c1", "Hypertension")],
            )
        ]

        def _fake_context_result(_sections):
            return SourceContextResult(
                text="=== Source-Indexed FHIR Context ===\n[S1] Condition/c1: Hypertension",
                sections=supplied_sections,
                supplied_source_ids={"S1"},
                truncated=False,
            )

        llm = MagicMock()
        llm.chat.completions.create.side_effect = [
            [_stream_chunk("## Current Issues\nHypertension remains elevated today")],
            {
                "choices": [
                    {
                        "message": {
                            "content": "## Current Issues\nHypertension remains elevated today"
                        }
                    }
                ]
            },
        ]
        monkeypatch.setattr("src.agent.build_source_context_result", _fake_context_result)
        agent = SummaryAgent(fhir_client=_make_fhir_client(), llm_client=llm)

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))

        assert chunks[-1][0] == "## Current Issues\nHypertension remains elevated today"
        assert chunks[-1][3] == CITATION_VALIDATION_WARNING

    def test_strict_mode_returns_error_when_citation_errors_remain(self, monkeypatch):
        monkeypatch.setenv("CITATION_STRICT_MODE", "true")
        monkeypatch.setenv("CITATION_REPAIR_ENABLED", "false")
        supplied_sections = [
            SourceSection(
                label="Conditions (1)",
                items=[_source_item("S1", "Condition", "c1", "Hypertension")],
            )
        ]

        def _fake_context_result(_sections):
            return SourceContextResult(
                text="=== Source-Indexed FHIR Context ===\n[S1] Condition/c1: Hypertension",
                sections=supplied_sections,
                supplied_source_ids={"S1"},
                truncated=False,
            )

        monkeypatch.setattr("src.agent.build_source_context_result", _fake_context_result)
        agent, _, _ = _make_agent(stream_contents=("## Current Issues\nDiabetes noted [S1]",))

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))

        assert chunks[-1][0].startswith("**Error:** Citation validation failed")
        assert chunks[-1][3] == ""

    def test_pagination_warnings_are_yielded_with_source_warning(self):
        fhir = _make_fhir_client()
        fhir.pagination_warnings = [
            "FHIR pagination stopped after 1 pages for Observation; additional matching resources may not be included."
        ]
        llm = _make_llm_stream("## Current Issues\n- Hypertension [S1]")
        agent = SummaryAgent(fhir_client=fhir, llm_client=llm)

        chunks = list(agent.generate_summary_stream("patient-001", "ED Doctor"))

        assert "FHIR pagination stopped" in chunks[0][2]


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
