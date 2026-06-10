"""
Prompt and constructor tests for SummaryAgent.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent import (
    CARE_MANAGER_PROMPT,
    DECEASED_RECORD_RULES,
    ED_DOCTOR_PROMPT,
    FAMILY_CAREGIVER_PROMPT,
    LIVING_PATIENT_RULES,
    PATIENT_PROMPT,
    SUMMARY_OUTPUT_RULES,
    SUPPORTED_ROLES,
    SummaryAgent,
    VOICE_AND_AUDIENCE_RULES,
    _utc_now_iso,
)


def _make_agent() -> tuple[SummaryAgent, MagicMock, MagicMock, MagicMock]:
    fhir_client = MagicMock()
    extractor = MagicMock()
    llm_client = MagicMock()
    agent = SummaryAgent(fhir_client=fhir_client, extractor=extractor, llm_client=llm_client)
    return agent, fhir_client, extractor, llm_client


class TestPromptConstants:
    def test_role_prompts_are_non_empty_strings(self):
        for prompt in (
            ED_DOCTOR_PROMPT,
            CARE_MANAGER_PROMPT,
            PATIENT_PROMPT,
            FAMILY_CAREGIVER_PROMPT,
        ):
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_ed_doctor_prompt_mentions_ed_focus(self):
        assert "Emergency Department" in ED_DOCTOR_PROMPT

    def test_care_manager_prompt_mentions_care_coordination(self):
        assert "Care Manager" in CARE_MANAGER_PROMPT

    def test_all_prompts_include_section_headers(self):
        for prompt in (
            ED_DOCTOR_PROMPT,
            CARE_MANAGER_PROMPT,
            PATIENT_PROMPT,
            FAMILY_CAREGIVER_PROMPT,
        ):
            assert "## Current Issues" in prompt
            assert "## Recent Changes" in prompt
            assert "## Risks and Follow-up" in prompt

    def test_prompts_are_distinct(self):
        prompts = {
            ED_DOCTOR_PROMPT,
            CARE_MANAGER_PROMPT,
            PATIENT_PROMPT,
            FAMILY_CAREGIVER_PROMPT,
        }
        assert len(prompts) == 4

    def test_supported_roles_include_all_required_roles(self):
        assert SUPPORTED_ROLES == (
            "ED Doctor",
            "Care Manager",
            "Patient",
            "Family Caregiver",
        )

    def test_all_prompts_include_deceased_record_guardrails(self):
        for prompt in (
            ED_DOCTOR_PROMPT,
            CARE_MANAGER_PROMPT,
            PATIENT_PROMPT,
            FAMILY_CAREGIVER_PROMPT,
        ):
            assert DECEASED_RECORD_RULES in prompt
            assert "summarize retrospectively only" in prompt
            assert "Do NOT recommend active treatment" in prompt
            assert "estate management" in prompt
            assert "FHIR-listed active diagnoses before death" in prompt

    def test_all_prompts_include_living_patient_death_field_guardrails(self):
        for prompt in (
            ED_DOCTOR_PROMPT,
            CARE_MANAGER_PROMPT,
            PATIENT_PROMPT,
            FAMILY_CAREGIVER_PROMPT,
        ):
            assert LIVING_PATIENT_RULES in prompt
            assert "Do NOT mention missing death certification" in prompt
            assert "Do NOT infer end-of-life care needs" in prompt

    def test_all_prompts_include_voice_and_audience_rules(self):
        for prompt in (
            ED_DOCTOR_PROMPT,
            CARE_MANAGER_PROMPT,
            PATIENT_PROMPT,
            FAMILY_CAREGIVER_PROMPT,
        ):
            assert VOICE_AND_AUDIENCE_RULES in prompt
            assert "Never mix voices within the same summary" in prompt

    def test_summary_output_rules_request_complete_summary(self):
        assert "Return one complete summary" in SUMMARY_OUTPUT_RULES
        assert "## Current Issues" in SUMMARY_OUTPUT_RULES
        assert "## Recent Changes" in SUMMARY_OUTPUT_RULES
        assert "## Risks and Follow-up" in SUMMARY_OUTPUT_RULES
        assert "Do not add any other headings" in SUMMARY_OUTPUT_RULES


class TestSummaryAgentInit:
    def test_init_stores_fhir_client(self):
        agent, fhir, _, _ = _make_agent()
        assert agent._fhir_client is fhir

    def test_init_stores_extractor(self):
        agent, _, extractor, _ = _make_agent()
        assert agent._extractor is extractor

    def test_init_stores_llm_client(self):
        agent, _, _, llm = _make_agent()
        assert agent._llm_client is llm


class TestRoleValidation:
    @pytest.mark.parametrize(
        "bad_role",
        [
            "Nurse",
            "ed doctor",
            "care manager",
            "",
            "ED Doctor ",
            " Care Manager",
            "Family caregiver",
            "admin",
            "123",
        ],
    )
    def test_unsupported_role_yields_error_without_calling_dependencies(self, bad_role):
        agent, fhir, extractor, llm = _make_agent()

        result = list(agent.generate_summary_stream("patient-001", bad_role))

        assert result == [(f"**Error:** Unsupported role: {bad_role}", [])]
        llm.chat.completions.create.assert_not_called()
        fhir.is_available.assert_not_called()
        extractor.extract.assert_not_called()


class TestUtcNowIso:
    def test_format_ends_with_z(self):
        ts = _utc_now_iso()
        assert ts.endswith("Z")

    def test_format_contains_t_separator(self):
        ts = _utc_now_iso()
        assert "T" in ts

    def test_format_length(self):
        assert len(_utc_now_iso()) == 20
