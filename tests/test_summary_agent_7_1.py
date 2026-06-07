"""
Unit tests for SummaryAgent:
  - Role-specific prompts (module-level constants)
  - SummaryAgent.__init__()
  - generate_summary() role validation (unsupported role -> immediate error result)
  - generate_summary() data-source determination via fhir_client.is_available()
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent import (
    CARE_MANAGER_PROMPT,
    DECEASED_RECORD_RULES,
    ED_DOCTOR_PROMPT,
    FAMILY_CAREGIVER_PROMPT,
    LIVING_PATIENT_RULES,
    PATIENT_PROMPT,
    SUPPORTED_ROLES,
    SummaryAgent,
    VOICE_AND_AUDIENCE_RULES,
    _SECTION_PROMPTS,
    _utc_now_iso,
)
from src.models import SummaryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(is_available: bool = True) -> tuple[SummaryAgent, MagicMock, MagicMock, MagicMock]:
    """Return a SummaryAgent wired with mock dependencies.

    The FHIRClient.is_available() mock returns *is_available*.
    _fetch_resources is patched so it raises NotImplementedError.
    """
    fhir_client = MagicMock()
    fhir_client.is_available.return_value = is_available
    extractor = MagicMock()
    llm_client = MagicMock()
    agent = SummaryAgent(fhir_client=fhir_client, extractor=extractor, llm_client=llm_client)
    return agent, fhir_client, extractor, llm_client


# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

class TestPromptConstants:
    def test_ed_doctor_prompt_is_non_empty_string(self):
        assert isinstance(ED_DOCTOR_PROMPT, str)
        assert len(ED_DOCTOR_PROMPT) > 0

    def test_care_manager_prompt_is_non_empty_string(self):
        assert isinstance(CARE_MANAGER_PROMPT, str)
        assert len(CARE_MANAGER_PROMPT) > 0

    def test_patient_prompt_is_non_empty_string(self):
        assert isinstance(PATIENT_PROMPT, str)
        assert len(PATIENT_PROMPT) > 0

    def test_family_caregiver_prompt_is_non_empty_string(self):
        assert isinstance(FAMILY_CAREGIVER_PROMPT, str)
        assert len(FAMILY_CAREGIVER_PROMPT) > 0

    def test_ed_doctor_prompt_mentions_ed_focus(self):
        """Prompt should reference ED / emergency physician focus per design."""
        assert "Emergency Department" in ED_DOCTOR_PROMPT

    def test_ed_doctor_prompt_instructs_omit_care_management(self):
        """Design doc says 'Do not include care management goals'."""
        assert "care management" in ED_DOCTOR_PROMPT.lower() or "care management goals" in ED_DOCTOR_PROMPT

    def test_care_manager_prompt_mentions_care_coordination(self):
        assert "Care Manager" in CARE_MANAGER_PROMPT

    def test_both_prompts_include_section_headers(self):
        """All prompts must instruct the LLM to emit the three section headers."""
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
            assert "unless explicitly documented" in prompt
            assert "FHIR-listed active diagnoses before death" in prompt
            assert "Do not say 'medications at the time of death'" in prompt
            assert "Do not use vague phrases like 'documentation gaps may exist'" in prompt
            assert "do not frame missing recent vitals/labs as an active concern" in prompt
            assert "Avoid repeating the same death date/cause in every section" in prompt

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
            assert "ED Doctor: write in concise third-person clinical chart style" in prompt
            assert "Care Manager: write in third-person care-coordination style" in prompt
            assert "Patient: for living patients, write directly to the patient using \"you\"" in prompt
            assert "If the patient is deceased, do not address the patient as \"you\"" in prompt
            assert "Family Caregiver: for living patients, write to the caregiver" in prompt
            assert "Never mix voices within the same summary" in prompt

    def test_care_manager_prompt_switches_deceased_patients_to_chart_review(self):
        assert "For deceased patients, switch to retrospective chart review only" in CARE_MANAGER_PROMPT
        assert "Do not create new care tasks" in CARE_MANAGER_PROMPT
        assert "actionable care coordination items only for living patients" in CARE_MANAGER_PROMPT
        assert "documented before death" in CARE_MANAGER_PROMPT
        assert "medications at the time of death" in CARE_MANAGER_PROMPT
        assert "unless a medication period overlaps the death date" in CARE_MANAGER_PROMPT
        assert "Do not use vague phrases like 'documentation gaps may exist'" in CARE_MANAGER_PROMPT
        assert "Do not list historical care plans as follow-up needs" in CARE_MANAGER_PROMPT

    def test_section_prompts_place_deceased_details_without_repetition(self):
        current = _SECTION_PROMPTS["Current Issues"]
        recent = _SECTION_PROMPTS["Recent Changes"]
        risks = _SECTION_PROMPTS["Risks and Follow-up"]

        assert "include deceased status and documented cause here once" in current
        assert "chronological event timeline" in recent
        assert "do not restate general deceased status from Current Issues" in recent
        assert "do not repeat death date/cause" in risks
        assert "Do not summarize diagnoses again" in risks
        assert "no active follow-up applies" in risks
        assert "missing vitals/labs raise concerns" in risks
        assert "retrospective documentation limitations" in risks


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Role validation - unsupported role returns error result immediately
# ---------------------------------------------------------------------------

class TestRoleValidation:
    @pytest.mark.parametrize("bad_role", [
        "Nurse",
        "ed doctor",          # wrong case
        "care manager",       # wrong case
        "",
        "ED Doctor ",         # trailing space
        " Care Manager",      # leading space
        "Family caregiver",   # wrong case
        "admin",
        "123",
    ])
    def test_unsupported_role_returns_error_without_calling_llm(self, bad_role):
        agent, fhir, extractor, llm = _make_agent()
        result = agent.generate_summary("patient-001", bad_role)

        assert isinstance(result, SummaryResult)
        assert result.error == f"Unsupported role: {bad_role}"
        # LLM must NOT be called
        llm.chat.completions.create.assert_not_called()
        # is_available / fhir calls also not needed
        fhir.is_available.assert_not_called()

    def test_unsupported_role_result_has_empty_section_fields(self):
        agent, _, _, _ = _make_agent()
        result = agent.generate_summary("p1", "Unknown Role")
        assert result.current_issues == ""
        assert result.recent_changes == ""
        assert result.risks_and_followup == ""

    def test_unsupported_role_result_preserves_patient_id(self):
        agent, _, _, _ = _make_agent()
        result = agent.generate_summary("patient-xyz", "BadRole")
        assert result.patient_id == "patient-xyz"

    def test_unsupported_role_result_preserves_role(self):
        agent, _, _, _ = _make_agent()
        result = agent.generate_summary("p1", "BadRole")
        assert result.role == "BadRole"

    def test_unsupported_role_result_has_generated_at(self):
        agent, _, _, _ = _make_agent()
        result = agent.generate_summary("p1", "BadRole")
        assert result.generated_at != ""
        # Must look like an ISO timestamp
        assert "T" in result.generated_at
        assert result.generated_at.endswith("Z")

    @pytest.mark.parametrize("good_role", list(SUPPORTED_ROLES))
    def test_valid_roles_proceed_past_validation(self, good_role):
        """Valid roles should NOT return the 'Unsupported role' error."""
        agent, _, _, _ = _make_agent()
        # _fetch_resources raises NotImplementedError - that's expected here;
        # we just verify the role error is NOT returned.
        result = agent.generate_summary("p1", good_role)
        # Either error is None or it's something other than "Unsupported role: ..."
        if result.error is not None:
            assert not result.error.startswith("Unsupported role:")


# ---------------------------------------------------------------------------
# Data-source determination
# ---------------------------------------------------------------------------

class TestDataSourceDetermination:
    def test_is_available_called_for_valid_role(self):
        """is_available() must be called once for each valid-role request."""
        agent, fhir, _, _ = _make_agent(is_available=True)
        agent.generate_summary("p1", "ED Doctor")
        fhir.is_available.assert_called_once()

    def test_data_source_fhir_server_when_available(self):
        """When is_available() is True, data_source must be 'fhir_server'."""
        agent, fhir, _, _ = _make_agent(is_available=True)
        result = agent.generate_summary("p1", "ED Doctor")
        assert result.data_source == "fhir_server"

    def test_data_source_local_fallback_when_unavailable(self):
        """When is_available() is False, data_source must be 'local_fallback'."""
        agent, fhir, _, _ = _make_agent(is_available=False)
        result = agent.generate_summary("p1", "Care Manager")
        assert result.data_source == "local_fallback"
        # is_available must have been consulted to reach this branch
        fhir.is_available.assert_called_once()

    def test_is_available_not_called_for_unsupported_role(self):
        """is_available() must NOT be called before role validation fails."""
        agent, fhir, _, _ = _make_agent()
        agent.generate_summary("p1", "BadRole")
        fhir.is_available.assert_not_called()


# ---------------------------------------------------------------------------
# _utc_now_iso helper
# ---------------------------------------------------------------------------

class TestUtcNowIso:
    def test_format_ends_with_z(self):
        ts = _utc_now_iso()
        assert ts.endswith("Z")

    def test_format_contains_t_separator(self):
        ts = _utc_now_iso()
        assert "T" in ts

    def test_format_length(self):
        # "YYYY-MM-DDTHH:MM:SSZ" = 20 characters
        ts = _utc_now_iso()
        assert len(ts) == 20
