"""
Unit tests for SummaryAgent task 7.1:
  - Role-specific prompts (module-level constants)
  - SummaryAgent.__init__()
  - generate_summary() role validation (unsupported role → immediate error result)
  - generate_summary() data-source determination via fhir_client.is_available()
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent import (
    CARE_MANAGER_PROMPT,
    ED_DOCTOR_PROMPT,
    SummaryAgent,
    _utc_now_iso,
)
from src.models import SummaryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(is_available: bool = True) -> tuple[SummaryAgent, MagicMock, MagicMock, MagicMock]:
    """Return a SummaryAgent wired with mock dependencies.

    The FHIRClient.is_available() mock returns *is_available*.
    _fetch_resources is patched so it raises NotImplementedError (the task 7.2 stub).
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

    def test_ed_doctor_prompt_mentions_ed_focus(self):
        """Prompt should reference ED / emergency physician focus per design."""
        assert "Emergency Department" in ED_DOCTOR_PROMPT

    def test_ed_doctor_prompt_instructs_omit_care_management(self):
        """Design doc says 'Do not include care management goals'."""
        assert "care management" in ED_DOCTOR_PROMPT.lower() or "care management goals" in ED_DOCTOR_PROMPT

    def test_care_manager_prompt_mentions_care_coordination(self):
        assert "Care Manager" in CARE_MANAGER_PROMPT

    def test_both_prompts_include_section_headers(self):
        """Both prompts must instruct the LLM to emit the three section headers."""
        for prompt in (ED_DOCTOR_PROMPT, CARE_MANAGER_PROMPT):
            assert "## Current Issues" in prompt
            assert "## Recent Changes" in prompt
            assert "## Risks and Follow-up" in prompt

    def test_prompts_are_distinct(self):
        assert ED_DOCTOR_PROMPT != CARE_MANAGER_PROMPT


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
# Role validation — unsupported role returns error result immediately
# ---------------------------------------------------------------------------

class TestRoleValidation:
    @pytest.mark.parametrize("bad_role", [
        "Nurse",
        "ed doctor",          # wrong case
        "care manager",       # wrong case
        "",
        "ED Doctor ",         # trailing space
        " Care Manager",      # leading space
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

    @pytest.mark.parametrize("good_role", ["ED Doctor", "Care Manager"])
    def test_valid_roles_proceed_past_validation(self, good_role):
        """Valid roles should NOT return the 'Unsupported role' error."""
        agent, _, _, _ = _make_agent()
        # _fetch_resources raises NotImplementedError — that's expected here;
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
