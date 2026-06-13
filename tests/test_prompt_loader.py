"""Prompt loader tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent import SUPPORTED_ROLES, SummaryAgent, _utc_now_iso
from src.tools.prompt_loader import get_role_prompt, get_supported_roles


def _make_agent() -> tuple[SummaryAgent, MagicMock, MagicMock]:
    fhir_client = MagicMock()
    llm_client = MagicMock()
    agent = SummaryAgent(fhir_client=fhir_client, llm_client=llm_client)
    return agent, fhir_client, llm_client


class TestPromptLoader:
    def test_supported_roles_include_all_required_roles(self):
        assert get_supported_roles() == (
            "ED Doctor",
            "Care Manager",
            "Patient",
            "Family Caregiver",
        )
        assert SUPPORTED_ROLES == get_supported_roles()

    @pytest.mark.parametrize("role", get_supported_roles())
    def test_role_prompt_includes_system_policy_and_role_content(self, role):
        prompt = get_role_prompt(role)

        assert "Use only the supplied FHIR context and source index" in prompt
        assert "Every factual claim in the summary should include one or more source ids" in prompt
        assert "## Current Issues" in prompt
        assert "## Recent Changes" in prompt
        assert "## Risks and Follow-up" in prompt
        assert f"# Role: {role}" in prompt

    def test_unsupported_role_raises_clear_error(self):
        with pytest.raises(ValueError, match="Unsupported role"):
            get_role_prompt("Radiologist")


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
        agent, fhir, llm = _make_agent()

        result = list(agent.generate_summary_stream("patient-001", bad_role))

        assert result == [(f"**Error:** Unsupported role: {bad_role}", [])]
        llm.chat.completions.create.assert_not_called()
        fhir.is_available.assert_not_called()


class TestUtcNowIso:
    def test_format_ends_with_z(self):
        ts = _utc_now_iso()
        assert ts.endswith("Z")

    def test_format_contains_t_separator(self):
        ts = _utc_now_iso()
        assert "T" in ts

    def test_format_length(self):
        assert len(_utc_now_iso()) == 20
