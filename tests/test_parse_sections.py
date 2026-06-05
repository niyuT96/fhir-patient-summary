"""
Unit tests for parse_sections() (task 4.1).

Covers Requirements 5.1–5.7:
- Always returns a dict with exactly the three expected keys
- Never raises an exception
- Correctly parses all three headers
- Handles missing headers (empty string values)
- Handles empty input
- Strips leading/trailing whitespace from section values
- Non-header fallback: full text goes to risks_and_followup
"""

import pytest

from src.agent import parse_sections


# ---------------------------------------------------------------------------
# Return-shape invariants (Req 5.1)
# ---------------------------------------------------------------------------

class TestReturnShape:
    REQUIRED_KEYS = {"current_issues", "recent_changes", "risks_and_followup"}

    def test_returns_dict_for_empty_string(self):
        result = parse_sections("")
        assert isinstance(result, dict)
        assert set(result.keys()) == self.REQUIRED_KEYS

    def test_returns_dict_for_well_formed_input(self):
        text = "## Current Issues\nfoo\n## Recent Changes\nbar\n## Risks and Follow-up\nbaz"
        result = parse_sections(text)
        assert set(result.keys()) == self.REQUIRED_KEYS

    def test_returns_dict_for_random_text(self):
        result = parse_sections("some random text with no headers at all")
        assert set(result.keys()) == self.REQUIRED_KEYS

    def test_never_raises_on_any_string(self):
        """parse_sections() must not raise for any string input."""
        inputs = [
            "",
            "   ",
            "\n\n\n",
            "## Unknown Header\ncontent",
            "## Current Issues",        # header with no content
            "no headers whatsoever",
            "## Risks and Follow-up\n## Current Issues\n## Recent Changes",
        ]
        for text in inputs:
            result = parse_sections(text)
            assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Empty input (Req 5.4)
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_string_returns_all_empty_values(self):
        result = parse_sections("")
        assert result["current_issues"] == ""
        assert result["recent_changes"] == ""
        assert result["risks_and_followup"] == ""


# ---------------------------------------------------------------------------
# Well-formed input with all three headers (Req 5.2)
# ---------------------------------------------------------------------------

class TestWellFormedInput:
    def _well_formed(self):
        return (
            "## Current Issues\n"
            "- Hypertension\n"
            "- Diabetes\n"
            "## Recent Changes\n"
            "- Started metformin\n"
            "## Risks and Follow-up\n"
            "- Follow up in 3 months\n"
        )

    def test_current_issues_parsed_correctly(self):
        result = parse_sections(self._well_formed())
        assert "Hypertension" in result["current_issues"]
        assert "Diabetes" in result["current_issues"]

    def test_recent_changes_parsed_correctly(self):
        result = parse_sections(self._well_formed())
        assert "metformin" in result["recent_changes"]

    def test_risks_and_followup_parsed_correctly(self):
        result = parse_sections(self._well_formed())
        assert "Follow up" in result["risks_and_followup"]

    def test_section_content_does_not_bleed_across_headers(self):
        result = parse_sections(self._well_formed())
        assert "Hypertension" not in result["recent_changes"]
        assert "metformin" not in result["current_issues"]

    def test_header_line_itself_not_included_in_value(self):
        result = parse_sections(self._well_formed())
        assert "## Current Issues" not in result["current_issues"]
        assert "## Recent Changes" not in result["recent_changes"]
        assert "## Risks and Follow-up" not in result["risks_and_followup"]


# ---------------------------------------------------------------------------
# Missing headers (Req 5.5) — absent keys default to empty string
# ---------------------------------------------------------------------------

class TestMissingHeaders:
    def test_missing_current_issues_defaults_to_empty(self):
        text = "## Recent Changes\nbar\n## Risks and Follow-up\nbaz"
        result = parse_sections(text)
        assert result["current_issues"] == ""

    def test_missing_recent_changes_defaults_to_empty(self):
        text = "## Current Issues\nfoo\n## Risks and Follow-up\nbaz"
        result = parse_sections(text)
        assert result["recent_changes"] == ""

    def test_missing_risks_and_followup_defaults_to_empty(self):
        text = "## Current Issues\nfoo\n## Recent Changes\nbar"
        result = parse_sections(text)
        assert result["risks_and_followup"] == ""

    def test_only_one_header_present(self):
        text = "## Current Issues\nsome issue"
        result = parse_sections(text)
        assert result["current_issues"] == "some issue"
        assert result["recent_changes"] == ""
        assert result["risks_and_followup"] == ""


# ---------------------------------------------------------------------------
# Header present but no content (Req 5.3)
# ---------------------------------------------------------------------------

class TestEmptySection:
    def test_header_with_no_content_gives_empty_string(self):
        text = "## Current Issues\n## Recent Changes\ncontent"
        result = parse_sections(text)
        assert result["current_issues"] == ""

    def test_final_header_with_no_content_gives_empty_string(self):
        text = "## Current Issues\ncontent\n## Recent Changes\n## Risks and Follow-up"
        result = parse_sections(text)
        assert result["recent_changes"] == ""
        assert result["risks_and_followup"] == ""


# ---------------------------------------------------------------------------
# Whitespace stripping (Req 5.6)
# ---------------------------------------------------------------------------

class TestWhitespaceStripping:
    def test_leading_blank_lines_stripped(self):
        text = "## Current Issues\n\n\n- issue one\n## Recent Changes\ncontent"
        result = parse_sections(text)
        assert not result["current_issues"].startswith("\n")

    def test_trailing_blank_lines_stripped(self):
        text = "## Current Issues\n- issue one\n\n\n## Recent Changes\ncontent"
        result = parse_sections(text)
        assert not result["current_issues"].endswith("\n")

    def test_internal_content_preserved(self):
        text = "## Current Issues\n\nfirst line\n\nsecond line\n\n## Recent Changes\ncontent"
        result = parse_sections(text)
        assert "first line" in result["current_issues"]
        assert "second line" in result["current_issues"]


# ---------------------------------------------------------------------------
# Non-header fallback (Req 5.7)
# ---------------------------------------------------------------------------

class TestNonHeaderFallback:
    def test_no_headers_puts_text_in_risks_and_followup(self):
        text = "This is a plain text response with no headers."
        result = parse_sections(text)
        assert result["current_issues"] == ""
        assert result["recent_changes"] == ""
        assert result["risks_and_followup"] == text.strip()

    def test_no_headers_full_text_preserved(self):
        text = "Line one.\nLine two.\nLine three."
        result = parse_sections(text)
        assert result["risks_and_followup"] == text.strip()

    def test_no_headers_whitespace_only_input(self):
        """Whitespace-only input: after strip it is empty, so all values empty."""
        text = "   \n\n   "
        result = parse_sections(text)
        # stripped text is "", so risks_and_followup should also be ""
        assert result["risks_and_followup"] == ""
        assert result["current_issues"] == ""
        assert result["recent_changes"] == ""

    def test_partial_header_not_recognised(self):
        """A line that almost matches a header must not trigger parsing."""
        text = "## current issues\nsome content"  # wrong case
        result = parse_sections(text)
        # no recognised headers → fallback
        assert result["current_issues"] == ""
        assert result["risks_and_followup"] == text.strip()

    def test_header_with_extra_whitespace_not_recognised(self):
        """A header with trailing spaces is NOT a match (case-sensitive, exact)."""
        text = "## Current Issues  \ncontent"
        result = parse_sections(text)
        # trailing space makes it non-matching → fallback
        assert result["current_issues"] == ""
        assert result["risks_and_followup"] == text.strip()
