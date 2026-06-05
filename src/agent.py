"""
SummaryAgent — orchestrates FHIR data retrieval, context extraction, and
LLM invocation to produce a SummaryResult.
"""

from src.models import PatientResources, SummaryResult  # noqa: F401


# ---------------------------------------------------------------------------
# Header → key mapping (case-sensitive, must match exactly on their own line)
# ---------------------------------------------------------------------------
_HEADER_MAP: dict[str, str] = {
    "## Current Issues": "current_issues",
    "## Recent Changes": "recent_changes",
    "## Risks and Follow-up": "risks_and_followup",
}


def parse_sections(raw_text: str) -> dict[str, str]:
    """Parse structured LLM output into the three summary sections.

    Always returns a dict with exactly the keys ``current_issues``,
    ``recent_changes``, and ``risks_and_followup``.  Never raises an
    exception.

    Parsing rules (Requirements 5.1–5.7):
    - If ``raw_text`` is empty → all three values are empty strings.
    - Lines that exactly equal a recognised ``## Header`` marker start a new
      section; content lines are accumulated until the next header or EOF.
    - Each section value is stripped of leading/trailing whitespace.
    - If ``raw_text`` is non-empty but contains *no* recognised headers, the
      full stripped input is placed in ``risks_and_followup`` and the other
      two keys are empty strings (Requirement 5.7 fallback).
    """
    result: dict[str, str] = {
        "current_issues": "",
        "recent_changes": "",
        "risks_and_followup": "",
    }

    # Empty input → return all-empty dict immediately (Req 5.4)
    if not raw_text:
        return result

    current_key: str | None = None
    buffer: list[str] = []
    found_any_header = False

    for line in raw_text.split("\n"):
        if line in _HEADER_MAP:
            # Flush the previous section's buffer
            if current_key is not None:
                result[current_key] = "\n".join(buffer).strip()
            current_key = _HEADER_MAP[line]
            buffer = []
            found_any_header = True
        elif current_key is not None:
            buffer.append(line)

    # Flush the final section's buffer
    if current_key is not None:
        result[current_key] = "\n".join(buffer).strip()

    # Req 5.7 fallback: non-empty input with no recognised headers
    if not found_any_header:
        result["risks_and_followup"] = raw_text.strip()

    return result
