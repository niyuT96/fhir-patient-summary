"""Repair generated summaries that contain missing or invalid source citations."""

from __future__ import annotations

from src.models import SourceSection
from src.tools.citation_validator import CitationValidationResult
from src.tools.source_items import build_source_context


def repair_summary_citations(
    *,
    llm_client,
    model: str,
    temperature: float,
    summary: str,
    source_sections: list[SourceSection],
    validation: CitationValidationResult,
) -> str:
    """Ask the LLM to repair citations without adding unsupported facts."""
    source_context = build_source_context(source_sections)
    errors = []
    if validation.invalid_source_ids:
        errors.append(f"Invalid source ids: {sorted(validation.invalid_source_ids)}")
    if validation.uncited_lines:
        errors.append("Lines that may need citations:\n" + "\n".join(validation.uncited_lines))

    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Repair source citations in the supplied summary. Use only the valid "
                    "source index. Add missing citations, replace invalid citations, or "
                    "remove unsupported factual claims. Preserve exactly these headings: "
                    "## Current Issues, ## Recent Changes, ## Risks and Follow-up."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summary to repair:\n"
                    f"{summary}\n\n"
                    "Validator findings:\n"
                    f"{chr(10).join(errors)}\n\n"
                    "Valid source index:\n"
                    f"{source_context}"
                ),
            },
        ],
        temperature=min(temperature, 0.2),
        max_tokens=1200,
        stream=False,
    )
    repaired = _extract_message_content(response)
    return repaired.strip() if repaired else summary


def _extract_message_content(response) -> str:
    if isinstance(response, dict):
        try:
            return response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        return getattr(message, "content", None) or ""
    except (AttributeError, IndexError, TypeError):
        return ""
