"""Repair generated summaries that contain missing or invalid source citations."""

from __future__ import annotations

from src.models import SourceSection
from src.tools.citation_validator import CitationValidationResult
from src.tools.source_items import build_source_context, flatten_source_items

_MAX_EVIDENCE_LINES_PER_SOURCE = 12


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
    if validation.unsupported_citations:
        unsupported_lines = [
            f"{finding.line} cited {sorted(finding.source_ids)}: {finding.reason}"
            for finding in validation.unsupported_citations
        ]
        errors.append(
            "Lines with citations that may not support the stated fact:\n"
            + "\n".join(unsupported_lines)
        )
    cited_evidence = _build_cited_source_evidence(source_sections, validation)

    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Repair source citations in the supplied summary. Use only the valid "
                    "source index. For each line, verify that the cited source evidence "
                    "supports the claim. Replace citations that point to unsupported "
                    "evidence, or remove factual claims that are unsupported by any "
                    "supplied source. Do not cite source ids absent from the source index. "
                    "Preserve exactly these headings: "
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
                    "Cited source evidence by summary line:\n"
                    f"{cited_evidence}\n\n"
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


def _build_cited_source_evidence(
    source_sections: list[SourceSection],
    validation: CitationValidationResult,
) -> str:
    source_by_id = {item.source_id: item for item in flatten_source_items(source_sections)}
    if not validation.cited_lines:
        return "(No cited summary lines.)"

    blocks: list[str] = []
    for cited_line in validation.cited_lines:
        blocks.append(f"- Summary line: {cited_line.text}")
        for source_id in sorted(cited_line.source_ids, key=_source_sort_key):
            item = source_by_id.get(source_id)
            if item is None:
                blocks.append(f"  - [{source_id}] invalid or not supplied")
                continue
            blocks.append(
                f"  - [{source_id}] {item.resource_type}/{item.resource_id}: {item.summary}"
            )
            for key, value in list(item.evidence.items())[:_MAX_EVIDENCE_LINES_PER_SOURCE]:
                blocks.append(f"    {key}: {value}")
    return "\n".join(blocks)


def _source_sort_key(source_id: str) -> int:
    try:
        return int(source_id[1:])
    except (ValueError, IndexError):
        return 0


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
