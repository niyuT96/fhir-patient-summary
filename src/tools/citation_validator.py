"""Validate source-id citations in generated summaries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.models import SourceSection
from src.tools.source_items import flatten_source_items

_CITATION_RE = re.compile(r"\[([Ss]\d+(?:\s*,\s*[Ss]\d+)*)\]")
_SOURCE_ID_RE = re.compile(r"S\d+")


@dataclass(frozen=True)
class CitationValidationResult:
    valid_source_ids: set[str]
    invalid_source_ids: set[str] = field(default_factory=set)
    uncited_lines: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.invalid_source_ids or self.uncited_lines)


def validate_citations(
    summary: str,
    source_sections: list[SourceSection],
) -> CitationValidationResult:
    """Check invalid source ids and likely factual lines without citations."""
    valid_ids = {item.source_id for item in flatten_source_items(source_sections)}
    cited_ids = set()
    for match in _CITATION_RE.finditer(summary):
        cited_ids.update(_SOURCE_ID_RE.findall(match.group(0).upper()))

    invalid_ids = {source_id for source_id in cited_ids if source_id not in valid_ids}
    uncited_lines = [
        line.strip()
        for line in summary.splitlines()
        if _looks_like_uncited_fact(line)
    ]
    return CitationValidationResult(
        valid_source_ids=valid_ids,
        invalid_source_ids=invalid_ids,
        uncited_lines=uncited_lines,
    )


def _looks_like_uncited_fact(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("##"):
        return False
    if _CITATION_RE.search(stripped):
        return False
    content = stripped.lstrip("-*0123456789. ").strip()
    if not content:
        return False
    lowered = content.lower()
    if lowered in {"none", "not documented", "unknown"}:
        return False
    return len(content.split()) >= 4
