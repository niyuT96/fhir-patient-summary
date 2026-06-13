"""Validate source-id citations in generated summaries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.models import SourceSection
from src.tools.source_items import flatten_source_items

_CITATION_RE = re.compile(r"\[([Ss]\d+(?:\s*,\s*[Ss]\d+)*)\]")
_SOURCE_ID_RE = re.compile(r"S\d+")


@dataclass(frozen=True)
class CitedLine:
    text: str
    source_ids: set[str]


@dataclass(frozen=True)
class UnsupportedCitationFinding:
    line: str
    source_ids: set[str]
    reason: str


@dataclass(frozen=True)
class CitationValidationResult:
    valid_source_ids: set[str]
    invalid_source_ids: set[str] = field(default_factory=set)
    uncited_lines: list[str] = field(default_factory=list)
    cited_lines: list[CitedLine] = field(default_factory=list)
    unsupported_citations: list[UnsupportedCitationFinding] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(
            self.invalid_source_ids
            or self.uncited_lines
            or self.unsupported_citations
        )


def validate_citations(
    summary: str,
    source_sections: list[SourceSection],
) -> CitationValidationResult:
    """Check invalid source ids and likely factual lines without citations."""
    source_items = flatten_source_items(source_sections)
    source_by_id = {item.source_id: item for item in source_items}
    valid_ids = set(source_by_id)
    cited_ids = set()
    cited_lines: list[CitedLine] = []
    for line in summary.splitlines():
        line_source_ids = _source_ids_in_text(line)
        if line_source_ids:
            cited_lines.append(CitedLine(text=line.strip(), source_ids=line_source_ids))
            cited_ids.update(line_source_ids)

    invalid_ids = {source_id for source_id in cited_ids if source_id not in valid_ids}
    uncited_lines = [
        line.strip()
        for line in summary.splitlines()
        if _looks_like_uncited_fact(line)
    ]
    unsupported_citations = [
        finding
        for cited_line in cited_lines
        if (finding := _unsupported_citation_finding(cited_line, source_by_id)) is not None
    ]
    return CitationValidationResult(
        valid_source_ids=valid_ids,
        invalid_source_ids=invalid_ids,
        uncited_lines=uncited_lines,
        cited_lines=cited_lines,
        unsupported_citations=unsupported_citations,
    )


def _source_ids_in_text(text: str) -> set[str]:
    source_ids: set[str] = set()
    for match in _CITATION_RE.finditer(text):
        source_ids.update(_SOURCE_ID_RE.findall(match.group(0).upper()))
    return source_ids


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


def _unsupported_citation_finding(cited_line: CitedLine, source_by_id: dict) -> UnsupportedCitationFinding | None:
    valid_source_ids = {source_id for source_id in cited_line.source_ids if source_id in source_by_id}
    if not valid_source_ids:
        return None

    line_tokens = _informative_tokens(cited_line.text)
    if not line_tokens:
        return None

    evidence_tokens: set[str] = set()
    for source_id in valid_source_ids:
        evidence_tokens.update(_source_item_tokens(source_by_id[source_id]))

    if line_tokens & evidence_tokens:
        return None

    return UnsupportedCitationFinding(
        line=cited_line.text,
        source_ids=valid_source_ids,
        reason="No informative token overlap between the cited line and cited source evidence.",
    )


_STOP_WORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "been",
    "before",
    "care",
    "change",
    "changes",
    "condition",
    "conditions",
    "current",
    "documented",
    "encounter",
    "encounters",
    "follow",
    "for",
    "from",
    "has",
    "have",
    "his",
    "her",
    "issue",
    "issues",
    "may",
    "medication",
    "medications",
    "not",
    "note",
    "noted",
    "observation",
    "observations",
    "patient",
    "plan",
    "plans",
    "recent",
    "request",
    "requests",
    "resource",
    "risk",
    "risks",
    "source",
    "still",
    "summary",
    "the",
    "this",
    "up",
    "was",
    "were",
    "with",
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _informative_tokens(text: str) -> set[str]:
    text = _CITATION_RE.sub("", text)
    text = text.lstrip("-*0123456789. #").strip().lower()
    tokens = set()
    for token in _TOKEN_RE.findall(text):
        if token in _STOP_WORDS:
            continue
        if token.isdigit() or len(token) >= 4:
            tokens.add(token)
    return tokens


def _source_item_tokens(item) -> set[str]:
    parts = [
        item.resource_type,
        item.resource_id,
        item.summary,
    ]
    for key, value in item.evidence.items():
        parts.append(str(key))
        parts.append(str(value))
    return _informative_tokens(" ".join(parts))
