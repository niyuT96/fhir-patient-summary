"""Patient-scoped retrieval over complete SourceItem objects."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from src.models import SourceItem, SourceSection
from src.tools.source_items import flatten_source_items

DEFAULT_VECTOR_SEARCH_ENABLED = True
DEFAULT_VECTOR_SEARCH_MAX_ITEMS = 40
DEFAULT_VECTOR_SEARCH_BACKEND = "local"
DEFAULT_VECTOR_SEARCH_EMBEDDING_MODEL = "text-embedding-3-small"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_LABEL_COUNT_RE = re.compile(r"^(.*?)\s+\(\d+\)$")

_STOP_WORDS = {
    "about",
    "after",
    "all",
    "and",
    "any",
    "are",
    "based",
    "before",
    "care",
    "current",
    "data",
    "doctor",
    "family",
    "fhir",
    "follow",
    "for",
    "from",
    "issue",
    "issues",
    "manager",
    "patient",
    "recent",
    "role",
    "source",
    "summary",
    "the",
    "this",
    "with",
}

_GENERAL_RETRIEVAL_QUERY = (
    "patient condition medicationrequest medication request allergyintolerance "
    "allergy observation encounter careplan care plan current issues active and "
    "historical conditions diagnoses medications allergies observations labs "
    "vitals encounters recent changes care plans risks follow-up"
)
_ROLE_RETRIEVAL_TERMS = {
    "ED Doctor": "emergency acute chief complaint vitals labs medications allergies encounters",
    "Care Manager": "care coordination care plans follow-up medications risks barriers",
    "Patient": "patient friendly conditions medications allergies follow-up recent changes",
    "Family Caregiver": "family caregiver support medications allergies care plans risks",
}


@dataclass(frozen=True)
class VectorSearchResult:
    """Retrieved patient-scoped SourceItems and retrieval metadata."""

    sections: list[SourceSection]
    retrieved_source_ids: set[str]
    warning: str = ""
    backend: str = DEFAULT_VECTOR_SEARCH_BACKEND
    fallback_used: bool = False


def build_patient_scope_retrieval_query(role: str) -> str:
    """Return a broad clinical retrieval query for the selected summary role."""
    role_terms = _ROLE_RETRIEVAL_TERMS.get(role, "")
    return " ".join(part for part in (_GENERAL_RETRIEVAL_QUERY, role_terms) if part)


def retrieve_patient_scoped_source_sections(
    sections: list[SourceSection],
    *,
    query: str,
    llm_client: Any | None = None,
    enabled: bool | None = None,
    max_items: int | None = None,
    backend: str | None = None,
    embedding_model: str | None = None,
) -> VectorSearchResult:
    """Retrieve relevant SourceItems from the current patient's source sections.

    The function never fetches or accepts global source data. Every returned
    item is the original complete SourceItem object from the supplied sections.
    """
    items = flatten_source_items(sections)
    all_ids = {item.source_id for item in items}
    if not items:
        return VectorSearchResult(sections=[], retrieved_source_ids=set())

    enabled = _env_bool("VECTOR_SEARCH_ENABLED", DEFAULT_VECTOR_SEARCH_ENABLED) if enabled is None else enabled
    max_items = _env_int("VECTOR_SEARCH_MAX_ITEMS", DEFAULT_VECTOR_SEARCH_MAX_ITEMS, minimum=1) if max_items is None else max_items
    backend = _clean_backend(backend or os.environ.get("VECTOR_SEARCH_BACKEND", DEFAULT_VECTOR_SEARCH_BACKEND))
    embedding_model = (
        embedding_model
        or os.environ.get("VECTOR_SEARCH_EMBEDDING_MODEL", DEFAULT_VECTOR_SEARCH_EMBEDDING_MODEL).strip()
        or DEFAULT_VECTOR_SEARCH_EMBEDDING_MODEL
    )

    if not enabled:
        return VectorSearchResult(
            sections=sections,
            retrieved_source_ids=all_ids,
            warning="Vector search is disabled; using all patient-scoped source items.",
            backend="disabled",
            fallback_used=True,
        )

    if backend not in {"local", "openai"}:
        warning = (
            f"Vector search backend '{backend}' is not available; using all "
            "patient-scoped source items."
        )
        return VectorSearchResult(
            sections=sections,
            retrieved_source_ids=all_ids,
            warning=warning,
            backend=backend,
            fallback_used=True,
        )

    if len(items) <= max_items:
        return VectorSearchResult(
            sections=sections,
            retrieved_source_ids=all_ids,
            backend=backend,
        )

    warning = ""
    fallback_used = False
    scores: list[float]
    effective_backend = backend

    if backend == "openai":
        try:
            scores = _rank_with_openai_embeddings(
                items,
                query=query,
                llm_client=llm_client,
                model=embedding_model,
            )
        except Exception as exc:  # noqa: BLE001
            warning = (
                f"OpenAI embedding search failed ({exc}); using local "
                "patient-scoped lexical vector search."
            )
            fallback_used = True
            effective_backend = "local"
            scores = _rank_with_local_vectors(items, query)
    else:
        scores = _rank_with_local_vectors(items, query)

    if not any(score > 0 for score in scores):
        no_match_warning = (
            "Vector retrieval found no matching source items; using all "
            "patient-scoped source items."
        )
        return VectorSearchResult(
            sections=sections,
            retrieved_source_ids=all_ids,
            warning=_join_warnings(warning, no_match_warning),
            backend=effective_backend,
            fallback_used=True,
        )

    selected_ids = _select_source_ids(items, scores, max_items)
    if not selected_ids:
        return VectorSearchResult(
            sections=sections,
            retrieved_source_ids=all_ids,
            warning=_join_warnings(
                warning,
                "Vector retrieval returned no source items; using all patient-scoped source items.",
            ),
            backend=effective_backend,
            fallback_used=True,
        )

    return VectorSearchResult(
        sections=_filter_sections(sections, selected_ids),
        retrieved_source_ids=selected_ids,
        warning=warning,
        backend=effective_backend,
        fallback_used=fallback_used,
    )


def _rank_with_local_vectors(items: list[SourceItem], query: str) -> list[float]:
    query_counts = Counter(_tokens(query))
    if not query_counts:
        return [0.0 for _ in items]
    query_norm = _norm(query_counts)
    scores: list[float] = []
    for item in items:
        item_counts = Counter(_tokens(_source_item_text(item)))
        scores.append(_cosine(query_counts, query_norm, item_counts))
    return scores


def _rank_with_openai_embeddings(
    items: list[SourceItem],
    *,
    query: str,
    llm_client: Any | None,
    model: str,
) -> list[float]:
    if llm_client is None:
        raise RuntimeError("OpenAI client is not available")

    inputs = [query] + [_source_item_text(item) for item in items]
    response = llm_client.embeddings.create(model=model, input=inputs)
    vectors = _extract_embedding_vectors(response)
    if len(vectors) != len(inputs):
        raise RuntimeError("embedding response did not include every input")

    query_vector = vectors[0]
    query_norm = _dense_norm(query_vector)
    return [
        _dense_cosine(query_vector, query_norm, item_vector)
        for item_vector in vectors[1:]
    ]


def _extract_embedding_vectors(response: Any) -> list[list[float]]:
    data = response.get("data", []) if isinstance(response, dict) else getattr(response, "data", [])
    vectors: list[list[float]] = []
    for item in data:
        embedding = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
        if embedding is None:
            continue
        vectors.append([float(value) for value in embedding])
    return vectors


def _select_source_ids(
    items: list[SourceItem],
    scores: list[float],
    max_items: int,
) -> set[str]:
    selected: list[str] = [
        item.source_id
        for item in items
        if item.resource_type == "Patient"
    ]
    selected_set = set(selected)

    ranked = sorted(
        enumerate(zip(items, scores)),
        key=lambda pair: (-pair[1][1], pair[0]),
    )
    for _, (item, score) in ranked:
        if len(selected_set) >= max_items:
            break
        if score <= 0:
            continue
        if item.source_id in selected_set:
            continue
        selected.append(item.source_id)
        selected_set.add(item.source_id)

    return selected_set


def _filter_sections(
    sections: list[SourceSection],
    selected_source_ids: set[str],
) -> list[SourceSection]:
    filtered: list[SourceSection] = []
    for section in sections:
        selected_items = [
            item
            for item in section.items
            if item.source_id in selected_source_ids
        ]
        if not selected_items:
            continue
        filtered.append(
            SourceSection(
                label=_retrieved_label(section.label, len(selected_items), len(section.items)),
                items=selected_items,
            )
        )
    return filtered


def _retrieved_label(label: str, selected_count: int, original_count: int) -> str:
    if selected_count == original_count:
        return label
    match = _LABEL_COUNT_RE.match(label)
    base_label = match.group(1) if match else label
    return f"{base_label} ({selected_count} retrieved of {original_count})"


def _source_item_text(item: SourceItem) -> str:
    parts = [
        item.resource_type,
        item.resource_id,
        item.summary,
    ]
    for key, value in item.evidence.items():
        parts.append(str(key))
        parts.append(str(value))
    return " ".join(parts)


def _tokens(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOP_WORDS:
            continue
        if token.isdigit() or len(token) >= 3:
            tokens.append(token)
    return tokens


def _cosine(
    query_counts: Counter[str],
    query_norm: float,
    item_counts: Counter[str],
) -> float:
    if not query_norm or not item_counts:
        return 0.0
    dot = sum(query_counts[token] * item_counts.get(token, 0) for token in query_counts)
    item_norm = _norm(item_counts)
    if not item_norm:
        return 0.0
    return dot / (query_norm * item_norm)


def _norm(counts: Counter[str]) -> float:
    return math.sqrt(sum(value * value for value in counts.values()))


def _dense_cosine(query_vector: list[float], query_norm: float, item_vector: list[float]) -> float:
    item_norm = _dense_norm(item_vector)
    if not query_norm or not item_norm:
        return 0.0
    return sum(a * b for a, b in zip(query_vector, item_vector)) / (query_norm * item_norm)


def _dense_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    if minimum is not None and value < minimum:
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name, "").strip().lower()
    if not raw_value:
        return default
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    return default


def _clean_backend(backend: str) -> str:
    return backend.strip().lower() or DEFAULT_VECTOR_SEARCH_BACKEND


def _join_warnings(*warnings: str) -> str:
    return "\n".join(warning for warning in warnings if warning)
