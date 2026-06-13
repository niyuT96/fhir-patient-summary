"""
Shared data models for the Smart Patient Summary Generator.

All modules import PatientResources and SummaryResult from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class PatientResources:
    """Holds the seven FHIR resource collections retrieved for a single patient."""

    patient: dict
    conditions: list[dict] = field(default_factory=list)
    medications: list[dict] = field(default_factory=list)
    allergies: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    encounters: list[dict] = field(default_factory=list)
    care_plans: list[dict] = field(default_factory=list)


@dataclass
class SourceItem:
    """One citeable FHIR evidence item shown in the UI and prompt context."""

    source_id: str
    label: str
    resource_type: str
    resource_id: str
    summary: str
    evidence: dict[str, Any]
    raw_resource: dict[str, Any]


@dataclass
class SourceSection:
    """One labelled group of citeable source data items shown in the UI."""

    label: str          # Display heading, e.g. "Active Conditions"
    items: list[SourceItem]
    hidden_items: list[SourceItem] = field(default_factory=list)


@dataclass
class SourceScopeInfo:
    """Source id scopes for retrieval, prompt supply, and final summary citations."""

    retrieved_source_ids: set[str] = field(default_factory=set)
    supplied_source_ids: set[str] = field(default_factory=set)
    cited_source_ids: set[str] = field(default_factory=set)
    retrieval_strategy: str = ""


@dataclass
class SummaryResult:
    """Holds the three summary sections, metadata, and an optional error field."""

    patient_name: str
    patient_id: str
    role: str
    current_issues: str
    recent_changes: str
    risks_and_followup: str
    data_source: Literal["fhir_server", "local_fallback"]
    generated_at: str  # ISO 8601 UTC timestamp, e.g. "2026-06-05T14:30:00Z"
    error: str | None = None
    # Structured source data shown in the "Data Sources" panel in the UI.
    # Each entry is one collapsible section (e.g. Conditions, Medications...).
    source_sections: list[SourceSection] = field(default_factory=list)
