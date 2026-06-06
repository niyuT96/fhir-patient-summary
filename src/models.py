"""
Shared data models for the Smart Patient Summary Generator.

All modules import PatientResources and SummaryResult from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


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
class SourceSection:
    """One labelled group of source data items shown in the UI.

    Example:
        label = "Recent Observations"
        items = ["Blood Pressure: 120/80 mmHg (2026-01-15)",
                 "Heart Rate: 72 /min (2026-01-15)"]
    """

    label: str          # Display heading, e.g. "Active Conditions"
    items: list[str]    # Formatted text lines extracted from FHIR resources
    hidden_items: list[str] = field(default_factory=list)


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
