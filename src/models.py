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
