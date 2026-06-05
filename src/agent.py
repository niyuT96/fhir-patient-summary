"""
SummaryAgent — orchestrates FHIR data retrieval, context extraction, and
LLM invocation to produce a SummaryResult.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.exceptions import FHIRClientError, FHIRUnavailableError
from src.models import PatientResources, SummaryResult  # noqa: F401

if TYPE_CHECKING:
    from openai import OpenAI

    from src.context_extractor import PatientContextExtractor
    from src.fhir_client import FHIRClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Role-specific system prompts (Requirements 4.1, 4.2)
# ---------------------------------------------------------------------------

ED_DOCTOR_PROMPT = """\
You are a clinical AI assistant generating a concise patient summary for an Emergency Department physician.
Focus on: active diagnoses, current medications and allergies (drug safety), the most recent labs and vitals,
and any acute concerns. Be brief, use medical shorthand where appropriate, and highlight anything
immediately actionable. Do not include care management goals or long-term follow-up plans.

Structure your response EXACTLY as:
## Current Issues
<bullet points>

## Recent Changes
<bullet points>

## Risks and Follow-up
<bullet points>"""

CARE_MANAGER_PROMPT = """\
You are a clinical AI assistant generating a patient summary for a Care Manager focused on chronic
disease management and care coordination. Focus on: chronic conditions, medication adherence,
pending care plan goals, upcoming follow-up needs, and social/functional risks.
Use plain clinical language. Include actionable care coordination items.

Structure your response EXACTLY as:
## Current Issues
<bullet points>

## Recent Changes
<bullet points>

## Risks and Follow-up
<bullet points>"""

# Map role names to their prompts (used in generate_summary)
_ROLE_PROMPTS: dict[str, str] = {
    "ED Doctor": ED_DOCTOR_PROMPT,
    "Care Manager": CARE_MANAGER_PROMPT,
}


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


# ---------------------------------------------------------------------------
# FHIR Fetch Algorithm (Requirements 7.1–7.5)
# ---------------------------------------------------------------------------

# Ordered list of (resource_type, base_params) pairs as specified in the
# design document §Algorithmic Pseudocode.  The patient filter param
# (_id for Patient, patient= for others) is injected at call time.
_RESOURCE_FETCH_PLAN: list[tuple[str, dict[str, str]]] = [
    ("Patient",            {}),
    ("Condition",          {"clinical-status": "active"}),
    ("MedicationRequest",  {"status": "active"}),
    ("AllergyIntolerance", {}),
    ("Observation",        {"_sort": "-date", "_count": "20"}),
    ("Encounter",          {"_sort": "-date", "_count": "5"}),
    ("CarePlan",           {"status": "active"}),
]


def _fetch_all_fhir_resources(
    fhir_client: "FHIRClient",
    patient_id: str,
) -> "PatientResources | SummaryResult":
    """Fetch all seven FHIR resource types for *patient_id* with graceful degradation.

    Implements the FHIR Fetch Algorithm from design §Algorithmic Pseudocode
    (Requirements 7.1–7.5):

    - Iterates all seven resource types sequentially, preserving previously
      fetched results throughout (Req 7.5).
    - For **non-Patient** types: on ``FHIRClientError`` or
      ``FHIRUnavailableError``, logs a warning, sets that field to ``[]``,
      and continues fetching remaining types (Req 7.1, 7.4).
    - If the **Patient** fetch raises an error, returns a ``SummaryResult``
      immediately with
      ``error="Failed to fetch Patient {patient_id}: {message}"`` (Req 7.3).
    - If the Patient resource list is empty (0 results), returns a
      ``SummaryResult`` with ``error="Patient {patient_id} not found"`` (Req 7.2).

    Returns:
        ``PatientResources`` on success, or a ``SummaryResult`` with ``error``
        set if the Patient fetch fails or returns no results.
    """
    results: dict[str, list[dict]] = {}

    for resource_type, base_params in _RESOURCE_FETCH_PLAN:
        # Build per-call params with the patient filter injected
        if resource_type == "Patient":
            params: dict[str, str] = {"_id": patient_id}
        else:
            params = dict(base_params)
            params["patient"] = patient_id

        try:
            entries = fhir_client.get_resource(resource_type, patient_id, params)
            results[resource_type] = entries
        except (FHIRClientError, FHIRUnavailableError) as exc:
            if resource_type == "Patient":
                # Patient fetch failure → abort and return error result (Req 7.3)
                return SummaryResult(
                    patient_name="",
                    patient_id=patient_id,
                    role="",
                    current_issues="",
                    recent_changes="",
                    risks_and_followup="",
                    data_source="fhir_server",
                    generated_at=_utc_now_iso(),
                    error=f"Failed to fetch Patient {patient_id}: {exc}",
                )
            # Non-Patient failure → warn, default to [], continue (Req 7.1, 7.4, 7.5)
            logger.warning("Failed to fetch %s: %s", resource_type, exc)
            results[resource_type] = []

    # Patient list empty → patient not found (Req 7.2)
    if not results.get("Patient"):
        return SummaryResult(
            patient_name="",
            patient_id=patient_id,
            role="",
            current_issues="",
            recent_changes="",
            risks_and_followup="",
            data_source="fhir_server",
            generated_at=_utc_now_iso(),
            error=f"Patient {patient_id} not found",
        )

    return PatientResources(
        patient=results["Patient"][0],
        conditions=results.get("Condition", []),
        medications=results.get("MedicationRequest", []),
        allergies=results.get("AllergyIntolerance", []),
        observations=results.get("Observation", []),
        encounters=results.get("Encounter", []),
        care_plans=results.get("CarePlan", []),
    )


# ---------------------------------------------------------------------------
# SummaryAgent
# ---------------------------------------------------------------------------

class SummaryAgent:
    """Orchestrates FHIR data retrieval, context extraction, and LLM invocation.

    Responsibilities:
    - Detect whether to use the live FHIR server or the local fallback bundle
    - Fetch all required FHIR resources (via ``_fetch_resources()``)
    - Build a compact patient context string via ``PatientContextExtractor``
    - Select the role-specific system prompt and call the OpenAI Chat API
    - Return a ``SummaryResult``; never propagate unhandled exceptions
    """

    def __init__(
        self,
        fhir_client: "FHIRClient",
        extractor: "PatientContextExtractor",
        llm_client: "OpenAI",
    ) -> None:
        """Initialise the agent.

        Args:
            fhir_client: Configured ``FHIRClient`` instance.
            extractor:   Configured ``PatientContextExtractor`` instance.
            llm_client:  Initialised ``openai.OpenAI`` client.
        """
        self._fhir_client = fhir_client
        self._extractor = extractor
        self._llm_client = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_summary(self, patient_id: str, role: str) -> SummaryResult:
        """Generate a role-specific clinical summary for the given patient.

        Args:
            patient_id: The FHIR patient ID to summarise.
            role:       Clinician role — must be ``"ED Doctor"`` or
                        ``"Care Manager"``.

        Returns:
            A ``SummaryResult`` containing the three summary sections and
            metadata.  Never raises an unhandled exception (Requirement 6.1).
        """
        generated_at = _utc_now_iso()

        # --- Role validation (Requirements 4.6, 6.7) ---------------------
        if role not in _ROLE_PROMPTS:
            return SummaryResult(
                patient_name="",
                patient_id=patient_id,
                role=role,
                current_issues="",
                recent_changes="",
                risks_and_followup="",
                data_source="local_fallback",  # placeholder; no fetch occurred
                generated_at=generated_at,
                error=f"Unsupported role: {role}",
            )

        # --- Data-source determination (Requirements 2.3, 2.4) -----------
        if self._fhir_client.is_available():
            data_source: str = "fhir_server"
            resources = self._fetch_resources(patient_id, use_live=True)
        else:
            data_source = "local_fallback"
            resources = self._fetch_resources(patient_id, use_live=False)

        # NOTE: LLM invocation is implemented in task 7.4.
        # For now we return a placeholder SummaryResult after the fetch step.
        # (The actual patient_name extraction and LLM call will be added in 7.4.)
        return SummaryResult(
            patient_name="",
            patient_id=patient_id,
            role=role,
            current_issues="",
            recent_changes="",
            risks_and_followup="",
            data_source=data_source,  # type: ignore[arg-type]
            generated_at=generated_at,
            error=None,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_resources(
        self, patient_id: str, *, use_live: bool
    ) -> "PatientResources | SummaryResult":
        """Fetch all required FHIR resources for *patient_id*.

        Args:
            patient_id: The patient's FHIR ID.
            use_live:   ``True`` → query the live FHIR server via
                        ``_fetch_all_fhir_resources()``;
                        ``False`` → load from the local fallback bundle
                        (implemented in task 7.1 / 7.4).

        Returns:
            A ``PatientResources`` on success, or a ``SummaryResult`` with
            ``error`` set if the Patient fetch fails or is not found.
        """
        if use_live:
            return _fetch_all_fhir_resources(self._fhir_client, patient_id)
        # Fallback path (local bundle) — implemented in task 7.4
        raise NotImplementedError(
            "_fetch_resources(use_live=False) is not yet implemented (task 7.4)"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string ending in 'Z'."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
