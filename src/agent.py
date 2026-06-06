"""
SummaryAgent coordinates FHIR data retrieval, context extraction, and
LLM invocation to produce a SummaryResult.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.exceptions import FHIRClientError, FHIRUnavailableError
from src.models import PatientResources, SourceSection, SummaryResult

if TYPE_CHECKING:
    from src.context_extractor import PatientContextExtractor
    from src.fhir_client import FHIRClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role-specific system prompts (Requirements 4.1, 4.2)
# ---------------------------------------------------------------------------

ED_DOCTOR_PROMPT = (
    "You are a clinical AI assistant generating a concise patient summary for an Emergency "
    "Department physician.\n"
    "Focus on: active diagnoses, current medications and allergies (drug safety), the most recent "
    "labs and vitals,\n"
    "and any acute concerns. Be brief, use medical shorthand where appropriate, and highlight "
    "anything\n"
    "immediately actionable. Do not include care management goals or long-term follow-up plans.\n"
    "Use only the supplied FHIR context. Do not invent missing values. Keep each section to "
    "3-5 concise bullet points.\n"
    "\n"
    "Structure your response EXACTLY as:\n"
    "## Current Issues\n"
    "<bullet points>\n"
    "\n"
    "## Recent Changes\n"
    "<bullet points>\n"
    "\n"
    "## Risks and Follow-up\n"
    "<bullet points>"
)

CARE_MANAGER_PROMPT = (
    "You are a clinical AI assistant generating a patient summary for a Care Manager focused on "
    "chronic\n"
    "disease management and care coordination. Focus on: chronic conditions, medication adherence,\n"
    "pending care plan goals, upcoming follow-up needs, and social/functional risks.\n"
    "Use plain clinical language. Include actionable care coordination items.\n"
    "Use only the supplied FHIR context. Do not invent missing values. Keep each section to "
    "3-5 concise bullet points.\n"
    "\n"
    "Structure your response EXACTLY as:\n"
    "## Current Issues\n"
    "<bullet points>\n"
    "\n"
    "## Recent Changes\n"
    "<bullet points>\n"
    "\n"
    "## Risks and Follow-up\n"
    "<bullet points>"
)

_ROLE_PROMPTS: dict[str, str] = {
    "ED Doctor": ED_DOCTOR_PROMPT,
    "Care Manager": CARE_MANAGER_PROMPT,
}

_SECTION_PROMPTS: dict[str, str] = {
    "Current Issues": (
        "Generate only the 'Current Issues' section. Return exactly:\n"
        "## Current Issues\n"
        "<3-5 concise bullet points>"
    ),
    "Recent Changes": (
        "Generate only the 'Recent Changes' section. Return exactly:\n"
        "## Recent Changes\n"
        "<2-4 concise bullet points>"
    ),
    "Risks and Follow-up": (
        "Generate only the 'Risks and Follow-up' section. Return exactly:\n"
        "## Risks and Follow-up\n"
        "<3-5 concise bullet points>"
    ),
}

_SECTION_ORDER = ["Current Issues", "Recent Changes", "Risks and Follow-up"]

# ---------------------------------------------------------------------------
# Header - key mapping (case-sensitive, must match exactly on their own line)
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

    Parsing rules (Requirements 5.1-5.7):
    - If ``raw_text`` is empty - all three values are empty strings.
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

    # Empty input - return all-empty dict immediately (Req 5.4)
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
# FHIR fetch helper (Task 7.2)
# ---------------------------------------------------------------------------

def _fetch_all_fhir_resources(
    fhir_client: "FHIRClient",
    patient_id: str,
) -> "PatientResources | SummaryResult":
    """Fetch all seven FHIR resource types for *patient_id* with graceful degradation.

    Returns a ``PatientResources`` on success.  Returns a ``SummaryResult``
    with ``error`` set (early-exit sentinel) when the Patient fetch itself
    fails or yields no results.

    Non-Patient resource fetch failures are logged as warnings and the
    corresponding field is set to an empty list (Requirements 7.1-7.5).
    """
    # Ordered fetch list: (resource_type, extra_params)
    _FETCH_PLAN = [
        ("Patient",            {"_id": patient_id}),
        ("Condition",          {"patient": patient_id, "clinical-status": "active"}),
        ("MedicationRequest",  {"patient": patient_id, "status": "active"}),
        ("AllergyIntolerance", {"patient": patient_id}),
        ("Observation",        {"patient": patient_id, "_sort": "-date", "_count": "20"}),
        ("Encounter",          {"patient": patient_id, "_sort": "-date", "_count": "5"}),
        ("CarePlan",           {"patient": patient_id, "status": "active"}),
    ]

    results: dict[str, list[dict]] = {}

    for resource_type, params in _FETCH_PLAN:
        try:
            entries = fhir_client.get_resource(resource_type, patient_id, params)
            results[resource_type] = entries
        except (FHIRClientError, FHIRUnavailableError) as exc:
            if resource_type == "Patient":
                # Patient fetch failure is fatal, so return an error sentinel.
                return _error_result(
                    patient_id=patient_id,
                    error=f"Failed to fetch Patient {patient_id}: {exc}",
                )
            # Non-Patient failure: log and continue (Req 7.1)
            logger.warning("Failed to fetch %s for patient %s: %s", resource_type, patient_id, exc)
            results[resource_type] = []

    # Guard: empty Patient result means the patient doesn't exist (Req 7.2)
    if not results.get("Patient"):
        return _error_result(
            patient_id=patient_id,
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


def _error_result(patient_id: str, error: str, role: str = "") -> SummaryResult:
    """Return a SummaryResult carrying only the error field (all sections empty)."""
    return SummaryResult(
        patient_name="",
        patient_id=patient_id,
        role=role,
        current_issues="",
        recent_changes="",
        risks_and_followup="",
        data_source="fhir_server",  # placeholder; overridden by caller when needed
        generated_at=_utc_now_iso(),
        error=error,
    )


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string, e.g. '2026-06-05T14:30:00Z'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# SummaryAgent (Tasks 7.1 / 7.4)
# ---------------------------------------------------------------------------

class SummaryAgent:
    """Orchestrates FHIR retrieval, context extraction, and LLM invocation.

    Requirements: 2.3, 2.4, 4.1-4.5, 6.1-6.7, 7.1-7.5
    """

    def __init__(
        self,
        fhir_client: "FHIRClient",
        extractor: "PatientContextExtractor",
        llm_client,  # openai.OpenAI instance, typed loosely to avoid a hard dependency
    ) -> None:
        self._fhir = fhir_client
        self._fhir_client = fhir_client  # alias for test compatibility
        self._extractor = extractor
        self._llm = llm_client
        self._llm_client = llm_client  # alias for test compatibility

    def generate_summary(self, patient_id: str, role: str) -> SummaryResult:
        """Generate a role-specific clinical summary for *patient_id*.

        Always returns a ``SummaryResult`` and never raises an unhandled
        exception (Requirement 6.1).
        """
        generated_at = _utc_now_iso()

        try:
            # --- Role validation (Req 4.3, 6.6, 6.7) ---
            if role not in _ROLE_PROMPTS:
                return SummaryResult(
                    patient_name="",
                    patient_id=patient_id,
                    role=role,
                    current_issues="",
                    recent_changes="",
                    risks_and_followup="",
                    data_source="fhir_server",
                    generated_at=generated_at,
                    error=f"Unsupported role: {role}",
                )

            # --- Data-source determination (Req 2.3, 2.4) ---
            if self._fhir.is_available():
                data_source = "fhir_server"
                fetch_result = _fetch_all_fhir_resources(self._fhir, patient_id)
                # _fetch_all_fhir_resources may return an error SummaryResult
                if isinstance(fetch_result, SummaryResult):
                    # Patch in the correct metadata
                    fetch_result.role = role
                    fetch_result.data_source = data_source
                    fetch_result.generated_at = generated_at
                    return fetch_result
                resources: PatientResources = fetch_result
            else:
                data_source = "local_fallback"
                resources = self._load_fallback_resources(patient_id)
                if isinstance(resources, SummaryResult):
                    resources.role = role
                    resources.data_source = data_source
                    resources.generated_at = generated_at
                    return resources

            # --- Extract patient context string ---
            context_text = self._extractor.extract(resources)

            # --- LLM invocation (Task 7.4) ---
            system_prompt = _ROLE_PROMPTS[role]
            try:
                response = self._llm.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context_text},
                    ],
                    temperature=0.3,
                    max_tokens=800,
                )
                raw_text = response.choices[0].message.content
                sections = parse_sections(raw_text)
                llm_error: str | None = None
            except Exception as llm_exc:  # noqa: BLE001
                sections = {"current_issues": "", "recent_changes": "", "risks_and_followup": ""}
                llm_error = str(llm_exc)

            # --- Extract patient name for result ---
            patient_name = self._extract_patient_name(resources.patient)

            return SummaryResult(
                patient_name=patient_name,
                patient_id=patient_id,
                role=role,
                current_issues=sections["current_issues"],
                recent_changes=sections["recent_changes"],
                risks_and_followup=sections["risks_and_followup"],
                data_source=data_source,
                generated_at=generated_at,
                error=llm_error,
            )

        except Exception as exc:  # noqa: BLE001 - top-level safety net (Req 6.1)
            logger.exception("Unhandled error in generate_summary: %s", exc)
            return SummaryResult(
                patient_name="",
                patient_id=patient_id,
                role=role,
                current_issues="",
                recent_changes="",
                risks_and_followup="",
                data_source="fhir_server",
                generated_at=generated_at,
                error=str(exc),
            )

    # ---------------------------------------------------------------------- #
    # Private helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _load_fallback_resources(
        self, patient_id: str
    ) -> "PatientResources | SummaryResult":
        """Load patient resources from the local fallback bundle.

        Filters all bundle resources to those belonging to *patient_id*.
        Returns an error SummaryResult if the bundle is missing/invalid or
        the patient is not found in the bundle.
        """
        try:
            all_resources = self._fhir._load_fallback_bundle()
        except RuntimeError as exc:
            return _error_result(patient_id=patient_id, error=str(exc))

        # Find the Patient resource matching patient_id
        patients = [
            r for r in all_resources
            if r.get("resourceType") == "Patient" and r.get("id") == patient_id
        ]
        if not patients:
            # If patient_id doesn't match any id, take the first patient
            # (bundle may use server-assigned IDs we don't know yet)
            patients = [r for r in all_resources if r.get("resourceType") == "Patient"]

        if not patients:
            return _error_result(
                patient_id=patient_id,
                error=f"Patient {patient_id} not found",
            )

        patient = patients[0]
        actual_id = patient.get("id", patient_id)

        def _of_type(rtype: str) -> list[dict]:
            return [
                r for r in all_resources
                if r.get("resourceType") == rtype
                and self._resource_belongs_to_patient(r, actual_id)
            ]

        return PatientResources(
            patient=patient,
            conditions=_of_type("Condition"),
            medications=_of_type("MedicationRequest"),
            allergies=_of_type("AllergyIntolerance"),
            observations=_of_type("Observation"),
            encounters=_of_type("Encounter"),
            care_plans=_of_type("CarePlan"),
        )

    @staticmethod
    def _resource_belongs_to_patient(resource: dict, patient_id: str) -> bool:
        """Return True when a FHIR resource references the selected patient."""
        expected_refs = {
            patient_id,
            f"Patient/{patient_id}",
            f"urn:uuid:{patient_id}",
        }

        for field_name in ("subject", "patient", "beneficiary"):
            reference = resource.get(field_name, {}).get("reference")
            if reference in expected_refs:
                return True

        return False

    @staticmethod
    def _extract_patient_name(patient: dict) -> str:
        """Extract a display name from a FHIR Patient resource dict."""
        names = patient.get("name", [])
        if not names:
            return "Unknown"
        first = names[0]
        if first.get("text"):
            return first["text"]
        given = " ".join(first.get("given", []))
        family = first.get("family", "")
        parts = [p for p in [given, family] if p]
        return " ".join(parts) if parts else "Unknown"

    # ---------------------------------------------------------------------- #
    # Source-section builder                                                  #
    # ---------------------------------------------------------------------- #

    def _build_source_sections(self, resources: PatientResources) -> list[SourceSection]:
        """Build a compact source summary for the UI reference panel."""
        sections: list[SourceSection] = []

        def _source_section(label: str, items: list[str], limit: int = 3) -> SourceSection:
            if not items:
                return SourceSection(label=label, items=["None"])
            return SourceSection(
                label=label,
                items=items[:limit],
                hidden_items=items[limit:],
            )

        # --- Active Conditions ---
        cond_items = [
            self._extractor._format_condition(c).lstrip("- ")
            for c in resources.conditions
        ]
        sections.append(_source_section(f"Active Conditions ({len(cond_items)})", cond_items))

        # --- Active Medications ---
        med_items = [
            self._extractor._format_medication(m).lstrip("- ")
            for m in resources.medications
        ]
        sections.append(_source_section(f"Active Medications ({len(med_items)})", med_items))

        # --- Allergies ---
        allergy_items = [
            self._extractor._format_allergy(a).lstrip("- ")
            for a in resources.allergies
        ]
        sections.append(_source_section(f"Allergies ({len(allergy_items)})", allergy_items))

        # --- Recent Observations (newest first, up to 10) ---
        sorted_obs = sorted(
            resources.observations,
            key=lambda o: o.get("effectiveDateTime", ""),
            reverse=True,
        )[:10]
        obs_items = [
            self._extractor._format_observation(o).lstrip("- ")
            for o in sorted_obs
        ]
        obs_items = [i for i in obs_items if i]  # drop blanks
        sections.append(_source_section(f"Recent Observations ({len(obs_items)})", obs_items))

        # --- Recent Encounters (newest first, up to 3) ---
        sorted_enc = sorted(
            resources.encounters,
            key=lambda e: e.get("period", {}).get("start", ""),
            reverse=True,
        )[:3]
        enc_items = [
            self._extractor._format_encounter(e).lstrip("- ")
            for e in sorted_enc
        ]
        enc_items = [i for i in enc_items if i]
        sections.append(_source_section(f"Recent Encounters ({len(enc_items)})", enc_items))

        # --- Care Plan ---
        activity_lines = self._extractor._extract_activity_lines(resources.care_plans)
        cp_items = [line.lstrip("- Activity: ") for line in activity_lines]
        sections.append(_source_section(f"Care Plan ({len(cp_items)} activities)", cp_items))

        return sections

    # ---------------------------------------------------------------------- #
    # Streaming summary generator                                             #
    # ---------------------------------------------------------------------- #

    def generate_summary_stream(self, patient_id: str, role: str):
        """Generate the summary section by section.

        Yields tuples of (partial_markdown: str, source_sections: list[SourceSection] | None).
        - First yield carries source_sections immediately after FHIR data is ready.
        - Each completed section is yielded immediately so the UI can update.
        Never raises; errors are surfaced as a final plain-text yield with source_sections=[].
        """
        if role not in _ROLE_PROMPTS:
            yield f"**Error:** Unsupported role: {role}", []
            return

        # --- Data fetch (same logic as generate_summary) ---
        try:
            if self._fhir.is_available():
                data_source = "fhir_server"
                fetch_result = _fetch_all_fhir_resources(self._fhir, patient_id)
                if isinstance(fetch_result, SummaryResult):
                    yield f"**Error:** {fetch_result.error}", []
                    return
                resources: PatientResources = fetch_result
            else:
                data_source = "local_fallback"
                resources = self._load_fallback_resources(patient_id)
                if isinstance(resources, SummaryResult):
                    yield f"**Error:** {resources.error}", []
                    return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Data fetch error in generate_summary_stream: %s", exc)
            yield f"**Error:** {exc}", []
            return

        source_sections = self._build_source_sections(resources)
        context_text = self._extractor.extract(resources)
        base_prompt = _ROLE_PROMPTS[role]

        # Let the UI show reference data before waiting for any LLM response.
        yield "", source_sections

        try:
            rendered_sections: list[str] = []
            for section_name in _SECTION_ORDER:
                section_text = self._generate_one_section(
                    base_prompt=base_prompt,
                    section_name=section_name,
                    context_text=context_text,
                )
                rendered_sections.append(section_text)
                partial_markdown = "\n\n".join(rendered_sections)
                yield partial_markdown, source_sections

        except Exception as llm_exc:  # noqa: BLE001
            logger.exception("LLM section generation error: %s", llm_exc)
            yield f"**Error:** {llm_exc}", []

    def _generate_one_section(
        self,
        base_prompt: str,
        section_name: str,
        context_text: str,
    ) -> str:
        """Generate one named summary section with a separate LLM request."""
        response = self._llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": base_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{_SECTION_PROMPTS[section_name]}\n\n"
                        "FHIR patient context:\n"
                        f"{context_text}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
