"""
SummaryAgent coordinates FHIR data retrieval, context extraction, and
LLM invocation to produce a SummaryResult.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.exceptions import FHIRClientError, FHIRUnavailableError
from src.models import PatientResources, SourceSection, SummaryResult

if TYPE_CHECKING:
    from src.context_extractor import PatientContextExtractor
    from src.fhir_client import FHIRClient

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_TEMPERATURE = 0.3
DEFAULT_OPENAI_MAX_TOKENS = 800
STREAM_THROTTLE_SECONDS = 0.2


@dataclass(frozen=True)
class ModelConfig:
    model: str
    temperature: float
    max_tokens: int
    stream_throttle_seconds: float


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        return default
    if minimum is not None and value < minimum:
        return default
    return value


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


def _get_model_config() -> ModelConfig:
    """Read OpenAI model settings from environment variables with safe fallbacks."""
    model = os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_OPENAI_MODEL
    return ModelConfig(
        model=model,
        temperature=_env_float("OPENAI_TEMPERATURE", DEFAULT_OPENAI_TEMPERATURE),
        max_tokens=_env_int("OPENAI_MAX_TOKENS", DEFAULT_OPENAI_MAX_TOKENS, minimum=1),
        stream_throttle_seconds=_env_float(
            "STREAM_THROTTLE_SECONDS",
            STREAM_THROTTLE_SECONDS,
            minimum=0,
        ),
    )

# ---------------------------------------------------------------------------
# Role-specific system prompts
# ---------------------------------------------------------------------------

DECEASED_RECORD_RULES = """
If Patient.deceasedDateTime, deceasedBoolean=true, death certification, or cause-of-death data is present:
- State first that the patient is deceased and summarize retrospectively only.
- Treat active conditions, medications, and care plans as historical unless clearly documented before death.
- For diagnoses in a deceased patient, prefer 'FHIR-listed active diagnoses before death' or 'conditions documented before death' instead of unqualified 'active diagnoses'.
- Do not imply that conditions, medications, or care plans were active at the exact time of death unless supplied dates explicitly support that.
- Do not say 'medications at the time of death' unless a medication period overlaps the death date.
- Do NOT recommend active treatment, monitoring, medication adherence work, chronic disease follow-up, self-management, or routine care coordination.
- Do NOT recommend family support, estate management, bereavement support, caregiver actions, or support for surviving family members unless explicitly documented in the supplied FHIR data.
- Mention documentation gaps only when the supplied context shows a specific missing or unclear field, such as missing cause of death, missing allergy data, unclear medication dates, or unclear terminal encounter details. Do not use vague phrases like 'documentation gaps may exist'.
- For deceased patients, do not frame missing recent vitals/labs as an active concern about current clinical status. If relevant, phrase it only as a retrospective documentation limitation, e.g. "No vitals/labs are documented in the supplied data between [date] and death."
- For Risks and Follow-up, limit content to retrospective chart review: death date/cause clarity, terminal clinical trajectory, medication/allergy safety facts, and specific documentation gaps.
- Avoid repeating the same death date/cause in every section. Place deceased status in Current Issues, place death/death-certification events in Recent Changes only as dated timeline events, and use Risks and Follow-up only for specific retrospective safety facts or documentation gaps.
"""

LIVING_PATIENT_RULES = """
If no Patient.deceasedDateTime, no deceasedBoolean=true, no death certification, and no cause-of-death data is present:
- Treat the patient as living or death status not documented.
- Do NOT mention missing death certification, missing cause of death, end-of-life documentation gaps, or death-related follow-up.
- Do NOT infer end-of-life care needs from the absence of death-related fields.
"""

VOICE_AND_AUDIENCE_RULES = """
Voice and audience rules:
- ED Doctor: write in concise third-person clinical chart style. Use "patient" or the patient's name. Do not address the reader as "you".
- Care Manager: write in third-person care-coordination style. Use "patient" or the patient's name. Do not address the reader as "you".
- Patient: for living patients, write directly to the patient using "you" and plain language. If the patient is deceased, do not address the patient as "you"; write retrospectively in third person.
- Family Caregiver: for living patients, write to the caregiver using "your family member" or "the patient". Do not imply legal authority or caregiving duties not documented in the data. If the patient is deceased, write retrospectively in third person.
- Never mix voices within the same summary.
"""

ED_DOCTOR_PROMPT = ( """
Summarize the supplied FHIR data for an Emergency Department physician in English.

Use only supplied data. Prioritize latest encounter, recent vitals/labs, active/recent diagnoses,
meds, allergies, and acute safety risks. Ignore billing/insurance/care management goals/care plans unless clinically actionable.
Recent data > old history. Include dates/key values. Do not invent missing data.

""" + DECEASED_RECORD_RULES + LIVING_PATIENT_RULES + VOICE_AND_AUDIENCE_RULES + """

Recent Changes rules:
- List clinically relevant events in chronological order, latest to earliest.
- The first bullet must be the latest documented clinical event.

Be concise: 3-5 bullets per section. Medical shorthand allowed.

Output exactly:

## Current Issues
- ...

## Recent Changes
- ...

## Risks and Follow-up
- ED safety risks, missing critical data, drug/allergy concerns, or retrospective disposition notes only.
"""
)

CARE_MANAGER_PROMPT = (
    "You are a clinical AI assistant generating a patient summary for a Care Manager focused on "
    "chronic\n"
    "disease management and care coordination. Focus on: chronic conditions, medication adherence,\n"
    "pending care plan goals, upcoming follow-up needs, and social/functional risks.\n"
    "For living patients, include actionable care coordination items.\n"
    "For deceased patients, switch to retrospective chart review only: death date/cause, final "
    "clinical trajectory, pre-death diagnoses, medications, allergies, and documentation gaps. "
    "Do not create new care tasks.\n"
    "Do not list historical care plans as follow-up needs. For deceased patients, mention care plans "
    "only as historical context if they directly clarify the terminal course or chart review.\n"
    "Use plain clinical language. Include actionable care coordination items only for living patients.\n"
    "Use only the supplied FHIR context. Do not invent missing values. Keep each section to "
    "3-5 concise bullet points.\n"
    + DECEASED_RECORD_RULES + LIVING_PATIENT_RULES + VOICE_AND_AUDIENCE_RULES +
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

PATIENT_PROMPT = (
    "You are a clinical AI assistant generating a patient-facing summary in English.\n"
    "Use the same supplied FHIR context, but explain it in plain language for the patient.\n"
    "Avoid medical jargon when possible; when jargon is necessary, briefly explain it.\n"
    "Focus on what the patient should understand about current health issues, recent changes, "
    "medicines, allergies, and what questions to ask their clinician.\n"
    "Use only supplied data. Do not diagnose new conditions, invent missing values, or give "
    "emergency instructions beyond advising urgent care for clearly documented serious risk.\n"
    + DECEASED_RECORD_RULES + LIVING_PATIENT_RULES + VOICE_AND_AUDIENCE_RULES +
    "Keep each section to 3-5 concise bullet points.\n"
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

FAMILY_CAREGIVER_PROMPT = (
    "You are a clinical AI assistant generating a family caregiver summary in English.\n"
    "Use the same supplied FHIR context, but explain it for a non-clinician who may help with "
    "appointments, medication awareness, and safety monitoring.\n"
    "Focus on practical caregiving implications: active problems, recent changes, medication "
    "or allergy risks, warning signs documented in the record, and questions to raise with the "
    "care team.\n"
    "Use only supplied data. Do not invent missing values. Avoid giving independent medical "
    "orders or replacing clinician advice.\n"
    + DECEASED_RECORD_RULES + LIVING_PATIENT_RULES + VOICE_AND_AUDIENCE_RULES +
    "Keep each section to 3-5 concise bullet points.\n"
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
    "Patient": PATIENT_PROMPT,
    "Family Caregiver": FAMILY_CAREGIVER_PROMPT,
}

SUPPORTED_ROLES = tuple(_ROLE_PROMPTS.keys())

SUMMARY_OUTPUT_RULES = """
Return one complete summary with exactly these sections:

## Current Issues
- bullets

## Recent Changes
- bullets, chronological order, latest to earliest

## Risks and Follow-up
- bullets

Do not add any other headings.
Use only the supplied FHIR context.
"""

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

    Parsing rules:
    - If ``raw_text`` is empty - all three values are empty strings.
    - Lines that exactly equal a recognised ``## Header`` marker start a new
      section; content lines are accumulated until the next header or EOF.
    - Each section value is stripped of leading/trailing whitespace.
    - If ``raw_text`` is non-empty but contains *no* recognised headers, the
      full stripped input is placed in ``risks_and_followup`` and the other
      two keys are empty strings.
    """
    result: dict[str, str] = {
        "current_issues": "",
        "recent_changes": "",
        "risks_and_followup": "",
    }

    # Empty input - return all-empty dict immediately.
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

    # Fallback: non-empty input with no recognised headers.
    if not found_any_header:
        result["risks_and_followup"] = raw_text.strip()

    return result


# ---------------------------------------------------------------------------
# FHIR fetch helper
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
    corresponding field is set to an empty list.
    """
    # Ordered fetch list: (resource_type, extra_params)
    _FETCH_PLAN = [
        ("Patient",            {"_id": patient_id}),
        ("Condition",          {"patient": patient_id, "_count": "100"}),
        ("MedicationRequest",  {"patient": patient_id, "_sort": "-authoredon", "_count": "150"}),
        ("AllergyIntolerance", {"patient": patient_id}),
        ("Observation",        {"patient": patient_id, "_sort": "-date", "_count": "150"}),
        ("Encounter",          {"patient": patient_id, "_sort": "-date", "_count": "75"}),
        ("CarePlan",           {"patient": patient_id, "_count": "50"}),
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
            # Non-Patient failure: log and continue.
            logger.warning("Failed to fetch %s for patient %s: %s", resource_type, patient_id, exc)
            results[resource_type] = []

    # Guard: empty Patient result means the patient does not exist.
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
# SummaryAgent
# ---------------------------------------------------------------------------

class SummaryAgent:
    """Orchestrates FHIR retrieval, context extraction, and LLM invocation.

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

        # --- Medication Requests ---
        med_items = [
            self._extractor._format_medication(m).lstrip("- ")
            for m in resources.medications
        ]
        sections.append(_source_section(f"Medication Requests ({len(med_items)})", med_items))

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
        """Stream a role-specific clinical summary.

        Yields tuples of (partial_markdown: str, source_sections: list[SourceSection] | None).
        - First yield carries source_sections immediately after FHIR data is ready.
        - Later yields carry accumulated markdown from a single OpenAI stream.
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

        # Let the UI show reference data before waiting for any LLM response.
        yield "", source_sections

        try:
            config = _get_model_config()
            stream = self._llm.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": _ROLE_PROMPTS[role]},
                    {
                        "role": "user",
                        "content": (
                            f"{SUMMARY_OUTPUT_RULES}\n\n"
                            "FHIR patient context:\n"
                            f"{context_text}"
                        ),
                    },
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                stream=True,
            )

            accumulated = ""
            first_chunk_seen = False
            last_yield_at = time.monotonic()

            for chunk in stream:
                delta = _extract_stream_delta(chunk)
                if not delta:
                    continue

                accumulated += delta
                now = time.monotonic()

                if not first_chunk_seen:
                    first_chunk_seen = True
                    last_yield_at = now
                    yield accumulated, source_sections
                    continue

                if now - last_yield_at >= config.stream_throttle_seconds:
                    last_yield_at = now
                    yield accumulated, source_sections

            if accumulated:
                yield accumulated, source_sections

        except Exception as llm_exc:  # noqa: BLE001
            logger.exception("LLM stream generation error: %s", llm_exc)
            yield f"**Error:** {llm_exc}", []


def _extract_stream_delta(chunk) -> str:
    """Extract text content from an OpenAI streaming chat completion chunk."""
    if isinstance(chunk, dict):
        try:
            choices = chunk.get("choices") or []
            delta = choices[0].get("delta") or {}
            return delta.get("content") or ""
        except (AttributeError, IndexError, TypeError):
            return ""

    try:
        choices = getattr(chunk, "choices", None)
        if not choices:
            return ""
        delta = getattr(choices[0], "delta", None)
        return getattr(delta, "content", None) or ""
    except (AttributeError, IndexError, TypeError):
        return ""
