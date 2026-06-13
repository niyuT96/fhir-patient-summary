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
from src.tools.citation_repair import repair_summary_citations
from src.tools.citation_validator import validate_citations
from src.tools.prompt_loader import get_role_prompt, get_supported_roles
from src.tools.source_items import build_source_context, build_source_sections

if TYPE_CHECKING:
    from src.fhir_client import FHIRClient

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_TEMPERATURE = 0.3
DEFAULT_OPENAI_MAX_TOKENS = 800
STREAM_THROTTLE_SECONDS = 0.2
DEFAULT_CITATION_REPAIR_ENABLED = True
DEFAULT_CITATION_REPAIR_MAX_ATTEMPTS = 1
DEFAULT_CITATION_STRICT_MODE = False


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


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name, "").strip().lower()
    if not raw_value:
        return default
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    return default


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

SUPPORTED_ROLES = get_supported_roles()

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
        llm_client,  # openai.OpenAI instance, typed loosely to avoid a hard dependency
    ) -> None:
        self._fhir = fhir_client
        self._llm = llm_client

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

    # ---------------------------------------------------------------------- #
    # Source-section builder                                                  #
    # ---------------------------------------------------------------------- #

    def _build_source_sections(self, resources: PatientResources) -> list[SourceSection]:
        """Build structured citeable source data for the UI reference panel."""
        return build_source_sections(resources)

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
        if role not in SUPPORTED_ROLES:
            yield f"**Error:** Unsupported role: {role}", []
            return

        # --- Data fetch (same logic as generate_summary) ---
        try:
            if self._fhir.is_available():
                fetch_result = _fetch_all_fhir_resources(self._fhir, patient_id)
                if isinstance(fetch_result, SummaryResult):
                    yield f"**Error:** {fetch_result.error}", []
                    return
                resources: PatientResources = fetch_result
            else:
                resources = self._load_fallback_resources(patient_id)
                if isinstance(resources, SummaryResult):
                    yield f"**Error:** {resources.error}", []
                    return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Data fetch error in generate_summary_stream: %s", exc)
            yield f"**Error:** {exc}", []
            return

        source_sections = self._build_source_sections(resources)
        source_context = build_source_context(source_sections)
        system_prompt = get_role_prompt(role)

        # Let the UI show reference data before waiting for any LLM response.
        yield "", source_sections

        try:
            config = _get_model_config()
            stream = self._llm.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            "FHIR source-indexed context:\n"
                            f"{source_context}"
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
                accumulated = self._repair_citations_if_needed(
                    accumulated,
                    source_sections,
                    config,
                )
                yield accumulated, source_sections

        except Exception as llm_exc:  # noqa: BLE001
            logger.exception("LLM stream generation error: %s", llm_exc)
            yield f"**Error:** {llm_exc}", []

    def _repair_citations_if_needed(
        self,
        summary: str,
        source_sections: list[SourceSection],
        config: ModelConfig,
    ) -> str:
        """Validate final summary citations and optionally repair them once."""
        validation = validate_citations(summary, source_sections)
        if not validation.has_errors:
            return summary

        max_attempts = _env_int(
            "CITATION_REPAIR_MAX_ATTEMPTS",
            DEFAULT_CITATION_REPAIR_MAX_ATTEMPTS,
            minimum=0,
        )
        repair_enabled = _env_bool("CITATION_REPAIR_ENABLED", DEFAULT_CITATION_REPAIR_ENABLED)
        strict_mode = _env_bool("CITATION_STRICT_MODE", DEFAULT_CITATION_STRICT_MODE)

        repaired = summary
        if repair_enabled:
            for _ in range(max_attempts):
                try:
                    repaired = repair_summary_citations(
                        llm_client=self._llm,
                        model=config.model,
                        temperature=config.temperature,
                        summary=repaired,
                        source_sections=source_sections,
                        validation=validation,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Citation repair failed: %s", exc)
                    break
                validation = validate_citations(repaired, source_sections)
                if not validation.has_errors:
                    return repaired

        if strict_mode and validation.has_errors:
            details = []
            if validation.invalid_source_ids:
                details.append(f"invalid source ids: {sorted(validation.invalid_source_ids)}")
            if validation.uncited_lines:
                details.append(f"uncited lines: {len(validation.uncited_lines)}")
            return "**Error:** Citation validation failed (" + "; ".join(details) + ")"

        return repaired


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
