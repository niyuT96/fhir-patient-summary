"""
PatientContextExtractor converts raw FHIR resource lists into a compact,
token-efficient plain-text string for LLM consumption.

Token budget: 3,000 tokens (cl100k_base / gpt-4o-mini).

Truncation priority order, from lowest priority to highest priority:
  CarePlan activities -> Encounters -> Observations -> Allergies ->
  Medications -> Conditions -> Demographics is never truncated
"""

from __future__ import annotations

import copy

import tiktoken

from src.models import PatientResources  # noqa: F401

_TOKEN_BUDGET = 3_000
_ENCODING_NAME = "cl100k_base"


def _count_tokens(text: str) -> int:
    """Return the cl100k_base token count for *text*."""
    enc = tiktoken.get_encoding(_ENCODING_NAME)
    return len(enc.encode(text))


class PatientContextExtractor:
    """Converts a PatientResources object into a structured plain-text context block."""

    def extract(self, resources: PatientResources) -> str:
        """Return a compact, labelled plain-text representation of the patient record.

        Sections included (in order):
          - Patient Demographics
          - Active Conditions
          - Active Medications
          - Allergies
          - Recent Observations
          - Recent Encounters
          - Care Plan

        Each section is always present; empty resource lists render as "None".
        Output is guaranteed to be at most 3,000 cl100k_base tokens.
        The input PatientResources is never mutated.
        """
        # Work on shallow copies of every list so we never mutate the caller's data.
        # Individual dicts are not copied - we only read them, never modify them.
        conditions = list(resources.conditions)
        medications = list(resources.medications)
        allergies = list(resources.allergies)

        # Sort observations/encounters on copies, newest first
        observations = sorted(
            resources.observations,
            key=lambda o: o.get("effectiveDateTime", ""),
            reverse=True,
        )
        encounters = sorted(
            resources.encounters,
            key=lambda e: e.get("period", {}).get("start", ""),
            reverse=True,
        )
        care_plans = list(resources.care_plans)

        # Build the initial (possibly oversized) context and enforce the budget.
        return self._build_with_budget(
            patient=resources.patient,
            conditions=conditions,
            medications=medications,
            allergies=allergies,
            observations=observations,
            encounters=encounters,
            care_plans=care_plans,
        )

    # ---------------------------------------------------------------------- #
    # Token-budget enforcement                                                #
    # ---------------------------------------------------------------------- #

    def _build_with_budget(
        self,
        patient: dict,
        conditions: list[dict],
        medications: list[dict],
        allergies: list[dict],
        observations: list[dict],
        encounters: list[dict],
        care_plans: list[dict],
    ) -> str:
        """Build the context string and truncate sections until it fits the token budget.

        Truncation priority (lowest priority removed first):
          1. CarePlan activities  (drop one activity at a time)
          2. Encounters           (drop one encounter at a time)
          3. Observations         (drop one observation at a time)
          4. Allergies            (drop one allergy at a time)
          5. Medications          (drop one medication at a time)
          6. Conditions           (drop one condition at a time)
          Demographics are never truncated.
        """
        # Work with mutable copies so we can shrink them during iteration.
        obs_working = list(observations[:10])    # cap at 10 up-front per spec
        enc_working = list(encounters[:3])       # cap at 3 up-front per spec
        cond_working = list(conditions)
        med_working = list(medications)
        allergy_working = list(allergies)

        # CarePlan activities extracted as individual strings (shallow copy)
        activity_lines = self._extract_activity_lines(care_plans)

        def _build() -> str:
            return self._assemble(
                patient=patient,
                conditions=cond_working,
                medications=med_working,
                allergies=allergy_working,
                observations=obs_working,
                encounters=enc_working,
                activity_lines=activity_lines,
            )

        # Truncation loop: remove one item at a time from the lowest-priority
        # section that is non-empty, until we're within budget.
        result = _build()
        while _count_tokens(result) > _TOKEN_BUDGET:
            if activity_lines:
                activity_lines.pop()
            elif enc_working:
                enc_working.pop()
            elif obs_working:
                obs_working.pop()
            elif allergy_working:
                allergy_working.pop()
            elif med_working:
                med_working.pop()
            elif cond_working:
                cond_working.pop()
            else:
                # Only demographics remain - cannot truncate further.
                break
            result = _build()

        return result

    # ---------------------------------------------------------------------- #
    # Section assembly                                                        #
    # ---------------------------------------------------------------------- #

    def _assemble(
        self,
        patient: dict,
        conditions: list[dict],
        medications: list[dict],
        allergies: list[dict],
        observations: list[dict],
        encounters: list[dict],
        activity_lines: list[str],
    ) -> str:
        """Combine all sections into the final plain-text string."""
        lines: list[str] = []

        # Demographics (never omitted, never truncated)
        lines.append("=== Patient Demographics ===")
        lines.extend(self._extract_demographics(patient))
        lines.append("")

        # Active Conditions
        lines.append("=== Active Conditions ===")
        if conditions:
            for cond in conditions:
                lines.append(self._format_condition(cond))
        else:
            lines.append("None")
        lines.append("")

        # Active Medications
        lines.append("=== Active Medications ===")
        if medications:
            for med in medications:
                lines.append(self._format_medication(med))
        else:
            lines.append("None")
        lines.append("")

        # Allergies
        lines.append("=== Allergies ===")
        if allergies:
            for allergy in allergies:
                lines.append(self._format_allergy(allergy))
        else:
            lines.append("None")
        lines.append("")

        # Recent Observations (up to 10 most recent, already sorted)
        lines.append("=== Recent Observations ===")
        if observations:
            rendered = [self._format_observation(o) for o in observations]
            rendered = [r for r in rendered if r]  # skip blank lines
            if rendered:
                lines.extend(rendered)
            else:
                lines.append("None")
        else:
            lines.append("None")
        lines.append("")

        # Recent Encounters (up to 3 most recent, already sorted)
        lines.append("=== Recent Encounters ===")
        if encounters:
            rendered = [self._format_encounter(e) for e in encounters]
            rendered = [r for r in rendered if r]
            if rendered:
                lines.extend(rendered)
            else:
                lines.append("None")
        else:
            lines.append("None")
        lines.append("")

        # Care Plan
        lines.append("=== Care Plan ===")
        if activity_lines:
            lines.extend(activity_lines)
        else:
            lines.append("None")
        lines.append("")

        return "\n".join(lines)

    # ---------------------------------------------------------------------- #
    # CarePlan helpers                                                        #
    # ---------------------------------------------------------------------- #

    def _extract_activity_lines(self, care_plans: list[dict]) -> list[str]:
        """Extract activity description lines from CarePlan resources.

        Returns a flat list of formatted strings (one per activity).
        Returns an empty list when no activities are found; the caller
        renders this section as "None".
        """
        result: list[str] = []
        for plan in care_plans:
            for activity in plan.get("activity", []):
                detail = activity.get("detail", {})
                description = detail.get("description")
                if description:
                    result.append(f"- Activity: {description}")
        return result

    # ---------------------------------------------------------------------- #
    # Private helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _extract_demographics(self, patient: dict) -> list[str]:
        """Extract and format patient demographic fields."""
        result: list[str] = []

        # Name: prefer name[0].text; fall back to given + family
        name = "Unknown"
        names = patient.get("name", [])
        if names:
            first_name = names[0]
            if first_name.get("text"):
                name = first_name["text"]
            else:
                given = " ".join(first_name.get("given", []))
                family = first_name.get("family", "")
                parts = [p for p in [given, family] if p]
                if parts:
                    name = " ".join(parts)

        result.append(f"Name: {name}")

        # Date of birth
        dob = patient.get("birthDate", "Unknown")
        result.append(f"DOB: {dob}")

        # Gender
        gender = patient.get("gender", "Unknown")
        result.append(f"Gender: {gender}")

        # MRN - identifier where type.coding[0].code == "MR"
        mrn = "Unknown"
        for identifier in patient.get("identifier", []):
            coding_list = identifier.get("type", {}).get("coding", [])
            if coding_list and coding_list[0].get("code") == "MR":
                mrn = identifier.get("value", "Unknown")
                break
        result.append(f"MRN: {mrn}")

        return result

    def _format_condition(self, condition: dict) -> str:
        """Format a single FHIR Condition resource as a brief string."""
        code = condition.get("code", {})
        text = code.get("text")
        if not text:
            codings = code.get("coding", [])
            if codings:
                text = codings[0].get("display")
        if not text:
            text = "Unknown condition"
        return f"- {text}"

    def _format_medication(self, med_request: dict) -> str:
        """Format a single FHIR MedicationRequest resource as a brief string."""
        med_concept = med_request.get("medicationCodeableConcept", {})
        drug_name = med_concept.get("text", "Unknown medication")

        dosage_instructions = med_request.get("dosageInstruction", [])
        if dosage_instructions:
            dosage_text = dosage_instructions[0].get("text")
            if dosage_text:
                return f"- {drug_name}: {dosage_text}"

        return f"- {drug_name}"

    def _format_allergy(self, allergy: dict) -> str:
        """Format a single FHIR AllergyIntolerance resource as a brief string."""
        parts: list[str] = []

        substance = allergy.get("code", {}).get("text")
        if substance:
            parts.append(substance)

        criticality = allergy.get("criticality")
        if criticality:
            parts.append(f"criticality: {criticality}")

        reactions = allergy.get("reaction", [])
        if reactions:
            manifestations = reactions[0].get("manifestation", [])
            if manifestations:
                reaction_text = manifestations[0].get("text")
                if reaction_text:
                    parts.append(f"reaction: {reaction_text}")

        if not parts:
            return "- Unknown allergy"

        return "- " + "; ".join(parts)

    def _format_observation(self, obs: dict) -> str:
        """Format a single FHIR Observation as '{name}: {value} {unit} ({date})'."""
        name = obs.get("code", {}).get("text", "")
        if not name:
            return ""

        value_quantity = obs.get("valueQuantity", {})
        value = value_quantity.get("value")
        unit = value_quantity.get("unit", "")

        effective = obs.get("effectiveDateTime", "")
        date = effective[:10] if effective else ""

        if value is None:
            return ""

        parts = [f"{name}: {value}"]
        if unit:
            parts.append(unit)
        if date:
            parts.append(f"({date})")

        return "- " + " ".join(parts)

    def _format_encounter(self, encounter: dict) -> str:
        """Format a single FHIR Encounter resource as a brief string."""
        parts: list[str] = []

        enc_types = encounter.get("type", [])
        if enc_types:
            enc_type_text = enc_types[0].get("text")
            if enc_type_text:
                parts.append(enc_type_text)

        period_start = encounter.get("period", {}).get("start", "")
        if period_start:
            parts.append(period_start[:10])

        reason_codes = encounter.get("reasonCode", [])
        if reason_codes:
            reason_text = reason_codes[0].get("text")
            if reason_text:
                parts.append(f"reason: {reason_text}")

        if not parts:
            return ""

        return "- " + "; ".join(parts)

    # Keep the legacy helper name so any existing callers don't break.
    def _extract_care_plan(self, care_plans: list[dict]) -> list[str]:
        """Legacy wrapper - delegates to _extract_activity_lines."""
        lines = self._extract_activity_lines(care_plans)
        return lines if lines else ["None"]
