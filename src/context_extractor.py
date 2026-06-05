"""
PatientContextExtractor — converts raw FHIR resource lists into a compact,
token-efficient plain-text string for LLM consumption.
"""

from __future__ import annotations

from src.models import PatientResources  # noqa: F401


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
        The output is guaranteed not to mutate the input PatientResources.
        """
        lines: list[str] = []

        # ------------------------------------------------------------------ #
        # Demographics                                                        #
        # ------------------------------------------------------------------ #
        lines.append("=== Patient Demographics ===")
        lines.extend(self._extract_demographics(resources.patient))
        lines.append("")

        # ------------------------------------------------------------------ #
        # Active Conditions                                                   #
        # ------------------------------------------------------------------ #
        lines.append("=== Active Conditions ===")
        if resources.conditions:
            for cond in resources.conditions:
                lines.append(self._format_condition(cond))
        else:
            lines.append("None")
        lines.append("")

        # ------------------------------------------------------------------ #
        # Active Medications                                                  #
        # ------------------------------------------------------------------ #
        lines.append("=== Active Medications ===")
        if resources.medications:
            for med in resources.medications:
                lines.append(self._format_medication(med))
        else:
            lines.append("None")
        lines.append("")

        # ------------------------------------------------------------------ #
        # Allergies                                                           #
        # ------------------------------------------------------------------ #
        lines.append("=== Allergies ===")
        if resources.allergies:
            for allergy in resources.allergies:
                lines.append(self._format_allergy(allergy))
        else:
            lines.append("None")
        lines.append("")

        # ------------------------------------------------------------------ #
        # Recent Observations (up to 10, sorted newest first)                #
        # ------------------------------------------------------------------ #
        lines.append("=== Recent Observations ===")
        if resources.observations:
            # Work on a copy — never mutate input
            obs_copy = list(resources.observations)
            obs_copy.sort(
                key=lambda o: o.get("effectiveDateTime", ""),
                reverse=True,
            )
            for obs in obs_copy[:10]:
                formatted = self._format_observation(obs)
                if formatted:
                    lines.append(formatted)
        else:
            lines.append("None")
        lines.append("")

        # ------------------------------------------------------------------ #
        # Recent Encounters (up to 3, sorted newest first)                   #
        # ------------------------------------------------------------------ #
        lines.append("=== Recent Encounters ===")
        if resources.encounters:
            enc_copy = list(resources.encounters)
            enc_copy.sort(
                key=lambda e: e.get("period", {}).get("start", ""),
                reverse=True,
            )
            for enc in enc_copy[:3]:
                formatted = self._format_encounter(enc)
                if formatted:
                    lines.append(formatted)
        else:
            lines.append("None")
        lines.append("")

        # ------------------------------------------------------------------ #
        # Care Plan                                                           #
        # ------------------------------------------------------------------ #
        lines.append("=== Care Plan ===")
        care_plan_lines = self._extract_care_plan(resources.care_plans)
        lines.extend(care_plan_lines)
        lines.append("")

        return "\n".join(lines)

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

        # MRN — identifier where type.coding[0].code == "MR"
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
        # Prefer code.text, fall back to code.coding[0].display, then fallback
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
        # Drug name from medicationCodeableConcept.text
        med_concept = med_request.get("medicationCodeableConcept", {})
        drug_name = med_concept.get("text", "Unknown medication")

        # Dosage from dosageInstruction[0].text (omit if absent)
        dosage_instructions = med_request.get("dosageInstruction", [])
        if dosage_instructions:
            dosage_text = dosage_instructions[0].get("text")
            if dosage_text:
                return f"- {drug_name}: {dosage_text}"

        return f"- {drug_name}"

    def _format_allergy(self, allergy: dict) -> str:
        """Format a single FHIR AllergyIntolerance resource as a brief string."""
        parts: list[str] = []

        # Substance from code.text
        substance = allergy.get("code", {}).get("text")
        if substance:
            parts.append(substance)

        # Severity from criticality
        criticality = allergy.get("criticality")
        if criticality:
            parts.append(f"criticality: {criticality}")

        # Reaction from reaction[0].manifestation[0].text
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
        """Format a single FHIR Observation resource as '{name}: {value} {unit} ({date})'."""
        name = obs.get("code", {}).get("text", "")
        if not name:
            return ""

        value_quantity = obs.get("valueQuantity", {})
        value = value_quantity.get("value")
        unit = value_quantity.get("unit", "")

        # ISO date portion only
        effective = obs.get("effectiveDateTime", "")
        date = effective[:10] if effective else ""

        if value is None:
            # Observation without a numeric value — skip
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

        # Type
        enc_types = encounter.get("type", [])
        if enc_types:
            enc_type_text = enc_types[0].get("text")
            if enc_type_text:
                parts.append(enc_type_text)

        # Date (ISO date portion of period.start)
        period_start = encounter.get("period", {}).get("start", "")
        if period_start:
            parts.append(period_start[:10])

        # Reason
        reason_codes = encounter.get("reasonCode", [])
        if reason_codes:
            reason_text = reason_codes[0].get("text")
            if reason_text:
                parts.append(f"reason: {reason_text}")

        if not parts:
            return ""

        return "- " + "; ".join(parts)

    def _extract_care_plan(self, care_plans: list[dict]) -> list[str]:
        """Extract goals and activities from active CarePlans.

        Returns a list of formatted strings, or ["None"] if nothing is found.
        """
        if not care_plans:
            return ["None"]

        result: list[str] = []

        for plan in care_plans:
            # Activities from activity[].detail.description
            activities = plan.get("activity", [])
            for activity in activities:
                detail = activity.get("detail", {})
                description = detail.get("description")
                if description:
                    result.append(f"- Activity: {description}")

        if not result:
            return ["None"]

        return result
