"""
PatientContextExtractor converts FHIR resources into an ED-focused clinical
context block for LLM consumption.

The extractor emphasizes:
- current ED-relevant issues
- recent clinical changes
- acute risks, missing safety data, and clinically actionable follow-up

It does not mutate caller-provided FHIR resources.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import tiktoken

from src.models import PatientResources  # noqa: F401

_TOKEN_BUDGET = 3_000
_ENCODING_NAME = "cl100k_base"

_ED_KEYWORDS = (
    "overdose",
    "detox",
    "bleeding",
    "hemorrhage",
    "cancer",
    "malign",
    "pain",
    "acute",
    "emergency",
    "ed ",
    "death",
    "death certification",
)

_SEVERE_CONDITION_KEYWORDS = (
    "cancer",
    "malign",
    "metast",
    "heart failure",
    "myocardial",
    "stroke",
    "sepsis",
    "renal failure",
    "kidney failure",
    "respiratory failure",
    "overdose",
    "bleeding",
    "hemorrhage",
)

_HIGH_RISK_MED_KEYWORDS = (
    "warfarin",
    "apixaban",
    "rivaroxaban",
    "dabigatran",
    "heparin",
    "enoxaparin",
    "insulin",
    "morphine",
    "oxycodone",
    "hydrocodone",
    "fentanyl",
    "methadone",
    "buprenorphine",
    "diazepam",
    "lorazepam",
    "alprazolam",
    "clonazepam",
    "chemo",
    "cisplatin",
    "doxorubicin",
    "vancomycin",
    "gentamicin",
    "tobramycin",
    "ibuprofen",
    "naproxen",
    "ketorolac",
    "antibiotic",
    "amoxicillin",
    "azithromycin",
    "ciprofloxacin",
    "levofloxacin",
)

_ACTIONABLE_CAREPLAN_KEYWORDS = (
    "cancer",
    "oncology",
    "overdose",
    "substance",
    "opioid",
    "detox",
    "end-of-life",
    "palliative",
    "hospice",
    "diabetes",
    "insulin",
    "ed",
    "emergency",
)

_VITAL_KEYWORDS = {
    "blood pressure": "BP",
    "systolic": "BP",
    "diastolic": "BP",
    "heart rate": "HR",
    "pulse": "HR",
    "respiratory rate": "RR",
    "temperature": "Temp",
    "oxygen saturation": "SpO2",
    "spo2": "SpO2",
    "pain": "Pain",
    "weight": "Weight/BMI",
    "body mass index": "Weight/BMI",
    "bmi": "Weight/BMI",
}

_LAB_KEYWORDS = {
    "hemoglobin": "CBC",
    "hematocrit": "CBC",
    "platelet": "CBC",
    "wbc": "CBC",
    "white blood": "CBC",
    "sodium": "BMP/CMP",
    "potassium": "BMP/CMP",
    "chloride": "BMP/CMP",
    "bicarbonate": "BMP/CMP",
    "creatinine": "Renal",
    "egfr": "Renal",
    "bun": "Renal",
    "glucose": "Glucose/HbA1c",
    "a1c": "Glucose/HbA1c",
    "hba1c": "Glucose/HbA1c",
    "cholesterol": "Lipids",
    "ldl": "Lipids",
    "hdl": "Lipids",
    "triglyceride": "Lipids",
}

_SOCIAL_HISTORY_KEYWORDS = (
    "smoking",
    "tobacco",
    "alcohol",
    "substance",
    "drug use",
    "opioid",
)


def _count_tokens(text: str) -> int:
    """Return the cl100k_base token count for the supplied text."""
    enc = tiktoken.get_encoding(_ENCODING_NAME)
    return len(enc.encode(text))


def _parse_date(value: str | None) -> datetime | None:
    """Parse common FHIR date/dateTime strings into a datetime."""
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        if "T" in normalized:
            return datetime.fromisoformat(normalized)
        return datetime.fromisoformat(f"{normalized}T00:00:00")
    except ValueError:
        return None


def _date_key(value: str | None) -> str:
    """Return a sortable string for FHIR date/dateTime fields."""
    parsed = _parse_date(value)
    return parsed.isoformat() if parsed else ""


def _date_only(value: str | None) -> str:
    """Return the yyyy-mm-dd portion of a FHIR date/dateTime value."""
    if not value:
        return ""
    return value[:10]


def _text(value: Any) -> str:
    """Return a compact display string for common FHIR coding structures."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_text(item) for item in value]
        return ", ".join(part for part in parts if part)
    if not isinstance(value, dict):
        return str(value)

    if value.get("text"):
        return str(value["text"])

    coding = value.get("coding")
    if isinstance(coding, list):
        for item in coding:
            display = _text(item)
            if display:
                return display

    if value.get("display"):
        return str(value["display"])
    if value.get("code"):
        return str(value["code"])
    return ""


def _status_text(value: Any) -> str:
    """Return status text from a FHIR CodeableConcept-like value."""
    return _text(value) or "Unknown"


def _contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    """Return True when text contains any keyword, case-insensitively."""
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


class PatientContextExtractor:
    """Convert PatientResources into a structured ED-focused context block."""

    def extract(self, resources: PatientResources) -> str:
        """Return a compact clinical context string.

        The output includes patient demographics, death-related data, sorted
        encounter chronology, ranked conditions, observations, medications,
        allergies, and actionable care plans.
        """
        encounters = sorted(
            list(resources.encounters),
            key=lambda e: _date_key(e.get("period", {}).get("start")),
        )
        observations = sorted(
            list(resources.observations),
            key=lambda o: _date_key(self._observation_date(o)),
            reverse=True,
        )

        lines: list[str] = []
        lines.extend(self._build_patient_section(resources.patient, encounters, observations))
        lines.extend(self._build_encounter_section(encounters))
        lines.extend(self._build_condition_section(list(resources.conditions)))
        lines.extend(self._build_observation_section(observations))
        lines.extend(self._build_medication_section(list(resources.medications), resources.patient))
        lines.extend(self._build_allergy_section(list(resources.allergies)))
        lines.extend(self._build_careplan_section(list(resources.care_plans)))

        result = "\n".join(lines).strip() + "\n"
        return self._enforce_budget(result)

    def _enforce_budget(self, text: str) -> str:
        """Trim low-priority tail lines until the context fits the token budget."""
        if _count_tokens(text) <= _TOKEN_BUDGET:
            return text

        lines = text.splitlines()
        protected_prefix = []
        for line in lines:
            protected_prefix.append(line)
            if line == "=== Encounters ===":
                break

        work = list(lines)
        while _count_tokens("\n".join(work)) > _TOKEN_BUDGET and len(work) > len(protected_prefix):
            work.pop()

        return "\n".join(work).strip() + "\n"

    # ------------------------------------------------------------------ #
    # Section builders                                                    #
    # ------------------------------------------------------------------ #

    def _build_patient_section(
        self,
        patient: dict,
        encounters: list[dict],
        observations: list[dict],
    ) -> list[str]:
        """Build patient demographics, death, and death-cause context."""
        lines = ["=== Patient Demographics ==="]
        lines.extend(self._extract_demographics(patient, encounters))

        deceased = patient.get("deceasedDateTime") or patient.get("deceasedBoolean")
        death_cert = self._find_death_certification_encounter(encounters)
        cause = self._find_cause_of_death(observations)
        has_death_context = bool(deceased or death_cert or cause)

        if deceased:
            lines.append(f"Deceased: {deceased}")

        if has_death_context:
            if death_cert:
                lines.append(
                    "Death certification encounter date: "
                    f"{_date_only(death_cert.get('period', {}).get('start')) or 'Unknown'}"
                )
            else:
                lines.append("Death certification encounter date: not documented")

            if cause:
                lines.append(f"Cause of death observation: {cause}")
            else:
                lines.append("Cause of death observation: not documented")

        lines.append("")
        return lines

    def _build_encounter_section(self, encounters: list[dict]) -> list[str]:
        """Build encounter chronology, latest encounter, and ED-relevant encounters."""
        lines = ["=== Encounters ==="]
        if encounters:
            dates = [_date_only(e.get("period", {}).get("start")) for e in encounters]
            lines.append("All encounter dates ascending: " + ", ".join(d for d in dates if d))
            lines.append("Latest encounter: " + self._format_encounter_detail(encounters[-1]))
        else:
            lines.append("All encounter dates ascending: None")
            lines.append("Latest encounter: None")

        ed_relevant = [e for e in encounters if self._is_ed_relevant_encounter(e)]
        lines.append("ED-relevant encounters:")
        if ed_relevant:
            for encounter in ed_relevant:
                lines.append(self._format_encounter_detail(encounter))
        else:
            lines.append("None")

        lines.append("")

        lines.append("=== Recent Encounters ===")
        if encounters:
            for encounter in reversed(encounters[-3:]):
                rendered = self._format_encounter(encounter)
                if rendered:
                    lines.append(rendered)
        else:
            lines.append("None")
        lines.append("")
        return lines

    def _build_condition_section(self, conditions: list[dict]) -> list[str]:
        """Build ranked active, recent, and severe historical condition context."""
        ranked = sorted(conditions, key=self._condition_rank, reverse=True)
        active = [c for c in ranked if self._is_active_condition(c)]
        recent = [c for c in ranked if self._is_recent_condition(c)]
        severe_historical = [
            c for c in ranked
            if not self._is_active_condition(c) and self._is_severe_condition(c)
        ]

        lines = ["=== Active Conditions ==="]
        if active:
            for condition in active[:8]:
                lines.append(self._format_condition_detail(condition))
        else:
            lines.append("None")
        lines.append("")

        lines.append("=== Recent Conditions ===")
        if recent:
            for condition in recent[:6]:
                lines.append(self._format_condition_detail(condition))
        else:
            lines.append("None")
        lines.append("")

        lines.append("=== Severe Historical Conditions ===")
        if severe_historical:
            for condition in severe_historical[:6]:
                lines.append(self._format_condition_detail(condition))
        else:
            lines.append("None")
        lines.append("")
        return lines

    def _build_observation_section(self, observations: list[dict]) -> list[str]:
        """Build vitals, labs, abnormal, death-cause, and social history observations."""
        vitals = self._latest_by_category(observations, self._vital_category)
        labs = self._latest_by_category(observations, self._lab_category)
        abnormal = [o for o in observations if self._is_abnormal_observation(o)]
        death_causes = [o for o in observations if self._is_death_cause_observation(o)]
        social = [o for o in observations if self._is_social_history_observation(o)]

        lines = ["=== Recent Observations ==="]
        if observations:
            for obs in observations[:10]:
                rendered = self._format_observation(obs)
                if rendered:
                    lines.append(rendered)
        else:
            lines.append("None")
        lines.append("")

        lines.append("=== Latest Vitals ===")
        if vitals:
            for label in ("BP", "HR", "RR", "Temp", "SpO2", "Pain", "Weight/BMI"):
                if label in vitals:
                    lines.append(self._format_observation(vitals[label]))
        else:
            lines.append("None")
        lines.append("")

        lines.append("=== Latest Labs ===")
        if labs:
            for label in ("CBC", "BMP/CMP", "Renal", "Glucose/HbA1c", "Lipids"):
                if label in labs:
                    lines.append(self._format_observation(labs[label]))
        else:
            lines.append("None")
        lines.append("")

        lines.append("=== Abnormal or Critical Observations ===")
        if abnormal:
            for obs in abnormal[:12]:
                lines.append(self._format_observation(obs))
        else:
            lines.append("None")
        lines.append("")

        lines.append("=== Death Cause Observations ===")
        if death_causes:
            for obs in death_causes:
                lines.append(self._format_observation(obs))
        else:
            lines.append("None")
        lines.append("")

        lines.append("=== Social History Observations ===")
        if social:
            for obs in social[:8]:
                lines.append(self._format_observation(obs))
        else:
            lines.append("None")
        lines.append("")
        return lines

    def _build_medication_section(self, medications: list[dict], patient: dict) -> list[str]:
        """Build active/recent, historical, and high-risk medication context."""
        deceased = bool(patient.get("deceasedDateTime") or patient.get("deceasedBoolean"))
        active_recent = [m for m in medications if self._is_active_or_recent_medication(m)]
        high_risk = [m for m in medications if self._is_high_risk_medication(m)]
        sorted_medications = sorted(
            medications,
            key=lambda m: m.get("authoredOn", ""),
            reverse=True,
        )

        lines = ["=== Active Medications ==="]
        if active_recent:
            for med in sorted(active_recent, key=lambda m: m.get("authoredOn", ""), reverse=True):
                lines.append(self._format_medication_detail(med, deceased))
        else:
            if deceased and medications:
                lines.append(
                    "None documented as currently active in the supplied context; "
                    "see Medication History for pre-death records."
                )
            else:
                lines.append("None")
        lines.append("")

        lines.append("=== Medication History ===")
        if sorted_medications:
            for med in sorted_medications[:12]:
                lines.append(self._format_medication_detail(med, deceased))
        else:
            lines.append("None")
        lines.append("")

        lines.append("=== High-Risk Medications ===")
        if high_risk:
            for med in high_risk:
                lines.append(self._format_medication_detail(med, deceased))
        else:
            lines.append("None")
        lines.append("")
        return lines

    def _build_allergy_section(self, allergies: list[dict]) -> list[str]:
        """Build allergy context with documented absence wording."""
        lines = ["=== Allergies ==="]
        if allergies:
            for allergy in allergies:
                lines.append(self._format_allergy_detail(allergy))
        else:
            lines.append("No allergies documented")
        lines.append("")
        return lines

    def _build_careplan_section(self, care_plans: list[dict]) -> list[str]:
        """Build only clinically actionable ED-relevant care plan context."""
        actionable = [plan for plan in care_plans if self._is_actionable_care_plan(plan)]
        lines = ["=== Care Plan ==="]
        if actionable:
            for plan in actionable:
                lines.append(self._format_care_plan(plan))
        else:
            lines.append("None")
        lines.append("")
        return lines

    # ------------------------------------------------------------------ #
    # Demographics                                                        #
    # ------------------------------------------------------------------ #

    def _extract_demographics(self, patient: dict, encounters: list[dict] | None = None) -> list[str]:
        """Extract patient sex, DOB, age at latest encounter, name, and MRN."""
        result: list[str] = []

        name = self._patient_name(patient)
        result.append(f"Name: {name}")

        dob = patient.get("birthDate", "Unknown")
        result.append(f"DOB: {dob}")

        gender = patient.get("gender", "Unknown")
        result.append(f"Gender: {gender}")
        result.append(f"Sex: {gender}")

        latest_encounter_date = ""
        if encounters:
            latest_encounter_date = encounters[-1].get("period", {}).get("start", "")
        age = self._age_at_date(patient.get("birthDate"), latest_encounter_date)
        if age is not None:
            result.append(f"Age at latest encounter: {age}")
        else:
            result.append("Age at latest encounter: Unknown")

        mrn = "Unknown"
        for identifier in patient.get("identifier", []):
            coding_list = identifier.get("type", {}).get("coding", [])
            if coding_list and coding_list[0].get("code") == "MR":
                mrn = identifier.get("value", "Unknown")
                break
        result.append(f"MRN: {mrn}")
        return result

    def _patient_name(self, patient: dict) -> str:
        """Extract the preferred patient display name."""
        names = patient.get("name", [])
        if not names:
            return "Unknown"
        first_name = names[0]
        if first_name.get("text"):
            return first_name["text"]
        given = " ".join(first_name.get("given", []))
        family = first_name.get("family", "")
        parts = [p for p in [given, family] if p]
        return " ".join(parts) if parts else "Unknown"

    def _age_at_date(self, birth_date: str | None, target_date: str | None) -> int | None:
        """Calculate age at a target date, usually the latest encounter date."""
        birth = _parse_date(birth_date)
        target = _parse_date(target_date) or datetime.combine(date.today(), datetime.min.time())
        if not birth:
            return None
        years = target.year - birth.year
        if (target.month, target.day) < (birth.month, birth.day):
            years -= 1
        return years

    # ------------------------------------------------------------------ #
    # Conditions                                                          #
    # ------------------------------------------------------------------ #

    def _format_condition(self, condition: dict) -> str:
        """Format a Condition name for compact source displays."""
        return "- " + (self._condition_name(condition) or "Unknown condition")

    def _format_condition_detail(self, condition: dict) -> str:
        """Format a Condition with status, verification, and dates."""
        parts = [self._condition_name(condition) or "Unknown condition"]
        parts.append(f"clinicalStatus: {_status_text(condition.get('clinicalStatus'))}")
        parts.append(f"verificationStatus: {_status_text(condition.get('verificationStatus'))}")
        for field in ("onsetDateTime", "recordedDate", "abatementDateTime"):
            if condition.get(field):
                parts.append(f"{field}: {condition[field]}")
        if self._is_severe_condition(condition):
            parts.append("ED relevance: high")
        return "- " + "; ".join(parts)

    def _condition_name(self, condition: dict) -> str:
        """Return a display name for a Condition."""
        return _text(condition.get("code"))

    def _is_active_condition(self, condition: dict) -> bool:
        """Return True when a Condition appears active."""
        clinical = _status_text(condition.get("clinicalStatus")).lower()
        if "inactive" in clinical or "resolved" in clinical or "remission" in clinical:
            return False
        if condition.get("abatementDateTime"):
            return False
        return "active" in clinical or clinical == "unknown" or not condition.get("clinicalStatus")

    def _is_recent_condition(self, condition: dict) -> bool:
        """Return True when a Condition has recent date fields."""
        latest = condition.get("recordedDate") or condition.get("onsetDateTime")
        parsed = _parse_date(latest)
        if not parsed:
            return False
        return parsed.year >= datetime.now().year - 2

    def _is_severe_condition(self, condition: dict) -> bool:
        """Return True when a Condition is historically severe or ED-relevant."""
        return _contains_keyword(self._condition_name(condition), _SEVERE_CONDITION_KEYWORDS)

    def _condition_rank(self, condition: dict) -> int:
        """Rank conditions by ED relevance for compact context ordering."""
        score = 0
        if self._is_active_condition(condition):
            score += 5
        if self._is_recent_condition(condition):
            score += 3
        if self._is_severe_condition(condition):
            score += 8
        return score

    # ------------------------------------------------------------------ #
    # Medications                                                         #
    # ------------------------------------------------------------------ #

    def _format_medication(self, med_request: dict) -> str:
        """Format a MedicationRequest name and free-text dosage."""
        drug_name = self._medication_name(med_request)
        dosage_text = self._dosage_text(med_request)
        if dosage_text:
            return f"- {drug_name}: {dosage_text}"
        return f"- {drug_name}"

    def _format_medication_detail(self, med_request: dict, deceased: bool = False) -> str:
        """Format a MedicationRequest with status, dosage, route, timing, and indication."""
        parts = [self._medication_name(med_request)]
        status = med_request.get("status")
        if status:
            parts.append(f"status: {status}")
        if deceased:
            parts.append("currentness: historical because patient is deceased")
        if med_request.get("authoredOn"):
            parts.append(f"authoredOn: {med_request['authoredOn']}")

        dosage = self._dosage_text(med_request)
        if dosage:
            parts.append(f"dosage: {dosage}")

        route = self._dosage_route(med_request)
        if route:
            parts.append(f"route: {route}")

        frequency = self._dosage_frequency(med_request)
        if frequency:
            parts.append(f"frequency: {frequency}")

        bounds = self._dosage_bounds(med_request)
        if bounds:
            parts.append(bounds)

        indication = _text(med_request.get("reasonCode")) or self._reference_text(med_request.get("reasonReference"))
        if indication:
            parts.append(f"indication: {indication}")

        if self._is_high_risk_medication(med_request):
            parts.append("high-risk: yes")

        return "- " + "; ".join(parts)

    def _medication_name(self, med_request: dict) -> str:
        """Return the MedicationRequest medication display name."""
        return _text(med_request.get("medicationCodeableConcept")) or "Unknown medication"

    def _dosage_text(self, med_request: dict) -> str:
        """Return the first dosageInstruction.text if present."""
        instructions = med_request.get("dosageInstruction", [])
        if instructions:
            return instructions[0].get("text", "")
        return ""

    def _dosage_route(self, med_request: dict) -> str:
        """Return dosage route text when present."""
        instructions = med_request.get("dosageInstruction", [])
        if not instructions:
            return ""
        return _text(instructions[0].get("route"))

    def _dosage_frequency(self, med_request: dict) -> str:
        """Return a readable timing frequency when present."""
        instructions = med_request.get("dosageInstruction", [])
        if not instructions:
            return ""
        repeat = instructions[0].get("timing", {}).get("repeat", {})
        frequency = repeat.get("frequency")
        period = repeat.get("period")
        unit = repeat.get("periodUnit")
        if frequency and period and unit:
            return f"{frequency} per {period} {unit}"
        return ""

    def _dosage_bounds(self, med_request: dict) -> str:
        """Return medication timing start/end if present."""
        instructions = med_request.get("dosageInstruction", [])
        if not instructions:
            return ""
        timing = instructions[0].get("timing", {})
        period = timing.get("repeat", {}).get("boundsPeriod") or timing.get("boundsPeriod")
        if not period:
            return ""
        parts = []
        if period.get("start"):
            parts.append(f"start: {period['start']}")
        if period.get("end"):
            parts.append(f"end: {period['end']}")
        return "; ".join(parts)

    def _is_active_or_recent_medication(self, med_request: dict) -> bool:
        """Return True for active or dated recent MedicationRequests."""
        status = str(med_request.get("status", "")).lower()
        if not status:
            return True
        if status in {"active", "on-hold", "unknown"}:
            return True
        authored = _parse_date(med_request.get("authoredOn"))
        return bool(authored and authored.year >= datetime.now().year - 2)

    def _is_high_risk_medication(self, med_request: dict) -> bool:
        """Return True when a medication name suggests ED-relevant risk."""
        return _contains_keyword(self._medication_name(med_request), _HIGH_RISK_MED_KEYWORDS)

    # ------------------------------------------------------------------ #
    # Allergies                                                           #
    # ------------------------------------------------------------------ #

    def _format_allergy(self, allergy: dict) -> str:
        """Format an AllergyIntolerance for compact source displays."""
        parts: list[str] = []
        substance = _text(allergy.get("code"))
        if substance:
            parts.append(substance)
        criticality = allergy.get("criticality")
        if criticality:
            parts.append(f"criticality: {criticality}")
        reactions = allergy.get("reaction", [])
        if reactions:
            reaction_text = _text(reactions[0].get("manifestation"))
            if reaction_text:
                parts.append(f"reaction: {reaction_text}")
        if not parts:
            return "- Unknown allergy"
        return "- " + "; ".join(parts)

    def _format_allergy_detail(self, allergy: dict) -> str:
        """Format AllergyIntolerance with status, reaction, severity, and date."""
        parts = [_text(allergy.get("code")) or "Unknown allergy"]
        parts.append(f"clinicalStatus: {_status_text(allergy.get('clinicalStatus'))}")
        parts.append(f"verificationStatus: {_status_text(allergy.get('verificationStatus'))}")
        if allergy.get("criticality"):
            parts.append(f"criticality: {allergy['criticality']}")
        if allergy.get("recordedDate"):
            parts.append(f"recordedDate: {allergy['recordedDate']}")

        reactions = allergy.get("reaction", [])
        if reactions:
            reaction = reactions[0]
            manifestation = _text(reaction.get("manifestation"))
            if manifestation:
                parts.append(f"manifestation: {manifestation}")
                parts.append(f"reaction: {manifestation}")
            if reaction.get("severity"):
                parts.append(f"severity: {reaction['severity']}")
        return "- " + "; ".join(parts)

    # ------------------------------------------------------------------ #
    # Observations                                                        #
    # ------------------------------------------------------------------ #

    def _format_observation(self, obs: dict) -> str:
        """Format an Observation with code, value, unit, date, and abnormal metadata."""
        name = _text(obs.get("code"))
        if not name:
            return ""

        value = self._observation_value(obs)
        if not value:
            return ""

        date_text = _date_only(self._observation_date(obs))
        parts = [f"{name}: {value}"]
        if date_text:
            parts.append(f"({date_text})")

        interpretation = _text(obs.get("interpretation"))
        if interpretation:
            parts.append(f"flag: {interpretation}")

        reference = self._reference_range(obs)
        if reference:
            parts.append(f"reference range: {reference}")

        return "- " + " ".join(parts)

    def _observation_date(self, obs: dict) -> str:
        """Return the best available Observation date."""
        return (
            obs.get("effectiveDateTime")
            or obs.get("issued")
            or obs.get("effectivePeriod", {}).get("start")
            or ""
        )

    def _observation_value(self, obs: dict) -> str:
        """Return a readable Observation value, including BP components."""
        if obs.get("component"):
            component_values = []
            for component in obs["component"]:
                label = _text(component.get("code")) or "component"
                value = self._quantity_value(component.get("valueQuantity", {}))
                if value:
                    component_values.append(f"{label} {value}")
            if component_values:
                return " / ".join(component_values)

        quantity = self._quantity_value(obs.get("valueQuantity", {}))
        if quantity:
            return quantity
        for field in ("valueString", "valueCodeableConcept", "valueInteger", "valueBoolean"):
            if field in obs:
                return _text(obs[field])
        return ""

    def _quantity_value(self, quantity: dict) -> str:
        """Return value plus unit from a FHIR Quantity."""
        if not quantity or quantity.get("value") is None:
            return ""
        value = str(quantity["value"])
        unit = quantity.get("unit") or quantity.get("code") or ""
        return f"{value} {unit}".strip()

    def _reference_range(self, obs: dict) -> str:
        """Return the first Observation reference range when present."""
        ranges = obs.get("referenceRange", [])
        if not ranges:
            return ""
        first = ranges[0]
        if first.get("text"):
            return first["text"]
        low = self._quantity_value(first.get("low", {}))
        high = self._quantity_value(first.get("high", {}))
        if low and high:
            return f"{low}-{high}"
        return low or high

    def _is_abnormal_observation(self, obs: dict) -> bool:
        """Return True for abnormal or critical Observation interpretation flags."""
        interpretation = _text(obs.get("interpretation")).lower()
        if any(flag in interpretation for flag in ("abnormal", "critical", "high", "low", "panic")):
            return True
        return any(
            coding.get("code") in {"A", "AA", "H", "HH", "L", "LL", "HU", "LU"}
            for item in obs.get("interpretation", [])
            for coding in item.get("coding", [])
        )

    def _is_death_cause_observation(self, obs: dict) -> bool:
        """Return True when an Observation appears to document cause of death."""
        name = _text(obs.get("code"))
        return "cause of death" in name.lower()

    def _is_social_history_observation(self, obs: dict) -> bool:
        """Return True for smoking, alcohol, or substance-use observations."""
        return _contains_keyword(_text(obs.get("code")), _SOCIAL_HISTORY_KEYWORDS)

    def _vital_category(self, obs: dict) -> str:
        """Classify an Observation as a vital sign category if possible."""
        name = _text(obs.get("code")).lower()
        for keyword, label in _VITAL_KEYWORDS.items():
            if keyword in name:
                return label
        return ""

    def _lab_category(self, obs: dict) -> str:
        """Classify an Observation as a lab category if possible."""
        name = _text(obs.get("code")).lower()
        for keyword, label in _LAB_KEYWORDS.items():
            if keyword in name:
                return label
        return ""

    def _latest_by_category(self, observations: list[dict], classifier) -> dict[str, dict]:
        """Return newest Observation per classifier label."""
        result: dict[str, dict] = {}
        for obs in observations:
            label = classifier(obs)
            if label and label not in result:
                result[label] = obs
        return result

    def _find_cause_of_death(self, observations: list[dict]) -> str:
        """Return formatted cause-of-death Observation value if present."""
        for obs in observations:
            if self._is_death_cause_observation(obs):
                return self._format_observation(obs).lstrip("- ")
        return ""

    # ------------------------------------------------------------------ #
    # Encounters                                                          #
    # ------------------------------------------------------------------ #

    def _format_encounter(self, encounter: dict) -> str:
        """Format a brief Encounter line for compact source displays."""
        parts: list[str] = []
        enc_type = _text(encounter.get("type"))
        if enc_type:
            parts.append(enc_type)

        period_start = encounter.get("period", {}).get("start", "")
        if period_start:
            parts.append(_date_only(period_start))

        reason = _text(encounter.get("reasonCode"))
        if reason:
            parts.append(f"reason: {reason}")

        if not parts:
            return ""
        return "- " + "; ".join(parts)

    def _format_encounter_detail(self, encounter: dict) -> str:
        """Format Encounter date, type, class, status, reason, and diagnosis."""
        parts: list[str] = []
        date_text = _date_only(encounter.get("period", {}).get("start"))
        if date_text:
            parts.append(f"date: {date_text}")
        enc_type = _text(encounter.get("type"))
        if enc_type:
            parts.append(f"type: {enc_type}")
        enc_class = _text(encounter.get("class"))
        if enc_class:
            parts.append(f"class: {enc_class}")
        if encounter.get("status"):
            parts.append(f"status: {encounter['status']}")
        reason = _text(encounter.get("reasonCode"))
        if reason:
            parts.append(f"reason: {reason}")
        diagnosis = self._encounter_diagnosis(encounter)
        if diagnosis:
            parts.append(f"diagnosis: {diagnosis}")
        return "- " + "; ".join(parts) if parts else "- Unknown encounter"

    def _encounter_diagnosis(self, encounter: dict) -> str:
        """Return diagnosis reference text from Encounter.diagnosis if present."""
        diagnosis = encounter.get("diagnosis", [])
        refs = []
        for item in diagnosis:
            ref = self._reference_text(item.get("condition"))
            if ref:
                refs.append(ref)
        return ", ".join(refs)

    def _is_ed_relevant_encounter(self, encounter: dict) -> bool:
        """Return True for encounters matching ED-relevant keywords."""
        haystack = " ".join(
            [
                _text(encounter.get("type")),
                _text(encounter.get("class")),
                _text(encounter.get("reasonCode")),
                self._encounter_diagnosis(encounter),
            ]
        )
        return _contains_keyword(haystack, _ED_KEYWORDS)

    def _find_death_certification_encounter(self, encounters: list[dict]) -> dict | None:
        """Return the first encounter that appears to be death certification."""
        for encounter in encounters:
            haystack = " ".join(
                [_text(encounter.get("type")), _text(encounter.get("reasonCode"))]
            ).lower()
            if "death certification" in haystack or "death certificate" in haystack:
                return encounter
        return None

    # ------------------------------------------------------------------ #
    # Care plans                                                          #
    # ------------------------------------------------------------------ #

    def _extract_activity_lines(self, care_plans: list[dict]) -> list[str]:
        """Extract activity description lines from actionable CarePlans."""
        result: list[str] = []
        for plan in care_plans:
            if not self._is_actionable_care_plan(plan):
                continue
            for activity in plan.get("activity", []):
                description = activity.get("detail", {}).get("description")
                if description:
                    result.append(f"- Activity: {description}")
        return result

    def _is_actionable_care_plan(self, plan: dict) -> bool:
        """Return True for clinically actionable, ED-relevant CarePlans."""
        haystack = " ".join(
            [
                plan.get("status", ""),
                plan.get("intent", ""),
                _text(plan.get("category")),
                _text(plan.get("reasonCode")),
                _text(plan.get("description")),
                " ".join(
                    activity.get("detail", {}).get("description", "")
                    for activity in plan.get("activity", [])
                ),
            ]
        )
        return _contains_keyword(haystack, _ACTIONABLE_CAREPLAN_KEYWORDS)

    def _format_care_plan(self, plan: dict) -> str:
        """Format actionable CarePlan status, intent, category, period, reason, activities."""
        parts = []
        if plan.get("status"):
            parts.append(f"status: {plan['status']}")
        if plan.get("intent"):
            parts.append(f"intent: {plan['intent']}")
        category = _text(plan.get("category"))
        if category:
            parts.append(f"category: {category}")
        period = plan.get("period", {})
        if period.get("start"):
            parts.append(f"period.start: {period['start']}")
        if period.get("end"):
            parts.append(f"period.end: {period['end']}")
        reason = _text(plan.get("reasonCode"))
        if reason:
            parts.append(f"reason: {reason}")
        activities = [
            activity.get("detail", {}).get("description")
            for activity in plan.get("activity", [])
            if activity.get("detail", {}).get("description")
        ]
        if activities:
            parts.append("key activities: " + " | ".join(activities[:4]))
        return "- " + "; ".join(parts) if parts else "- Actionable care plan"

    # ------------------------------------------------------------------ #
    # Shared helpers                                                      #
    # ------------------------------------------------------------------ #

    def _reference_text(self, value: Any) -> str:
        """Return display/reference text from FHIR Reference-like values."""
        if isinstance(value, list):
            return ", ".join(filter(None, [self._reference_text(item) for item in value]))
        if not isinstance(value, dict):
            return ""
        return value.get("display") or value.get("reference") or ""

    # Keep the legacy helper name so any existing callers do not break.
    def _extract_care_plan(self, care_plans: list[dict]) -> list[str]:
        """Legacy wrapper that returns actionable care plan activity lines."""
        lines = self._extract_activity_lines(care_plans)
        return lines if lines else ["None"]
