"""Build citeable source items from the supported FHIR resource types."""

from __future__ import annotations

from typing import Any

from src.context_extractor import _date_only, _text
from src.models import PatientResources, SourceItem, SourceSection

SUPPORTED_SOURCE_RESOURCE_TYPES = (
    "Patient",
    "Condition",
    "MedicationRequest",
    "AllergyIntolerance",
    "Observation",
    "Encounter",
    "CarePlan",
)


def build_source_sections(resources: PatientResources) -> list[SourceSection]:
    """Return source sections for the seven supported patient-scoped FHIR types."""
    counter = _SourceIdCounter()
    return [
        _section("Patient (1)", [_patient_item(resources.patient, counter)]),
        _section(
            f"Conditions ({len(resources.conditions)})",
            [_condition_item(resource, counter) for resource in resources.conditions],
        ),
        _section(
            f"Medication Requests ({len(resources.medications)})",
            [_medication_item(resource, counter) for resource in resources.medications],
        ),
        _section(
            f"Allergies ({len(resources.allergies)})",
            [_allergy_item(resource, counter) for resource in resources.allergies],
        ),
        _section(
            f"Observations ({len(resources.observations)})",
            [
                _observation_item(resource, counter)
                for resource in sorted(
                    resources.observations,
                    key=lambda item: _observation_date(item),
                    reverse=True,
                )
            ],
        ),
        _section(
            f"Encounters ({len(resources.encounters)})",
            [
                _encounter_item(resource, counter)
                for resource in sorted(
                    resources.encounters,
                    key=lambda item: item.get("period", {}).get("start", ""),
                    reverse=True,
                )
            ],
        ),
        _section(
            f"Care Plans ({len(resources.care_plans)})",
            [_care_plan_item(resource, counter) for resource in resources.care_plans],
        ),
    ]


def build_source_context(sections: list[SourceSection]) -> str:
    """Return compact source-index context for the LLM."""
    lines = ["=== Citeable FHIR Source Index ==="]
    for section in sections:
        if not section.items:
            continue
        lines.append(f"{section.label}:")
        for item in section.items:
            lines.append(
                f"[{item.source_id}] {item.resource_type}/{item.resource_id}: {item.summary}"
            )
        lines.append("")
    return "\n".join(lines).strip()


def flatten_source_items(sections: list[SourceSection]) -> list[SourceItem]:
    """Return all SourceItem objects in display order."""
    return [item for section in sections for item in section.items]


class _SourceIdCounter:
    def __init__(self) -> None:
        self._next = 1

    def next(self) -> str:
        source_id = f"S{self._next}"
        self._next += 1
        return source_id


def _section(label: str, items: list[SourceItem]) -> SourceSection:
    return SourceSection(label=label, items=items)


def _item(
    *,
    counter: _SourceIdCounter,
    resource: dict[str, Any],
    summary: str,
    evidence: dict[str, Any],
) -> SourceItem:
    resource_type = str(resource.get("resourceType") or "Unknown")
    resource_id = str(resource.get("id") or "unknown")
    source_id = counter.next()
    label = f"{source_id} | {resource_type}/{_short_id(resource_id)} | {summary}"
    return SourceItem(
        source_id=source_id,
        label=label,
        resource_type=resource_type,
        resource_id=resource_id,
        summary=summary,
        evidence={
            "resourceType": resource_type,
            "id": resource_id,
            **{key: value for key, value in evidence.items() if value not in ("", None, [])},
        },
        raw_resource=dict(resource),
    )


def _short_id(resource_id: str) -> str:
    return resource_id[:8] if resource_id else "unknown"


def _patient_item(patient: dict[str, Any], counter: _SourceIdCounter) -> SourceItem:
    name = _patient_name(patient)
    birth_date = patient.get("birthDate", "")
    gender = patient.get("gender", "")
    deceased = patient.get("deceasedDateTime") or patient.get("deceasedBoolean")
    parts = [name]
    if birth_date:
        parts.append(f"DOB: {birth_date}")
    if gender:
        parts.append(f"gender: {gender}")
    if deceased:
        parts.append(f"deceased: {deceased}")
    return _item(
        counter=counter,
        resource=patient,
        summary="; ".join(parts),
        evidence={
            "name": name,
            "birthDate": birth_date,
            "gender": gender,
            "deceased": deceased,
        },
    )


def _condition_item(resource: dict[str, Any], counter: _SourceIdCounter) -> SourceItem:
    code = _text(resource.get("code")) or "Unknown condition"
    onset = resource.get("onsetDateTime") or resource.get("onsetPeriod", {}).get("start", "")
    recorded = resource.get("recordedDate", "")
    clinical = _text(resource.get("clinicalStatus"))
    summary_parts = [code]
    if onset:
        summary_parts.append(f"onset: {_date_only(onset) or onset}")
    if recorded:
        summary_parts.append(f"recorded: {_date_only(recorded) or recorded}")
    return _item(
        counter=counter,
        resource=resource,
        summary="; ".join(summary_parts),
        evidence={
            "code": code,
            "clinicalStatus": clinical,
            "onsetDateTime": onset,
            "recordedDate": recorded,
        },
    )


def _medication_item(resource: dict[str, Any], counter: _SourceIdCounter) -> SourceItem:
    medication = _text(resource.get("medicationCodeableConcept")) or "Unknown medication"
    dosage = ""
    if resource.get("dosageInstruction"):
        dosage = resource["dosageInstruction"][0].get("text", "")
    authored = resource.get("authoredOn", "")
    status = resource.get("status", "")
    summary_parts = [medication]
    if status:
        summary_parts.append(f"status: {status}")
    if authored:
        summary_parts.append(f"authored: {_date_only(authored) or authored}")
    if dosage:
        summary_parts.append(f"dosage: {dosage}")
    return _item(
        counter=counter,
        resource=resource,
        summary="; ".join(summary_parts),
        evidence={
            "medication": medication,
            "status": status,
            "authoredOn": authored,
            "dosage": dosage,
        },
    )


def _allergy_item(resource: dict[str, Any], counter: _SourceIdCounter) -> SourceItem:
    substance = _text(resource.get("code")) or "Unknown allergy"
    clinical = _text(resource.get("clinicalStatus"))
    criticality = resource.get("criticality", "")
    reaction = ""
    if resource.get("reaction"):
        reaction = _text(resource["reaction"][0].get("manifestation"))
    summary_parts = [substance]
    if criticality:
        summary_parts.append(f"criticality: {criticality}")
    if reaction:
        summary_parts.append(f"reaction: {reaction}")
    return _item(
        counter=counter,
        resource=resource,
        summary="; ".join(summary_parts),
        evidence={
            "code": substance,
            "clinicalStatus": clinical,
            "criticality": criticality,
            "reaction": reaction,
            "recordedDate": resource.get("recordedDate", ""),
        },
    )


def _observation_item(resource: dict[str, Any], counter: _SourceIdCounter) -> SourceItem:
    code = _text(resource.get("code")) or "Unknown observation"
    value, unit = _observation_value_and_unit(resource)
    date = _observation_date(resource)
    summary = code
    if value != "":
        summary += f": {value}{(' ' + unit) if unit else ''}"
    if date:
        summary += f" ({_date_only(date) or date})"
    return _item(
        counter=counter,
        resource=resource,
        summary=summary,
        evidence={
            "code": code,
            "value": value,
            "unit": unit,
            "effectiveDateTime": resource.get("effectiveDateTime", ""),
            "issued": resource.get("issued", ""),
            "interpretation": _text(resource.get("interpretation")),
        },
    )


def _encounter_item(resource: dict[str, Any], counter: _SourceIdCounter) -> SourceItem:
    enc_type = _text(resource.get("type")) or "Encounter"
    status = resource.get("status", "")
    start = resource.get("period", {}).get("start", "")
    reason = _text(resource.get("reasonCode"))
    summary_parts = [enc_type]
    if status:
        summary_parts.append(f"status: {status}")
    if start:
        summary_parts.append(f"start: {_date_only(start) or start}")
    if reason:
        summary_parts.append(f"reason: {reason}")
    return _item(
        counter=counter,
        resource=resource,
        summary="; ".join(summary_parts),
        evidence={
            "type": enc_type,
            "status": status,
            "class": _text(resource.get("class")),
            "period.start": start,
            "period.end": resource.get("period", {}).get("end", ""),
            "reason": reason,
        },
    )


def _care_plan_item(resource: dict[str, Any], counter: _SourceIdCounter) -> SourceItem:
    category = _text(resource.get("category")) or resource.get("description", "") or "CarePlan"
    status = resource.get("status", "")
    intent = resource.get("intent", "")
    start = resource.get("period", {}).get("start", "")
    summary_parts = [f"CarePlan: {category}"]
    if status:
        summary_parts.append(f"status: {status}")
    if intent:
        summary_parts.append(f"intent: {intent}")
    if start:
        summary_parts.append(f"start: {_date_only(start) or start}")
    return _item(
        counter=counter,
        resource=resource,
        summary="; ".join(summary_parts),
        evidence={
            "category": category,
            "status": status,
            "intent": intent,
            "period.start": start,
            "period.end": resource.get("period", {}).get("end", ""),
        },
    )


def _patient_name(patient: dict[str, Any]) -> str:
    names = patient.get("name", [])
    if not names:
        return "Unknown"
    first = names[0]
    if first.get("text"):
        return str(first["text"])
    given = " ".join(first.get("given", []))
    family = first.get("family", "")
    return " ".join(part for part in (given, family) if part) or "Unknown"


def _observation_date(resource: dict[str, Any]) -> str:
    return (
        resource.get("effectiveDateTime")
        or resource.get("issued")
        or resource.get("effectivePeriod", {}).get("start")
        or ""
    )


def _observation_value_and_unit(resource: dict[str, Any]) -> tuple[Any, str]:
    quantity = resource.get("valueQuantity")
    if isinstance(quantity, dict) and quantity.get("value") is not None:
        return quantity["value"], str(quantity.get("unit") or quantity.get("code") or "")
    for field in ("valueString", "valueInteger", "valueBoolean"):
        if field in resource:
            return resource[field], ""
    if "valueCodeableConcept" in resource:
        return _text(resource["valueCodeableConcept"]), ""
    if resource.get("component"):
        values = []
        for component in resource["component"]:
            label = _text(component.get("code")) or "component"
            value, unit = _observation_value_and_unit(component)
            if value != "":
                values.append(f"{label} {value}{(' ' + unit) if unit else ''}")
        return " / ".join(values), ""
    return "", ""
