"""Build citeable source items from the supported FHIR resource types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.context_extractor import _date_only, _text
from src.models import PatientResources, SourceItem, SourceSection

_SKIPPED_EVIDENCE_KEYS = {"extension"}
_SOURCE_CONTEXT_MAX_CHARS = 24000
_SOURCE_CONTEXT_MAX_VALUE_CHARS = 500
_TRUNCATION_NOTICE = "[Context truncated due to source context budget.]"


@dataclass(frozen=True)
class SourceContextResult:
    """Source context text plus the exact SourceItems supplied in that text."""

    text: str
    sections: list[SourceSection]
    supplied_source_ids: set[str]
    truncated: bool


def build_source_sections(resources: PatientResources) -> list[SourceSection]:
    """Return source sections for the seven supported patient-scoped FHIR types."""
    counter = _SourceIdCounter()
    return [
        SourceSection(label="Patient (1)", items=[_patient_item(resources.patient, counter)]),
        SourceSection(
            label=f"Conditions ({len(resources.conditions)})",
            items=[_condition_item(resource, counter) for resource in resources.conditions],
        ),
        SourceSection(
            label=f"Medication Requests ({len(resources.medications)})",
            items=[_medication_item(resource, counter) for resource in resources.medications],
        ),
        SourceSection(
            label=f"Allergies ({len(resources.allergies)})",
            items=[_allergy_item(resource, counter) for resource in resources.allergies],
        ),
        SourceSection(
            label=f"Observations ({len(resources.observations)})",
            items=[
                _observation_item(resource, counter)
                for resource in sorted(
                    resources.observations,
                    key=lambda item: _observation_date(item),
                    reverse=True,
                )
            ],
        ),
        SourceSection(
            label=f"Encounters ({len(resources.encounters)})",
            items=[
                _encounter_item(resource, counter)
                for resource in sorted(
                    resources.encounters,
                    key=lambda item: item.get("period", {}).get("start", ""),
                    reverse=True,
                )
            ],
        ),
        SourceSection(
            label=f"Care Plans ({len(resources.care_plans)})",
            items=[_care_plan_item(resource, counter) for resource in resources.care_plans],
        ),
    ]


def build_source_context(sections: list[SourceSection]) -> str:
    """Return source-indexed factual context for the LLM."""
    return build_source_context_result(sections).text


def build_source_context_result(
    sections: list[SourceSection],
    *,
    max_chars: int = _SOURCE_CONTEXT_MAX_CHARS,
) -> SourceContextResult:
    """Return LLM context and the exact SourceItems included in that context."""
    lines = [
        "=== Source-Indexed FHIR Context ===",
        "Use only the source-indexed FHIR facts below.",
        "Every factual claim based on these facts should cite one or more source ids.",
        "Do not cite source ids that are not listed here.",
        "",
    ]
    current_chars = sum(len(line) + 1 for line in lines)
    supplied_sections: list[SourceSection] = []
    supplied_ids: set[str] = set()
    truncated = False

    for section in sections:
        if not section.items:
            continue
        section_lines = [f"{section.label}:"]
        section_chars = _lines_char_count(section_lines)
        section_added = False
        supplied_items: list[SourceItem] = []

        for item in section.items:
            item_lines = [
                f"[{item.source_id}] {item.resource_type}/{item.resource_id}: {item.summary}"
            ]
            item_lines.extend(
                f"  {key}: {_format_evidence_value(value)}"
                for key, value in item.evidence.items()
            )
            item_chars = _lines_char_count(item_lines)
            needed_chars = item_chars
            if not section_added:
                needed_chars += section_chars
            if current_chars + needed_chars > max_chars:
                truncated = True
                break
            if not section_added:
                lines.extend(section_lines)
                current_chars += section_chars
                section_added = True
            lines.extend(item_lines)
            current_chars += item_chars
            supplied_items.append(item)
            supplied_ids.add(item.source_id)

        if section_added:
            supplied_sections.append(SourceSection(label=section.label, items=supplied_items))

        if truncated:
            break

        if section_added:
            blank_chars = 1
            if current_chars + blank_chars <= max_chars:
                lines.append("")
                current_chars += blank_chars

    if truncated:
        notice_chars = len(_TRUNCATION_NOTICE) + 1
        if current_chars + notice_chars <= max_chars:
            lines.append(_TRUNCATION_NOTICE)

    return SourceContextResult(
        text="\n".join(lines).strip(),
        sections=supplied_sections,
        supplied_source_ids=supplied_ids,
        truncated=truncated,
    )


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


def _item(
    *,
    counter: _SourceIdCounter,
    resource: dict[str, Any],
    summary: str,
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
        evidence=_flatten_resource(resource),
        raw_resource=dict(resource),
    )


def _flatten_resource(resource: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    _flatten_value(resource, "", result)
    return result


def _flatten_value(value: Any, path: str, result: dict[str, Any]) -> None:
    if _is_empty_evidence_value(value):
        return

    if isinstance(value, dict):
        for key, child in value.items():
            if key in _SKIPPED_EVIDENCE_KEYS:
                continue
            child_path = f"{path}.{key}" if path else str(key)
            _flatten_value(child, child_path, result)
        return

    if isinstance(value, list):
        for index, child in enumerate(value):
            _flatten_value(child, f"{path}[{index}]", result)
        return

    if path:
        result[path] = value


def _is_empty_evidence_value(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _format_evidence_value(value: Any) -> str:
    text = str(value)
    if len(text) <= _SOURCE_CONTEXT_MAX_VALUE_CHARS:
        return text
    return text[: _SOURCE_CONTEXT_MAX_VALUE_CHARS - 3] + "..."


def _lines_char_count(lines: list[str]) -> int:
    return sum(len(line) + 1 for line in lines)


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
    )


def _condition_item(resource: dict[str, Any], counter: _SourceIdCounter) -> SourceItem:
    code = _text(resource.get("code")) or "Unknown condition"
    onset = resource.get("onsetDateTime") or resource.get("onsetPeriod", {}).get("start", "")
    recorded = resource.get("recordedDate", "")
    summary_parts = [code]
    if onset:
        summary_parts.append(f"onset: {_date_only(onset) or onset}")
    if recorded:
        summary_parts.append(f"recorded: {_date_only(recorded) or recorded}")
    return _item(
        counter=counter,
        resource=resource,
        summary="; ".join(summary_parts),
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
    )


def _allergy_item(resource: dict[str, Any], counter: _SourceIdCounter) -> SourceItem:
    substance = _text(resource.get("code")) or "Unknown allergy"
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
