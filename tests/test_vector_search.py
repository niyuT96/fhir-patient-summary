from unittest.mock import MagicMock

from src.models import SourceItem, SourceSection
from src.tools.vector_search import (
    retrieve_patient_scoped_source_sections,
)


def _source_item(source_id: str, resource_type: str, resource_id: str, summary: str) -> SourceItem:
    return SourceItem(
        source_id=source_id,
        label=f"{source_id} | {resource_type}/{resource_id} | {summary}",
        resource_type=resource_type,
        resource_id=resource_id,
        summary=summary,
        evidence={"summary": summary},
        raw_resource={"resourceType": resource_type, "id": resource_id},
    )


def test_local_vector_search_returns_complete_source_items_from_current_sections():
    patient = _source_item("S1", "Patient", "p1", "Jane Doe")
    hypertension = _source_item("S2", "Condition", "c1", "Hypertension")
    diabetes = _source_item("S3", "Condition", "c2", "Diabetes")
    sections = [
        SourceSection(label="Patient (1)", items=[patient]),
        SourceSection(label="Conditions (2)", items=[hypertension, diabetes]),
    ]

    result = retrieve_patient_scoped_source_sections(
        sections,
        query="hypertension",
        max_items=2,
        enabled=True,
        backend="local",
    )

    assert result.warning == ""
    assert result.retrieved_source_ids == {"S1", "S2"}
    assert result.sections[0].items[0] is patient
    assert result.sections[1].items[0] is hypertension
    assert result.sections[1].label == "Conditions (1 retrieved of 2)"
    assert all(item is not diabetes for section in result.sections for item in section.items)


def test_vector_search_disabled_falls_back_to_all_patient_scoped_items():
    sections = [
        SourceSection(
            label="Conditions (2)",
            items=[
                _source_item("S1", "Condition", "c1", "Hypertension"),
                _source_item("S2", "Condition", "c2", "Diabetes"),
            ],
        )
    ]

    result = retrieve_patient_scoped_source_sections(
        sections,
        query="hypertension",
        max_items=1,
        enabled=False,
    )

    assert result.fallback_used is True
    assert "using all patient-scoped source items" in result.warning
    assert result.retrieved_source_ids == {"S1", "S2"}
    assert [item.source_id for item in result.sections[0].items] == ["S1", "S2"]


def test_openai_embedding_failure_falls_back_to_local_vector_search():
    llm = MagicMock()
    llm.embeddings.create.side_effect = RuntimeError("embedding outage")
    sections = [
        SourceSection(
            label="Conditions (2)",
            items=[
                _source_item("S1", "Condition", "c1", "Hypertension"),
                _source_item("S2", "Condition", "c2", "Diabetes"),
            ],
        )
    ]

    result = retrieve_patient_scoped_source_sections(
        sections,
        query="diabetes",
        llm_client=llm,
        max_items=1,
        enabled=True,
        backend="openai",
    )

    assert result.fallback_used is True
    assert result.backend == "local"
    assert "OpenAI embedding search failed" in result.warning
    assert result.retrieved_source_ids == {"S2"}


def test_unavailable_backend_falls_back_to_all_patient_scoped_items():
    sections = [
        SourceSection(
            label="Conditions (1)",
            items=[_source_item("S1", "Condition", "c1", "Hypertension")],
        )
    ]

    result = retrieve_patient_scoped_source_sections(
        sections,
        query="hypertension",
        max_items=1,
        enabled=True,
        backend="iris",
    )

    assert result.fallback_used is True
    assert result.backend == "iris"
    assert "not available" in result.warning
    assert result.retrieved_source_ids == {"S1"}
