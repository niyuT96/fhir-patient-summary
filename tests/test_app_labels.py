import importlib
from datetime import date
from unittest.mock import MagicMock, patch

from src.models import SourceItem, SourceSection


def _load_app_module(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("src.app.FHIRClient") as mock_fhir_client:
        mock_fhir_client.return_value.is_available.return_value = False
        mock_fhir_client.return_value.list_patients.return_value = []
        with patch("src.app.OpenAI", return_value=MagicMock()):
            import src.app as app
            return importlib.reload(app)


def test_patient_label_uses_name_and_age(monkeypatch):
    app = _load_app_module(monkeypatch)
    patient = {
        "id": "synthea-patient-001",
        "name": [{"text": "Eleanor M. Voss"}],
        "birthDate": "1968-04-12",
    }

    with patch("src.app.date") as mock_date:
        mock_date.today.return_value = date(2026, 6, 6)
        mock_date.fromisoformat.side_effect = date.fromisoformat
        assert app._patient_label(patient) == "Eleanor M. Voss | DOB: 1968-04-12 | Age: 58"


def test_patient_label_hides_patient_id_by_default(monkeypatch):
    app = _load_app_module(monkeypatch)
    patient = {
        "id": "secret-id",
        "name": [{"given": ["Jane"], "family": "Doe"}],
        "birthDate": "1980-01-01",
    }

    label = app._patient_label(patient)

    assert "secret-id" not in label
    assert label == f"Jane Doe | DOB: 1980-01-01 | Age: {app._patient_age(patient)}"


def test_patient_label_handles_missing_birth_date(monkeypatch):
    app = _load_app_module(monkeypatch)
    patient = {"name": [{"text": "Jane Doe"}]}

    assert app._patient_label(patient) == "Jane Doe | DOB: unknown | Age: unknown"


def test_patient_choices_append_short_id_only_for_duplicate_base_labels(monkeypatch):
    app = _load_app_module(monkeypatch)
    patients = [
        {
            "id": "alpha-123456",
            "name": [{"text": "Jane Doe"}],
            "birthDate": "1980-01-01",
        },
        {
            "id": "beta-456789",
            "name": [{"text": "Jane Doe"}],
            "birthDate": "1980-01-01",
        },
        {
            "id": "gamma-789012",
            "name": [{"text": "John Roe"}],
            "birthDate": "1980-01-01",
        },
    ]

    with patch("src.app.date") as mock_date:
        mock_date.today.return_value = date(2026, 6, 6)
        mock_date.fromisoformat.side_effect = date.fromisoformat
        choices, id_map = app._build_patient_choices_and_id_map(patients)

    assert len(choices) == len(set(choices))
    assert choices[0] == "Jane Doe | DOB: 1980-01-01 | Age: 46 | ID: alpha-12"
    assert choices[1] == "Jane Doe | DOB: 1980-01-01 | Age: 46 | ID: beta-456"
    assert choices[0] != choices[1]
    assert "ID:" not in choices[2]
    assert id_map[choices[0]] == "alpha-123456"
    assert id_map[choices[1]] == "beta-456789"


def _source_item(source_id="S1", summary="Hypertension"):
    return SourceItem(
        source_id=source_id,
        label=f"{source_id} | Condition/c1 | {summary}",
        resource_type="Condition",
        resource_id="c1",
        summary=summary,
        evidence={"resourceType": "Condition", "id": "c1", "code": summary},
        raw_resource={"resourceType": "Condition", "id": "c1", "code": {"text": summary}},
    )


def test_sources_html_expands_source_item_evidence(monkeypatch):
    app = _load_app_module(monkeypatch)
    html = app._build_sources_html(
        [
            SourceSection(
                label="Conditions (1)",
                items=[_source_item()],
            )
        ],
        "local_fallback",
    )

    assert "[S1]" in html
    assert "Evidence" in html
    assert "Hypertension" in html
    assert "Show raw FHIR resource" in html
    assert "Show 1 more" not in html


def test_sources_html_displays_source_warning(monkeypatch):
    app = _load_app_module(monkeypatch)
    html = app._build_sources_html(
        [
            SourceSection(
                label="Conditions (1)",
                items=[_source_item()],
            )
        ],
        "local_fallback",
        "Source context was truncated.",
    )

    assert "Source context was truncated." in html


def test_on_generate_uses_generator_end_as_completion_signal(monkeypatch):
    app = _load_app_module(monkeypatch)

    class FakeAgent:
        def generate_summary_stream(self, patient_id, role):
            assert patient_id == "patient-001"
            assert role == "ED Doctor"
            yield "", [
                SourceSection(
                    label="Conditions (1)",
                    items=[_source_item()],
                )
            ]
            yield "## Current Issues\n- Hypertension", [
                SourceSection(
                    label="Conditions (1)",
                    items=[_source_item()],
                )
            ]

    monkeypatch.setattr(app, "_agent", FakeAgent())
    monkeypatch.setattr(app, "_patient_id_map", {"Jane Doe": "patient-001"})
    monkeypatch.setattr(app, "_data_source_label", "local_fallback")

    outputs = list(app.on_generate("Jane Doe", "ED Doctor"))

    assert outputs[0][1] == "Preparing FHIR reference data..."
    assert outputs[0][4]["interactive"] is False
    assert outputs[1][1] == "Generating summary..."
    assert "Conditions" in outputs[1][3]
    assert outputs[2][0] == "## Current Issues\n- Hypertension"
    assert outputs[2][1] == "Generating summary..."
    assert outputs[2][4]["interactive"] is False
    assert outputs[-1][0] == "## Current Issues\n- Hypertension"
    assert outputs[-1][1] == ""
    assert "Generated:" in outputs[-1][2]
    assert outputs[-1][4]["interactive"] is True


def test_on_generate_displays_truncated_source_warning(monkeypatch):
    app = _load_app_module(monkeypatch)

    class FakeAgent:
        def generate_summary_stream(self, patient_id, role):
            yield "", [
                SourceSection(
                    label="Conditions (1)",
                    items=[_source_item()],
                )
            ], "Source context was truncated."
            yield "## Current Issues\n- Hypertension", [
                SourceSection(
                    label="Conditions (1)",
                    items=[_source_item()],
                )
            ], "Source context was truncated."

    monkeypatch.setattr(app, "_agent", FakeAgent())
    monkeypatch.setattr(app, "_patient_id_map", {"Jane Doe": "patient-001"})
    monkeypatch.setattr(app, "_data_source_label", "local_fallback")

    outputs = list(app.on_generate("Jane Doe", "ED Doctor"))

    assert "Source context was truncated." in outputs[1][3]
    assert "Source context was truncated." in outputs[-1][3]


def test_on_generate_recovers_button_if_agent_yields_nothing(monkeypatch):
    app = _load_app_module(monkeypatch)

    class EmptyAgent:
        def generate_summary_stream(self, patient_id, role):
            return
            yield

    monkeypatch.setattr(app, "_agent", EmptyAgent())
    monkeypatch.setattr(app, "_patient_id_map", {"Jane Doe": "patient-001"})

    outputs = list(app.on_generate("Jane Doe", "ED Doctor"))

    assert outputs[0][4]["interactive"] is False
    assert outputs[-1][0] == ""
    assert outputs[-1][1] == ""
    assert outputs[-1][2] == ""
    assert outputs[-1][4]["interactive"] is True


def test_on_generate_displays_error_and_restores_button(monkeypatch):
    app = _load_app_module(monkeypatch)

    class ErrorAgent:
        def generate_summary_stream(self, patient_id, role):
            yield "**Error:** Rate limit exceeded", []

    monkeypatch.setattr(app, "_agent", ErrorAgent())
    monkeypatch.setattr(app, "_patient_id_map", {"Jane Doe": "patient-001"})

    outputs = list(app.on_generate("Jane Doe", "ED Doctor"))

    assert outputs[1][0] == "**Error:** Rate limit exceeded"
    assert outputs[1][1] == ""
    assert outputs[1][4]["interactive"] is False
    assert outputs[-1][0] == "**Error:** Rate limit exceeded"
    assert outputs[-1][1] == ""
    assert outputs[-1][4]["interactive"] is True
