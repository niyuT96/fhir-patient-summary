import importlib
from datetime import date
from unittest.mock import MagicMock, patch

from src.models import SourceSection


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


def test_sources_html_keeps_hidden_items_expandable(monkeypatch):
    app = _load_app_module(monkeypatch)
    html = app._build_sources_html(
        [
            SourceSection(
                label="Active Conditions (4)",
                items=["A", "B", "C"],
                hidden_items=["D"],
            )
        ],
        "local_fallback",
    )

    assert "Show 1 more" in html
    assert "<li>D</li>" in html
    assert "...and" not in html
