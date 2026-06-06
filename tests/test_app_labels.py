import importlib
from datetime import date
from unittest.mock import MagicMock, patch


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
        assert app._patient_label(patient) == "Eleanor M. Voss (58)"


def test_patient_label_hides_patient_id(monkeypatch):
    app = _load_app_module(monkeypatch)
    patient = {
        "id": "secret-id",
        "name": [{"given": ["Jane"], "family": "Doe"}],
        "birthDate": "1980-01-01",
    }

    label = app._patient_label(patient)

    assert "secret-id" not in label
    assert label.startswith("Jane Doe")


def test_patient_label_handles_missing_birth_date(monkeypatch):
    app = _load_app_module(monkeypatch)
    patient = {"name": [{"text": "Jane Doe"}]}

    assert app._patient_label(patient) == "Jane Doe (age unknown)"
