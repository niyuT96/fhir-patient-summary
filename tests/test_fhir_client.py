"""
Unit tests for FHIRClient (tasks 2.1, 2.2, 2.5).

Covers:
- is_available() returns False when the server is not reachable
- get_resource() raises ValueError for invalid resource types
- get_resource() raises FHIRClientError for HTTP 4xx / 5xx responses
- _parse_bundle_entries() silently skips entries without a 'resource' key
- list_patients() falls back to the local bundle when the server is unavailable
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.exceptions import FHIRClientError, FHIRUnavailableError
from src.fhir_client import FHIRClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_client(fallback_path: str = "data/sample-patient-bundle.json") -> FHIRClient:
    """Create a FHIRClient pointed at a local test base URL."""
    return FHIRClient(
        base_url="http://localhost:52773/fhir/r4",
        username="test_user",
        password="test_pass",
        fallback_path=fallback_path,
    )


def _bundle_with(entries: list[dict]) -> dict:
    """Build a minimal FHIR Bundle dict from a list of entry dicts."""
    return {"resourceType": "Bundle", "type": "searchset", "entry": entries}


# ---------------------------------------------------------------------------
# Task 2.1 — is_available()
# ---------------------------------------------------------------------------

class TestIsAvailable:
    def test_returns_true_on_http_200(self):
        client = make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("src.fhir_client.requests.get", return_value=mock_response):
            assert client.is_available() is True

    def test_returns_false_on_non_200(self):
        client = make_client()
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("src.fhir_client.requests.get", return_value=mock_response):
            assert client.is_available() is False

    def test_returns_false_on_connection_error(self):
        client = make_client()

        with patch(
            "src.fhir_client.requests.get",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            assert client.is_available() is False

    def test_returns_false_on_timeout(self):
        client = make_client()

        with patch(
            "src.fhir_client.requests.get",
            side_effect=requests.exceptions.Timeout("timed out"),
        ):
            assert client.is_available() is False

    def test_probes_metadata_endpoint(self):
        """is_available() must probe the /metadata endpoint."""
        client = make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("src.fhir_client.requests.get", return_value=mock_response) as mock_get:
            client.is_available()
            called_url = mock_get.call_args[0][0]
            assert called_url.endswith("/metadata")


# ---------------------------------------------------------------------------
# Task 2.2 — get_resource() validation and error handling
# ---------------------------------------------------------------------------

class TestGetResourceValidation:
    def test_raises_value_error_for_invalid_resource_type(self):
        client = make_client()
        with pytest.raises(ValueError, match="Invalid resource type"):
            client.get_resource("Invoice", "patient-1")

    def test_raises_value_error_does_not_make_network_request(self):
        """ValueError must be raised before any network call."""
        client = make_client()
        with patch("src.fhir_client.requests.get") as mock_get:
            with pytest.raises(ValueError):
                client.get_resource("BadType", "patient-1")
            mock_get.assert_not_called()

    @pytest.mark.parametrize(
        "resource_type",
        [
            "Patient",
            "Condition",
            "MedicationRequest",
            "AllergyIntolerance",
            "Observation",
            "Encounter",
            "CarePlan",
        ],
    )
    def test_valid_resource_types_are_accepted(self, resource_type: str):
        """No ValueError should be raised for the seven allowed types."""
        client = make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _bundle_with([])

        with patch("src.fhir_client.requests.get", return_value=mock_response):
            result = client.get_resource(resource_type, "patient-1")
            assert isinstance(result, list)


class TestGetResourceHTTPErrors:
    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 500, 503])
    def test_raises_fhir_client_error_on_http_error(self, status_code: int):
        client = make_client()
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.text = f"Error {status_code}"

        with patch("src.fhir_client.requests.get", return_value=mock_response):
            with pytest.raises(FHIRClientError) as exc_info:
                client.get_resource("Condition", "patient-1")
            assert exc_info.value.status_code == status_code

    def test_raises_fhir_unavailable_error_on_timeout(self):
        client = make_client()

        with patch(
            "src.fhir_client.requests.get",
            side_effect=requests.exceptions.Timeout("timeout"),
        ):
            with pytest.raises(FHIRUnavailableError):
                client.get_resource("Condition", "patient-1")

    def test_raises_fhir_unavailable_error_on_connection_refused(self):
        client = make_client()

        with patch(
            "src.fhir_client.requests.get",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            with pytest.raises(FHIRUnavailableError):
                client.get_resource("Observation", "patient-1")


class TestGetResourceBundleParsing:
    def test_returns_only_matching_resource_type(self):
        """Entries whose resourceType differs from the request must be filtered out."""
        client = make_client()
        bundle = _bundle_with([
            {"resource": {"resourceType": "Condition", "id": "c1"}},
            {"resource": {"resourceType": "Observation", "id": "o1"}},  # different type
            {"resource": {"resourceType": "Condition", "id": "c2"}},
        ])
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = bundle

        with patch("src.fhir_client.requests.get", return_value=mock_response):
            result = client.get_resource("Condition", "patient-1")

        assert len(result) == 2
        assert all(r["resourceType"] == "Condition" for r in result)

    def test_skips_entries_without_resource_key(self):
        """Entries missing the 'resource' key must be silently skipped."""
        client = make_client()
        bundle = _bundle_with([
            {"fullUrl": "urn:uuid:no-resource"},  # no 'resource' key
            {"resource": {"resourceType": "Observation", "id": "o1"}},
        ])
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = bundle

        with patch("src.fhir_client.requests.get", return_value=mock_response):
            result = client.get_resource("Observation", "patient-1")

        assert len(result) == 1
        assert result[0]["id"] == "o1"

    def test_returns_empty_list_for_empty_bundle(self):
        client = make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"resourceType": "Bundle", "entry": []}

        with patch("src.fhir_client.requests.get", return_value=mock_response):
            result = client.get_resource("Patient", "patient-1")

        assert result == []

    def test_bundle_missing_entry_key_returns_empty_list(self):
        """A bundle with no 'entry' key at all should return an empty list."""
        client = make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"resourceType": "Bundle"}

        with patch("src.fhir_client.requests.get", return_value=mock_response):
            result = client.get_resource("Patient", "patient-1")

        assert result == []


class TestGetResourceURLBuilding:
    def test_patient_resource_uses_id_param(self):
        """Patient queries must use _id= rather than patient=."""
        client = make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _bundle_with([])

        with patch("src.fhir_client.requests.get", return_value=mock_response) as mock_get:
            client.get_resource("Patient", "patient-abc")
            _, kwargs = mock_get.call_args
            assert kwargs["params"].get("_id") == "patient-abc"
            assert "patient" not in kwargs["params"]

    def test_non_patient_resource_uses_patient_param(self):
        """Non-Patient queries must pass patient= as query parameter."""
        client = make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _bundle_with([])

        with patch("src.fhir_client.requests.get", return_value=mock_response) as mock_get:
            client.get_resource("Condition", "patient-abc")
            _, kwargs = mock_get.call_args
            assert kwargs["params"].get("patient") == "patient-abc"

    def test_type_specific_defaults_are_applied(self):
        """Observation defaults (_sort=-date, _count=20) must be sent."""
        client = make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _bundle_with([])

        with patch("src.fhir_client.requests.get", return_value=mock_response) as mock_get:
            client.get_resource("Observation", "patient-abc")
            _, kwargs = mock_get.call_args
            assert kwargs["params"].get("_sort") == "-date"
            assert kwargs["params"].get("_count") == "20"

    def test_caller_params_override_defaults(self):
        """Caller-supplied params must win over type-specific defaults."""
        client = make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _bundle_with([])

        with patch("src.fhir_client.requests.get", return_value=mock_response) as mock_get:
            client.get_resource("Observation", "patient-abc", params={"_count": "5"})
            _, kwargs = mock_get.call_args
            # caller override
            assert kwargs["params"].get("_count") == "5"
            # default preserved when not overridden
            assert kwargs["params"].get("_sort") == "-date"


# ---------------------------------------------------------------------------
# Task 2.5 — Fallback bundle loading and list_patients()
# ---------------------------------------------------------------------------

class TestParseBundleEntries:
    def test_skips_entries_without_resource_key(self):
        client = make_client()
        bundle = _bundle_with([
            {"fullUrl": "no-resource"},
            {"resource": {"resourceType": "Patient", "id": "p1"}},
            {},  # empty entry
        ])
        result = client._parse_bundle_entries(bundle, "Patient")
        assert len(result) == 1
        assert result[0]["id"] == "p1"

    def test_filters_by_resource_type(self):
        client = make_client()
        bundle = _bundle_with([
            {"resource": {"resourceType": "Patient", "id": "p1"}},
            {"resource": {"resourceType": "Condition", "id": "c1"}},
        ])
        result = client._parse_bundle_entries(bundle, "Patient")
        assert len(result) == 1
        assert result[0]["resourceType"] == "Patient"

    def test_empty_bundle_returns_empty_list(self):
        client = make_client()
        result = client._parse_bundle_entries({"resourceType": "Bundle"}, "Patient")
        assert result == []


class TestLoadFallbackBundle:
    def test_raises_runtime_error_for_missing_file(self):
        client = make_client(fallback_path="/nonexistent/path/bundle.json")
        with pytest.raises(RuntimeError, match="/nonexistent/path/bundle.json"):
            client._load_fallback_bundle()

    def test_raises_runtime_error_for_invalid_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write("{ invalid json }")
            tmp_path = tmp.name

        try:
            client = make_client(fallback_path=tmp_path)
            with pytest.raises(RuntimeError, match=tmp_path):
                client._load_fallback_bundle()
        finally:
            os.unlink(tmp_path)

    def test_parses_valid_bundle_file(self):
        bundle = _bundle_with([
            {"resource": {"resourceType": "Patient", "id": "p1"}},
            {"resource": {"resourceType": "Condition", "id": "c1"}},
            {"fullUrl": "no-resource"},  # should be skipped
        ])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(bundle, tmp)
            tmp_path = tmp.name

        try:
            client = make_client(fallback_path=tmp_path)
            result = client._load_fallback_bundle()
            # entry without 'resource' key is skipped
            assert len(result) == 2
            resource_types = {r["resourceType"] for r in result}
            assert resource_types == {"Patient", "Condition"}
        finally:
            os.unlink(tmp_path)


class TestListPatients:
    def test_uses_live_server_when_available(self):
        client = make_client()
        patient = {"resourceType": "Patient", "id": "p1"}
        bundle = _bundle_with([{"resource": patient}])
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = bundle

        with patch.object(client, "is_available", return_value=True):
            with patch("src.fhir_client.requests.get", return_value=mock_response):
                result = client.list_patients()

        assert len(result) == 1
        assert result[0]["id"] == "p1"

    def test_falls_back_to_bundle_when_server_unavailable(self):
        bundle = _bundle_with([
            {"resource": {"resourceType": "Patient", "id": "p1"}},
            {"resource": {"resourceType": "Patient", "id": "p2"}},
            {"resource": {"resourceType": "Condition", "id": "c1"}},  # filtered out
        ])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(bundle, tmp)
            tmp_path = tmp.name

        try:
            client = make_client(fallback_path=tmp_path)
            with patch.object(client, "is_available", return_value=False):
                result = client.list_patients()

            assert len(result) == 2
            assert all(r["resourceType"] == "Patient" for r in result)
        finally:
            os.unlink(tmp_path)

    def test_fallback_returns_only_patient_resources(self):
        """list_patients() must filter out non-Patient resources from the bundle."""
        bundle = _bundle_with([
            {"resource": {"resourceType": "Patient", "id": "p1"}},
            {"resource": {"resourceType": "Observation", "id": "o1"}},
            {"resource": {"resourceType": "Condition", "id": "c1"}},
        ])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(bundle, tmp)
            tmp_path = tmp.name

        try:
            client = make_client(fallback_path=tmp_path)
            with patch.object(client, "is_available", return_value=False):
                result = client.list_patients()

            assert len(result) == 1
            assert result[0]["resourceType"] == "Patient"
        finally:
            os.unlink(tmp_path)

    def test_list_patients_live_sends_no_id_filter(self):
        """When listing all patients, no _id param should be sent."""
        client = make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _bundle_with([])

        with patch.object(client, "is_available", return_value=True):
            with patch("src.fhir_client.requests.get", return_value=mock_response) as mock_get:
                client.list_patients()
                _, kwargs = mock_get.call_args
                # empty patient_id means no _id filter
                assert "_id" not in kwargs["params"] or kwargs["params"]["_id"] == ""
