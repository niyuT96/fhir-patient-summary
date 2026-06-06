import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.exceptions import FHIRLoaderError
from src.fhir_loader import _resolve_bundle_files, load_bundle


def _transaction_response(patient_id: str) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "resourceType": "Bundle",
        "entry": [
            {"response": {"location": f"Patient/{patient_id}/_history/1"}},
            {"response": {"location": "Observation/o1/_history/1"}},
        ],
    }
    return response


def test_resolve_bundle_files_accepts_single_file():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        assert _resolve_bundle_files(tmp_path)[0].name == os.path.basename(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_resolve_bundle_files_accepts_directory_in_filename_order():
    with tempfile.TemporaryDirectory() as tmp_dir:
        open(os.path.join(tmp_dir, "b.json"), "w", encoding="utf-8").close()
        open(os.path.join(tmp_dir, "a.json"), "w", encoding="utf-8").close()

        result = _resolve_bundle_files(tmp_dir)

        assert [path.name for path in result] == ["a.json", "b.json"]


def test_resolve_bundle_files_rejects_empty_directory():
    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(FHIRLoaderError, match="No JSON bundle files"):
            _resolve_bundle_files(tmp_dir)


def test_load_bundle_posts_every_json_file_in_directory():
    with tempfile.TemporaryDirectory() as tmp_dir:
        for filename in ["a.json", "b.json"]:
            with open(os.path.join(tmp_dir, filename), "w", encoding="utf-8") as fh:
                json.dump({"resourceType": "Bundle", "type": "transaction", "entry": []}, fh)

        responses = [_transaction_response("p1"), _transaction_response("p2")]

        with patch("src.fhir_loader._IDEMPOTENCY_FILE", os.path.join(tmp_dir, "loaded.json")):
            with patch("src.fhir_loader.requests.post", side_effect=responses) as mock_post:
                patient_ids = load_bundle(
                    fhir_base_url="http://localhost:52773/fhir/r4",
                    bundle_path=tmp_dir,
                    username="superuser",
                    password="SYS",
                )

        assert patient_ids == ["p1", "p2"]
        assert mock_post.call_count == 2
        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Accept"] == "application/fhir+json"
        assert kwargs["headers"]["Content-Type"] == "application/fhir+json"
