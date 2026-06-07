"""
Container startup entry point.

The web app is designed to connect to a user-provided IRIS FHIR endpoint.
By default, startup does not write sample data into that endpoint. If
LOAD_SAMPLE_BUNDLE=true, this script attempts to load local FHIR sample bundles
before starting the Gradio UI. If IRIS is unavailable or loading fails, the UI
still starts and uses the local JSON fallback.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from src.fhir_client import FHIRClient
from src.fhir_loader import load_bundle

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def maybe_load_sample_bundle() -> None:
    """Load sample FHIR data into IRIS when the endpoint is reachable."""
    should_load = os.environ.get("LOAD_SAMPLE_BUNDLE", "false").strip().lower()
    if should_load in {"0", "false", "no"}:
        logger.info("LOAD_SAMPLE_BUNDLE is disabled; skipping FHIR bundle load.")
        return

    base_url = os.environ.get("IRIS_BASE_URL", "http://iris:52773/fhir/r4")
    username = os.environ.get("IRIS_USERNAME", "superuser")
    password = os.environ.get("IRIS_PASSWORD", "SYS")
    fallback_path = os.environ.get("FHIR_FALLBACK_PATH", "data/sample-patient-bundle.json")

    client = FHIRClient(
        base_url=base_url,
        username=username,
        password=password,
        fallback_path=fallback_path,
    )
    if not client.is_available():
        logger.warning("FHIR server is unavailable; using local fallback data.")
        return

    try:
        patient_ids = load_bundle(
            fhir_base_url=base_url,
            bundle_path=fallback_path,
            username=username,
            password=password,
        )
        logger.info("FHIR sample bundle is ready. Patient IDs: %s", patient_ids)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FHIR sample bundle load failed; using local fallback data: %s", exc)


def run_app() -> None:
    """Start the Gradio web UI."""
    maybe_load_sample_bundle()

    from src.app import demo

    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=server_port)


if __name__ == "__main__":
    run_app()
