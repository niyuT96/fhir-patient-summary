"""
Container startup entry point.

When the IRIS FHIR endpoint is reachable, this script loads the bundled sample
FHIR transaction once before starting the Gradio UI. If IRIS is unavailable or
the load fails, the UI still starts and uses the local JSON fallback.
"""

from __future__ import annotations

import logging
import os

from src.fhir_client import FHIRClient
from src.fhir_loader import load_bundle

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def maybe_load_sample_bundle() -> None:
    """Load sample FHIR data into IRIS when the endpoint is reachable."""
    should_load = os.environ.get("LOAD_SAMPLE_BUNDLE", "true").strip().lower()
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


if __name__ == "__main__":
    maybe_load_sample_bundle()

    from src.app import demo

    demo.launch(server_name="0.0.0.0", server_port=7860)
