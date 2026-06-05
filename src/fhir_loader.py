"""
FHIRLoader — one-time startup script that POSTs the local synthetic FHIR
bundle to the IRIS FHIR R4 server.

Requirements: 9.1–9.5
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict

import requests
from requests.exceptions import ConnectionError, Timeout

from src.exceptions import FHIRLoaderError

logger = logging.getLogger(__name__)

_IDEMPOTENCY_FILE = "data/loaded-patient-ids.json"


def load_bundle(
    fhir_base_url: str,
    bundle_path: str,
    username: str,
    password: str,
) -> list[str]:
    """POST the local FHIR bundle to the IRIS FHIR server.

    Idempotency guard: if ``data/loaded-patient-ids.json`` already exists
    and is non-empty, return the previously assigned patient IDs without
    re-POSTing (Requirement 9.4).

    Args:
        fhir_base_url: Base URL of the FHIR R4 server (e.g.
                       ``"http://localhost:52773/fhir/r4"``).
        bundle_path:   Path to the local ``sample-patient-bundle.json`` file.
        username:      HTTP Basic auth username.
        password:      HTTP Basic auth password.

    Returns:
        List of assigned patient ID strings extracted from the server response.

    Raises:
        FHIRLoaderError: On non-200 HTTP response, connection error, or timeout.
    """
    # --- Idempotency guard (Req 9.4) ---
    if os.path.exists(_IDEMPOTENCY_FILE):
        try:
            with open(_IDEMPOTENCY_FILE, "r", encoding="utf-8") as fh:
                existing_ids: list[str] = json.load(fh)
            if existing_ids:
                logger.info(
                    "Bundle already loaded (%d patient IDs). Skipping POST.",
                    len(existing_ids),
                )
                return existing_ids
        except (json.JSONDecodeError, OSError):
            pass  # fall through and re-load if file is corrupt / empty

    # --- Read bundle file ---
    with open(bundle_path, "r", encoding="utf-8") as fh:
        bundle_data = json.load(fh)

    # --- POST transaction bundle (Req 9.1) ---
    url = fhir_base_url.rstrip("/")
    try:
        response = requests.post(
            url,
            json=bundle_data,
            auth=(username, password),
            headers={"Content-Type": "application/fhir+json"},
            timeout=30,  # Req 9.5
        )
    except Timeout as exc:
        raise FHIRLoaderError(
            f"POST to {url} timed out after 30 seconds: {exc}"
        ) from exc
    except ConnectionError as exc:
        raise FHIRLoaderError(
            f"Connection error while POSTing bundle to {url}: {exc}"
        ) from exc

    # --- Handle non-200 (Req 9.3) ---
    if response.status_code != 200:
        raise FHIRLoaderError(
            f"FHIR server returned HTTP {response.status_code} when loading bundle. "
            f"Body: {response.text[:500]}"
        )

    # --- Parse response and extract assigned IDs (Req 9.2) ---
    response_bundle: dict = response.json()
    patient_ids: list[str] = []
    resource_counts: dict[str, int] = defaultdict(int)

    for entry in response_bundle.get("entry", []):
        location: str = entry.get("response", {}).get("location", "")
        if location:
            # location looks like "Patient/12345/_history/1"
            parts = location.split("/")
            if len(parts) >= 2:
                resource_type = parts[0]
                resource_id = parts[1]
                resource_counts[resource_type] += 1
                if resource_type == "Patient":
                    patient_ids.append(resource_id)

    # Log success per resource type (Req 9.2)
    for rtype, count in sorted(resource_counts.items()):
        logger.info("Loaded %d %s resource(s).", count, rtype)

    # --- Persist patient IDs (Req 9.2) ---
    os.makedirs(os.path.dirname(_IDEMPOTENCY_FILE), exist_ok=True)
    with open(_IDEMPOTENCY_FILE, "w", encoding="utf-8") as fh:
        json.dump(patient_ids, fh)

    logger.info("Bundle loaded successfully. Patient IDs: %s", patient_ids)
    return patient_ids
