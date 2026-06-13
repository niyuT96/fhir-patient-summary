"""
FHIRClient handles HTTP communication with the IRIS FHIR R4 endpoint.

Handles authentication, resource retrieval, fallback to local bundle,
and patient listing.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from urllib.parse import urljoin

import requests
from requests.exceptions import ConnectionError, Timeout

from src.exceptions import FHIRClientError, FHIRUnavailableError  # noqa: F401
from src.models import PatientResources  # noqa: F401

logger = logging.getLogger(__name__)

# Resource types allowed by the FHIR R4 interface
_ALLOWED_RESOURCE_TYPES = {
    "Patient",
    "Condition",
    "MedicationRequest",
    "AllergyIntolerance",
    "Observation",
    "Encounter",
    "CarePlan",
}

# Type-specific default query parameters
_RESOURCE_DEFAULTS: dict[str, dict[str, str]] = {
    "Condition": {"_count": "100"},
    "MedicationRequest": {"_sort": "-authoredon", "_count": "150"},
    "Observation": {"_sort": "-date", "_count": "150"},
    "Encounter": {"_sort": "-date", "_count": "75"},
    "CarePlan": {"_count": "50"},
}

_FHIR_JSON_HEADERS = {"Accept": "application/fhir+json"}
DEFAULT_FHIR_MAX_PAGES = 10


class FHIRClient:
    """HTTP client for the IRIS FHIR R4 endpoint with local bundle fallback."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        fallback_path: str = "data/sample-patient-bundle.json",
    ) -> None:
        """
        Initialise the client.

        Args:
            base_url:      Root URL of the FHIR R4 server, e.g.
                           ``"http://localhost:52773/fhir/r4"``.
            username:      HTTP Basic auth username.
            password:      HTTP Basic auth password.
            fallback_path: Path to a local FHIR bundle JSON file or a directory
                           containing FHIR bundle JSON files used when the
                           server is unavailable.
        """
        # Normalise: strip any trailing slash so URL concatenation is uniform
        self._base_url = base_url.rstrip("/")
        self._auth = (username, password)
        self._fallback_path = fallback_path
        self.pagination_warnings: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return ``True`` when the FHIR server responds HTTP 200 to a
        capability-statement probe; ``False`` on any other outcome."""
        try:
            url = f"{self._base_url}/metadata"
            response = requests.get(
                url,
                auth=self._auth,
                headers=_FHIR_JSON_HEADERS,
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_resource(
        self,
        resource_type: str,
        patient_id: str,
        params: dict[str, str] | None = None,
    ) -> list[dict]:
        """Fetch FHIR resources of *resource_type* for the given *patient_id*
        from the live server.

        Args:
            resource_type: One of the seven allowed FHIR resource types.
            patient_id:    The patient's FHIR ID.  Pass an empty string when
                           calling without a patient filter (e.g. listing all
                           patients).
            params:        Extra query parameters; override any type-specific
                           defaults when keys collide.

        Returns:
            A list of resource dicts whose ``resourceType`` matches
            *resource_type*.

        Raises:
            ValueError:            For disallowed *resource_type* values.
            FHIRClientError:       On HTTP 4xx / 5xx.
            FHIRUnavailableError:  On connection timeout or refusal.
        """
        if resource_type not in _ALLOWED_RESOURCE_TYPES:
            raise ValueError(
                f"Invalid resource type '{resource_type}'. "
                f"Allowed types: {sorted(_ALLOWED_RESOURCE_TYPES)}"
            )
        self._clear_pagination_warnings_for_resource(resource_type)

        # Build query parameters: start from type-specific defaults, then
        # overlay caller-supplied params (caller takes precedence).
        query: dict[str, str] = dict(_RESOURCE_DEFAULTS.get(resource_type, {}))
        if params:
            query.update(params)

        # Determine URL and patient filter param
        if resource_type == "Patient":
            url = f"{self._base_url}/Patient"
            if patient_id:
                query["_id"] = patient_id
        else:
            url = f"{self._base_url}/{resource_type}"
            if patient_id:
                query["patient"] = patient_id

        max_pages = _env_int("FHIR_MAX_PAGES", DEFAULT_FHIR_MAX_PAGES, minimum=1)
        bundles: list[dict] = []
        next_url: str | None = url
        next_params: dict[str, str] | None = query
        pages_fetched = 0

        while next_url and pages_fetched < max_pages:
            bundle = self._request_bundle_page(next_url, next_params)
            bundles.append(bundle)
            pages_fetched += 1
            next_link = _find_next_link(bundle)
            next_url = _normalize_next_url(self._base_url, next_link) if next_link else None
            next_params = None

        if next_url:
            warning = (
                f"FHIR pagination stopped after {max_pages} pages for {resource_type}; "
                "additional matching resources may not be included."
            )
            logger.warning(
                warning,
            )
            self.pagination_warnings.append(warning)

        results: list[dict] = []
        for bundle in bundles:
            results.extend(self._parse_bundle_entries(bundle, resource_type))
        return results

    def list_patients(self) -> list[dict]:
        """Return a list of Patient resources.

        Uses the live server when available; falls back to the local bundle
        otherwise.
        """
        if self.is_available():
            # get_resource with empty patient_id fetches all patients
            return self.get_resource("Patient", "")
        # Server unavailable - parse local fallback bundle
        all_resources = self._load_fallback_bundle()
        return [r for r in all_resources if r.get("resourceType") == "Patient"]

    def clear_pagination_warnings(self) -> None:
        """Clear pagination warnings accumulated by live FHIR requests."""
        self.pagination_warnings.clear()

    def _clear_pagination_warnings_for_resource(self, resource_type: str) -> None:
        """Remove stale pagination warnings for a new request of the same type."""
        prefix = "FHIR pagination stopped after "
        suffix = f" pages for {resource_type};"
        self.pagination_warnings = [
            warning
            for warning in self.pagination_warnings
            if not (warning.startswith(prefix) and suffix in warning)
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_bundle_entries(
        self, bundle: dict, resource_type: str
    ) -> list[dict]:
        """Extract resources from a FHIR Bundle dict.

        Iterates ``entry[]``, reads ``entry[].resource``, and returns only
        those whose ``resourceType`` matches *resource_type*.  Entries that
        lack a ``resource`` key are silently skipped.

        Args:
            bundle:        Parsed FHIR Bundle JSON as a dict.
            resource_type: The resource type to keep.

        Returns:
            Filtered list of resource dicts.
        """
        results: list[dict] = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource")
            if resource is None:
                continue
            if resource.get("resourceType") == resource_type:
                results.append(resource)
        return results

    def _request_bundle_page(
        self,
        url: str,
        params: dict[str, str] | None,
    ) -> dict:
        """Request one FHIR Bundle page and return parsed JSON."""
        try:
            response = requests.get(
                url,
                auth=self._auth,
                headers=_FHIR_JSON_HEADERS,
                params=params,
                timeout=10,
            )
        except (Timeout, ConnectionError) as exc:
            raise FHIRUnavailableError(
                f"Could not reach FHIR server at {url}: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise FHIRClientError(response.status_code, response.text)

        return response.json()

    def _load_fallback_bundle(self) -> list[dict]:
        """Read and parse local fallback bundle file(s).

        Returns a flat list of all resource dicts found in the bundle
        (entries without a ``resource`` key are silently skipped).

        Raises:
            RuntimeError: If the path is missing or contains invalid JSON,
                          including the path and the reason.
        """
        path = Path(self._fallback_path)
        if path.is_dir():
            json_files = sorted(path.glob("*.json"))
            if not json_files:
                raise RuntimeError(
                    f"No JSON fallback bundles found in directory '{self._fallback_path}'"
                )

            results: list[dict] = []
            for json_file in json_files:
                results.extend(self._load_fallback_bundle_file(json_file, str(json_file)))
            return results

        return self._load_fallback_bundle_file(path, self._fallback_path)

    @staticmethod
    def _load_fallback_bundle_file(path: Path, display_path: str) -> list[dict]:
        """Read one FHIR JSON bundle and return its entry resources."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as exc:
            raise RuntimeError(
                f"Cannot read fallback bundle at '{display_path}': {exc}"
            ) from exc

        try:
            bundle = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid JSON in fallback bundle at '{display_path}': {exc}"
            ) from exc

        results: list[dict] = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource")
            if resource is not None:
                results.append(resource)
        return results


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    if minimum is not None and value < minimum:
        return default
    return value


def _find_next_link(bundle: dict) -> str | None:
    for link in bundle.get("link", []):
        if link.get("relation") == "next" and link.get("url"):
            return str(link["url"])
    return None


def _normalize_next_url(base_url: str, next_url: str) -> str:
    if next_url.startswith(("http://", "https://")):
        return next_url
    return urljoin(base_url.rstrip("/") + "/", next_url.lstrip("/"))
