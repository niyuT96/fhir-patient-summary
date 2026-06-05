"""
Custom exceptions for the Smart Patient Summary Generator.
"""


class FHIRClientError(Exception):
    """Raised when the FHIR server returns an HTTP 4xx or 5xx response.

    Attributes:
        status_code: The HTTP status code returned by the server.
        body: The response body text returned by the server.
    """

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"FHIR server returned HTTP {status_code}: {body}")


class FHIRUnavailableError(Exception):
    """Raised when the FHIR server is unreachable (connection timeout or refusal)."""


class FHIRLoaderError(Exception):
    """Raised when the FHIR bundle loader fails to POST the bundle to the server."""
