"""
FHIRLoader — one-time startup script that POSTs the local synthetic FHIR
bundle to the IRIS FHIR server.
"""

from src.models import PatientResources  # noqa: F401
from src.exceptions import FHIRLoaderError  # noqa: F401
