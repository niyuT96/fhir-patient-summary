# Implementation Plan: Smart Patient Summary Generator

## Overview

Implement the Smart Patient Summary Generator as a Python application packaged in Docker. The system retrieves FHIR R4 patient data from an InterSystems IRIS for Health server (with local bundle fallback), synthesizes a role-specific clinical summary via OpenAI gpt-4o-mini, and presents the result through a Gradio web UI.

The implementation follows the component order: data models and shared types → FHIRClient → PatientContextExtractor → SummaryAgent (+ section parser) → GradioUI → FHIRLoader → Docker Compose + infrastructure.

## Tasks

- [x] 1. Set up project structure, shared data models, and error types
  - Create the `src/` directory layout with `__init__.py` files for `fhir_client.py`, `context_extractor.py`, `agent.py`, `app.py`, and `fhir_loader.py`
  - Define `PatientResources` dataclass with fields: `patient: dict`, `conditions: list[dict]`, `medications: list[dict]`, `allergies: list[dict]`, `observations: list[dict]`, `encounters: list[dict]`, `care_plans: list[dict]`
  - Define `SummaryResult` dataclass with fields: `patient_name`, `patient_id`, `role`, `current_issues`, `recent_changes`, `risks_and_followup`, `data_source`, `generated_at`, `error`
  - Define custom exceptions: `FHIRClientError(Exception)` (carries HTTP status code and body), `FHIRUnavailableError(Exception)`, `FHIRLoaderError(Exception)`
  - Create `requirements.txt` pinning: `gradio>=4.0`, `openai>=1.0`, `requests>=2.31`, `python-dotenv>=1.0`, `hypothesis>=6.0`, `pytest>=8.0`, `tiktoken>=0.6`
  - Create `.env.example` with placeholder values for `OPENAI_API_KEY` and `IRIS_PASSWORD`
  - Add `.env` to `.gitignore`
  - _Requirements: 10.5, 10.6, 11.2, 11.3_

- [x] 2. Implement FHIRClient
  - [x] 2.1 Implement `FHIRClient.__init__()` and `is_available()`
    - Accept `base_url`, `username`, `password`, and `fallback_path` constructor parameters
    - Implement `is_available()`: GET `/fhir/r4/metadata` with 5-second timeout; return `True` on HTTP 200, `False` on any other outcome including exceptions
    - _Requirements: 2.2_

  - [x] 2.2 Implement `FHIRClient.get_resource()` for live server
    - Validate `resource_type` against the seven allowed types; raise `ValueError` immediately for any other value without making a network request
    - Build the FHIR R4 query URL and merge caller-supplied `params` with any type-specific defaults (Observation: `_sort=-date&_count=20`; Encounter: `_sort=-date&_count=5`; Condition: `clinical-status=active`; MedicationRequest: `status=active`; CarePlan: `status=active`)
    - Send authenticated HTTP GET with HTTP Basic auth; raise `FHIRClientError` on HTTP 4xx/5xx; raise `FHIRUnavailableError` on connection timeout (10-second timeout) or connection refusal
    - Parse the FHIR Bundle response: iterate `entry[]`, extract `entry[].resource`, and filter to only entries whose `resourceType` matches the requested type (silently skip non-matching entries)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 12.1_

  - [ ]* 2.3 Write property test for FHIRClient resource type filter (Property 1)
    - **Property 1: Resource type filter correctness** — for any valid resource type and non-empty patient ID, every element in the returned list must have `resourceType` equal to the requested type
    - **Validates: Requirements 1.1, 1.5**

  - [ ]* 2.4 Write property test for FHIRClient HTTP error raising (Property 2)
    - **Property 2: HTTP error codes raise FHIRClientError** — for any HTTP status code in 400–599, `get_resource()` must raise `FHIRClientError` and must not return a resource list
    - **Validates: Requirements 1.3**

  - [x] 2.5 Implement `FHIRClient` fallback bundle loading
    - Implement local bundle parsing: read `data/sample-patient-bundle.json`, parse JSON, extract `entry[].resource` using the same extraction logic as the live server path (silently skip entries without `resource` key)
    - Raise `RuntimeError` with path and parse failure reason if the file is missing or contains invalid JSON
    - Implement `list_patients()`: return list of Patient resources from the live server (GET `/fhir/r4/Patient`) or from the fallback bundle if the server is unavailable
    - _Requirements: 2.1, 2.6, 2.7, 12.2_

  - [ ]* 2.6 Write property test for FHIR bundle parsing equivalence (Property 13)
    - **Property 13: FHIR bundle parsing equivalence and round-trip** — for any valid FHIR R4 Bundle JSON from either source, `FHIRClient` must produce a structurally identical `list[dict]`; patient identity fields must survive a serialize-then-parse cycle without data loss
    - **Validates: Requirements 12.1, 12.2, 12.3, 12.4**

- [x] 3. Checkpoint — Ensure all FHIRClient tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement `parse_sections()`
  - [x] 4.1 Implement the `parse_sections()` function
    - Accept a single string argument; always return a dict with exactly the three keys `current_issues`, `recent_changes`, `risks_and_followup`; never raise an exception
    - Parse line-by-line: detect `## Current Issues`, `## Recent Changes`, `## Risks and Follow-up` headers matched case-sensitively on their own lines; accumulate following lines into the corresponding section value; strip leading/trailing whitespace from each section value
    - Handle the fallback case per Requirement 5.7: if the input is non-empty and contains no recognized `## Header` markers, set `current_issues` and `recent_changes` to empty strings and assign the full stripped input to `risks_and_followup`
    - Handle empty input: return all three keys as empty strings
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [ ]* 4.2 Write property test for section parsing completeness (Property 8)
    - **Property 8: Section parsing completeness** — for any string input (including empty strings, random text, and malformed output), `parse_sections()` must always return a dict with exactly the three keys and must never raise an exception
    - **Validates: Requirements 5.1, 5.3, 5.4, 5.5**

  - [ ]* 4.3 Write property test for section parsing correctness (Property 9)
    - **Property 9: Section parsing correctness** — for any string containing the three expected headers with arbitrary content between them, `parse_sections()` must extract content into the correct keys with whitespace stripped; for strings without those headers, the raw text must be preserved in `risks_and_followup`
    - **Validates: Requirements 5.2, 5.6, 5.7**

- [x] 5. Implement PatientContextExtractor
  - [x] 5.1 Implement demographics, conditions, medications, and allergies extraction
    - Extract patient demographics from `patient.name[0].text` (or `given` + `family` if `text` absent), `patient.birthDate`, `patient.gender`, and MRN from `patient.identifier` where `type.coding[0].code == "MR"`
    - Summarize conditions using `condition.code.text`, falling back to `condition.code.coding[0].display`, then `"Unknown condition"`
    - List medications with `medicationCodeableConcept.text` for drug name and `dosageInstruction[0].text` for dosage (omit dosage field if absent)
    - List allergies with `code.text` for substance, `criticality` for severity, and `reaction[0].manifestation[0].text` for reaction (omit absent fields)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 5.2 Implement observations, encounters, CarePlan extraction, and token-budget enforcement
    - Format up to the 10 most recent Observation resources ordered by `effectiveDateTime` descending as `"{name}: {value} {unit} ({date})"`
    - Summarize up to the 3 most recent Encounter resources ordered by `period.start` descending using type, date, and reason (omit absent fields)
    - Include active CarePlan goals (resolved within the same `PatientResources`) and activities from `activity[].detail.description`; render as `"None"` if neither is present
    - Enforce 3,000-token budget using `tiktoken` cl100k_base tokenizer; if exceeded, truncate sections in priority order (lowest first): CarePlan activities → Encounters → Observations → Allergies → Medications → Conditions; never truncate Demographics
    - Render empty resource lists as `"None"` for that section rather than omitting the section label
    - Ensure `PatientResources` is not mutated: operate on copies of lists, never modify input dicts
    - _Requirements: 3.6, 3.7, 3.8, 3.9, 3.10, 3.11_

  - [ ]* 5.3 Write property test for context extraction completeness (Property 4)
    - **Property 4: Context extraction completeness** — for any `PatientResources` with a non-empty `patient` dict, `extract()` must return a non-empty string containing demographics, and each non-empty resource list must produce its corresponding clinical section in the output
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8**

  - [ ]* 5.4 Write property test for context extraction token budget (Property 5)
    - **Property 5: Context extraction token budget** — for any `PatientResources` value with a non-empty patient dict (including inputs with large numbers of resources), `extract()` must return a string whose token count is at most 3,000
    - **Validates: Requirements 3.9**

  - [ ]* 5.5 Write property test for empty resource list rendering (Property 6)
    - **Property 6: Empty resource list renders as "None"** — for any `PatientResources` where one or more resource lists are empty, `extract()` must include each empty section in the output with the marker `"None"` rather than omitting the section
    - **Validates: Requirements 3.10**

  - [ ]* 5.6 Write property test for PatientResources immutability (Property 7)
    - **Property 7: PatientResources immutability** — for any `PatientResources` value, calling `extract()` must leave all fields of the input object unchanged (same resource dicts, same list lengths, same field values) before and after the call
    - **Validates: Requirements 3.11**

- [x] 6. Checkpoint — Ensure all PatientContextExtractor and parse_sections tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement SummaryAgent
  - [x] 7.1 Implement role-specific prompts and `generate_summary()` core logic
    - Define ED Doctor and Care Manager system prompts exactly as specified in the design document
    - Implement `SummaryAgent.__init__()` accepting `fhir_client`, `extractor`, and `llm_client`
    - Implement the data-source determination: call `fhir_client.is_available()`; if `True` proceed with live fetch and set `data_source="fhir_server"`; if `False` load from local bundle and set `data_source="local_fallback"`
    - Return `SummaryResult` with `error="Unsupported role: {role}"` and empty section fields immediately (without calling LLM) for any role other than `"ED Doctor"` or `"Care Manager"`
    - _Requirements: 2.3, 2.4, 4.1, 4.2, 4.3, 6.6, 6.7_

  - [x] 7.2 Implement FHIR resource fetching with graceful degradation
    - Fetch all seven resource types sequentially using the query parameters defined in the FHIR Fetch Algorithm (design §Algorithmic Pseudocode)
    - On `FHIRClientError` or `FHIRUnavailableError` for non-Patient resource types: log a warning with the resource type and error message, set that field to `[]`, and continue fetching remaining types; preserve all previously fetched results
    - If the Patient fetch itself raises an error, return `SummaryResult` with `error="Failed to fetch Patient {id}: {error_message}"` and do not call the LLM
    - If the Patient resource list is empty (server returned 0 results), return `SummaryResult` with `error="Patient {id} not found"` and do not call the LLM
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 7.3 Write property test for partial FHIR fetch graceful degradation (Property 12)
    - **Property 12: Partial FHIR fetch graceful degradation** — for any non-empty subset of non-Patient resource types configured to return errors, the fetch step must return `PatientResources` where Patient is non-empty, failed types default to `[]`, and successfully fetched types retain their values
    - **Validates: Requirements 7.1, 7.3, 7.4**

  - [x] 7.4 Implement LLM invocation and result assembly
    - Call `extractor.extract(resources)` to build the patient context string
    - Invoke OpenAI Chat API with model `gpt-4o-mini`, `temperature=0.3`, `max_tokens=800`, and a two-message payload: `system` message with the role-specific prompt and `user` message with the context string
    - Pass the full response text to `parse_sections()` and populate the three `SummaryResult` section fields
    - On any OpenAI API error (API error, rate limit, timeout): set `SummaryResult.error` to a descriptive string and set all three section fields to empty strings
    - Always set `SummaryResult.generated_at` to the current UTC time in ISO 8601 format (e.g., `"2026-06-05T14:30:00Z"`) regardless of success or failure
    - Always set `SummaryResult.patient_id` and `SummaryResult.role` to the exact input values
    - Wrap the entire `generate_summary()` body so that no unhandled exception propagates to the caller
    - _Requirements: 4.3, 4.4, 4.5, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [ ]* 7.5 Write property test for data source invariant (Property 3)
    - **Property 3: Data source invariant** — for any `patient_id` and server availability state, `SummaryResult.data_source` must equal `"fhir_server"` iff `is_available()` returned `True`, and `"local_fallback"` iff it returned `False`
    - **Validates: Requirements 2.3, 2.4, 2.5**

  - [ ]* 7.6 Write property test for SummaryAgent never raises unhandled exception (Property 10)
    - **Property 10: SummaryAgent never raises an unhandled exception** — for any valid `patient_id` and role, and for any combination of mocked FHIR client or LLM client failures, `generate_summary()` must always return a `SummaryResult` and must never propagate an unhandled exception; `SummaryResult.generated_at` must always be a valid ISO 8601 UTC timestamp
    - **Validates: Requirements 6.1, 6.4**

  - [ ]* 7.7 Write property test for SummaryResult success contract (Property 11)
    - **Property 11: SummaryResult success contract** — for any valid `patient_id` and role, when the LLM call succeeds, the returned `SummaryResult` must have `error=None`, all three section fields non-empty, `patient_id` equal to the input, and `role` equal to the input
    - **Validates: Requirements 6.2, 6.5, 6.6**

- [x] 8. Checkpoint — Ensure all SummaryAgent tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement GradioUI (`app.py`)
  - [x] 9.1 Implement UI layout, patient dropdown, and role selection
    - Replace the existing `ui.py` mock with `app.py`
    - On startup: call `FHIRClient.is_available()`; populate `patient_dropdown` from `FHIRClient.list_patients()` displaying patient ID and name; show green `"FHIR Server"` badge or amber `"Local Fallback"` badge accordingly
    - If `is_available()` returned `False`, display the fallback mode banner above the patient selector
    - If `list_patients()` returns an empty list, display `"No patients available"` in the dropdown and disable the Generate button
    - Present a `gr.Radio` with exactly `["ED Doctor", "Care Manager"]`
    - _Requirements: 8.1, 8.2, 8.3, 8.6, 8.10_

  - [x] 9.2 Implement summary generation flow, status bar, and error display
    - On Generate button click: validate patient selection (display `"Please select a patient before generating a summary"` in status bar and do not call the agent if none is selected); show `"Generating summary…"` in status bar and disable the Generate button while generation is in progress; re-enable and clear the status bar on completion or failure
    - Call `SummaryAgent.generate_summary()` with the selected patient ID and role; render returned `SummaryResult` sections as formatted Markdown in the summary output panel
    - Display `data_source` as `"Source: FHIR Server"` or `"Source: Local Fallback"` and `generated_at` as a human-readable UTC datetime in a footer below the summary output
    - On any error condition (`SummaryResult.error` non-null, network failures, validation errors): display `"Error: {message}"` in the summary panel without exposing Python stack traces or internal exception details
    - _Requirements: 8.4, 8.5, 8.7, 8.8, 8.9_

- [x] 10. Implement FHIRLoader (`fhir_loader.py`)
  - [x] 10.1 Implement `load_bundle()` with idempotency guard and error handling
    - Check if `data/loaded-patient-ids.json` exists and is non-empty; if so, return the previously assigned patient IDs from that file without re-posting the bundle
    - Read `data/sample-patient-bundle.json` and POST it as a FHIR R4 transaction bundle to `{base_url}` using HTTP Basic auth with a 30-second timeout
    - On HTTP 200: extract assigned patient IDs from `entry[].response.location` fields, log a success line per resource type showing the count of resources created, and write patient IDs as a JSON array to `data/loaded-patient-ids.json`
    - On non-200 HTTP response, connection error, or timeout: raise `FHIRLoaderError` with the HTTP status code (if available) and response body (if available); for timeout specifically, include a message indicating the timeout
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 11. Create synthetic FHIR bundle and Docker Compose configuration
  - [x] 11.1 Create `data/sample-patient-bundle.json`
    - Author a valid FHIR R4 transaction bundle (`"type": "transaction"`) with `"request": {"method": "POST", "url": "{ResourceType}"}` entries for each resource
    - Include at minimum: 1 `Patient`, 3 `Condition` (active), 5 `MedicationRequest` (active), 1 `AllergyIntolerance`, 10 `Observation` (lab values + vitals with `effectiveDateTime`), 3 `Encounter` (with `period.start`), 1 `CarePlan` (active with goals and activities)
    - Ensure all resources use synthetic Synthea-style data; no real patient records
    - _Requirements: 2.8, 9.1, 11.1_

  - [x] 11.2 Write `Dockerfile` for the Python app container and update `docker-compose.yml`
    - Write a `Dockerfile` based on `python:3.11-slim` that installs dependencies from `requirements.txt`, copies source files, and runs `app.py`
    - Add an `EnvironmentError` check at app startup that raises with a descriptive message naming `OPENAI_API_KEY` if it is absent or empty, halting before Gradio launches
    - Update `docker-compose.yml`: define `iris` service (`intersystems/irishealth-community:latest-em`, port `52773:52773`, health check probing `/fhir/r4/metadata` every 5 seconds with 60-second start period and 12 retries) and `app` service (custom image, port `7860:7860`, reads `OPENAI_API_KEY` and `IRIS_PASSWORD` from environment, `depends_on: iris` with `condition: service_healthy`)
    - Ensure no literal secret values appear in `docker-compose.yml` or `Dockerfile`
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.7, 10.8_

- [x] 12. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP delivery
- Each task references specific requirements for full traceability
- Property tests use `hypothesis` with `pytest`; unit tests use `pytest` alone
- Checkpoints ensure incremental validation at each component boundary
- The `tiktoken` library (cl100k_base tokenizer) is required for token-budget enforcement in `PatientContextExtractor`
- `OPENAI_API_KEY` and `IRIS_PASSWORD` must only be passed via environment variables; use `.env` locally (never committed)
- `data/loaded-patient-ids.json` should be added to `.gitignore` to avoid accidentally committing server-assigned IDs

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "4.1"] },
    { "id": 2, "tasks": ["2.2", "5.1"] },
    { "id": 3, "tasks": ["2.3", "2.4", "2.5", "4.2", "4.3", "5.2"] },
    { "id": 4, "tasks": ["2.6", "5.3", "5.4", "5.5", "5.6", "7.1"] },
    { "id": 5, "tasks": ["7.2", "11.1"] },
    { "id": 6, "tasks": ["7.3", "7.4"] },
    { "id": 7, "tasks": ["7.5", "7.6", "7.7", "9.1", "10.1"] },
    { "id": 8, "tasks": ["9.2", "11.2"] }
  ]
}
```
