# Requirements Document

## Introduction

The Smart Patient Summary Generator is a clinician-facing AI agent application built for the InterSystems Programming Contest: AI Agents for FHIR. The system retrieves FHIR R4 patient data from an InterSystems IRIS for Health server, synthesizes it into a concise role-specific summary using an OpenAI LLM, and presents the result through a Gradio web UI. When the IRIS FHIR server is unreachable, the system falls back to a local synthetic FHIR bundle so that the demo workflow remains fully functional without a live server. The application targets two clinician roles — ED Doctor and Care Manager — and structures every summary into three sections: Current Issues, Recent Changes, and Risks and Follow-up. All patient data used is synthetic (Synthea-generated); no real patient data is permitted.

---

## Glossary

- **FHIRClient**: The Python component (`fhir_client.py`) responsible for all HTTP communication with the IRIS FHIR R4 endpoint and for loading the local fallback bundle.
- **PatientContextExtractor**: The Python component (`context_extractor.py`) that converts raw FHIR resource lists into a compact, token-bounded plain-text string for LLM consumption.
- **SummaryAgent**: The Python orchestration component (`agent.py`) that coordinates FHIR retrieval, context extraction, and LLM invocation to produce a `SummaryResult`.
- **GradioUI**: The web UI component (`app.py`) that allows clinicians to select a patient and role, trigger summary generation, and view the result.
- **FHIRLoader**: The one-time startup script (`fhir_loader.py`) that POSTs the local synthetic FHIR bundle to the IRIS FHIR server.
- **IRIS_FHIR_Server**: The InterSystems IRIS for Health Community Edition container hosting the FHIR R4 endpoint at `/fhir/r4`.
- **Local_Bundle**: The synthetic FHIR R4 transaction bundle stored at `data/sample-patient-bundle.json`, used when the IRIS_FHIR_Server is unreachable.
- **PatientResources**: A runtime dataclass holding the seven FHIR resource collections retrieved for a single patient: Patient, Condition, MedicationRequest, AllergyIntolerance, Observation, Encounter, and CarePlan.
- **SummaryResult**: A runtime dataclass holding the three summary sections, metadata (patient name, role, data source, timestamp), and an optional error field.
- **Role**: One of two clinician roles supported by the system: `"ED Doctor"` or `"Care Manager"`.
- **System**: The Smart Patient Summary Generator application as a whole, comprising all five components above.
- **LLM**: The OpenAI `gpt-4o-mini` model used for natural-language summary generation.
- **Token_Budget**: The maximum number of LLM tokens allowed for the patient context string, set at 3,000.

---

## Requirements

### Requirement 1: FHIR Data Retrieval

**User Story:** As a clinician, I want the system to retrieve up-to-date patient data from the IRIS FHIR server, so that the summary reflects the most current clinical record.

#### Acceptance Criteria

1. WHEN `FHIRClient.get_resource()` is called with a valid resource type and a non-empty patient ID, THE FHIRClient SHALL send an authenticated HTTP GET request to the IRIS_FHIR_Server FHIR R4 endpoint and return a list of matching FHIR resource dicts.
2. THE FHIRClient SHALL support retrieval of exactly these seven resource types: `Patient`, `Condition`, `MedicationRequest`, `AllergyIntolerance`, `Observation`, `Encounter`, and `CarePlan`. IF `get_resource()` is called with any other resource type, THE FHIRClient SHALL raise a `ValueError` immediately without making a network request.
3. IF the IRIS_FHIR_Server returns an HTTP 4xx or 5xx response, THEN THE FHIRClient SHALL raise a `FHIRClientError` containing the HTTP status code and response body, and SHALL NOT return a resource list.
4. IF the IRIS_FHIR_Server is unreachable due to a connection timeout or connection refusal after a 10-second timeout without receiving an HTTP response, THEN THE FHIRClient SHALL raise a `FHIRUnavailableError` and SHALL NOT raise `FHIRClientError`.
5. WHEN `FHIRClient.get_resource()` returns successfully, THE FHIRClient SHALL return only resource dicts whose `resourceType` field matches the requested resource type; any entry whose `resourceType` does not match SHALL be silently excluded from the returned list.
6. THE FHIRClient SHALL retrieve Observations using query parameters `_sort=-date` and `_count=20`, and shall retrieve Encounters using `_sort=-date` and `_count=5`, appended to the base resource query.
7. THE FHIRClient SHALL retrieve Conditions with the additional query parameter `clinical-status=active`, MedicationRequests with `status=active`, and CarePlans with `status=active`.

---

### Requirement 2: Fallback to Local Bundle

**User Story:** As a demo user, I want the system to work without a live FHIR server, so that the application can be demonstrated in environments where IRIS is not running.

#### Acceptance Criteria

1. WHEN `FHIRClient.is_available()` returns `False`, THE SummaryAgent SHALL load patient data from the Local_Bundle at `data/sample-patient-bundle.json` instead of querying the IRIS_FHIR_Server.
2. THE FHIRClient SHALL expose an `is_available()` method that sends an HTTP GET to `/fhir/r4/metadata` with a 5-second timeout; it SHALL return `True` if the response is HTTP 200, and SHALL return `False` for any non-200 response, connection timeout, connection refusal, or any exception raised during the probe.
3. WHEN operating in fallback mode, THE SummaryAgent SHALL set `SummaryResult.data_source` to `"local_fallback"`.
4. WHEN operating against the IRIS_FHIR_Server, THE SummaryAgent SHALL set `SummaryResult.data_source` to `"fhir_server"`.
5. THE SummaryAgent SHALL enforce that `SummaryResult.data_source` always matches the actual operating mode used for the request: `"fhir_server"` when data was retrieved from the IRIS_FHIR_Server and `"local_fallback"` when data was loaded from the Local_Bundle.
6. WHEN in fallback mode, THE FHIRClient SHALL parse the Local_Bundle JSON file, extract the `entry[].resource` array, and return resources as a list of resource dicts — the same structure produced by a live server response.
7. IF the Local_Bundle file is missing or cannot be parsed as valid JSON, THEN THE System SHALL raise a `RuntimeError` with a message identifying the file path and the parse failure reason, and SHALL NOT attempt to generate a summary with incomplete data.
8. THE Local_Bundle SHALL contain at minimum: 1 `Patient` resource, 3 `Condition` resources, 5 `MedicationRequest` resources, 1 `AllergyIntolerance` resource, 10 `Observation` resources, 3 `Encounter` resources, and 1 `CarePlan` resource.

---

### Requirement 3: Patient Context Extraction

**User Story:** As the system, I want to convert raw FHIR resources into a compact plain-text representation, so that the LLM receives a focused, token-efficient summary of the patient record.

#### Acceptance Criteria

1. WHEN `PatientContextExtractor.extract()` is called with a valid `PatientResources` value, THE PatientContextExtractor SHALL return a non-empty string containing labelled sections for patient demographics, active conditions, active medications, allergies, recent observations, recent encounters, and active CarePlan goals.
2. THE PatientContextExtractor SHALL include patient demographics from `patient.name[0].text` (or `given` + `family` concatenated if `text` is absent), `patient.birthDate`, `patient.gender`, and the MRN from `patient.identifier` where `type.coding[0].code == "MR"`.
3. THE PatientContextExtractor SHALL summarize active conditions using `condition.code.text` if present; otherwise using `condition.code.coding[0].display`; if neither is present, the condition SHALL be rendered as `"Unknown condition"`.
4. THE PatientContextExtractor SHALL list active medications using `medicationRequest.medicationCodeableConcept.text` for the drug name and `medicationRequest.dosageInstruction[0].text` for dosage; if dosage is absent the medication SHALL be listed without a dosage field.
5. THE PatientContextExtractor SHALL list allergies using `allergyIntolerance.code.text` for the substance, `allergyIntolerance.criticality` for severity, and `allergyIntolerance.reaction[0].manifestation[0].text` for the reaction; fields absent in the resource SHALL be omitted from the rendered line.
6. THE PatientContextExtractor SHALL format up to the 10 most recent Observation resources — ordered by `observation.effectiveDateTime` descending — as the string `"{name}: {value} {unit} ({date})"` where name comes from `observation.code.text`, value from `observation.valueQuantity.value`, unit from `observation.valueQuantity.unit`, and date is the ISO date portion of `observation.effectiveDateTime`.
7. THE PatientContextExtractor SHALL summarize up to the 3 most recent Encounter resources — ordered by `encounter.period.start` descending — using `encounter.type[0].text` for type, the ISO date portion of `encounter.period.start` for date, and `encounter.reasonCode[0].text` for reason; fields absent in the resource SHALL be omitted from the rendered line.
8. THE PatientContextExtractor SHALL include active CarePlan goals from `carePlan.goal` references resolved within the same `PatientResources`, and active activities from `carePlan.activity[].detail.description`; if neither is present, the section SHALL render as `"None"`.
9. THE PatientContextExtractor SHALL enforce the Token_Budget by counting tokens using the same tokenizer as the target LLM (cl100k_base for gpt-4o-mini) and, if the total exceeds 3,000 tokens, SHALL truncate sections in this priority order from lowest to highest priority: CarePlan activities, Encounters, Observations, Allergies, Medications, Conditions, then Demographics; demographics SHALL never be truncated.
10. WHEN any resource list within `PatientResources` is empty, THE PatientContextExtractor SHALL include that section's label in the output string with the value `"None"` rather than omitting the section entirely.
11. THE PatientContextExtractor SHALL NOT mutate any field of the `PatientResources` input; all list lengths and dict contents SHALL be identical before and after the call.

---

### Requirement 4: Role-Specific Summary Generation

**User Story:** As a clinician, I want the summary to be tailored to my role, so that I receive only the information relevant to my clinical workflow.

#### Acceptance Criteria

1. WHEN `SummaryAgent.generate_summary()` is called with `role="ED Doctor"`, THE SummaryAgent SHALL use the ED Doctor system prompt that instructs the LLM to focus on active diagnoses, current medications and allergies, recent labs and vitals, and acute concerns, and to omit care management goals and long-term follow-up plans.
2. WHEN `SummaryAgent.generate_summary()` is called with `role="Care Manager"`, THE SummaryAgent SHALL use the Care Manager system prompt that instructs the LLM to focus on chronic conditions, medication adherence, care plan goals, upcoming follow-up needs, and social or functional risks.
3. THE SummaryAgent SHALL invoke the LLM using model `gpt-4o-mini` with `temperature=0.3` and `max_tokens=800` for every call, regardless of role.
4. THE SummaryAgent SHALL pass the message payload to the LLM as an array containing exactly two messages: a `system` message with the role-specific prompt and a `user` message with the patient context string produced by `PatientContextExtractor.extract()`.
5. WHEN the LLM returns a response, THE SummaryAgent SHALL pass the full response text to `parse_sections()` and populate the three `SummaryResult` section fields from the returned dict.
6. IF `role` is a value other than `"ED Doctor"` or `"Care Manager"`, THEN THE SummaryAgent SHALL return a `SummaryResult` with `error` set to `"Unsupported role: {role}"` and all three section fields set to empty strings, without calling the LLM.

---

### Requirement 5: Section Parsing

**User Story:** As the system, I want to reliably parse structured LLM output into discrete sections, so that the UI can render each section independently.

#### Acceptance Criteria

1. WHEN `parse_sections()` is called with any string input, THE System SHALL return a dict containing exactly the three keys `current_issues`, `recent_changes`, and `risks_and_followup`, and SHALL NOT raise an exception.
2. WHEN the input string contains the headers `## Current Issues`, `## Recent Changes`, and `## Risks and Follow-up` matched case-sensitively and on their own lines, THE System SHALL populate the corresponding dict values with the text following each header up to the next recognized header or end of string; any key whose header is absent SHALL default to an empty string.
3. WHEN a recognized header exists but has no content following it before the next recognized header or end of string, THE System SHALL set that section's value to an empty string.
4. WHEN the input string is empty, THE System SHALL return a dict with all three keys set to empty strings.
5. WHEN the LLM omits one or more expected section headers, THE System SHALL set the missing keys to empty strings without raising an exception.
6. WHEN the input string contains extra whitespace around section content, THE System SHALL strip leading and trailing whitespace from each section value.
7. IF the input string is non-empty and contains no recognized `## Header` markers, THEN THE System SHALL set `current_issues` and `recent_changes` to empty strings and SHALL set `risks_and_followup` to the full raw input string with leading and trailing whitespace stripped.

---

### Requirement 6: Summary Result Contract

**User Story:** As the system, I want `SummaryAgent.generate_summary()` to always return a valid `SummaryResult`, so that the UI never receives an unhandled exception from the agent layer.

#### Acceptance Criteria

1. WHEN `SummaryAgent.generate_summary()` is called with a non-empty `patient_id` string and a `role` value of `"ED Doctor"` or `"Care Manager"`, THE SummaryAgent SHALL always return a `SummaryResult` and SHALL NOT raise an unhandled exception under any internal failure condition, including FHIR retrieval errors, LLM errors, and parsing errors.
2. WHEN the LLM call succeeds, THE SummaryAgent SHALL return a `SummaryResult` where `error` is `None` and all three fields `current_issues`, `recent_changes`, and `risks_and_followup` are non-empty strings.
3. WHEN the LLM call fails due to an API error, rate limit, or timeout, THE SummaryAgent SHALL return a `SummaryResult` where `error` is a non-empty string indicating the failure cause, and all three section fields (`current_issues`, `recent_changes`, `risks_and_followup`) are empty strings.
4. THE SummaryAgent SHALL always set `SummaryResult.generated_at` to the current UTC time in ISO 8601 format (e.g., `"2026-06-05T14:30:00Z"`), regardless of whether an error occurred.
5. THE SummaryAgent SHALL set `SummaryResult.patient_id` to the exact string value passed as `patient_id` in the input.
6. THE SummaryAgent SHALL set `SummaryResult.role` to the exact string value passed as `role` in the input.
7. IF `role` is a value other than `"ED Doctor"` or `"Care Manager"`, THEN THE SummaryAgent SHALL return a `SummaryResult` with `error` set to `"Unsupported role: {role}"` and all three section fields set to empty strings.

---

### Requirement 7: FHIR Resource Fetch Graceful Degradation

**User Story:** As the system, I want FHIR resource retrieval to tolerate partial failures, so that a summary can still be generated when some resource types are unavailable.

#### Acceptance Criteria

1. WHEN the IRIS_FHIR_Server returns a `FHIRClientError` or `FHIRUnavailableError` for one or more resource types other than `Patient`, THE SummaryAgent SHALL log a warning containing the resource type and the error message, set the corresponding field in `PatientResources` to an empty list, and continue fetching the remaining resource types.
2. IF the IRIS_FHIR_Server returns zero `Patient` resources for the requested patient ID, THEN THE SummaryAgent SHALL return a `SummaryResult` with `error` set to `"Patient {id} not found"` — where `{id}` is the literal patient ID string — all three section fields set to empty strings, and SHALL NOT call the LLM.
3. IF the IRIS_FHIR_Server raises a `FHIRClientError` or `FHIRUnavailableError` on the `Patient` resource fetch itself, THEN THE SummaryAgent SHALL return a `SummaryResult` with `error` set to `"Failed to fetch Patient {id}: {error_message}"` and SHALL NOT call the LLM.
4. THE `PatientResources` value returned by the fetch step SHALL have all non-Patient fields set to empty lists for any resource type whose fetch raised an exception; fields for resource types fetched successfully before a failure SHALL retain their fetched values.
5. WHEN iterating over the seven resource types during fetch, THE SummaryAgent SHALL preserve all previously fetched resource entries in `PatientResources` regardless of failures on subsequent resource types.

---

### Requirement 8: Gradio Web UI

**User Story:** As a clinician or demo user, I want a browser-based UI to select a patient and role and view the generated summary, so that I can use the application without writing code.

#### Acceptance Criteria

1. WHEN the GradioUI loads, THE GradioUI SHALL populate the patient dropdown by calling `FHIRClient.list_patients()` and display each patient as a selectable option showing at minimum the patient ID and, where available, the patient's name.
2. WHEN the GradioUI loads, THE GradioUI SHALL call `FHIRClient.is_available()` and display a green badge labeled `"FHIR Server"` if it returns `True`, or an amber badge labeled `"Local Fallback"` if it returns `False`.
3. THE GradioUI SHALL present a radio button group with exactly two options: `"ED Doctor"` and `"Care Manager"`, with no other options selectable.
4. WHEN the user clicks the Generate button, THE GradioUI SHALL call `SummaryAgent.generate_summary()` with the selected patient ID and role, then render the returned `SummaryResult` sections as formatted Markdown in the summary output panel.
5. IF no patient is selected when the user clicks Generate, THEN THE GradioUI SHALL display an inline error message `"Please select a patient before generating a summary"` in the status bar and SHALL NOT call `SummaryAgent.generate_summary()`.
6. IF `FHIRClient.list_patients()` returns an empty list, THEN THE GradioUI SHALL display the message `"No patients available"` in the patient dropdown and SHALL disable the Generate button.
7. WHEN any error condition occurs — including `SummaryResult.error` being non-null, network failures, or validation errors — THE GradioUI SHALL display a message of the form `"Error: {message}"` in the summary panel and SHALL NOT expose a Python stack trace or internal exception details.
8. THE GradioUI SHALL display the `data_source` value (formatted as `"Source: FHIR Server"` or `"Source: Local Fallback"`) and the `generated_at` value (formatted as a human-readable UTC datetime) in a footer below the summary output.
9. WHEN summary generation is in progress, THE GradioUI SHALL show the text `"Generating summary…"` in the status bar and SHALL disable the Generate button; WHEN generation completes or fails, THE GradioUI SHALL clear the status bar text and re-enable the Generate button.
10. WHEN the GradioUI starts in fallback mode (`FHIRClient.is_available()` returned `False`), THE GradioUI SHALL display a banner message reading `"Running in local fallback mode — summaries are generated from local sample data, not the live FHIR server"` above the patient selector.

---

### Requirement 9: FHIR Loader

**User Story:** As a developer or judge, I want the synthetic FHIR bundle to be automatically loaded into the IRIS FHIR server on first startup, so that the live-server workflow works without manual data entry.

#### Acceptance Criteria

1. WHEN `FHIRLoader.load_bundle()` is called, THE FHIRLoader SHALL read the Local_Bundle from `data/sample-patient-bundle.json`, and SHALL POST it as a single FHIR R4 transaction bundle to the IRIS_FHIR_Server base URL using HTTP Basic authentication with the configured credentials.
2. WHEN the POST succeeds with HTTP 200, THE FHIRLoader SHALL extract the assigned resource IDs from the transaction response bundle's `entry[].response.location` fields, log a success line per resource type showing the count of resources created, and write the list of assigned patient IDs as a JSON array to `data/loaded-patient-ids.json`.
3. IF the POST fails with a non-200 HTTP response or a connection error, THEN THE FHIRLoader SHALL log an error message containing the HTTP status code (if available) and the response body (if available), and SHALL raise a `FHIRLoaderError`.
4. IF `FHIRLoader.load_bundle()` is called when `data/loaded-patient-ids.json` already exists and is non-empty, THEN THE FHIRLoader SHALL skip the POST and return the previously assigned patient IDs from that file without re-loading data.
5. THE FHIRLoader SHALL apply a 30-second timeout to the POST request; IF the timeout expires before a response is received, THE FHIRLoader SHALL raise a `FHIRLoaderError` with a message indicating the timeout.

---

### Requirement 10: Docker Compose Deployment

**User Story:** As a judge or developer, I want to run the entire application with a single Docker Compose command, so that setup is reproducible and requires minimal manual steps.

#### Acceptance Criteria

1. THE System SHALL be deployable using two Docker containers managed by Docker Compose: one named `iris` running `intersystems/irishealth-community:latest-em` as the IRIS_FHIR_Server, and one named `app` running the Python application.
2. THE System SHALL expose the GradioUI on host port `7860` mapped to container port `7860`.
3. THE System SHALL expose the IRIS_FHIR_Server FHIR R4 endpoint on host port `52773` mapped to container port `52773`.
4. IF `OPENAI_API_KEY` is absent or empty in the runtime environment when the `app` container starts, THEN THE System SHALL raise an `EnvironmentError` with a message that names the missing variable and halts the process before the GradioUI launches.
5. THE System SHALL read `OPENAI_API_KEY` and `IRIS_PASSWORD` exclusively from environment variables at runtime; these values SHALL NOT appear as literal strings in any committed source file, Dockerfile, docker-compose.yml, or documentation file other than `.env.example` with clearly marked placeholder values.
6. THE System SHALL include an `.env.example` file at the repository root listing, at minimum, the variables `OPENAI_API_KEY` and `IRIS_PASSWORD` with placeholder values (e.g., `OPENAI_API_KEY=your-key-here`).
7. THE docker-compose.yml SHALL define a health check for the `iris` container that probes `http://localhost:52773/fhir/r4/metadata` every 5 seconds with a 60-second start period and a maximum of 12 retries before the container is marked unhealthy.
8. WHEN the `iris` container is marked unhealthy or does not reach healthy status within the health-check window, THE `app` container SHALL start in fallback mode using the Local_Bundle rather than failing to start.

---

### Requirement 11: Data Safety and Privacy

**User Story:** As a contest judge and responsible developer, I want the application to use only synthetic patient data, so that no real patient information is stored, transmitted, or exposed.

#### Acceptance Criteria

1. THE System SHALL use only synthetic FHIR patient data generated by Synthea or an equivalent synthetic data tool; no real patient records SHALL be included in the repository, in any Docker image, or loaded into the IRIS_FHIR_Server.
2. THE System SHALL NOT include `OPENAI_API_KEY`, `IRIS_PASSWORD`, or any other secret credential as a literal value in any committed file, including source code, configuration files, Dockerfiles, and documentation.
3. THE System SHALL include `.env` in `.gitignore` so that a locally created `.env` file containing real credentials cannot be accidentally committed.

---

### Requirement 12: FHIR Data Parser and Serializer

**User Story:** As the system, I want to reliably parse FHIR Bundle JSON and extract typed resource collections, so that all downstream components operate on well-structured data.

#### Acceptance Criteria

1. WHEN the FHIRClient receives a FHIR R4 Bundle JSON response from the IRIS_FHIR_Server, THE FHIRClient SHALL iterate over the `entry` array and return a list containing the `resource` object from each entry that has a `resource` key; entries without a `resource` key SHALL be silently skipped.
2. WHEN the FHIRClient loads the Local_Bundle from disk, THE FHIRClient SHALL parse the JSON file and apply the same `entry[].resource` extraction logic as criterion 12.1, producing an identical list-of-resource-dicts structure.
3. THE FHIRClient SHALL produce resource lists from both the IRIS_FHIR_Server and the Local_Bundle that, when passed to `PatientContextExtractor.extract()`, yield a non-empty output string containing patient demographics — confirming structural equivalence between the two sources.
4. FOR ALL valid `PatientResources` values derived from either source, the patient identity fields extracted by `PatientContextExtractor.extract()` — specifically patient name, patient ID, and date of birth — SHALL match the values present in the original `patient` dict, with no data loss or transformation applied.
