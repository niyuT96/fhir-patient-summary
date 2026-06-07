# Smart Patient Summary Generator

Smart Patient Summary Generator is an InterSystems IRIS for Health FHIR demo
application. It reads FHIR R4 patient data, builds a compact clinical context,
and uses an AI agent to generate role-specific summaries.

Supported roles:

- ED Doctor
- Care Manager
- Patient
- Family Caregiver

## Contest Task

This project implements the suggested task **Smart Patient Summary Generator**
for the **InterSystems Programming Contest: AI Agents for FHIR**.

Contest announcement:

https://community.intersystems.com/post/intersystems-programming-contest-ai-agents-fhir

The application generates:

- Current Issues
- Recent Changes
- Risks and Follow-up

The workflow uses these FHIR resource types:

- Patient
- Condition
- MedicationRequest
- AllergyIntolerance
- Observation
- Encounter
- CarePlan

## Team

| Name | Role | Developer Community Profile |
|---|---|---|
| Jan Rheineck | Developer | https://community.intersystems.com/user/jan-rheineck |
| Niyu Tong | Developer | https://community.intersystems.com/user/niyu-tong |

## Features

- Reads FHIR R4 patient bundles from InterSystems IRIS for Health when the FHIR
  endpoint is available.
- Falls back to local synthetic/public FHIR sample bundles when IRIS is not
  available.
- Lets the user select a patient and a summary role.
- Generates the three required summary sections.
- Shows compact Reference Data Sources so judges can inspect which FHIR values
  were used.
- Supports multiple local patient JSON bundles in the `data/` directory.
- Runs as a Dockerized web app that connects to a user-provided IRIS for
  Health or HealthShare FHIR endpoint.

## Architecture

```mermaid
flowchart LR
    User["User / Browser"] --> UI["Gradio Web UI"]
    UI --> App["Python App\nFHIR client + context extractor + AI agent"]

    App -->|FHIR API if available| IRIS["External IRIS for Health /\nHealthShare FHIR endpoint"]
    App -->|fallback if unavailable| Data["Local data/*.json\nFHIR Bundles"]

    App --> Context["Extracted patient context"]
    Context --> OpenAI["OpenAI API"]
    OpenAI --> Summary["Role-specific summary"]
    Summary --> UI
```

Main components:

- `src.start`: startup entry point. It checks whether the configured user
  IRIS FHIR endpoint is available and optionally loads local FHIR bundles into
  IRIS when explicitly enabled.
- `src.fhir_loader`: posts local FHIR transaction bundles into IRIS.
- `src.fhir_client`: reads FHIR resources from IRIS or from local fallback JSON.
- `src.context_extractor`: converts FHIR resources into compact clinical text.
- `src.agent`: selects the role prompt and calls the OpenAI model.
- `src.app`: Gradio web UI.

For Open Exchange usage, the Docker Compose default starts only the web app.
Users connect it to their own IRIS for Health or HealthShare FHIR endpoint with
`IRIS_BASE_URL`.

## How It Works

1. The user starts the application.
2. The application checks the configured IRIS FHIR R4 endpoint.
3. If IRIS is available, the application uses the FHIR API.
4. If IRIS is unavailable, the application reads local FHIR bundle JSON files.
5. The user selects a patient from the dropdown.
6. The user selects `ED Doctor`, `Care Manager`, `Patient`, or
   `Family Caregiver`.
7. The application extracts patient demographics, conditions, medications,
   allergies, observations, encounters, and care plans.
8. The AI agent generates three sections:
   - Current Issues
   - Recent Changes
   - Risks and Follow-up
9. The UI displays the generated summary and expandable reference source data.

## Using the App

The app is intended to help demo users and clinicians quickly turn FHIR patient
records into a concise, role-specific summary. It does not replace clinical
judgment.

To use the app:

1. Start it with either Python or Docker.
2. Open `http://localhost:7860`.
3. Select a patient from the dropdown.
4. Select a role: `ED Doctor`, `Care Manager`, `Patient`, or
   `Family Caregiver`.
5. Click `Generate Summary`.
6. Review the generated `Current Issues`, `Recent Changes`, and
   `Risks and Follow-up` sections.
7. Expand `Reference Data Sources` to inspect the FHIR values used by the
   summary.

## Demo

Video demo:

https://youtu.be/rOKGINwaqDU

Demo steps:

1. Start the application.
2. Open `http://localhost:7860`.
3. Select a patient from the dropdown.
4. Select a summary role.
5. Click `Generate Summary`.
6. Review `Current Issues`, `Recent Changes`, and `Risks and Follow-up`.
7. Expand `Reference Data Sources` to inspect the FHIR values used by the agent.

## Prerequisites

- Python 3.11 or newer
- Docker and Docker Compose, if running the container workflow
- A user-provided InterSystems IRIS for Health or HealthShare FHIR endpoint
- OpenAI API key

## Installation

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/niyuT96/fhir-patient-summary.git
cd fhir-patient-summary
```

Create a local environment file:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `OPENAI_API_KEY` and adjust the IRIS connection settings if you want to
connect to a live IRIS for Health or HealthShare FHIR endpoint. If the endpoint
is unavailable, the app can still run with the local fallback data in `data/`.

For local Python usage, install dependencies:

```powershell
pip install -r requirements.txt
```

For Docker usage, no local Python package installation is required. Docker
installs the Python dependencies inside the image when you run:

```powershell
docker compose up --build
```

## Configuration

Make sure `.env` contains at least:

```env
OPENAI_API_KEY=your-openai-api-key-here
IRIS_USERNAME=superuser
IRIS_PASSWORD=SYS
IRIS_BASE_URL=http://localhost:52773/csp/healthshare/fhir/fhir/r4
FHIR_FALLBACK_PATH=data
LOAD_SAMPLE_BUNDLE=false
```

Important environment variables:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required. Used by the OpenAI client. |
| `IRIS_BASE_URL` | IRIS FHIR R4 endpoint. |
| `IRIS_USERNAME` | IRIS basic-auth username. |
| `IRIS_PASSWORD` | IRIS basic-auth password. |
| `FHIR_FALLBACK_PATH` | File or directory used when IRIS is unavailable. |
| `LOAD_SAMPLE_BUNDLE` | Whether startup should POST local bundles into IRIS. Default is `false`. |

Do not commit `.env`. It may contain secrets.

Choose `IRIS_BASE_URL` based on how the app is started:

- Python running on the host machine:
  `http://localhost:52773/csp/healthshare/fhir/fhir/r4`
- Docker app container connecting to an IRIS server on the host machine:
  `http://host.docker.internal:52773/csp/healthshare/fhir/fhir/r4`

If the configured IRIS endpoint is unavailable, the app starts in local fallback
mode and reads FHIR bundles from `FHIR_FALLBACK_PATH`.

The application sends `Accept: application/fhir+json` for FHIR GET requests.
This is required by some IRIS for Health and HealthShare FHIR endpoints.

## Running Locally

This option runs the Python app directly on your computer.

Create and edit `.env` as described above. If your IRIS server is also running
on your computer, use `localhost` in `IRIS_BASE_URL`:

```env
IRIS_BASE_URL=http://localhost:52773/csp/healthshare/fhir/fhir/r4
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the app:

```powershell
python -m src
```

Open:

```text
http://localhost:7860
```

## Running With Docker

The Docker image packages the web application only. It does not require users
to run a bundled IRIS container. Set `IRIS_BASE_URL` to the user's own IRIS for
Health or HealthShare FHIR R4 endpoint.

Create and edit `.env` as described above. For an IRIS server running on the
host machine, use `host.docker.internal` because `localhost` inside the app
container refers to the container itself:

```env
IRIS_BASE_URL=http://host.docker.internal:52773/csp/healthshare/fhir/fhir/r4
```

If no IRIS server is reachable, the Dockerized app still starts and uses the
local FHIR bundles copied into the image from `data/`.

Check the configured IRIS FHIR endpoint from PowerShell:

```powershell
$env:IRIS_BASE_URL="http://localhost:52773/csp/healthshare/fhir/fhir/r4"
$env:IRIS_USERNAME="superuser"
$env:IRIS_PASSWORD="SYS"
.\scripts\check-iris.ps1
```

Start the Dockerized web app:

```powershell
docker compose up --build
```

Or use the helper script:

```powershell
.\scripts\run-docker.ps1
```

Detached mode:

```powershell
.\scripts\run-docker.ps1 -Detached
```

The UI is available at:

```text
http://localhost:7860
```

Optional local IRIS development profile:

```powershell
docker compose --profile local-iris up --build
```

The `local-iris` profile is for development only. Open Exchange users are
expected to connect the web app to their own IRIS FHIR endpoint. Starting this
profile creates an IRIS container, but the app can also be tested without it by
using local fallback data.

## Sample Data

The app supports both local fallback data and IRIS FHIR Server data.

Local fallback can point to either:

- One FHIR Bundle JSON file
- One directory containing multiple `*.json` FHIR Bundle files

Current recommended setting:

```env
FHIR_FALLBACK_PATH=data
```

With this setting, the app reads every JSON file directly under `data/`.

Each JSON file should be a FHIR Bundle. The code reads resources from:

```text
entry[].resource
```

The FHIR Bundle files in `data/` are sourced from the InterSystems
`samples-FHIR-resource-repository`:
https://github.com/intersystems/samples-FHIR-resource-repository

When IRIS is unavailable, the app uses local fallback data and lists all
`Patient` resources found. When IRIS is available and `LOAD_SAMPLE_BUNDLE=true`,
startup attempts to POST each JSON bundle into the configured IRIS FHIR
endpoint. Keep `LOAD_SAMPLE_BUNDLE=false` when connecting to a user's existing
FHIR server and no sample data should be written.

The local sample data is synthetic/public FHIR sample data. Do not add real
patient data to this repository.

## Model and Prompts

The model is configured in `src.agent`.

Current model:

```text
gpt-4o-mini
```

The prompt is role-specific:

- `ED_DOCTOR_PROMPT`
- `CARE_MANAGER_PROMPT`
- `PATIENT_PROMPT`
- `FAMILY_CAREGIVER_PROMPT`

The application sends the same extracted FHIR patient context to the model for
all roles. The selected role changes the system prompt, so the output is framed
for an ED doctor, care manager, patient, or family caregiver.

## Testing

Run:

```powershell
pytest -q
```

## Development Tools

This project was developed with assistance from:

- Kiro, used for requirements, design, and task planning.
- OpenAI Codex, used for implementation assistance, code review, testing
  support, and documentation refinement.

All generated code and documentation were reviewed and adapted by the project
team.

## Limitations

- The application is a contest MVP, not a production clinical decision support
  system.
- The generated summary must be reviewed by a qualified clinician.
- If the configured user IRIS endpoint is unavailable, the app uses local
  fallback FHIR bundles.
- The application does not write generated summaries back to FHIR resources.
- The application does not include full medication interaction checking.

## Open Exchange Submission Notes

| Field | Value |
|---|---|
| GitHub or GitLab URL | https://github.com/niyuT96/fhir-patient-summary |
| Open Exchange URL | To be added after publication |
| Demo video URL | https://youtu.be/rOKGINwaqDU |
| Team member DC profiles | Jan Rheineck: https://community.intersystems.com/user/jan-rheineck; Niyu Tong: https://community.intersystems.com/user/niyu-tong |
| Contest task | Smart Patient Summary Generator |
| InterSystems product | InterSystems IRIS for Health / HealthShare FHIR |

Suggested short description:

```text
An AI agent for InterSystems IRIS for Health that reads FHIR patient data and
generates role-specific patient summaries for emergency doctors, care managers,
patients, and family caregivers.
```


## Data References

- InterSystems FHIR sample data:
  https://github.com/intersystems/samples-FHIR-resource-repository
- Synthea synthetic patient population simulator:
  https://github.com/synthetichealth/synthea
- FHIR R4 specification:
  https://hl7.org/fhir/R4/

## License

This project is licensed under the MIT License. See `LICENSE`.
