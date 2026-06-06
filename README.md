# Smart Patient Summary Generator

Smart Patient Summary Generator is an InterSystems IRIS for Health FHIR demo
application. It reads FHIR R4 patient data, builds a compact clinical context,
and uses an AI agent to generate role-specific summaries for clinicians.

Supported roles:

- ED Doctor
- Care Manager

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
- Lets the user select a patient and a clinician role.
- Generates the three required summary sections.
- Shows compact Reference Data Sources so judges can inspect which FHIR values
  were used.
- Supports multiple local patient JSON bundles in the `data/` directory.
- Runs locally with Python or through Docker Compose.

## Architecture

Main components:

- `src.start`: startup entry point. It checks whether the IRIS FHIR endpoint is
  available and optionally loads local FHIR bundles into IRIS.
- `src.fhir_loader`: posts local FHIR transaction bundles into IRIS.
- `src.fhir_client`: reads FHIR resources from IRIS or from local fallback JSON.
- `src.context_extractor`: converts FHIR resources into compact clinical text.
- `src.agent`: selects the role prompt and calls the OpenAI model.
- `src.app`: Gradio web UI.

## How It Works

1. The user starts the application.
2. The application checks the configured IRIS FHIR R4 endpoint.
3. If IRIS is available, the application uses the FHIR API.
4. If IRIS is unavailable, the application reads local FHIR bundle JSON files.
5. The user selects a patient from the dropdown.
6. The user selects either `ED Doctor` or `Care Manager`.
7. The application extracts patient demographics, conditions, medications,
   allergies, observations, encounters, and care plans.
8. The AI agent generates three sections:
   - Current Issues
   - Recent Changes
   - Risks and Follow-up
9. The UI displays the generated summary and expandable reference source data.

## Demo

Video demo:

https://youtu.be/rOKGINwaqDU

Demo steps:

1. Start the application.
2. Open `http://localhost:7860`.
3. Select a patient from the dropdown.
4. Select `ED Doctor` or `Care Manager`.
5. Click `Generate Summary`.
6. Review `Current Issues`, `Recent Changes`, and `Risks and Follow-up`.
7. Expand `Reference Data Sources` to inspect the FHIR values used by the agent.

## Prerequisites

- Python 3.11 or newer
- Docker and Docker Compose, if running the container workflow
- InterSystems IRIS for Health or HealthShare FHIR endpoint
- OpenAI API key

## Configuration

Create a local environment file:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set at least:

```env
OPENAI_API_KEY=your-openai-api-key-here
IRIS_PASSWORD=SYS
IRIS_BASE_URL=http://localhost:52773/csp/healthshare/fhir/fhir/r4
FHIR_FALLBACK_PATH=data
```

Important environment variables:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required. Used by the OpenAI client. |
| `IRIS_BASE_URL` | IRIS FHIR R4 endpoint. |
| `IRIS_USERNAME` | IRIS basic-auth username. |
| `IRIS_PASSWORD` | IRIS basic-auth password. |
| `FHIR_FALLBACK_PATH` | File or directory used when IRIS is unavailable. |
| `LOAD_SAMPLE_BUNDLE` | Whether startup should POST local bundles into IRIS. |

Do not commit `.env`. It may contain secrets.

The application sends `Accept: application/fhir+json` for FHIR GET requests.
This is required by some IRIS for Health and HealthShare FHIR endpoints.

## Running Locally

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

```powershell
docker compose up --build
```

The UI is available at:

```text
http://localhost:7860
```

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

When IRIS is unavailable, the app uses local fallback data and lists all
`Patient` resources found. When IRIS is available and `LOAD_SAMPLE_BUNDLE=true`,
startup attempts to POST each JSON bundle into the IRIS FHIR endpoint.

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

The application sends the selected role prompt plus the extracted FHIR patient
context to the model.

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
- If the configured IRIS endpoint is unavailable, the app uses local fallback
  FHIR bundles.
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
generates clinician-ready, role-specific patient summaries for emergency
doctors and care managers.
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
