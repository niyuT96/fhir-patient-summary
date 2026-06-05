# Contest Plan: AI Agent for FHIR

## Recommended Direction

Build the **Smart Patient Summary Generator**.

This is the best fit for the current repository, the two-day deadline, and the official contest bonus structure. It is one of the suggested contest tasks, so implementing it can earn **5 bonus points**.

The application should take FHIR data for one patient and generate:

- Current issues
- Recent changes
- Risks / follow-up items
- Role-specific summaries for at least:
  - ED Doctor
  - Care Manager

## Contest Requirements

The application must:

- Be submitted by **June 7, 2026, 23:59 EST**.
- Be fully functional.
- Work on **InterSystems IRIS Community Edition** or **InterSystems IRIS for Health Community Edition**.
- Be open source and published on GitHub or GitLab.
- Have an English README.
- Include installation steps.
- Include either a video demo or a detailed description of how the application works.
- Be published on Open Exchange before applying to the contest.

Contest page:

https://community.intersystems.com/post/intersystems-programming-contest-ai-agents-fhir

Open Exchange apply docs:

https://docs.openexchange.intersystems.com/contest/apply/

## Target FHIR Resources

Use these resources for the MVP:

- Patient
- Condition
- MedicationRequest
- AllergyIntolerance
- Observation
- Encounter
- CarePlan

Optional useful resources:

- DiagnosticReport
- Procedure
- Immunization

## Bonus Points To Target

Realistic bonuses for a two-day submission:

- Suggested task implementation: **5 points**
- InterSystems FHIR Server usage: **2 points**
- Docker container usage: **2 points**
- LLM / LangChain usage: **3 points**
- Embedded Python: **3 points**, if easy to integrate
- YouTube demo video: **3 points**
- Online demo: **2 points**, only if deployment is quick
- First-time contribution: **3 points**, if applicable

Technology bonuses page:

https://community.intersystems.com/post/technology-bonuses-intersystems-programming-contest-ai-agents-fhir

## Current Repository Status

Current files show:

- `docker-compose.yml` uses `intersystems/irishealth-community:latest-em`.
- `src/Demo/Hello.cls` is only a small placeholder.
- `ui.py` is currently a mock Gradio UI.
- The UI text has encoding problems.
- Planning documents exist under `Plan/`.
- No local FHIR patient bundle exists yet.
- No real summary generation exists yet.
- The root README is too short for contest submission.

Conclusion: the project is not submission-ready yet.

## MVP Scope

Do not build a broad healthcare product. Build a narrow, working, easy-to-demo application.

Minimum MVP:

- Include one synthetic FHIR patient bundle.
- Parse the relevant FHIR resources.
- Extract a structured patient profile.
- Generate:
  - Current issues
  - Recent changes
  - Risks / follow-up items
- Generate summaries for:
  - ED Doctor
  - Care Manager
- Provide a simple UI or CLI demo.
- Document how IRIS for Health is used.
- Document how sample FHIR data is loaded or read.

## Implementation Strategy

1. Add sample synthetic FHIR data.
   - Use one Synthea or InterSystems sample FHIR bundle.
   - Store it locally, for example: `data/sample-patient-bundle.json`.
   - Do not include real patient data.

2. Build the FHIR extraction layer.
   - Parse the bundle.
   - Group resources by `resourceType`.
   - Extract patient demographics, active conditions, medications, allergies, recent labs, encounters, and care plans.

3. Build the summary layer.
   - First implement deterministic summaries so the demo works without an API key.
   - Add optional LLM-based generation if an API key is available.
   - Keep output structured and predictable.

4. Connect the UI.
   - Replace the current mock Gradio response.
   - Let the user select a role.
   - Display extracted FHIR data and generated summary.

5. Keep IRIS for Health visible in the project.
   - Keep Docker Compose.
   - Document the IRIS for Health container.
   - If possible, load the FHIR bundle into IRIS.
   - If loading into IRIS takes too long, keep a JSON fallback and document the intended IRIS FHIR path clearly.

6. Finish submission materials.
   - Rewrite README in English.
   - Add setup commands.
   - Add demo steps.
   - Add team members and Developer Community profile links.
   - Add contest task reference.
   - Add screenshots or a video link.

## Two-Day Schedule

### Day 1

- Confirm Smart Patient Summary Generator as the final scope.
- Add synthetic FHIR sample data.
- Implement FHIR bundle parsing.
- Implement first deterministic summary output.
- Fix the Gradio UI text.
- Generate one complete working patient summary.

### Day 2

- Add role-specific summaries for ED Doctor and Care Manager.
- Stabilize the UI or CLI demo.
- Add error handling for missing data and missing API keys.
- Update README in English.
- Prepare screenshots.
- Record a short demo video if possible.
- Run a clean-start test using the documented commands.
- Publish to GitHub or GitLab.
- Publish on Open Exchange.
- Apply to the contest.

## Submission Checklist

- [ ] Application runs locally.
- [ ] Docker Compose works or the documented fallback works.
- [ ] IRIS for Health usage is documented.
- [ ] Synthetic FHIR data is included or reproducibly loaded.
- [ ] No real patient data is included.
- [ ] No API keys, passwords, or tokens are committed.
- [ ] README is in English.
- [ ] README includes installation steps.
- [ ] README includes demo steps.
- [ ] README includes selected contest task.
- [ ] README includes team members and Developer Community profile links.
- [ ] README includes video link or detailed application description.
- [ ] Repository is public on GitHub or GitLab.
- [ ] App is published on Open Exchange.
- [ ] App is submitted to the contest before the deadline.

## Project Positioning

Use this description:

> An AI agent for InterSystems IRIS for Health that reads FHIR patient data and generates clinician-ready, role-specific patient summaries for emergency doctors and care managers.

This gives the judges a clear healthcare problem, a clear FHIR workflow, and a concrete AI agent output.

