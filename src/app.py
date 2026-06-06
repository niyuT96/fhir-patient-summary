"""
Gradio UI for patient selection, role selection, and summary display.

Replaces the root-level ui.py mock.
Requirements: 8.1-8.10, 10.4
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from src.agent import SummaryAgent
from src.context_extractor import PatientContextExtractor
from src.fhir_client import FHIRClient
from src.models import SummaryResult

load_dotenv()

# ---------------------------------------------------------------------------
# Startup guard - Req 10.4
# ---------------------------------------------------------------------------
_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
if not _api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set or empty. "
        "Please set this environment variable before starting the application."
    )

# ---------------------------------------------------------------------------
# Shared clients
# ---------------------------------------------------------------------------
_fhir_client = FHIRClient(
    base_url=os.environ.get("IRIS_BASE_URL", "http://iris:52773/fhir/r4"),
    username=os.environ.get("IRIS_USERNAME", "superuser"),
    password=os.environ.get("IRIS_PASSWORD", "SYS"),
    fallback_path=os.environ.get("FHIR_FALLBACK_PATH", "data/sample-patient-bundle.json"),
)
_extractor = PatientContextExtractor()
_llm_client = OpenAI(api_key=_api_key)
_agent = SummaryAgent(
    fhir_client=_fhir_client,
    extractor=_extractor,
    llm_client=_llm_client,
)


def _format_generated_at(iso_str: str) -> str:
    """Convert an ISO 8601 UTC string to a human-readable label."""
    try:
        dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return dt.strftime("%B %d, %Y %H:%M UTC")
    except Exception:
        return iso_str


def _patient_label(patient: dict) -> str:
    """Build a dropdown label like 'patient-001 - Jane Doe'."""
    patient_id = patient.get("id", "unknown")
    names = patient.get("name", [])
    if names:
        first_name = names[0]
        name = first_name.get("text") or (
            " ".join(first_name.get("given", []) + [first_name.get("family", "")]).strip()
        )
        if name:
            return f"{patient_id} - {name}"
    return patient_id


# ---------------------------------------------------------------------------
# Startup: probe server and build patient list
# ---------------------------------------------------------------------------
_server_available = _fhir_client.is_available()
_patients: list[dict] = _fhir_client.list_patients()

_source_badge_html = (
    '<span style="background:#16a34a;color:#fff;padding:3px 10px;'
    'border-radius:12px;font-size:0.85em;font-weight:600;">FHIR Server</span>'
    if _server_available
    else '<span style="background:#d97706;color:#fff;padding:3px 10px;'
    'border-radius:12px;font-size:0.85em;font-weight:600;">Local Fallback</span>'
)

_patient_choices: list[str] = [_patient_label(patient) for patient in _patients]
_patient_id_map: dict[str, str] = {
    _patient_label(patient): patient.get("id", "") for patient in _patients
}


def on_generate(patient_label: str | None, role: str):
    """Generate and render a summary for the selected patient and role."""
    if not patient_label:
        return (
            "",
            "Please select a patient before generating a summary",
            "",
            gr.update(interactive=True),
        )

    patient_id = _patient_id_map.get(patient_label, patient_label)
    result: SummaryResult = _agent.generate_summary(patient_id, role)

    if result.error:
        summary_markdown = f"**Error:** {result.error}"
    else:
        summary_markdown = (
            f"## Current Issues\n{result.current_issues}\n\n"
            f"## Recent Changes\n{result.recent_changes}\n\n"
            f"## Risks and Follow-up\n{result.risks_and_followup}"
        )

    source_label = (
        "Source: FHIR Server"
        if result.data_source == "fhir_server"
        else "Source: Local Fallback"
    )
    footer_html = (
        f'<p style="font-size:0.8em;color:#6b7280;">'
        f"{source_label} &nbsp;|&nbsp; Generated: {_format_generated_at(result.generated_at)}"
        "</p>"
    )

    return summary_markdown, "", footer_html, gr.update(interactive=True)


with gr.Blocks(title="Smart Patient Summary Generator") as demo:
    gr.Markdown("# Smart Patient Summary Generator")

    if not _server_available:
        gr.HTML(
            '<div style="background:#fef3c7;border:1px solid #f59e0b;border-radius:6px;'
            'padding:10px 14px;margin-bottom:8px;">'
            "Running in local fallback mode - summaries are generated from local sample "
            "data, not the live FHIR server."
            "</div>"
        )

    with gr.Row():
        gr.HTML(value=_source_badge_html)

    with gr.Row():
        with gr.Column(scale=1):
            patient_dropdown = gr.Dropdown(
                choices=_patient_choices if _patient_choices else ["No patients available"],
                label="Patient",
                value=None,
                interactive=bool(_patient_choices),
            )
            role_radio = gr.Radio(
                choices=["ED Doctor", "Care Manager"],
                label="Clinician Role",
                value="ED Doctor",
            )
            generate_btn = gr.Button(
                "Generate Summary",
                variant="primary",
                interactive=bool(_patient_choices),
            )
            status_bar = gr.Textbox(
                label="Status",
                value="",
                interactive=False,
                visible=True,
            )

        with gr.Column(scale=2):
            summary_output = gr.Markdown(label="Summary", value="")
            footer = gr.HTML(value="")

    generate_btn.click(
        fn=lambda: (gr.update(interactive=False), "Generating summary..."),
        inputs=[],
        outputs=[generate_btn, status_bar],
        queue=False,
    ).then(
        fn=on_generate,
        inputs=[patient_dropdown, role_radio],
        outputs=[summary_output, status_bar, footer, generate_btn],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
