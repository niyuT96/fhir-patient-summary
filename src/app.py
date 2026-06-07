"""
Gradio UI for patient selection, role selection, and summary display.

"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Generator

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from src.agent import SUPPORTED_ROLES, SummaryAgent
from src.context_extractor import PatientContextExtractor
from src.fhir_client import FHIRClient
from src.models import SourceSection

load_dotenv()

# ---------------------------------------------------------------------------
# Startup guard
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
    fallback_path=os.environ.get("FHIR_FALLBACK_PATH", "data"),
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


def _patient_name(patient: dict) -> str:
    """Extract the best available patient display name."""
    names = patient.get("name", [])
    if names:
        first_name = names[0]
        name = first_name.get("text") or (
            " ".join(first_name.get("given", []) + [first_name.get("family", "")]).strip()
        )
        if name:
            return name
    return "Unknown patient"


def _patient_age(patient: dict, today: date | None = None) -> int | None:
    """Calculate age in years from the FHIR Patient.birthDate field."""
    birth_date = patient.get("birthDate")
    if not birth_date:
        return None

    try:
        born = date.fromisoformat(birth_date)
    except ValueError:
        return None

    today = today or date.today()
    age = today.year - born.year
    if (today.month, today.day) < (born.month, born.day):
        age -= 1
    return age


def _patient_label(patient: dict) -> str:
    """Build a dropdown label using only patient name and age."""
    name = _patient_name(patient)
    age = _patient_age(patient)
    if age is None:
        return f"{name} (age unknown)"
    return f"{name} ({age})"


def _build_sources_html(sections: list[SourceSection], data_source: str) -> str:
    """Render reference FHIR values as a styled HTML block for the UI."""
    if not sections:
        return ""

    badge_color = "#16a34a" if data_source == "fhir_server" else "#d97706"
    badge_label = "FHIR Server" if data_source == "fhir_server" else "Local Fallback"

    lines: list[str] = [
        '<div style="font-size:0.88em;line-height:1.6;border:1px solid #e5e7eb;'
        'border-radius:8px;padding:12px 16px;background:#f9fafb;">',
        f'<div style="margin-bottom:10px;">'
        f'<span style="background:{badge_color};color:#fff;padding:2px 9px;'
        f'border-radius:10px;font-size:0.82em;font-weight:600;">'
        f"Data Source: {badge_label}</span>"
        f"</div>",
    ]

    for section in sections:
        if section.items == ["None"]:
            continue

        lines.append(
            f'<details style="margin-bottom:6px;">'
            f'<summary style="cursor:pointer;font-weight:600;color:#374151;'
            f'padding:3px 0;">{section.label}</summary>'
            f'<ul style="margin:4px 0 4px 18px;padding:0;color:#4b5563;">'
        )
        for item in section.items:
            lines.append(f"<li>{item}</li>")
        if section.hidden_items:
            hidden_count = len(section.hidden_items)
            lines.append(
                f'<li style="list-style:none;margin-top:4px;">'
                f'<details>'
                f'<summary style="cursor:pointer;color:#2563eb;">'
                f"Show {hidden_count} more"
                f"</summary>"
                f'<ul style="margin:4px 0 4px 18px;padding:0;color:#4b5563;">'
            )
            for hidden_item in section.hidden_items:
                lines.append(f"<li>{hidden_item}</li>")
            lines.append("</ul></details></li>")
        lines.append("</ul></details>")

    lines.append("</div>")
    return "\n".join(lines)


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

_data_source_label = "fhir_server" if _server_available else "local_fallback"


def on_generate(
    patient_label: str | None,
    role: str,
) -> Generator[tuple[str, str, str, str, gr.update], None, None]:
    """Stream summary text and populate reference sources on completion."""
    if not patient_label:
        yield (
            "",
            "Please select a patient before generating a summary",
            "",
            "",
            gr.update(interactive=True),
        )
        return

    yield (
        "",
        "Preparing FHIR reference data...",
        "",
        "",
        gr.update(interactive=False),
    )

    patient_id = _patient_id_map.get(patient_label, patient_label)
    data_source = _data_source_label
    accumulated_sources: list[SourceSection] = []

    for partial_text, source_sections in _agent.generate_summary_stream(patient_id, role):
        section_count = partial_text.count("## ")
        is_error = partial_text.startswith("**Error:**")
        is_final = is_error or section_count >= 3

        if source_sections is not None:
            accumulated_sources = source_sections or []

        sources_html = (
            _build_sources_html(accumulated_sources, data_source)
            if accumulated_sources
            else ""
        )

        footer_html = ""
        if is_final:
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            source_label = (
                "Source: FHIR Server" if data_source == "fhir_server" else "Source: Local Fallback"
            )
            footer_html = (
                f'<p style="font-size:0.8em;color:#6b7280;">'
                f"{source_label} &nbsp;|&nbsp; Generated: {_format_generated_at(now_iso)}"
                "</p>"
            )

        if not partial_text:
            status_text = "Reference data ready. Generating Current Issues..."
        elif section_count <= 1:
            status_text = "Current Issues ready. Generating Recent Changes..."
        elif section_count == 2:
            status_text = "Recent Changes ready. Generating Risks and Follow-up..."
        else:
            status_text = ""

        yield (
            partial_text,
            "" if is_final else status_text,
            footer_html,
            sources_html,
            gr.update(interactive=is_final),
        )


# ---------------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------------
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
                label="Patient (Age)",
                value=None,
                interactive=bool(_patient_choices),
            )
            role_radio = gr.Radio(
                choices=list(SUPPORTED_ROLES),
                label="Summary Role",
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

            with gr.Accordion("Reference Data Sources", open=False):
                sources_panel = gr.HTML(value="")

    generate_btn.click(
        fn=on_generate,
        inputs=[patient_dropdown, role_radio],
        outputs=[summary_output, status_bar, footer, sources_panel, generate_btn],
    )

demo.queue()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
