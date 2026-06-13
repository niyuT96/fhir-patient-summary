"""Load and assemble system prompts from markdown and role YAML files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
ROLE_DIR = PROMPT_DIR / "roles"
SYSTEM_POLICY_PATH = PROMPT_DIR / "system_policy.md"

REQUIRED_ROLE_FIELDS = {
    "role",
    "audience",
    "system_focus",
    "style_rules",
    "content_priorities",
    "avoid",
    "section_guidance",
}


def get_supported_roles() -> tuple[str, ...]:
    """Return supported role names in a stable UI order."""
    roles = _load_roles()
    preferred = ("ED Doctor", "Care Manager", "Patient", "Family Caregiver")
    ordered = [role for role in preferred if role in roles]
    ordered.extend(sorted(role for role in roles if role not in preferred))
    return tuple(ordered)


def get_role_prompt(role: str) -> str:
    """Return the assembled system prompt for a supported role."""
    roles = _load_roles()
    if role not in roles:
        raise ValueError(f"Unsupported role: {role}")

    system_policy = _load_system_policy()
    data = roles[role]
    parts = [
        system_policy.strip(),
        "",
        f"# Role: {data['role']}",
        f"Audience: {data['audience']}",
        "",
        "## Role Focus",
        str(data["system_focus"]).strip(),
        "",
        _render_list("Style Rules", data["style_rules"]),
        _render_list("Content Priorities", data["content_priorities"]),
        _render_list("Avoid", data["avoid"]),
        _render_section_guidance(data["section_guidance"]),
    ]
    return "\n".join(part for part in parts if part).strip() + "\n"


@lru_cache(maxsize=1)
def _load_system_policy() -> str:
    try:
        return SYSTEM_POLICY_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot read system prompt policy: {SYSTEM_POLICY_PATH}") from exc


@lru_cache(maxsize=1)
def _load_roles() -> dict[str, dict[str, Any]]:
    if not ROLE_DIR.exists():
        raise RuntimeError(f"Role prompt directory does not exist: {ROLE_DIR}")

    roles: dict[str, dict[str, Any]] = {}
    for path in sorted(ROLE_DIR.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise RuntimeError(f"Role prompt must be a YAML mapping: {path}")
        missing = sorted(REQUIRED_ROLE_FIELDS - set(data))
        if missing:
            raise RuntimeError(
                f"Role prompt {path.name} is missing required field(s): {', '.join(missing)}"
            )
        role = str(data["role"]).strip()
        if not role:
            raise RuntimeError(f"Role prompt {path.name} has an empty role field")
        roles[role] = data

    if not roles:
        raise RuntimeError(f"No role prompt YAML files found in {ROLE_DIR}")
    return roles


def _render_list(title: str, values: Any) -> str:
    if not isinstance(values, list):
        raise RuntimeError(f"{title} must be a list in role prompt YAML")
    lines = [f"## {title}"]
    lines.extend(f"- {value}" for value in values)
    return "\n".join(lines)


def _render_section_guidance(value: Any) -> str:
    if not isinstance(value, dict):
        raise RuntimeError("section_guidance must be a mapping in role prompt YAML")
    lines = ["## Section Guidance"]
    for key in ("current_issues", "recent_changes", "risks_followup"):
        if key not in value:
            raise RuntimeError(f"section_guidance is missing required key: {key}")
        lines.append(f"- {key}: {value[key]}")
    return "\n".join(lines)
