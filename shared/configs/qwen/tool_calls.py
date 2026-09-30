"""Qwen / Hermes tool-call markup for GRPO rewards.

Chat template format::

    <tool_call>
    {"name": "<function-name>", "arguments": {...}}
    </tool_call>
"""
from __future__ import annotations

import json
import re
from typing import Any

from shared.config import ToolCallFormat

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)


def _as_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return value if value is not None else {}
    stripped = value.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract ``{name, arguments}`` dicts from ``<tool_call>`` XML blocks."""
    parsed: list[dict[str, Any]] = []
    for match in _TOOL_CALL_RE.finditer(text):
        raw = match.group(1).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        if not name:
            continue
        parsed.append(
            {
                "name": str(name),
                "arguments": _as_arguments(payload.get("arguments", {})),
            }
        )
    return parsed


def has_tool_call_markup(text: str) -> bool:
    """True if a ``<tool_call>`` block is present (even when JSON is broken)."""
    return bool(_TOOL_CALL_RE.search(text))


HERMES_TOOL_CALLS = ToolCallFormat(
    parse=parse_tool_calls,
    has_markup=has_tool_call_markup,
)
