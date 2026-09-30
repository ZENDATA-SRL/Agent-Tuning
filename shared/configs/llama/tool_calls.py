"""Llama 3.x JSON tool-call markup for GRPO rewards.

Chat template format (custom tools)::

    {"name": "<function-name>", "parameters": {...}}

``arguments`` is also accepted, matching the vLLM ``llama3_json`` parser.
Builtin-tool form ``<|python_tag|>name.call(...)`` counts as markup only.
"""
from __future__ import annotations

import json
import re
from typing import Any

from shared.config import ToolCallFormat

_TOOL_ATTEMPT_RE = re.compile(r'\{\s*"name"\s*:')


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
    """Extract tool calls from Llama JSON objects with ``name`` + args."""
    parsed: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            break
        try:
            payload, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        index = start + consumed
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        if not name:
            continue
        if "parameters" in payload:
            arguments = payload["parameters"]
        elif "arguments" in payload:
            arguments = payload["arguments"]
        else:
            continue
        parsed.append(
            {
                "name": str(name),
                "arguments": _as_arguments(arguments),
            }
        )
    return parsed


def has_tool_call_markup(text: str) -> bool:
    """True if the text looks like a Llama tool-call attempt."""
    return "<|python_tag|>" in text or bool(_TOOL_ATTEMPT_RE.search(text))


LLAMA3_JSON_TOOL_CALLS = ToolCallFormat(
    parse=parse_tool_calls,
    has_markup=has_tool_call_markup,
)
