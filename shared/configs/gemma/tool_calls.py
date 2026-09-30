"""Gemma 4 tool-call markup for GRPO rewards.

Chat template format::

    <|tool_call>call:<name>{args}<tool_call|>

Some checkpoints emit ``<turn|>`` as the closer, or a bare ``call:name{args}``
without the special tokens.
"""
from __future__ import annotations

import re
from typing import Any

from shared.config import ToolCallFormat

_STANDARD_RE = re.compile(r"<\|tool_call\>call:(\w+)\{")
_FALLBACK_RE = re.compile(r"(?:<call>|(?:^|\s)call:)(\w+)\{")
_END_TOKENS = ("<tool_call|>", "<turn|>")


def _closing_brace(text: str, open_at: int) -> int | None:
    """Return the index of the ``}`` that closes ``text[open_at] == '{'``."""
    depth = 0
    for index in range(open_at, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _gemma_calls(
    text: str,
    pattern: re.Pattern[str],
    *,
    require_end: bool,
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        open_at = match.end() - 1
        close_at = _closing_brace(text, open_at)
        if close_at is None:
            continue
        if require_end:
            tail = text[close_at + 1 :]
            if not tail.startswith(_END_TOKENS):
                continue
        parsed.append(
            {
                "name": match.group(1),
                "arguments": text[open_at + 1 : close_at],
            }
        )
    return parsed


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract Gemma 4 tool calls from decoded assistant text."""
    parsed = _gemma_calls(text, _STANDARD_RE, require_end=True)
    if parsed:
        return parsed
    return _gemma_calls(text, _FALLBACK_RE, require_end=False)


def has_tool_call_markup(text: str) -> bool:
    """True if standard or fallback Gemma tool-call markup is present."""
    return "<|tool_call>" in text or bool(_FALLBACK_RE.search(text))


GEMMA4_TOOL_CALLS = ToolCallFormat(
    parse=parse_tool_calls,
    has_markup=has_tool_call_markup,
)
