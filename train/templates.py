"""Chat-template helper.

Rendering goes through the tokenizer's bundled Jinja template, which is
what supports `tools=[...]` and `role:"tool"` on the agent traces.
Model-specific kwargs (`enable_thinking`, markers, …) come from the
recipe's `ChatTemplateSpec` and are forwarded here unchanged.
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping

from shared.tool_calls import split_parallel_tool_calls

__all__ = [
    "coerce_tool_arguments_for_template",
    "render_chat",
    "split_parallel_tool_calls",
]


def _parse_tool_arguments(arguments: Any) -> Any:
    """Return tool arguments as a mapping when they are a JSON object string.

    OpenAI-canonical traces store ``function.arguments`` as a JSON string.
    HF chat templates (Gemma DSL, Qwen/Llama ``tojson``) render correctly
    when arguments are a mapping. Leaving a JSON string in place makes
    templates that cannot parse JSON in Jinja dump raw quotes into a
    non-JSON tool DSL (Gemma4).
    """
    if not isinstance(arguments, str):
        return arguments
    text = arguments.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return arguments
    return parsed if isinstance(parsed, dict) else arguments


def coerce_tool_arguments_for_template(messages: list[dict]) -> list[dict]:
    """Copy messages with tool-call argument JSON strings parsed to dicts."""
    prepared = deepcopy(messages)
    for message in prepared:
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if isinstance(function, dict) and "arguments" in function:
                function["arguments"] = _parse_tool_arguments(function["arguments"])
            elif "arguments" in tool_call:
                tool_call["arguments"] = _parse_tool_arguments(tool_call["arguments"])
    return prepared


def render_chat(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None,
    *,
    add_generation_prompt: bool = False,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Render a conversation with the tokenizer's chat template."""
    return tokenizer.apply_chat_template(
        coerce_tool_arguments_for_template(messages),
        tools=tools if tools else None,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **dict(chat_template_kwargs or {}),
    )
