"""Chat-template helper.

Rendering goes through the tokenizer's bundled Jinja template, which is
what supports `tools=[...]` and `role:"tool"` on the agent traces.
Model-specific kwargs (`enable_thinking`, markers, …) come from the
recipe's `ChatTemplateSpec` and are forwarded here unchanged.
"""
from __future__ import annotations

from typing import Any, Mapping


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
        messages,
        tools=tools if tools else None,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **dict(chat_template_kwargs or {}),
    )
