"""Chat-template helper for Qwen3.

Qwen3 ships with its own Jinja chat template in the tokenizer config
(ChatML-style: `<|im_start|>role\\n...<|im_end|>`). It supports
`tools=[...]` and `role:"tool"` natively, which is what we need to
train on full agentic traces (system → user → assistant(tool_calls) →
tool → assistant).

Loss-masking markers used by Unsloth's `train_on_responses_only`:
    instruction = "<|im_start|>user\\n"
    response    = "<|im_start|>assistant\\n"
"""
from __future__ import annotations

from typing import Any


# Markers consumed by `train_on_responses_only` to split the loss between
# the prompt prefix (masked, ignored by the loss) and the assistant turns
# (kept, learned). They must appear verbatim in the rendered text.
QWEN3_INSTRUCTION_PART: str = "<|im_start|>user\n"
QWEN3_RESPONSE_PART: str = "<|im_start|>assistant\n"


def render_chat(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None,
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Render a conversation into Qwen3's wire format using the tokenizer's
    bundled chat template. Returns the rendered string.
    """
    return tokenizer.apply_chat_template(
        messages,
        tools=tools if tools else None,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
