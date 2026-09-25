"""Reward functions (policies) for GRPO on agent tool-calling traces.

Each function follows the TRL GRPO contract: keyword args include
`prompts`, `completions`, `completion_ids`, plus any extra dataset columns,
and must return a list of floats (or `None` per sample to skip).

Register new rewards in `REWARD_REGISTRY` and select them by name from the
launch script via `resolve_reward_funcs`.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

RewardFunc = Callable[..., list[float | None]]

# Qwen / ChatML tool-call block produced by the chat template.
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)


def _completion_text(completion: Any) -> str:
    """Normalise TRL completion (string or conversational message list) to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for msg in completion:
            if isinstance(msg, dict):
                content = msg.get("content") or ""
                if isinstance(content, str):
                    parts.append(content)
        return "".join(parts)
    return str(completion)


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract `{name, arguments}` dicts from `<tool_call>` XML blocks."""
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
                "arguments": payload.get("arguments", {}),
            }
        )
    return parsed


def tool_call_format_reward(
    completions: list[Any],
    reference_has_tool_calls: list[bool],
    **kwargs: Any,
) -> list[float]:
    """Reward well-formed tool-call XML when the reference used tools.

    +1.0 if reference expects tools and completion has ≥1 parseable
    `<tool_call>`; +0.5 if reference expects no tools and none appear;
    0.0 otherwise.
    """
    rewards: list[float] = []
    for completion, expects in zip(completions, reference_has_tool_calls):
        text = _completion_text(completion)
        calls = _parse_tool_calls(text)
        has_block = bool(_TOOL_CALL_RE.search(text))
        if expects:
            rewards.append(1.0 if calls else (0.25 if has_block else 0.0))
        else:
            rewards.append(0.5 if not has_block else 0.0)
    return rewards


def tool_name_match_reward(
    completions: list[Any],
    reference_tool_names: list[list[str]],
    **kwargs: Any,
) -> list[float]:
    """Fraction of reference tool names that appear (in order) in the completion.

    Returns `None` when the reference has no tool calls so other rewards
    can own those samples.
    """
    rewards: list[float | None] = []
    for completion, ref_names in zip(completions, reference_tool_names):
        if not ref_names:
            rewards.append(None)
            continue
        pred_names = [c["name"] for c in _parse_tool_calls(_completion_text(completion))]
        if not pred_names:
            rewards.append(0.0)
            continue
        matched = sum(
            1
            for i, name in enumerate(ref_names)
            if i < len(pred_names) and pred_names[i] == name
        )
        rewards.append(matched / len(ref_names))
    return rewards


def non_empty_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    """Small reward for producing a non-empty completion."""
    return [
        1.0 if _completion_text(c).strip() else 0.0
        for c in completions
    ]


REWARD_REGISTRY: dict[str, RewardFunc] = {
    "tool_call_format": tool_call_format_reward,
    "tool_name_match": tool_name_match_reward,
    "non_empty": non_empty_reward,
}


def available_rewards() -> tuple[str, ...]:
    return tuple(sorted(REWARD_REGISTRY))


def resolve_reward_funcs(names: Sequence[str]) -> list[RewardFunc]:
    """Resolve reward names to callables. Unknown names raise `KeyError`."""
    resolved: list[RewardFunc] = []
    for name in names:
        try:
            resolved.append(REWARD_REGISTRY[name])
        except KeyError:
            known = ", ".join(available_rewards()) or "(nessuna)"
            raise KeyError(
                f"Reward sconosciuta {name!r}. Disponibili: {known}."
            ) from None
    if not resolved:
        raise ValueError("Serve almeno una reward function.")
    return resolved
