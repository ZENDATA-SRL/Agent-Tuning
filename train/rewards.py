"""Reward functions (policies) for GRPO on agent tool-calling traces.

Each function follows the TRL GRPO contract: keyword args include
`prompts`, `completions`, `completion_ids`, plus any extra dataset columns,
and must return a list of floats (or `None` per sample to skip).

Tool-call markup is model-specific: pass the recipe's
``chat.tool_calls`` (a ``ToolCallFormat``) to ``resolve_reward_funcs``.
This module stays format-agnostic.

Register new rewards in `REWARD_REGISTRY` and select them by name from the
launch script via `resolve_reward_funcs`.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from shared.config import ToolCallFormat

RewardFunc = Callable[..., list[float | None]]

# Same checkpoint as the benchmark answer metric, so GRPO and eval agree.
# CPU: the training GPU is already occupied by the policy model.
_ANSWER_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_COSINE_REWARD_STEP = 0.1
_embedding_model: Any = None


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


def _tool_calls_from_kwargs(kwargs: dict[str, Any]) -> ToolCallFormat:
    tool_calls = kwargs.get("tool_calls")
    if isinstance(tool_calls, list):
        tool_calls = tool_calls[0] if tool_calls else None
    if not isinstance(tool_calls, ToolCallFormat):
        raise ValueError(
            "Manca tool_calls (ToolCallFormat). Passalo a resolve_reward_funcs "
            "da recipe.chat.tool_calls."
        )
    return tool_calls


def _normalize_arg_value(value: Any) -> Any:
    """Canonicalise nested argument values for order-insensitive dict compare."""
    if isinstance(value, dict):
        return {str(k): _normalize_arg_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_arg_value(v) for v in value]
    return value


def _normalize_args(raw: Any) -> dict[str, Any] | None:
    """Return a dict of arguments, or ``None`` if ``raw`` is not map-like."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return {}
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None
    return {str(k): _normalize_arg_value(v) for k, v in raw.items()}


def _args_exact(reference: Any, predicted: Any) -> bool:
    """True if argument maps match exactly (key order ignored)."""
    ref = _normalize_args(reference)
    pred = _normalize_args(predicted)
    if ref is None or pred is None:
        return False
    return ref == pred


def tool_call_format_reward(
    completions: list[Any],
    reference_has_tool_calls: list[bool],
    **kwargs: Any,
) -> list[float]:
    """Reward parseable tool calls when the reference used tools.

    +1.0 if the reference expects tools and the completion has ≥1 parseable
    call; +0.5 if the reference expects no tools and no tool markup appears;
    0.0 otherwise.

    Requires ``tool_calls`` (a ``ToolCallFormat``) bound by
    ``resolve_reward_funcs``.
    """
    fmt = _tool_calls_from_kwargs(kwargs)
    rewards: list[float] = []
    for completion, expects in zip(completions, reference_has_tool_calls):
        text = _completion_text(completion)
        calls = fmt.parse(text)
        if expects:
            rewards.append(1.0 if calls else 0.0)
        else:
            rewards.append(0.5 if not fmt.has_markup(text) else 0.0)
    return rewards


def tool_name_match_reward(
    completions: list[Any],
    reference_tool_names: list[list[str]],
    **kwargs: Any,
) -> list[float | None]:
    """Fraction of reference tool names that appear (in order) in the completion.

    Returns `None` when the reference has no tool calls so other rewards
    can own those samples. Names are read via the bound ``ToolCallFormat``.
    """
    fmt = _tool_calls_from_kwargs(kwargs)
    rewards: list[float | None] = []
    for completion, ref_names in zip(completions, reference_tool_names):
        if not ref_names:
            rewards.append(None)
            continue
        pred_names = [
            call["name"] for call in fmt.parse(_completion_text(completion))
        ]
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


def _decode_reference_tool_arguments(
    raw: str | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Decode GRPO column values produced by ``format_grpo_dataset``.

    Arguments are stored as a JSON string so Arrow can hold heterogeneous
    tool schemas; older in-memory list-of-dict values are still accepted.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        if not raw.strip():
            return []
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def tool_args_match_reward(
    completions: list[Any],
    reference_tool_arguments: list[str | list[dict[str, Any]]],
    **kwargs: Any,
) -> list[float | None]:
    """1.0 if every reference call has exact argument key/values (order-free).

    Pairs calls by position. Returns 0.0 if the counts differ or any argument
    map differs. Returns `None` when the reference has no tool calls.

    ``reference_tool_arguments`` is a JSON string per row (list of arg dicts).
    """
    fmt = _tool_calls_from_kwargs(kwargs)
    rewards: list[float | None] = []
    for completion, raw_ref_args in zip(
        completions, reference_tool_arguments
    ):
        ref_args_list = _decode_reference_tool_arguments(raw_ref_args)
        if not ref_args_list:
            rewards.append(None)
            continue
        pred_calls = fmt.parse(_completion_text(completion))
        if len(pred_calls) != len(ref_args_list):
            rewards.append(0.0)
            continue
        exact = all(
            _args_exact(ref_args, pred.get("arguments"))
            for ref_args, pred in zip(ref_args_list, pred_calls)
        )
        rewards.append(1.0 if exact else 0.0)
    return rewards


def _get_embedding_model() -> Any:
    """Lazy-load the answer embedding model on CPU."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(
            _ANSWER_EMBEDDING_MODEL,
            device="cpu",
        )
    return _embedding_model


def _quantize_similarity(score: float, step: float = _COSINE_REWARD_STEP) -> float:
    """Snap ``score`` onto ``0.0, 0.1, ..., 1.0``.

    Values outside ``[0, 1]`` are clipped first, so a negative cosine (texts
    pointing apart) is ``0.0``. Halfway cases round away from zero
    (``0.05`` → ``0.1``) instead of using banker's rounding.
    """
    clipped = min(1.0, max(0.0, score))
    n_steps = int(round(1.0 / step))
    # 1e-9 absorbs binary error on exact halfway values (0.05 / 0.1 can be
    # 0.49999999999999994, which would otherwise fall into the lower bucket).
    bucket = int(clipped / step + 0.5 + 1e-9)
    if bucket > n_steps:
        bucket = n_steps
    return bucket / n_steps


def answer_cosine_reward(
    completions: list[Any],
    reference_content: list[str],
    **kwargs: Any,
) -> list[float | None]:
    """Discrete cosine similarity between the completion and the reference text.

    Embeddings come from ``paraphrase-multilingual-MiniLM-L12-v2``. The raw
    cosine is snapped to tenths so GRPO does not chase embedding noise
    smaller than ``0.1``.

    Args:
        completions: Model generations for the batch.
        reference_content: Target assistant text, one string per completion.

    Returns:
        One score per completion on ``{0.0, 0.1, ..., 1.0}``, or ``None``
        when the reference text is empty (tool-call turns stay with the
        tool rewards). An empty completion against a non-empty target is
        ``0.0``.
    """
    del kwargs  # tool_calls is bound for every reward; this one ignores it.
    rewards: list[float | None] = [None] * len(completions)
    pending: list[tuple[int, str, str]] = []
    for index, (completion, reference) in enumerate(
        zip(completions, reference_content)
    ):
        target = (reference or "").strip()
        if not target:
            continue
        generated = _completion_text(completion).strip()
        if not generated:
            rewards[index] = 0.0
            continue
        pending.append((index, target, generated))

    if not pending:
        return rewards

    model = _get_embedding_model()
    texts = [
        text
        for _, target, generated in pending
        for text in (target, generated)
    ]
    embeddings = np.asarray(
        model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )
    for offset, (index, _, _) in enumerate(pending):
        cosine = float(np.dot(embeddings[offset * 2], embeddings[offset * 2 + 1]))
        rewards[index] = _quantize_similarity(cosine)
    return rewards


REWARD_REGISTRY: dict[str, RewardFunc] = {
    "tool_call_format": tool_call_format_reward,
    "tool_name_match": tool_name_match_reward,
    "tool_args_match": tool_args_match_reward,
    "answer_cosine": answer_cosine_reward,
}


def available_rewards() -> tuple[str, ...]:
    """Registered reward names, sorted."""
    return tuple(sorted(REWARD_REGISTRY))


def resolve_reward_funcs(
    names: Sequence[str],
    *,
    tool_calls: ToolCallFormat,
) -> list[RewardFunc]:
    """Resolve reward names to callables bound to one ``ToolCallFormat``.

    Pass ``recipe.chat.tool_calls``. Unknown reward names raise `KeyError`.
    """
    if not isinstance(tool_calls, ToolCallFormat):
        raise TypeError(
            f"tool_calls deve essere un ToolCallFormat, "
            f"non {type(tool_calls).__name__}."
        )
    resolved: list[RewardFunc] = []
    for name in names:
        try:
            fn = REWARD_REGISTRY[name]
        except KeyError:
            known = ", ".join(available_rewards()) or "(nessuna)"
            raise KeyError(
                f"Reward sconosciuta {name!r}. Disponibili: {known}."
            ) from None

        def bound(
            fn: RewardFunc = fn,
            fmt: ToolCallFormat = tool_calls,
            **kwargs: Any,
        ) -> list[float | None]:
            return fn(tool_calls=fmt, **kwargs)

        bound.__name__ = getattr(fn, "__name__", name)
        bound.__doc__ = fn.__doc__
        resolved.append(bound)
    if not resolved:
        raise ValueError("Serve almeno una reward function.")
    return resolved
