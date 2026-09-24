"""Dataset loading and chat-template rendering for agent traces."""
from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any, Mapping

from datasets import Dataset, load_dataset

from train.templates import render_chat

# Hold out test and eval from a single JSON file. The caller only passes
# `dataset_path`; the split happens here, before any per-turn expansion.
_TEST_RATIO = 0.1
_EVAL_RATIO = 0.1
_SPLIT_SEED = 42


def load_json_dataset(path: str) -> Dataset:
    """Load a local JSON array (or JSONL) file via HuggingFace `datasets`."""
    return load_dataset("json", data_files=path, split="train")


def _subset_by_fraction(rows: list[dict], dataset_fraction: float) -> list[dict]:
    """Keep the first N rows after shuffle, where N = fraction * len(rows)."""
    if dataset_fraction >= 1.0:
        return rows
    if dataset_fraction <= 0.0 or dataset_fraction > 1.0:
        raise ValueError(
            "dataset_fraction deve essere in (0, 1], "
            f"ricevuto {dataset_fraction}."
        )
    n_keep = max(1, int(len(rows) * dataset_fraction))
    if n_keep >= len(rows):
        return rows
    return rows[:n_keep]


def split_records(
    records: list[dict],
    *,
    dataset_fraction: float = 1.0,
) -> dict[str, list[dict]]:
    """Shuffle once and cut eval, then test, then train (10% / 10% / 80%)."""
    rows = list(records)
    random.Random(_SPLIT_SEED).shuffle(rows)
    rows = _subset_by_fraction(rows, dataset_fraction)
    n_eval = int(len(rows) * _EVAL_RATIO)
    n_test = int(len(rows) * _TEST_RATIO)
    return {
        "eval": rows[:n_eval],
        "test": rows[n_eval:n_eval + n_test],
        "train": rows[n_eval + n_test:],
    }


def _without_nulls(value: Any) -> Any:
    """Drop dict keys whose value is None, including nested dicts and lists."""
    if isinstance(value, dict):
        return {
            key: _without_nulls(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_without_nulls(item) for item in value]
    return value


def _record_for_export(row: dict) -> dict:
    """Copy a record for JSONL export without schema-alignment nulls."""
    return _without_nulls(row)


def write_split_jsonl(
    dataset_path: str,
    splits: Mapping[str, list[dict]],
    temp_id: str,
) -> Path:
    """Write one JSONL per split under `<dataset_dir>/<temp_id>/`."""
    if (
        not temp_id
        or temp_id in {".", ".."}
        or "/" in temp_id
        or "\\" in temp_id
    ):
        raise ValueError(f"temp_id non valido: {temp_id!r}")
    out_dir = Path(dataset_path).parent / temp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "test", "eval"):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in splits[name]:
                handle.write(
                    json.dumps(_record_for_export(row), ensure_ascii=False) + "\n"
                )
    return out_dir


def _traces_ending_with_assistant(record: dict) -> list[dict]:
    """One copy per assistant turn: messages are the prefix ending on that turn."""
    messages = list(record.get("messages") or [])
    pieces: list[dict] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if _normalize_role(str(message.get("role", ""))) != "assistant":
            continue
        piece = dict(record)
        piece["messages"] = messages[: index + 1]
        pieces.append(piece)
    return pieces


def load_prepared_splits(
    path: str,
    *,
    dataset_fraction: float = 1.0,
    temp_id: str | None = None,
) -> dict[str, list[dict]]:
    """Split one dataset file, then cut traces so each ends on an assistant turn.

    The conversation-level split is written next to `path` as
    `<temp_id>/{train,test,eval}.jsonl` before per-turn expansion.
    """
    rows = [dict(row) for row in load_json_dataset(path)]
    splits = split_records(rows, dataset_fraction=dataset_fraction)
    split_dir = write_split_jsonl(path, splits, temp_id or str(uuid.uuid4()))
    print(f"Wrote split files to {split_dir}")
    return {
        name: [
            piece
            for row in split
            for piece in _traces_ending_with_assistant(row)
        ]
        for name, split in splits.items()
    }


def limit_traces(dataset: Dataset, max_traces: int | None) -> Dataset:
    """Keep at most `max_traces` rows. `None` leaves the dataset unchanged."""
    if max_traces is None:
        return dataset
    if max_traces < 1:
        raise ValueError(f"max_traces deve essere >= 1, ricevuto {max_traces}.")
    if max_traces >= len(dataset):
        return dataset
    return dataset.select(range(max_traces))


# Langchain-style traces use "human"/"ai", OpenAI uses "user"/"assistant".
# Normalise to OpenAI canonical so `apply_chat_template` understands them.
_ROLE_MAPPING = {
    "human": "user",
    "ai": "assistant",
    "tool": "tool",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
}


def _normalize_role(role: str) -> str:
    return _ROLE_MAPPING.get(role.lower(), role.lower())


def _assert_openai_message(msg: Any) -> None:
    """Assert that a message already follows the OpenAI chat schema."""
    assert isinstance(msg, dict), "message must be a dict"
    role = msg.get("role")
    assert role in {"system", "user", "assistant", "tool"}, (
        f"invalid OpenAI message role: {role!r}"
    )

    content = msg.get("content", "")
    assert content is None or isinstance(content, str), (
        "OpenAI message content must be a string or None"
    )

    if role == "assistant" and msg.get("tool_calls") is not None:
        tool_calls = msg["tool_calls"]
        assert isinstance(tool_calls, list), "tool_calls must be a list"
        for tool_call in tool_calls:
            assert isinstance(tool_call, dict), "tool_call must be a dict"
            assert tool_call.get("id"), "tool_call must have an id"
            assert tool_call.get("type") == "function", (
                "tool_call type must be 'function'"
            )
            function = tool_call.get("function")
            assert isinstance(function, dict), "tool_call function must be a dict"
            assert function.get("name"), "tool_call function must have a name"
            assert isinstance(function.get("arguments"), str), (
                "tool_call function arguments must be a JSON string"
            )

    if role == "tool":
        assert msg.get("tool_call_id"), "tool message must have tool_call_id"


def _build_openai_message(msg: dict) -> dict | None:
    """Convert a raw trace message into an OpenAI-canonical message dict."""
    role = _normalize_role(msg.get("role", ""))
    content = msg.get("content", "") or ""

    if role == "assistant":
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            formatted_tcs = []
            for tc in tool_calls:
                name = tc.get("name", tc.get("function", {}).get("name", ""))
                args = tc.get("args", tc.get("function", {}).get("arguments", {}))
                if isinstance(args, dict):
                    args_str = json.dumps(args)
                else:
                    args_str = str(args)
                formatted_tcs.append({
                    "id": tc.get("id", tc.get("call_id", "")),
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                })
            return {"role": "assistant", "content": content, "tool_calls": formatted_tcs}
        return {"role": "assistant", "content": content}

    if role == "tool":
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": msg.get("tool_call_id", msg.get("id", "")),
        }

    if role in ("user", "system"):
        return {"role": role, "content": content}

    return None


def trace_to_messages(record: dict) -> list[dict]:
    """Project a dataset record into a list of OpenAI-canonical messages."""
    raw_messages = record.get("messages", [])
    messages: list[dict] = []
    for msg in raw_messages:
        _assert_openai_message(msg)
        m = _build_openai_message(msg)
        if m is not None:
            messages.append(m)
    return messages


def format_dataset(
    records: Dataset | list[dict],
    tokenizer: Any,
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> Dataset:
    """Render each record into a single training string (the "text" column)."""
    if len(records) == 0:
        raise ValueError("format_dataset received an empty dataset.")

    texts: list[str] = []
    for rec in records:
        messages = trace_to_messages(rec)
        tools = rec.get("tools") or []
        text = render_chat(
            tokenizer,
            messages,
            tools=tools if tools else None,
            add_generation_prompt=False,
            chat_template_kwargs=chat_template_kwargs,
        )
        texts.append(text)

    return Dataset.from_dict({"text": texts})


def _last_assistant_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            return i
    raise ValueError(
        "Trace has no assistant message; cannot build a GRPO prompt."
    )


def _reference_tool_names(assistant_msg: dict) -> list[str]:
    tool_calls = assistant_msg.get("tool_calls") or []
    names: list[str] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            names.append(str(name))
    return names


def format_grpo_dataset(
    records: Dataset | list[dict],
    tokenizer: Any,
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> Dataset:
    """Build a GRPO dataset with a rendered `prompt` plus reference columns.

    For each trace, the prompt is every message before the last assistant
    turn, rendered with the chat template (tools included) and a generation
    prompt so the model continues as assistant. Reference fields are kept for
    reward functions (TRL forwards every column except `prompt`).
    """
    if len(records) == 0:
        raise ValueError("format_grpo_dataset received an empty dataset.")

    prompts: list[str] = []
    reference_contents: list[str] = []
    reference_tool_names: list[list[str]] = []
    reference_has_tool_calls: list[bool] = []

    for rec in records:
        messages = trace_to_messages(rec)
        last_idx = _last_assistant_index(messages)
        prefix = messages[:last_idx]
        if not prefix:
            raise ValueError(
                "Trace has no messages before the last assistant turn."
            )
        reference = messages[last_idx]
        tools = rec.get("tools") or []
        prompt = render_chat(
            tokenizer,
            prefix,
            tools=tools if tools else None,
            add_generation_prompt=True,
            chat_template_kwargs=chat_template_kwargs,
        )
        tool_names = _reference_tool_names(reference)
        prompts.append(prompt)
        reference_contents.append(reference.get("content") or "")
        reference_tool_names.append(tool_names)
        reference_has_tool_calls.append(bool(tool_names))

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "reference_content": reference_contents,
            "reference_tool_names": reference_tool_names,
            "reference_has_tool_calls": reference_has_tool_calls,
        }
    )