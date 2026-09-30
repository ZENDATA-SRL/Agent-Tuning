"""Dataset loading and chat-template rendering for agent traces."""
from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from datasets import Dataset, load_dataset

from train.templates import render_chat, split_parallel_tool_calls

# Hold out test and eval per source file, then merge. The split happens
# here, before any per-turn expansion. Combined JSONL lands in data/merged.
_TEST_RATIO = 0.1
_EVAL_RATIO = 0.1
_SPLIT_SEED = 42
_MERGED_ROOT = Path(__file__).resolve().parent.parent / "data" / "merged"
# Provenance for the post-filter log. Stripped again before JSONL export.
_SOURCE_KEY = "_dataset_source"


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
    exported = _without_nulls(row)
    exported.pop(_SOURCE_KEY, None)
    return exported


def write_split_jsonl(
    splits: Mapping[str, list[dict]],
    temp_id: str,
) -> Path:
    """Write one JSONL per split under ``data/merged/<temp_id>/``."""
    if (
        not temp_id
        or temp_id in {".", ".."}
        or "/" in temp_id
        or "\\" in temp_id
    ):
        raise ValueError(f"temp_id non valido: {temp_id!r}")
    out_dir = _MERGED_ROOT / temp_id
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
        if index == 0:
            continue
        if not isinstance(message, dict):
            continue
        if _normalize_role(str(message.get("role", ""))) != "assistant":
            continue
        piece = dict(record)
        piece["messages"] = messages[: index + 1]
        pieces.append(piece)
    return pieces


def _maybe_split_parallel_tool_calls(
    record: dict,
    *,
    support_multi_tool_calls: bool,
) -> dict:
    """Return a copy with parallel tool_calls expanded when the model forbids them."""
    if support_multi_tool_calls:
        return record
    messages = record.get("messages")
    if not messages:
        return record
    expanded = dict(record)
    expanded["messages"] = split_parallel_tool_calls(list(messages))
    return expanded


def load_prepared_splits(
    sources: Sequence[tuple[str, float]],
    *,
    temp_id: str | None = None,
    support_multi_tool_calls: bool = True,
) -> dict[str, list[dict]]:
    """Split each source file, merge, then cut traces so each ends on an assistant turn.

    Each pair is ``(path, fraction of that file)``. Fractions are independent.
    Conversation-level splits are shuffled together and written once as
    ``data/merged/<temp_id>/{train,test,eval}.jsonl`` before per-turn expansion.

    When ``support_multi_tool_calls`` is False, parallel tool_calls are split
    into sequential single-call turns before the JSONL write and before the
    per-assistant-turn expansion, so each call becomes its own training piece.

    Args:
        sources: Dataset paths and the fraction of each file to keep.
        temp_id: Subfolder of ``data/merged``. A uuid is used when omitted.
        support_multi_tool_calls: When False, expand parallel tool calls first.

    Returns:
        Train, test, and eval traces. Each row keeps ``_dataset_source``.
    """
    if not sources:
        raise ValueError("datasets non può essere vuoto.")
    merged: dict[str, list[dict]] = {"eval": [], "test": [], "train": []}
    for path, fraction in sources:
        rows = [
            _maybe_split_parallel_tool_calls(
                dict(row),
                support_multi_tool_calls=support_multi_tool_calls,
            )
            for row in load_json_dataset(path)
        ]
        for row in rows:
            row[_SOURCE_KEY] = path
        parts = split_records(rows, dataset_fraction=fraction)
        for name in merged:
            merged[name].extend(parts[name])
    for name in merged:
        random.Random(_SPLIT_SEED).shuffle(merged[name])
    split_dir = write_split_jsonl(merged, temp_id or str(uuid.uuid4()))
    print(f"Wrote split files to {split_dir}")
    return {
        name: [
            piece
            for row in split
            for piece in _traces_ending_with_assistant(row)
        ]
        for name, split in merged.items()
    }


def _format_tool_text(cells: Mapping[str, Mapping[str, int]]) -> str:
    """Render tool/text counts for train, eval, and test on one line."""
    return "\n" + "  ".join(
        f"{cells[split]['tool']:7d} {cells[split]['text']:7d}"
        for split in ("train", "eval", "test")
    ) + "\n"


def log_source_counts(
    prefix: str,
    sources: Sequence[str],
    *,
    train: Dataset,
    eval_dataset: Dataset | None,
    test: Sequence[Mapping[str, Any]],
) -> None:
    """Print how many post-filter traces from each source land in each split.

    Counts are assistant-turn traces after ``tool_trace_fraction`` and
    ``max_traces``. ``tool`` is a turn whose last message has tool calls;
    ``text`` is a normal assistant message. The JSONL export is still one
    row per conversation, so its line counts differ from this table. Test
    is not passed through those two filters.

    Args:
        prefix: Log tag, ``sft`` or ``grpo``.
        sources: Dataset paths, in the order they should appear.
        train: Train split after filters.
        eval_dataset: Eval split after filters, if any.
        test: Test traces. Tool-mix and ``max_traces`` do not apply.
    """
    splits = ("train", "eval", "test")
    order: list[str] = []
    seen: set[str] = set()
    for src in sources:
        if src not in seen:
            order.append(src)
            seen.add(src)
    empty = {split: {"tool": 0, "text": 0} for split in splits}
    counts: dict[str, dict[str, dict[str, int]]] = {
        src: {split: dict(cells) for split, cells in empty.items()}
        for src in order
    }

    def _add(split: str, rows: Dataset | Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            src = str(row[_SOURCE_KEY])
            if src not in counts:
                counts[src] = {name: {"tool": 0, "text": 0} for name in splits}
                order.append(src)
            kind = "tool" if _ends_with_tool_call(dict(row)) else "text"
            counts[src][split][kind] += 1

    _add("train", train)
    if eval_dataset is not None:
        _add("eval", eval_dataset)
    _add("test", test)

    label = "dataset"
    width = max(len(label), *(len(src) for src in order))
    pair = f"{'tool':>7} {'text':>7}"
    print(f"[{prefix}] traces after filters")
    print(
        f"{'':<{width}}  "
        + "  ".join(f"{name:^{len(pair)}}" for name in splits)
    )
    print(f"{label:<{width}}  " + "  ".join(pair for _ in splits))
    totals = {split: {"tool": 0, "text": 0} for split in splits}
    for src in order:
        row = counts[src]
        print(f"{src:<{width}}  " + _format_tool_text(row))
        for split in splits:
            totals[split]["tool"] += row[split]["tool"]
            totals[split]["text"] += row[split]["text"]
    print(f"{'total':<{width}}  " + _format_tool_text(totals))


def limit_traces(dataset: Dataset, max_traces: int | None) -> Dataset:
    """Keep at most `max_traces` rows. `None` leaves the dataset unchanged."""
    if max_traces is None:
        return dataset
    if max_traces < 1:
        raise ValueError(f"max_traces deve essere >= 1, ricevuto {max_traces}.")
    if max_traces >= len(dataset):
        return dataset
    return dataset.select(range(max_traces))


def _ends_with_tool_call(record: dict) -> bool:
    """True if the last message is an assistant turn with at least one tool call."""
    messages = record.get("messages") or []
    if not messages:
        return False
    last = messages[-1]
    if not isinstance(last, dict):
        return False
    if _normalize_role(str(last.get("role", ""))) != "assistant":
        return False
    tool_calls = last.get("tool_calls") or []
    return isinstance(tool_calls, list) and len(tool_calls) > 0


def filter_tool_call_traces(dataset: Dataset) -> Dataset:
    """Keep only traces whose last message is an assistant tool call."""
    keep = [
        index
        for index, row in enumerate(dataset)
        if _ends_with_tool_call(dict(row))
    ]
    if not keep:
        return dataset.select([])
    return dataset.select(keep)


def mix_tool_trace_fraction(
    dataset: Dataset,
    tool_trace_fraction: float,
    *,
    seed: int,
) -> Dataset:
    """Resample so tool-ending traces are ``tool_trace_fraction`` of the result.

    ``tool_trace_fraction`` must be in ``[0.0, 1.0]``:
    - ``1.0`` → only traces ending in an assistant tool call
    - ``0.0`` → only traces ending in a text assistant turn
    - in between → largest subset with that tool / non-tool ratio

    Sampling is seeded. The returned row order is shuffled.
    """
    if not 0.0 <= tool_trace_fraction <= 1.0:
        raise ValueError(
            "tool_trace_fraction deve essere in [0.0, 1.0], "
            f"ricevuto {tool_trace_fraction}."
        )

    tool_idx: list[int] = []
    non_tool_idx: list[int] = []
    for index, row in enumerate(dataset):
        if _ends_with_tool_call(dict(row)):
            tool_idx.append(index)
        else:
            non_tool_idx.append(index)

    rng = random.Random(seed)
    rng.shuffle(tool_idx)
    rng.shuffle(non_tool_idx)

    n_tool_avail = len(tool_idx)
    n_non_avail = len(non_tool_idx)

    if tool_trace_fraction >= 1.0:
        keep = tool_idx
    elif tool_trace_fraction <= 0.0:
        keep = non_tool_idx
    else:
        # Largest T with floor(f*T) tools and T - that many non-tools available.
        max_total = min(
            int(n_tool_avail / tool_trace_fraction),
            int(n_non_avail / (1.0 - tool_trace_fraction)),
        )
        if max_total < 1:
            raise ValueError(
                "Impossibile raggiungere tool_trace_fraction="
                f"{tool_trace_fraction:g}: disponibili "
                f"{n_tool_avail} tool e {n_non_avail} non-tool."
            )
        n_tool = int(round(tool_trace_fraction * max_total))
        n_tool = min(max(n_tool, 0), n_tool_avail, max_total)
        n_non = min(max_total - n_tool, n_non_avail)
        # Prefer hitting the ratio over leaving one side empty when both exist.
        if n_tool == 0 and n_tool_avail > 0 and tool_trace_fraction > 0:
            n_tool = 1
            n_non = min(max_total - 1, n_non_avail)
        if n_non == 0 and n_non_avail > 0 and tool_trace_fraction < 1:
            n_non = 1
            n_tool = min(max_total - 1, n_tool_avail)
        keep = tool_idx[:n_tool] + non_tool_idx[:n_non]
        rng.shuffle(keep)

    if not keep:
        return dataset.select([])
    return dataset.select(keep)


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
    support_multi_tool_calls: bool = True,
) -> Dataset:
    """Render each record into a single training string (the "text" column)."""
    if len(records) == 0:
        raise ValueError("format_dataset received an empty dataset.")

    texts: list[str] = []
    for rec in records:
        messages = trace_to_messages(rec)
        if not support_multi_tool_calls:
            messages = split_parallel_tool_calls(messages)
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


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """Normalise OpenAI-style tool arguments to a dict."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _reference_tool_calls(assistant_msg: dict) -> list[dict[str, Any]]:
    """Return ``[{name, arguments}, ...]`` from the last assistant turn."""
    tool_calls = assistant_msg.get("tool_calls") or []
    parsed: list[dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        parsed.append(
            {
                "name": str(name),
                "arguments": _parse_tool_arguments(fn.get("arguments")),
            }
        )
    return parsed


def format_grpo_dataset(
    records: Dataset | list[dict],
    tokenizer: Any,
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    support_multi_tool_calls: bool = True,
) -> Dataset:
    """Build a GRPO dataset with a rendered `prompt` plus reference columns.

    For each trace, the prompt is every message before the last assistant
    turn, rendered with the chat template (tools included) and a generation
    prompt so the model continues as assistant. Reference fields are kept for
    reward functions (TRL forwards every column except `prompt`).

    ``reference_tool_arguments`` is stored as a JSON string per row so Arrow
    does not need a uniform nested schema across heterogeneous tool APIs.
    """
    if len(records) == 0:
        raise ValueError("format_grpo_dataset received an empty dataset.")

    prompts: list[str] = []
    reference_contents: list[str] = []
    reference_tool_names: list[list[str]] = []
    reference_tool_arguments: list[str] = []
    reference_has_tool_calls: list[bool] = []

    for rec in records:
        messages = trace_to_messages(rec)
        if not support_multi_tool_calls:
            messages = split_parallel_tool_calls(messages)
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
        ref_calls = _reference_tool_calls(reference)
        tool_names = [call["name"] for call in ref_calls]
        tool_args = [call["arguments"] for call in ref_calls]
        prompts.append(prompt)
        reference_contents.append(reference.get("content") or "")
        reference_tool_names.append(tool_names)
        reference_tool_arguments.append(json.dumps(tool_args, ensure_ascii=False))
        reference_has_tool_calls.append(bool(tool_names))

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "reference_content": reference_contents,
            "reference_tool_names": reference_tool_names,
            "reference_tool_arguments": reference_tool_arguments,
            "reference_has_tool_calls": reference_has_tool_calls,
        }
    )