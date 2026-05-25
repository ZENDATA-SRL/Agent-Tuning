from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from train.templates import ChatTemplateAdapter


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL dataset into a list of dicts.

    Mirrors `_load_dataset` in `benchmark/runner.py` so that the SFT and
    benchmark pipelines stay in sync on dataset ingestion semantics.
    """
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


_ROLE_MAPPING = {
    "human": "user",
    "ai": "assistant",
    "tool": "tool",
    "system": "system",
}


def _normalize_role(role: str) -> str:
    return _ROLE_MAPPING.get(role.lower(), role.lower())


def _build_openai_message(msg: dict) -> dict | None:
    """Convert a raw trace message into an OpenAI-canonical message dict.

    Same shape produced by `benchmark.replay._build_openai_message`, kept
    in sync but duplicated here to avoid pulling the benchmark module (and
    its langchain dependency tree) into the training entrypoint.
    """
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
    """Project a dataset record into a list of OpenAI-canonical messages.

    Skips entries that don't map to a valid role. The returned list is
    suitable as input to `tokenizer.apply_chat_template(...)`.
    """
    full_trace: list[dict] = record.get("full_trace", [])
    messages: list[dict] = []
    for msg in full_trace:
        m = _build_openai_message(msg)
        if m is not None:
            messages.append(m)
    return messages


def format_dataset(
    records: list[dict],
    adapter: ChatTemplateAdapter,
) -> tuple[Dataset, list[str]]:
    """Serialize each record into the model-specific wire format.

    Delegates the actual rendering to `adapter`, which encapsulates all
    model-family-specific knowledge (special tokens, tool-call syntax,
    whether to keep the native template or swap it for an Unsloth one).

    Returns a ``(Dataset, raw_texts)`` pair so callers that need the raw
    rendered strings (e.g. for sample-logging callbacks) can access them
    without re-running the template.
    """
    if not records:
        raise ValueError("format_dataset received an empty record list.")

    texts: list[str] = []
    for rec in records:
        messages = trace_to_messages(rec)
        tools = rec.get("tools") or []
        text = adapter.apply_template(
            messages,
            tools=tools if tools else None,
            add_generation_prompt=False,
        )
        texts.append(text)

    return Dataset.from_dict({"text": texts}), texts


# ---------------------------------------------------------------------------
# Selective loss masking
# ---------------------------------------------------------------------------

def _find_all(text: str, marker: str) -> list[int]:
    """Return the start offsets of all non-overlapping occurrences of *marker*."""
    offsets: list[int] = []
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx == -1:
            break
        offsets.append(idx)
        start = idx + len(marker)
    return offsets


def mask_labels_selective(
    tokenizer: Any,
    text: str,
    adapter: ChatTemplateAdapter,
    *,
    include_final_response: bool = False,
) -> list[int]:
    """Build a label tensor that covers only tool-call spans (and optionally
    the last assistant turn) in *text*.

    The function works at the **character level**:

    1. Collect all ``[start, end)`` character spans that should contribute to
       the loss:
         a. every ``<|tool_call>…<tool_call|>`` block in any assistant turn;
         b. if *include_final_response* is True, also the character span of
            the **last** assistant turn (from ``<|turn>model\\n`` to the
            closing ``<turn|>\\n``).

    2. Tokenise *text* with ``return_offsets_mapping=True`` to get the
       character→token mapping.

    3. For each token, emit ``token_id`` if its character span overlaps at
       least one of the selected regions, else emit ``-100``.

    Returns a list of integers ready to be used as ``labels`` in a TRL/HF
    training sample.
    """
    # ── Step 1: collect character spans that should be loss-bearing ──────────
    keep_spans: list[tuple[int, int]] = []  # (char_start, char_end) inclusive-exclusive

    # (a) all tool-call blocks
    tc_starts = _find_all(text, adapter.tool_call_start)
    tc_end_marker = adapter.tool_call_end
    for tc_start in tc_starts:
        tc_end = text.find(tc_end_marker, tc_start)
        if tc_end == -1:
            # Malformed template output — skip gracefully
            continue
        span_end = tc_end + len(tc_end_marker)
        keep_spans.append((tc_start, span_end))

    # (b) optionally the last assistant turn
    if include_final_response:
        resp_marker = adapter.response_part  # e.g. "<|turn>model\n"
        turn_end_marker = adapter.turn_end   # e.g. "<turn|>\n"
        # Find the *last* occurrence of the response marker
        last_resp = text.rfind(resp_marker)
        if last_resp != -1:
            turn_content_start = last_resp + len(resp_marker)
            turn_end_pos = text.find(turn_end_marker, turn_content_start)
            if turn_end_pos == -1:
                # Turn goes to end of string (no closing marker)
                turn_content_end = len(text)
            else:
                turn_content_end = turn_end_pos + len(turn_end_marker)
            keep_spans.append((turn_content_start, turn_content_end))

    # ── Step 2: tokenise with offset mapping ─────────────────────────────────
    # Gemma4 wraps the fast tokenizer inside a multimodal Processor.
    # The outer Processor doesn't support `return_offsets_mapping` and raises
    # TypeError when called with a positional string (images-first dispatch).
    # Unwrap to the inner fast tokenizer when available; it supports both
    # `text=` as keyword and `return_offsets_mapping`.
    _tok = getattr(tokenizer, "tokenizer", tokenizer)
    encoding = _tok(
        text=text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids: list[int] = encoding["input_ids"]
    offset_mapping: list[tuple[int, int]] = encoding["offset_mapping"]

    # ── Step 3: build labels ──────────────────────────────────────────────────
    labels: list[int] = []
    for token_id, (tok_start, tok_end) in zip(input_ids, offset_mapping):
        # Check overlap: a token overlaps a span if tok_start < span_end AND tok_end > span_start
        in_span = any(
            tok_start < span_end and tok_end > span_start
            for span_start, span_end in keep_spans
        )
        labels.append(token_id if in_span else -100)

    return labels
