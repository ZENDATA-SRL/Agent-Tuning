"""Dataset loading + chat-template rendering for Qwen3 SFT."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from train.templates import render_chat


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL dataset into a list of dicts."""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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
    full_trace: list[dict] = record.get("full_trace", [])
    messages: list[dict] = []
    for msg in full_trace:
        m = _build_openai_message(msg)
        if m is not None:
            messages.append(m)
    return messages


def format_dataset(
    records: list[dict],
    tokenizer: Any,
) -> Dataset:
    """Render each record into a single training string (the "text" column)."""
    if not records:
        raise ValueError("format_dataset received an empty record list.")

    texts: list[str] = []
    for rec in records:
        messages = trace_to_messages(rec)
        tools = rec.get("tools") or []
        text = render_chat(
            tokenizer,
            messages,
            tools=tools if tools else None,
            add_generation_prompt=False,
        )
        texts.append(text)

    return Dataset.from_dict({"text": texts})
