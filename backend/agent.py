from __future__ import annotations

import json
import re
import uuid
from typing import Any

from backend.model import generate
from backend.prompts import SYSTEM_PROMPT
from backend.tools import TOOLS, run_tool

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def _strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[-1].lstrip("\n")
    return text


def _loads_args(args: Any) -> dict:
    if isinstance(args, dict):
        return args
    if isinstance(args, str) and args.strip():
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = payload.get("name")
        args = _loads_args(payload.get("arguments", {}))
        if not name:
            continue
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
            "name": name,
            "args": args,
        })
    return calls


def _content_without_tool_calls(text: str) -> str:
    cleaned = _TOOL_CALL_RE.sub("", text).strip()
    cleaned = cleaned.replace("<|im_end|>", "").strip()
    return cleaned


def _render_prompt(tokenizer, messages: list[dict]) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "tools": TOOLS,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def run_agent(
    adapter_id: str,
    history: list[dict],
    *,
    max_steps: int = 6,
    max_new_tokens: int = 512,
) -> dict:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        role = msg.get("role")
        if role not in ("user", "assistant", "tool"):
            continue
        entry: dict[str, Any] = {"role": role, "content": msg.get("content") or ""}
        if role == "assistant" and msg.get("tool_calls"):
            entry["tool_calls"] = msg["tool_calls"]
        if role == "tool":
            entry["tool_call_id"] = msg.get("tool_call_id", "")
            if msg.get("name"):
                entry["name"] = msg["name"]
        messages.append(entry)

    from backend.model import ensure_loaded

    _, tokenizer = ensure_loaded(adapter_id)
    trace: list[dict] = []
    final_text = ""

    for _ in range(max_steps):
        prompt = _render_prompt(tokenizer, messages)
        if "<tools>" not in prompt:
            print("[agent] WARNING: rendered prompt has no <tools> block")
        raw = generate(adapter_id, prompt, max_new_tokens=max_new_tokens)
        text = _strip_think(raw)
        tool_calls = _parse_tool_calls(text)
        content = _content_without_tool_calls(text)
        if not tool_calls and "<tool_call>" in text:
            print("[agent] WARNING: saw <tool_call> but parse failed:", repr(text[:400]))
        elif not tool_calls:
            print("[agent] no tool_calls; raw=", repr(raw[:300]))

        if tool_calls:
            assistant_msg = {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                tc["args"]
                                if isinstance(tc["args"], str)
                                else json.dumps(tc["args"], ensure_ascii=False)
                            ),
                        },
                    }
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_msg)
            trace.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {"id": tc["id"], "name": tc["name"], "args": tc["args"]}
                    for tc in tool_calls
                ],
                "raw_completion": raw,
            })
            for tc in tool_calls:
                result = run_tool(tc["name"], tc["args"])
                tool_msg = {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                }
                messages.append(tool_msg)
                trace.append(tool_msg)
            continue

        final_text = content
        assistant_msg = {
            "role": "assistant",
            "content": content,
            "raw_completion": raw,
        }
        messages.append({"role": "assistant", "content": content})
        trace.append(assistant_msg)
        break
    else:
        final_text = final_text or "(max tool steps reached)"

    return {
        "reply": final_text,
        "trace": trace,
    }
