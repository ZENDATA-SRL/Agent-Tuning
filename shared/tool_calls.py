"""Shared helpers for OpenAI-style tool-calling message lists.

Used by both training (chat-template rendering) and benchmarking (trace
replay). Keep this module free of heavy ML dependencies.
"""
from __future__ import annotations


def _is_assistant_role(role: str) -> bool:
    return role.lower() in {"assistant", "ai"}


def _is_tool_role(role: str) -> bool:
    return role.lower() == "tool"


def split_parallel_tool_calls(messages: list[dict]) -> list[dict]:
    """Expand parallel tool_calls into sequential single-call assistant turns.

    OpenAI traces may pack several calls in one assistant message, followed by
    one ``role:"tool"`` result per call. Templates that only allow a single
    call (Llama 3.1) need that block rewritten as::

        assistant[tc1] → tool[r1] → assistant[tc2] → tool[r2] → …

    The original assistant ``content`` is kept only on the first split turn.
    Tool results are matched by ``tool_call_id``; unmatched tool messages that
    followed the original assistant are appended after the expanded block.
    Idempotent when every assistant already has at most one tool call.
    """
    output: list[dict] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict):
            output.append(message)
            index += 1
            continue

        role = str(message.get("role", ""))
        tool_calls = message.get("tool_calls")
        if (
            not _is_assistant_role(role)
            or not isinstance(tool_calls, list)
            or len(tool_calls) <= 1
        ):
            output.append(message)
            index += 1
            continue

        tool_by_id: dict[str, dict] = {}
        cursor = index + 1
        while cursor < len(messages):
            follower = messages[cursor]
            if not isinstance(follower, dict) or not _is_tool_role(
                str(follower.get("role", ""))
            ):
                break
            tool_call_id = follower.get("tool_call_id")
            if tool_call_id is not None:
                tool_by_id[str(tool_call_id)] = follower
            cursor += 1

        content = message.get("content") or ""
        matched_ids: set[str] = set()
        for call_index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            output.append(
                {
                    **{key: value for key, value in message.items() if key != "tool_calls"},
                    "content": content if call_index == 0 else "",
                    "tool_calls": [tool_call],
                }
            )
            tool_call_id = tool_call.get("id")
            if tool_call_id is None:
                continue
            tool_call_id = str(tool_call_id)
            tool_message = tool_by_id.get(tool_call_id)
            if tool_message is not None:
                output.append(tool_message)
                matched_ids.add(tool_call_id)

        for follower_index in range(index + 1, cursor):
            follower = messages[follower_index]
            follower_id = (
                follower.get("tool_call_id") if isinstance(follower, dict) else None
            )
            # Already emitted matched results next to their call; keep the rest.
            if follower_id is None or str(follower_id) not in matched_ids:
                output.append(follower)

        index = cursor

    return output
