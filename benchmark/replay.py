from __future__ import annotations

from dataclasses import dataclass, field

from benchmark.inference.base import GenerationResult, InferenceClient, ToolCall


@dataclass
class TurnResult:
    """Outcome of a single evaluated AI turn during replay."""

    turn_index: int
    reference_text: str
    reference_tool_calls: list[ToolCall]
    generated: GenerationResult
    is_final: bool = False   # True when this is the last AI turn (final answer)


@dataclass
class TraceReplayResult:
    """All turn results for a single replayed trace."""

    trace_id: str
    turns: list[TurnResult] = field(default_factory=list)
    reference_final_answer: str = ""
    generated_final_answer: str = ""
    total_turns_reference: int = 0
    total_turns_generated: int = 0
    error: str | None = None


def _extract_reference_tool_calls(msg: dict) -> list[ToolCall]:
    """Parse tool_calls from a reference AI message dict."""
    raw = msg.get("tool_calls", [])
    result = []
    for tc in raw:
        if isinstance(tc, dict):
            name = tc.get("name", tc.get("function", {}).get("name", ""))
            args = tc.get("args", tc.get("function", {}).get("arguments", {}))
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            call_id = tc.get("id", tc.get("call_id", ""))
            result.append(ToolCall(name=name, args=args, call_id=call_id))
    return result


def _normalize_role(role: str) -> str:
    """Normalise LangChain-style role names to OpenAI-style roles."""
    mapping = {
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
        "system": "system",
    }
    return mapping.get(role.lower(), role.lower())


def _build_openai_message(msg: dict) -> dict | None:
    """
    Convert a raw trace message dict to an OpenAI-style message dict
    suitable for passing to an inference client.
    Returns None if the message should be skipped.
    """
    role = _normalize_role(msg.get("role", ""))
    content = msg.get("content", "") or ""

    if role == "assistant":
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            # Format tool calls in OpenAI schema so clients can forward them properly
            formatted_tcs = []
            for tc in tool_calls:
                import json
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


def _is_ai_role(role: str) -> bool:
    return role in ("ai", "assistant")


def _last_ai_index(full_trace: list[dict]) -> int:
    """Return the index of the last AI/assistant message in the trace."""
    for i in range(len(full_trace) - 1, -1, -1):
        if _is_ai_role(full_trace[i].get("role", "").lower()):
            return i
    return -1


class ReplayEngine:
    """
    Replays a single agent trace against an InferenceClient.

    Generation is called only for the two kinds of AI turns that matter:

      1. Tool-calling turns — the AI message contains tool_calls.
         These are the decision points where we evaluate whether the model
         picks the right tool with the right arguments.

      2. The final answer turn — the last AI message in the trace.
         This is always generated regardless of whether it has tool calls,
         so we can measure answer quality.

    All other intermediate text-only AI messages (e.g. clarifying questions,
    availability summaries) are injected verbatim from the reference, keeping
    the conversation trajectory intact without burning inference budget on
    turns that carry no evaluation signal.

    For every generated turn the REFERENCE message is still injected into the
    history afterwards, so the replay stays fully deterministic and offline.
    """

    def __init__(
        self,
        client: InferenceClient,
        tools: list[dict] | None = None,
    ) -> None:
        self._client = client
        self._tools = tools or []

    def replay(self, trace: dict) -> TraceReplayResult:
        trace_id = trace.get("trace_id", "")
        full_trace: list[dict] = trace.get("full_trace", [])
        reference_final_answer: str = trace.get("final_answer", "")

        result = TraceReplayResult(
            trace_id=trace_id,
            reference_final_answer=reference_final_answer,
        )

        final_ai_idx = _last_ai_index(full_trace)

        # Working history — what we feed to the model (OpenAI format)
        history: list[dict] = []
        turn_index = 0
        reference_ai_turns = 0
        generated_ai_turns = 0
        last_generated_text = ""

        try:
            for msg_idx, msg in enumerate(full_trace):
                role = msg.get("role", "").lower()

                if role == "system":
                    history.append({"role": "system", "content": msg.get("content", "")})
                    continue

                if role in ("human", "user"):
                    openai_msg = _build_openai_message(msg)
                    if openai_msg:
                        history.append(openai_msg)
                    continue

                if _is_ai_role(role):
                    ref_msg = _build_openai_message(msg)
                    reference_ai_turns += 1

                    has_tool_calls = bool(msg.get("tool_calls"))
                    is_final = (msg_idx == final_ai_idx)

                    if has_tool_calls or is_final:
                        # Generate: tool-calling turn or final answer
                        generated = self._client.generate(
                            messages=list(history),
                            tools=self._tools if self._tools else None,
                        )
                        generated_ai_turns += 1
                        last_generated_text = generated.text

                        ref_tool_calls = _extract_reference_tool_calls(msg)
                        result.turns.append(TurnResult(
                            turn_index=turn_index,
                            reference_text=msg.get("content", "") or "",
                            reference_tool_calls=ref_tool_calls,
                            generated=generated,
                            is_final=is_final,
                        ))
                        turn_index += 1
                    # Always inject the reference message to keep history on track
                    if ref_msg:
                        history.append(ref_msg)
                    continue

                if role == "tool":
                    openai_msg = _build_openai_message(msg)
                    if openai_msg:
                        history.append(openai_msg)
                    continue

        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)

        result.total_turns_reference = reference_ai_turns
        result.total_turns_generated = generated_ai_turns
        result.generated_final_answer = last_generated_text

        return result
