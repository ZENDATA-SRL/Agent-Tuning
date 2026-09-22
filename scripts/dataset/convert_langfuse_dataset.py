"""Convert the Langfuse observation export to OpenAI chat format.

Input:
    test_dataset.json

Output:
    test_dataset_openai.json
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "test_dataset.json"
OUTPUT_PATH = ROOT / "test_dataset_openai.json"
TOOLS_PATH = ROOT / "examples" / "tools.json"


def decode_json(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def normalize_tool_call(tool_call):
    function = tool_call.get("function", {})
    name = tool_call.get("name") or function.get("name", "")
    arguments = tool_call.get("args", function.get("arguments", {}))
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)

    return {
        "id": tool_call.get("id") or tool_call.get("call_id", ""),
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def normalize_message(message):
    role = {
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
    }.get(message.get("type", message.get("role", "")).lower())
    if role is None:
        role = message.get("role")
    if role not in {"system", "user", "assistant", "tool"}:
        return None

    normalized = {
        "role": role,
        "content": message.get("content", "") or "",
    }

    if role == "assistant" and message.get("tool_calls"):
        normalized["tool_calls"] = [
            normalize_tool_call(tool_call)
            for tool_call in message["tool_calls"]
        ]

    if role == "tool":
        normalized["tool_call_id"] = message.get(
            "tool_call_id", message.get("id", "")
        )
        if message.get("name"):
            normalized["name"] = message["name"]

    return normalized


def choose_message_source(trace):
    candidates = []
    for observation in trace.get("observations", []):
        output = decode_json(observation.get("output"))
        if isinstance(output, dict) and isinstance(output.get("messages"), list):
            candidates.append((observation, output))

    if not candidates:
        return None

    # The root output normally contains the complete final message history.
    return max(
        candidates,
        key=lambda item: (
            bool(item[0].get("is_root")),
            len(item[1]["messages"]),
        ),
    )[1]


def convert_trace(trace, tools):
    source = choose_message_source(trace)
    if source is None:
        return None

    messages = [
        normalized
        for raw_message in source["messages"]
        if (normalized := normalize_message(raw_message)) is not None
    ]

    system_prompt = source.get("system_prompt") or ""
    if system_prompt and not any(
        message["role"] == "system" for message in messages
    ):
        messages.insert(0, {"role": "system", "content": system_prompt})

    return {
        "tools": tools,
        "messages": messages,
    }


def main():
    traces = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    tools = json.loads(TOOLS_PATH.read_text(encoding="utf-8"))
    dataset = [
        converted
        for trace in traces
        if (converted := convert_trace(trace, tools)) is not None
    ]
    OUTPUT_PATH.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Salvate {len(dataset)} tracce in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
