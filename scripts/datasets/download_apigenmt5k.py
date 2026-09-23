"""Download the ApigenMT5k dataset from HuggingFace."""

import json
import os
import random

from datasets import load_dataset

# APIGen-MT-5k is published as a single train split. Hold out test and eval
# so we write three JSON array files HuggingFace `datasets` can load natively.
_SPLIT_NAMES = ("train", "test", "eval")
_TEST_RATIO = 0.1
_EVAL_RATIO = 0.1
_SPLIT_SEED = 42
_OUTPUT_DIR = "data/apigenmt5k"

# ShareGPT-style turns in APIGen-MT-5k -> OpenAI chat roles.
# function_call / observation are not roles: they become an assistant
# tool_calls message and the matching role:"tool" result.
_ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
}


def _arguments_as_json(arguments) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False)


def _parse_function_call(value: str | dict) -> dict:
    payload = json.loads(value) if isinstance(value, str) else value
    return {
        "name": payload["name"],
        "arguments": _arguments_as_json(payload.get("arguments", {})),
    }


def _openai_tools(tools: str | list) -> list[dict]:
    """Wrap APIGen `{name, description, parameters}` entries as OpenAI tools."""
    if isinstance(tools, str):
        tools = json.loads(tools)
    formatted = []
    for tool in tools:
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            formatted.append(tool)
            continue
        formatted.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            },
        })
    return formatted


def conversations_to_messages(record: dict) -> list[dict]:
    """Convert one APIGen-MT conversation into OpenAI chat messages.

    `human` and `gpt` become `user` and `assistant`. Each `function_call`
    becomes its own assistant message with a single tool call (the policy
    forbids answering the user in the same turn), and the following
    `observation` becomes the `role:"tool"` result bound to that call id.
    """
    messages: list[dict] = []
    system = record.get("system")
    if system:
        messages.append({"role": "system", "content": system})

    pending_calls: list[tuple[str, str]] = []
    call_index = 0

    for turn in record["conversations"]:
        speaker = turn["from"]
        value = turn.get("value") or ""

        if speaker == "function_call":
            call = _parse_function_call(value)
            call_index += 1
            call_id = f"call_{call_index}"
            pending_calls.append((call_id, call["name"]))
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }],
            })
            continue

        if speaker == "observation":
            if not pending_calls:
                raise ValueError("observation without a preceding function_call")
            call_id, name = pending_calls.pop(0)
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": value,
            })
            continue

        role = _ROLE_MAPPING.get(speaker)
        if role is None:
            raise ValueError(f"unknown conversation role: {speaker!r}")
        messages.append({"role": role, "content": value})

    if pending_calls:
        raise ValueError("function_call without a following observation")
    return messages


def convert_record(record: dict) -> dict:
    return {
        "messages": conversations_to_messages(record),
        "tools": _openai_tools(record["tools"]),
    }


def _split_records(records: list[dict]) -> dict[str, list[dict]]:
    """Shuffle once and cut eval, then test, then train."""
    rows = list(records)
    random.Random(_SPLIT_SEED).shuffle(rows)
    n_eval = int(len(rows) * _EVAL_RATIO)
    n_test = int(len(rows) * _TEST_RATIO)
    return {
        "eval": rows[:n_eval],
        "test": rows[n_eval:n_eval + n_test],
        "train": rows[n_eval + n_test:],
    }


def download_apigenmt5k() -> None:
    dataset = load_dataset("Salesforce/APIGen-MT-5k")
    splits = {
        split: [convert_record(record) for record in dataset[split]]
        for split in _SPLIT_NAMES
        if split in dataset
    }
    if set(splits) == {"train"}:
        splits = _split_records(splits["train"])

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    for split in _SPLIT_NAMES:
        if split not in splits:
            continue
        path = os.path.join(_OUTPUT_DIR, f"{split}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(splits[split], f, ensure_ascii=False)


if __name__ == "__main__":
    download_apigenmt5k()
