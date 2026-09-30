"""Launchable entrypoint for a vLLM-served benchmark.

Edit the constants below, then run this file. No CLI flags.
Serving knobs (tool-call parser, hf_overrides, …) come from the matching
``shared/configs`` recipe for ``MODEL_ID``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.vllm_server import run_served_benchmark  # noqa: E402


# ---------------------------------------------------------------------------
# Run knobs — change these, then launch the script.
# ---------------------------------------------------------------------------

# Recipe id / Hugging Face checkpoint passed to ``vllm serve``.
# Examples:
#   "unsloth/Qwen3-8B-unsloth-bnb-4bit"
#   "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
#   "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
#   "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit"
MODEL_ID = "unsloth/Qwen3-8B-unsloth-bnb-4bit"

DATASET_PATH = "data/umore/f818d980-45c4-4adc-9140-b491c7d6aa56/test.jsonl"

# Set both for a LoRA run; leave both None for the base checkpoint.
LORA_PATH: str | None = "outputs/sft-qwen3-8b-umore-july/checkpoint-320"
LORA_NAME: str | None = "qwen-umore"

# "thinking" or "no_thinking". The recipe must define that key.
DECODING_MODE = "no_thinking"

# Optional smoke-test cap. None = full dataset.
MAX_TRACES: int | None = None


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    run_served_benchmark(
        base_model=MODEL_ID,
        dataset_path=DATASET_PATH,
        lora_name=LORA_NAME,
        lora_path=LORA_PATH,
        max_traces=MAX_TRACES,
        decoding_mode=DECODING_MODE,
    )


if __name__ == "__main__":
    main()
