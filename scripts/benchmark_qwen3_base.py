"""Start vLLM with the base Qwen3 checkpoint and benchmark it."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.vllm_server import run_served_benchmark  # noqa: E402


_BASE_MODEL = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
_DATASET = "data/umore/f818d980-45c4-4adc-9140-b491c7d6aa56/test.jsonl"


def main() -> None:
    """Serve the base Qwen3 model and replay the same traces as the LoRA run."""
    from dotenv import load_dotenv

    load_dotenv()
    run_served_benchmark(
        base_model=_BASE_MODEL,
        dataset_path=_DATASET,
        tool_call_parser="hermes",
    )


if __name__ == "__main__":
    main()
