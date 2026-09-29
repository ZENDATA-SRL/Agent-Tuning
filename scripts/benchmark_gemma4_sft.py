"""Start vLLM with the Gemma4 LoRA and benchmark it."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.vllm_server import gemma4_bnb_hf_overrides, run_served_benchmark  # noqa: E402


_BASE_MODEL = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
_DATASET = "data/umore/f818d980-45c4-4adc-9140-b491c7d6aa56/test.jsonl"
_LORA = (
    "outputs/sft-gemma4-e2b-umore-f818d980-45c4-4adc-9140-b491c7d6aa56/checkpoint-120"
)


def main() -> None:
    """Serve ``gemma4-umore`` and replay the held-out traces."""
    from dotenv import load_dotenv

    load_dotenv()
    run_served_benchmark(
        base_model=_BASE_MODEL,
        dataset_path=_DATASET,
        tool_call_parser="gemma4",
        lora_name="gemma4-umore",
        lora_path=_LORA,
        hf_overrides=gemma4_bnb_hf_overrides(_BASE_MODEL),
    )


if __name__ == "__main__":
    main()
