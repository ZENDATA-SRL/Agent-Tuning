"""Launchable entrypoint for GRPO training.

Edit the constants below, then run this file. No CLI flags.
Supported model ids are discovered from ``shared/configs/``
(see ``shared.configs.available_models()``).

Tool-call rewards use the recipe's ``chat.tool_calls`` format
(defined per model family under ``shared/configs/``).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when this script is run directly
# (so `from shared` / `from train` resolve without installing the package).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.configs import load_model  # noqa: E402
from train import run_grpo  # noqa: E402
from train.rewards import resolve_reward_funcs  # noqa: E402


# ---------------------------------------------------------------------------
# Run knobs — change these, then launch the script.
# ---------------------------------------------------------------------------

# Recipe id from shared/configs (examples):
#   "unsloth/Qwen3-8B-unsloth-bnb-4bit"
#   "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
#   "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
#   "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit"
MODEL_ID = "unsloth/Qwen3-8B-unsloth-bnb-4bit"

# (path, fraction of that file). Fractions are independent.
DATASETS = (
    ("data/apigenmt5k/dataset.json", 0.20),
)
OUTPUT_DIR = "outputs/grpo-qwen3-8b-umore"

SHUFFLE = True
# MAX_TRACES = None  # optional hard cap after shuffle
# Share of traces ending in a tool call: 0.0 / 1.0 / in-between, or None = natural mix.
TOOL_TRACE_FRACTION: float | None = 0.75

# Reward policy: names from `train.rewards.REWARD_REGISTRY`.
# Order matches `REWARD_WEIGHTS` below.
REWARD_NAMES = (
    "tool_call_format",
    "tool_name_match",
    "tool_args_match",
    "answer_cosine",
)
REWARD_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# "thinking" or "no_thinking". The recipe must define that key.
# GRPO applies the recipe's generation_override on top of this mode.
DECODING_MODE = "no_thinking"

# Optional path to an SFT LoRA directory (`adapter_config.json` + weights)
# for this MODEL_ID, e.g. `outputs/sft-.../best_eval_model`.
# None → attach a fresh LoRA on the base recipe checkpoint.
LORA_ADAPTER_PATH: str | None = "outputs/sft-qwen3-8b-umore-july/checkpoint-320"

# Optional resume path, or None for a fresh run.
RESUME_FROM_CHECKPOINT: str | None = None


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    recipe = load_model(MODEL_ID)
    config = recipe.grpo(
        datasets=DATASETS,
        output_dir=OUTPUT_DIR,
        lora_adapter_path=LORA_ADAPTER_PATH,
        shuffle=SHUFFLE,
        tool_trace_fraction=TOOL_TRACE_FRACTION,
        reward_weights=list(REWARD_WEIGHTS),
        decoding_mode=DECODING_MODE,
        resume_from_checkpoint=RESUME_FROM_CHECKPOINT,
        # Override recipe defaults for this run if needed:
        # learning_rate=1e-4,
        # report_to="none",
    )
    run_grpo(
        config,
        reward_funcs=resolve_reward_funcs(
            REWARD_NAMES,
            tool_calls=recipe.chat.tool_calls,
        ),
    )


if __name__ == "__main__":
    main()
