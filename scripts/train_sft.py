"""Launchable entrypoint for SFT training.

Edit the constants below, then run this file. No CLI flags.
Supported model ids are discovered from ``shared/configs/``
(see ``shared.configs.available_models()``).
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
from train import run_sft  # noqa: E402


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
    ("data/apigenmt5k/dataset.json", 0.12),
    ("data/umore/umore_july.json", 1.00),
)
OUTPUT_DIR = "outputs/sft-qwen3-8b-apigenmt5k-umore"

SHUFFLE = True
# MAX_TRACES = None  # optional hard cap after shuffle
# Share of traces ending in a tool call: 0.0 / 1.0 / in-between, or None = natural mix.
TOOL_TRACE_FRACTION: float | None = 0.6

# "thinking" or "no_thinking". The recipe must define that key.
DECODING_MODE = "no_thinking"

# Share of the shortest train traces to keep when retrying the current batch
# size, before that size is lowered again. 1.0 drops nothing. Range [0.0, 1.0].
BATCH_KEEP_PERCENTILE = 0.85

# Optional resume path, or None for a fresh run.
RESUME_FROM_CHECKPOINT: str | None = None


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    recipe = load_model(MODEL_ID)
    config = recipe.sft(
        datasets=DATASETS,
        output_dir=OUTPUT_DIR,
        shuffle=SHUFFLE,
        tool_trace_fraction=TOOL_TRACE_FRACTION,
        decoding_mode=DECODING_MODE,
        resume_from_checkpoint=RESUME_FROM_CHECKPOINT,
        batch_keep_percentile=BATCH_KEEP_PERCENTILE,
        per_device_train_batch_size=4,
        # Override recipe defaults for this run if needed:
        # lora_dropout=0.05,
        # learning_rate=1e-4,
        # save_steps=20,
        # eval_steps=20,
        # report_to="none",
    )
    run_sft(config)


if __name__ == "__main__":
    main()
