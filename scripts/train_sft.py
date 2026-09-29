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

DATASET_PATH = "data/umore/umore_openai.json"
OUTPUT_DIR = "outputs/sft-qwen3-8b-umore-prova"

SHUFFLE = True
DATASET_FRACTION = 0.50  # 1.0 = full file
# MAX_TRACES = None  # optional hard cap after shuffle
TRAIN_ON_TOOL_CALLS_ONLY = False

# Optional resume path, or None for a fresh run.
RESUME_FROM_CHECKPOINT: str | None = None


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    recipe = load_model(MODEL_ID)
    config = recipe.sft(
        dataset_path=DATASET_PATH,
        output_dir=OUTPUT_DIR,
        shuffle=SHUFFLE,
        dataset_fraction=DATASET_FRACTION,
        train_on_tool_calls_only=TRAIN_ON_TOOL_CALLS_ONLY,
        resume_from_checkpoint=RESUME_FROM_CHECKPOINT,
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
