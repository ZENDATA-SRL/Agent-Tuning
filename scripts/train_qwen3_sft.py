"""Launchable entrypoint for Qwen3-8B SFT training."""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when this script is run directly
# (so `from train import ...` resolves without installing the package).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train import load_model, run_sft  # noqa: E402


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    recipe = load_model("qwen3_8b_unsloth_bnb_4bit")
    config = recipe.sft(
        dataset_path="data/umore/umore_openai.json",
        output_dir="outputs/sft-qwen3-8b-umore",
        shuffle=True,
        dataset_fraction=0.50, # use only 50% of the dataset to speed up training
        lora_dropout=0.05,

        # Override default params of the model
        # learning_rate=1e-4,
        # report_to="none", # if you want to disable reporting to wandb
        # resume_from_checkpoint="outputs/sft-qwen3-8b-umore-overfit/checkpoint-120",
    )
    run_sft(config)


if __name__ == "__main__":
    main()
