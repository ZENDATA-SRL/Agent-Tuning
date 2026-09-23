"""Launchable entrypoint for Qwen3-8B SFT training.

Model, LoRA, sampling and chat-template defaults come from the recipe
`qwen3_8b_unsloth_bnb_4bit`. This script only sets the run: dataset,
output directory, resume checkpoint.
"""
from __future__ import annotations

import sys
import uuid
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
    temp_id = str(uuid.uuid4())
    recipe = load_model("qwen3_8b_unsloth_bnb_4bit")
    config = recipe.sft(
        dataset_path="data/umore_july_dataset",
        output_dir=f"outputs/sft-qwen3-8b-umore-{temp_id}",
        # None per una nuova esecuzione. Il percorso deve puntare a una
        # directory checkpoint.
        resume_from_checkpoint="outputs/sft-qwen3-8b-umore-overfit/checkpoint-120",
    )
    run_sft(config)


if __name__ == "__main__":
    main()
