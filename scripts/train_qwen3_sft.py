"""Launchable entrypoint for Qwen3-8B SFT training.

Edit the `SFTConfig(...)` instantiation below to change dataset, output
directory, or any hyperparameter. Run via the "Train: Qwen3-8B SFT"
configuration in `.vscode/launch.json`, or just `python scripts/train_qwen3_sft.py`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import uuid
# Make the repo root importable when this script is run directly
# (so `from train import ...` resolves without installing the package).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train import SFTConfig, run_sft  # noqa: E402


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    temp_id = str(uuid.uuid4())
    config = SFTConfig(
        dataset_path="data/umore_july_dataset",
        output_dir=f"outputs/sft-qwen3-8b-umore-{temp_id}",
    )
    run_sft(config)


if __name__ == "__main__":
    main()
