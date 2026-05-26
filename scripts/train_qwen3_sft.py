"""Launchable entrypoint for Qwen3-8B SFT training.

Edit the `SFTConfig(...)` instantiation below to change dataset, output
directory, or any hyperparameter. Run via the "Train: Qwen3-8B SFT"
configuration in `.vscode/launch.json`, or just `python scripts/train_qwen3_sft.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when this script is run directly
# (so `from train import ...` resolves without installing the package).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train import SFTConfig, run_sft  # noqa: E402


def main() -> None:
    config = SFTConfig(
        dataset_path="examples/traces-2026-05-18.jsonl",
        output_dir="outputs/sft-qwen3-8b-v1",
    )
    run_sft(config)


if __name__ == "__main__":
    main()
