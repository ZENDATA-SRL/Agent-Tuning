"""Runtime environment shared by SFT and GRPO.

Call `configure_env` before Hugging Face or Unsloth imports, and again at
the start of each training entrypoint so values loaded later (for example
from `.env`) are applied before the first download.
"""
from __future__ import annotations

import os

_EMPTY_WANDB_KEYS = (
    "WANDB_TAGS",
    "WANDB_PROJECT",
    "WANDB_NAME",
    "WANDB_ENTITY",
    "WANDB_MODE",
)


def configure_env() -> None:
    """Set training environment variables."""

    # Enable high-performance downloads through Xet.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    # Reduce CUDA allocator fragmentation when VRAM is nearly full.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    for key in _EMPTY_WANDB_KEYS:
        if os.environ.get(key, "").strip() == "":
            os.environ.pop(key, None)
