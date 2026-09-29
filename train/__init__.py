"""Training loops for Agent-Tuning (SFT / GRPO)."""
from __future__ import annotations

from typing import Any

__all__ = [
    "run_grpo",
    "run_sft",
]


def __getattr__(name: str) -> Any:
    """Lazy-load training loops so importing the package stays light."""
    if name == "run_sft":
        from train.sft import run_sft

        return run_sft
    if name == "run_grpo":
        from train.grpo import run_grpo

        return run_grpo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
