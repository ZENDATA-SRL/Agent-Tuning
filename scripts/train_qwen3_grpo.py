"""Launchable entrypoint for Qwen3-8B GRPO training."""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when this script is run directly
# (so `from train import ...` resolves without installing the package).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train import load_model, run_grpo  # noqa: E402
from train.rewards import resolve_reward_funcs  # noqa: E402


# Reward policy: names from `train.rewards.REWARD_REGISTRY`.
# Order matches `reward_weights` below when weights are set.
REWARD_NAMES = (
    "tool_call_format",
    "tool_name_match",
    "non_empty",
)
REWARD_WEIGHTS = (1.0, 1.0, 0.1)

# Optional path to an SFT LoRA directory (`adapter_config.json` + weights),
# e.g. `outputs/sft-qwen3-8b-umore-<id>/best_eval_model`.
# None → attach a fresh LoRA on the base recipe checkpoint.
LORA_ADAPTER_PATH: str | None = "outputs/sft-qwen3-8b-umore-july/checkpoint-320"


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    recipe = load_model("qwen3_8b_unsloth_bnb_4bit")
    config = recipe.grpo(
        dataset_path="data/umore/umore_openai.json",
        output_dir="outputs/grpo-qwen3-8b-umore",
        lora_adapter_path=LORA_ADAPTER_PATH,
        shuffle=True,
        dataset_fraction=0.20,
        reward_weights=list(REWARD_WEIGHTS),

        # Override default params of the model
        # learning_rate=1e-4,
        # report_to="none", # if you want to disable reporting to wandb
    )
    run_grpo(config, reward_funcs=resolve_reward_funcs(REWARD_NAMES))


if __name__ == "__main__":
    main()
