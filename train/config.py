"""Minimal SFT config for Qwen3-8B QLoRA fine-tuning.

Designed to be edited in-place (no CLI). The launchable entrypoint in
`scripts/train_qwen3_sft.py` simply instantiates this config and calls
`train.sft.run_sft`. Tweak fields here, hit F5 in VSCode.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SFTConfig:
    # I/O
    dataset_path: str = "examples/traces-2026-05-18.jsonl"
    output_dir: str = "outputs/sft-qwen3-8b-v1"

    # Model — Unsloth's pre-quantized Qwen3-8B 4-bit checkpoint.
    # Using a "*-unsloth-bnb-4bit" repo avoids downloading ~16 GB of fp16
    # weights and quantizing them at load time on every fresh machine.
    model_name: str = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    max_seq_length: int = 4096
    load_in_4bit: bool = True

    # LoRA / PEFT
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    # Training hyperparameters
    learning_rate: float = 2e-4
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    max_grad_norm: float = 1.0
    seed: int = 3407
    logging_steps: int = 5
    save_steps: int = 50
    # bf16 for Ampere+ (3090/4090/A100). Use "fp16" on older GPUs.
    mixed_precision: str = "bf16"
    # Unsloth's optimised gradient checkpointing (recommended).
    gradient_checkpointing: str | bool = "unsloth"
