"""Supervised fine-tuning loop for Qwen3-8B on agent traces.

Minimal, single-purpose: load Qwen3 in 4-bit, attach LoRA adapters,
render the JSONL traces with the tokenizer's chat template, mask the
loss to assistant turns only, train, save the LoRA adapter.
"""
from __future__ import annotations

import os
from pathlib import Path

from train.config import SFTConfig
from train.data import format_dataset, load_jsonl
from train.templates import QWEN3_INSTRUCTION_PART, QWEN3_RESPONSE_PART


def _print_masking_check(trainer, tokenizer) -> None:
    """Decode the first training example before/after loss masking.

    Catches the most common silent bug of agentic SFT: a misconfigured
    masking that hides the assistant content. If the masked decode does
    not contain the expected assistant content, abort early.
    """
    sample = trainer.train_dataset[0]

    print("\n" + "=" * 70)
    print("[sft] First example — full decoded input:")
    print("=" * 70)
    print(tokenizer.decode(sample["input_ids"]))

    pad_id = tokenizer.pad_token_id
    masked = [pad_id if x == -100 else x for x in sample["labels"]]
    pad_str = tokenizer.pad_token if tokenizer.pad_token else "<pad>"
    print("\n" + "=" * 70)
    print("[sft] First example — masked labels (only loss-bearing tokens):")
    print("=" * 70)
    print(tokenizer.decode(masked).replace(pad_str, " "))
    print("=" * 70 + "\n")


def run_sft(config: SFTConfig) -> None:
    """Run a full SFT loop and persist the LoRA adapter to `config.output_dir`."""
    # Reduce CUDA allocator fragmentation when VRAM is nearly full.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    # Pop any malformed WANDB_* values so they can't propagate through
    # subprocesses or unexpected callbacks.
    for _k in ("WANDB_TAGS", "WANDB_PROJECT", "WANDB_NAME", "WANDB_ENTITY"):
        if os.environ.get(_k, "").strip() == "":
            os.environ.pop(_k, None)

    # Unsloth must be imported before transformers/trl to install its patches.
    from unsloth import FastLanguageModel  # type: ignore
    from unsloth.chat_templates import train_on_responses_only  # type: ignore
    from trl import SFTConfig as TRLSFTConfig  # type: ignore
    from trl import SFTTrainer  # type: ignore

    print(f"[sft] Loading model: {config.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        full_finetuning=False,
    )

    print("[sft] Attaching LoRA adapters")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing=config.gradient_checkpointing,
        random_state=config.seed,
        max_seq_length=config.max_seq_length,
    )

    print(f"[sft] Loading dataset: {config.dataset_path}")
    records = load_jsonl(config.dataset_path)
    print(f"[sft] {len(records)} records loaded")

    print("[sft] Rendering dataset with Qwen3 chat template")
    train_dataset = format_dataset(records, tokenizer)

    output_dir = str(Path(config.output_dir))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        args=TRLSFTConfig(
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_checkpointing=bool(config.gradient_checkpointing),
            bf16=config.mixed_precision == "bf16",
            fp16=config.mixed_precision == "fp16",
            warmup_ratio=config.warmup_ratio,
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_strategy="steps",
            optim=config.optim,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            max_grad_norm=config.max_grad_norm,
            seed=config.seed,
            output_dir=output_dir,
            report_to="wandb",
        ),
    )

    print("[sft] Wiring loss masking (train_on_responses_only)")
    trainer = train_on_responses_only(
        trainer,
        instruction_part=QWEN3_INSTRUCTION_PART,
        response_part=QWEN3_RESPONSE_PART,
    )

    _print_masking_check(trainer, tokenizer)

    print("[sft] Starting training")
    trainer.train()

    print(f"[sft] Saving LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[sft] Done.")
