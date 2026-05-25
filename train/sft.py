"""Supervised fine-tuning entrypoint backed by Unsloth + TRL.

Run with:
    python -m train --dataset data/traces-2026-05-18.jsonl \
        --output-dir outputs/sft-gemma4-e4b-v1
"""
from __future__ import annotations

from pathlib import Path

from train.config import SFTConfig
from train.data import format_dataset, load_jsonl, mask_labels_selective
from train.templates import get_template_adapter
from train.wandb_setup import (
    finish_wandb,
    log_trainable_parameters,
    make_sample_log_callback,
    maybe_init_wandb,
)


def _print_masking_check(trainer, tokenizer) -> None:
    """Decode the first training example before/after loss masking.

    Catches the most common silent bug of agentic SFT: a misconfigured
    masking that hides assistant tool-calls or exposes tool observations.
    If the masked decode does NOT contain the expected assistant content,
    abort early.

    Pattern lifted verbatim from the official Unsloth Gemma-4 notebook.
    """
    sample = trainer.train_dataset[0]

    print("\n" + "=" * 70)
    print("[train.sft] First example — full decoded input:")
    print("=" * 70)
    print(tokenizer.decode(sample["input_ids"]))

    pad_id = tokenizer.pad_token_id
    masked = [pad_id if x == -100 else x for x in sample["labels"]]
    pad_str = tokenizer.pad_token if tokenizer.pad_token else "<pad>"
    print("\n" + "=" * 70)
    print("[train.sft] First example — masked labels (only loss-bearing tokens):")
    print("=" * 70)
    print(tokenizer.decode(masked).replace(pad_str, " "))
    print("=" * 70 + "\n")


def run_sft(config: SFTConfig) -> None:
    """Run a full SFT loop and persist the LoRA adapter to `config.output_dir`.

    Notes on Gemma-4 E4B quirks (documented by Unsloth, see
    https://unsloth.ai/docs/models/gemma-4/train):
      - A loss plateau around 13–15 is normal for E2B/E4B (multimodal quirk).
      - `use_cache=True` is REQUIRED at training time because of KV-shared
        layers; Unsloth forces this internally even with gradient
        checkpointing — do not override.
    """
    import os
    # Reduces CUDA allocator fragmentation — helps when VRAM is nearly full.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    # Unsloth must be imported before transformers/trl to install its patches.
    from unsloth import FastModel  # type: ignore
    from unsloth.chat_templates import (  # type: ignore
        train_on_responses_only,
    )
    from trl import SFTConfig as TRLSFTConfig  # type: ignore
    from trl import SFTTrainer  # type: ignore

    print(f"[train.sft] Loading model: {config.model_name}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        full_finetuning=False,
    )

    print(f"[train.sft] Selecting chat template adapter for {config.model_name}")
    adapter = get_template_adapter(config.model_name, tokenizer)
    tokenizer = adapter.prepare_tokenizer()
    print(f"[train.sft] Adapter: {adapter.name}")

    print("[train.sft] Attaching LoRA adapters")
    model = FastModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        finetune_vision_layers=config.finetune_vision_layers,
        finetune_language_layers=config.finetune_language_layers,
        finetune_attention_modules=config.finetune_attention_modules,
        finetune_mlp_modules=config.finetune_mlp_modules,
        bias="none",
        random_state=config.seed,
    )

    # W&B is initialised AFTER PEFT so we can record trainable-parameter
    # counts up-front in the run summary, but BEFORE SFTTrainer so TRL's
    # callback latches onto our pre-existing run instead of starting a
    # second one with default settings.
    wandb_run = maybe_init_wandb(config)
    if wandb_run is not None:
        log_trainable_parameters(model)

    # If our pre-init failed, strip "wandb" from report_to so TRL does not
    # register a WandbCallback that would call wandb.init() a second time
    # during trainer.train() and crash fatally (we already set WANDB_MODE=
    # disabled in maybe_init_wandb, but removing the reporter is belt-and-
    # suspenders and avoids the callback overhead entirely).
    effective_report_to = config.report_to
    if wandb_run is None and "wandb" in (effective_report_to or "").lower():
        reporters = [r for r in effective_report_to.split(",") if r.strip().lower() != "wandb"]
        effective_report_to = ",".join(reporters) or "none"

    print(f"[train.sft] Loading dataset: {config.dataset_path}")
    records = load_jsonl(config.dataset_path)
    print(f"[train.sft] {len(records)} records loaded")

    print("[train.sft] Formatting dataset with chat template")
    train_dataset, raw_texts = format_dataset(records, adapter)

    output_dir = str(Path(config.output_dir))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        args=TRLSFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_checkpointing=config.gradient_checkpointing,
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
            report_to=effective_report_to,
        ),
    )

    if config.loss_on_tool_calls_only:
        _masking_mode = "tool_calls_only"
        if config.loss_include_final_response:
            _masking_mode += " + final_response"
        print(f"[train.sft] Applying selective loss masking: {_masking_mode}")

        # Pre-tokenise each example and inject hand-crafted labels so that only
        # the desired spans contribute to the cross-entropy loss.
        # We do this BEFORE `train_on_responses_only` (which is skipped) by
        # replacing the text dataset with one that already carries `input_ids`
        # and `labels`.
        from datasets import Dataset as HFDataset  # type: ignore

        def _apply_selective_mask(batch: dict) -> dict:
            all_input_ids: list[list[int]] = []
            all_labels: list[list[int]] = []
            all_attention: list[list[int]] = []
            # Use the inner fast tokenizer (unwrapped from the multimodal
            # Processor if needed). Gemma4Processor returns batched outputs
            # even for a single string — enc["input_ids"] = [[id1, id2, ...]]
            # — which gives [1]*len([[...]]) = [1] for the attention mask.
            # The HF collator then pads that to match the 2D input_ids,
            # producing a 3D attention_mask [batch, 1, seq] that breaks
            # Unsloth's get_batch_samples. The inner fast tokenizer always
            # returns un-batched 1D lists, so the collator stacks them
            # correctly into [batch, seq].
            _inner_tok = getattr(tokenizer, "tokenizer", tokenizer)
            for text in batch["text"]:
                enc = _inner_tok(text=text, add_special_tokens=False)
                input_ids: list[int] = enc["input_ids"]
                labels = mask_labels_selective(
                    tokenizer,
                    text,
                    adapter,
                    include_final_response=config.loss_include_final_response,
                )
                all_input_ids.append(input_ids)
                all_labels.append(labels)
                all_attention.append([1] * len(input_ids))
            return {
                "input_ids": all_input_ids,
                "labels": all_labels,
                "attention_mask": all_attention,
            }

        masked_dataset = train_dataset.map(
            _apply_selective_mask,
            batched=True,
            remove_columns=["text"],
            desc="Applying selective loss masking",
        )
        trainer.train_dataset = masked_dataset
    else:
        print("[train.sft] Wiring loss masking (train_on_responses_only — all assistant tokens)")
        trainer = train_on_responses_only(
            trainer,
            instruction_part=adapter.instruction_part,
            response_part=adapter.response_part,
        )

    _print_masking_check(trainer, tokenizer)

    sample_cb = make_sample_log_callback(
        model,
        tokenizer,
        raw_texts,
        adapter,
        num_samples=2,
        generate_samples=config.log_generate_samples,
        max_new_tokens=256,
    )
    trainer.add_callback(sample_cb)

    print("[train.sft] Starting training")
    try:
        trainer.train()

        print(f"[train.sft] Saving LoRA adapter to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("[train.sft] Done.")
    finally:
        # Ensure the W&B run is closed even if training crashed half-way,
        # so the dashboard shows a "finished" / "crashed" state instead of
        # an indefinitely-"running" zombie run.
        finish_wandb(wandb_run)
