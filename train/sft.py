"""Supervised fine-tuning loop on agent traces."""
from __future__ import annotations

import os
from pathlib import Path

# Unsloth must be imported before transformers/trl to install its patches.
from unsloth import FastLanguageModel  # type: ignore
from unsloth.chat_templates import train_on_responses_only  # type: ignore
from trl import SFTConfig as TRLSFTConfig  # type: ignore
from trl import SFTTrainer  # type: ignore

from datasets import Dataset

from train.config import SFTConfig, partition_trainer_kwargs
from train.data import format_dataset, limit_traces, load_prepared_splits
from train.utils import assert_masking_ok


def run_sft(config: SFTConfig) -> None:
    """Run a full SFT loop and persist the LoRA adapter to `config.output_dir`."""
    # Reduce CUDA allocator fragmentation when VRAM is nearly full.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    # Pop any malformed WANDB_* values so they can't propagate through
    # subprocesses or unexpected callbacks. Empty WANDB_MODE is especially
    # bad: wandb Settings requires one of online/offline/shared/disabled/…
    for _k in (
        "WANDB_TAGS",
        "WANDB_PROJECT",
        "WANDB_NAME",
        "WANDB_ENTITY",
        "WANDB_MODE",
    ):
        if os.environ.get(_k, "").strip() == "":
            os.environ.pop(_k, None)

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
        target_modules=config.lora_target_modules,
        bias="none",
        use_gradient_checkpointing=config.gradient_checkpointing,
        random_state=config.seed,
        max_seq_length=config.max_seq_length,
    )

    model.generation_config.do_sample = config.generation_do_sample
    model.generation_config.temperature = config.generation_temperature
    model.generation_config.top_p = config.generation_top_p
    model.generation_config.top_k = config.generation_top_k
    model.generation_config.min_p = config.generation_min_p

    print(f"[sft] Loading dataset: {config.dataset_path}")
    if config.dataset_fraction < 1.0:
        print(
            f"[sft] Using {config.dataset_fraction:g} of records from file"
        )
    splits = load_prepared_splits(
        config.dataset_path,
        dataset_fraction=config.dataset_fraction,
        temp_id=config.temp_id,
    )
    print(
        "[sft] Split sizes: "
        f"train={len(splits['train'])}, "
        f"test={len(splits['test'])}, "
        f"eval={len(splits['eval'])}"
    )

    train_dataset = Dataset.from_list(splits["train"])
    if config.shuffle:
        print(f"[sft] Shuffling train dataset (seed={config.seed})")
        train_dataset = train_dataset.shuffle(seed=config.seed)
    if config.max_traces is not None:
        before = len(train_dataset)
        train_dataset = limit_traces(train_dataset, config.max_traces)
        print(f"[sft] Train traces: {len(train_dataset)}/{before}")

    eval_dataset = (
        Dataset.from_list(splits["eval"]) if splits["eval"] else None
    )

    print("[sft] Rendering dataset with chat template")
    train_dataset = format_dataset(
        train_dataset, tokenizer, chat_template_kwargs=config.chat_template_kwargs
    )
    if eval_dataset is not None:
        eval_dataset = format_dataset(
            eval_dataset, tokenizer, chat_template_kwargs=config.chat_template_kwargs
        )

    rendered = [f"train={len(train_dataset)}", f"test={len(splits['test'])}"]
    if eval_dataset is not None:
        rendered.append(f"eval={len(eval_dataset)}")
    print(f"[sft] Dataset sizes: {', '.join(rendered)}")

    output_dir = str(Path(config.output_dir))
    has_eval = eval_dataset is not None

    trainer_args: dict = dict(
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=bool(config.gradient_checkpointing),
        bf16=config.mixed_precision == "bf16",
        fp16=config.mixed_precision == "fp16",
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        save_steps=config.save_steps,
        save_strategy="steps",
        eval_strategy="steps" if has_eval else "no",
        load_best_model_at_end=has_eval,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        seed=config.seed,
        output_dir=output_dir,
        report_to="wandb",
        logging_steps=config.logging_steps,
        # TRL 1.13 defaults to chunked_nll, which replaces model.forward and
        # skips Unsloth's fused cross-entropy. nll keeps that path.
        # loss_type="nll",
        # Unsloth treats a missing padding_free as "auto". This machine has no
        # FlashAttention 2, and padding-free then concatenates the batch into
        # one sequence whose attention matrix does not fit in 24 GB.
        padding_free=False,
        # TRL 1.13 sets use_reentrant=False on transformers 4.x, which skips
        # Unsloth's gradient offload.
        # gradient_checkpointing_kwargs={"use_reentrant": True},
    )
    if has_eval:
        # Eval loss does not need logits. Leaving prediction_loss_only off
        # materializes [batch, seq, vocab] and concatenates it on the 3090.
        trainer_args.update(
            eval_steps=config.eval_steps,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            prediction_loss_only=True,
            eval_do_concat_batches=False,
            eval_accumulation_steps=1,
        )

    config_kwargs, ctor_kwargs = partition_trainer_kwargs(
        SFTTrainer, config.trainer_kwargs
    )
    trainer_args.update(config_kwargs)
    if config.trainer_kwargs:
        print(f"[sft] Trainer kwargs: {sorted(config.trainer_kwargs)}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=TRLSFTConfig(**trainer_args),
        #callbacks=[GenerationCallback(
        #    model,
        #    tokenizer,
        #    eval_dataset,
        #    response_part=config.response_part,
        #    generation_prompt_suffix=config.generation_prompt_suffix,
        #)],
        **ctor_kwargs,
    )

    if config.loss_masking == "unsloth_responses_only":
        print("[sft] Wiring loss masking (train_on_responses_only)")
        trainer = train_on_responses_only(
            trainer,
            instruction_part=config.instruction_part,
            response_part=config.response_part,
            last_response_only=config.last_response_only,
        )
    else:
        raise ValueError(
            f"loss_masking sconosciuto: {config.loss_masking!r}."
        )

    assert_masking_ok(
        trainer,
        tokenizer,
        forbidden_in_loss=config.forbidden_in_loss,
        turn_end_marker=config.turn_end_marker,
    )

    if config.resume_from_checkpoint:
        print(
            f"[sft] Resuming training from checkpoint: "
            f"{config.resume_from_checkpoint}"
        )
    else:
        print("[sft] Starting training")
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    print(f"[sft] Saving best model LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir + "/best_eval_model")
    tokenizer.save_pretrained(output_dir + "/best_eval_model")
    print("[sft] Done.")
