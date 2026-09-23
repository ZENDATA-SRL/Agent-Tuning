"""Supervised fine-tuning loop on agent traces."""
from __future__ import annotations

import os
from pathlib import Path

from train.config import SFTConfig
from train.data import format_dataset, load_json_dataset
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

    print(f"[sft] Loading train dataset: {config.train_dataset_path}")
    train_dataset = load_json_dataset(config.train_dataset_path)
    if config.shuffle:
        print(f"[sft] Shuffling train dataset (seed={config.seed})")
        train_dataset = train_dataset.shuffle(seed=config.seed)

    test_dataset = None
    if config.test_dataset_path:
        print(f"[sft] Loading test dataset: {config.test_dataset_path}")
        test_dataset = load_json_dataset(config.test_dataset_path)

    eval_dataset = None
    if config.eval_dataset_path:
        print(f"[sft] Loading eval dataset: {config.eval_dataset_path}")
        eval_dataset = load_json_dataset(config.eval_dataset_path)

    print("[sft] Rendering dataset with chat template")
    train_dataset = format_dataset(
        train_dataset, tokenizer, chat_template_kwargs=config.chat_template_kwargs
    )
    if test_dataset is not None:
        test_dataset = format_dataset(
            test_dataset, tokenizer, chat_template_kwargs=config.chat_template_kwargs
        )
    if eval_dataset is not None:
        eval_dataset = format_dataset(
            eval_dataset, tokenizer, chat_template_kwargs=config.chat_template_kwargs
        )

    splits = [f"train={len(train_dataset)}"]
    if test_dataset is not None:
        splits.append(f"test={len(test_dataset)}")
    if eval_dataset is not None:
        splits.append(f"eval={len(eval_dataset)}")
    print(f"[sft] Dataset sizes: {', '.join(splits)}")

    output_dir = str(Path(config.output_dir))
    has_eval = eval_dataset is not None

    trainer_args: dict = dict(
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
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
    )
    if has_eval:
        trainer_args.update(
            eval_steps=config.eval_steps,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

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
