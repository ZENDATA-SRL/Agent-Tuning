"""Group Relative Policy Optimization loop (TRL GRPOTrainer + Unsloth LoRA)."""
from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset

from train.config import GRPOConfig, partition_trainer_kwargs
from train.data import format_grpo_dataset, limit_traces, load_prepared_splits

RewardFunc = Callable[..., list[float | None]]


def _sanitize_wandb_env() -> None:
    for key in (
        "WANDB_TAGS",
        "WANDB_PROJECT",
        "WANDB_NAME",
        "WANDB_ENTITY",
        "WANDB_MODE",
    ):
        if os.environ.get(key, "").strip() == "":
            os.environ.pop(key, None)


def _load_model_and_tokenizer(config: GRPOConfig) -> tuple[Any, Any]:
    """Load base (+ fresh LoRA) or an existing SFT LoRA adapter for GRPO."""
    # Unsloth must be imported before transformers/trl to install its patches.
    from unsloth import FastLanguageModel  # type: ignore

    if config.lora_adapter_path:
        adapter = Path(config.lora_adapter_path)
        if not adapter.is_dir():
            raise FileNotFoundError(
                f"lora_adapter_path non trovato: {adapter}"
            )
        if not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"lora_adapter_path senza adapter_config.json: {adapter}"
            )
        print(f"[grpo] Loading trainable LoRA adapter: {adapter}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter),
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            full_finetuning=False,
        )
        # Unsloth may load adapters in eval/inference mode; re-enable LoRA grads.
        model.train()
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
    else:
        print(f"[grpo] Loading base model: {config.model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            full_finetuning=False,
        )
        print("[grpo] Attaching fresh LoRA adapters")
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

    # GRPO left-pads prompts for batched generation.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Some Qwen/Unsloth checkpoints ship generation_config.json with
    # max_length=40960. GRPO passes an explicit completion limit, so keeping
    # both values makes Transformers emit a precedence warning on every batch.
    model.generation_config.max_length = None

    return model, tokenizer


def run_grpo(
    config: GRPOConfig,
    reward_funcs: RewardFunc | Sequence[RewardFunc],
) -> None:
    """Run GRPO and persist the LoRA adapter to `config.output_dir`."""
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    _sanitize_wandb_env()

    from trl import GRPOConfig as TRLGRPOConfig  # type: ignore
    from trl import GRPOTrainer  # type: ignore

    if callable(reward_funcs):
        rewards: list[RewardFunc] = [reward_funcs]
    else:
        rewards = list(reward_funcs)
    if not rewards:
        raise ValueError("run_grpo richiede almeno una reward function.")

    if config.reward_weights is not None and len(config.reward_weights) != len(
        rewards
    ):
        raise ValueError(
            f"reward_weights ({len(config.reward_weights)}) deve avere la "
            f"stessa lunghezza di reward_funcs ({len(rewards)})."
        )

    model, tokenizer = _load_model_and_tokenizer(config)

    print(f"[grpo] Loading dataset: {config.dataset_path}")
    if config.dataset_fraction < 1.0:
        print(
            f"[grpo] Using {config.dataset_fraction:g} of records from file"
        )
    splits = load_prepared_splits(
        config.dataset_path,
        dataset_fraction=config.dataset_fraction,
        temp_id=config.temp_id,
    )
    print(
        "[grpo] Split sizes: "
        f"train={len(splits['train'])}, "
        f"test={len(splits['test'])}, "
        f"eval={len(splits['eval'])}"
    )

    train_dataset = Dataset.from_list(splits["train"])
    if config.shuffle:
        print(f"[grpo] Shuffling train dataset (seed={config.seed})")
        train_dataset = train_dataset.shuffle(seed=config.seed)
    if config.max_traces is not None:
        before = len(train_dataset)
        train_dataset = limit_traces(train_dataset, config.max_traces)
        print(f"[grpo] Train traces: {len(train_dataset)}/{before}")

    eval_dataset = (
        Dataset.from_list(splits["eval"]) if splits["eval"] else None
    )

    print("[grpo] Rendering GRPO prompts with chat template")
    train_dataset = format_grpo_dataset(
        train_dataset,
        tokenizer,
        chat_template_kwargs=config.chat_template_kwargs,
    )
    if eval_dataset is not None:
        eval_dataset = format_grpo_dataset(
            eval_dataset,
            tokenizer,
            chat_template_kwargs=config.chat_template_kwargs,
        )

    rendered = [f"train={len(train_dataset)}", f"test={len(splits['test'])}"]
    if eval_dataset is not None:
        rendered.append(f"eval={len(eval_dataset)}")
    print(f"[grpo] Dataset sizes: {', '.join(rendered)}")
    print(
        f"[grpo] Reward functions: "
        f"{[getattr(fn, '__name__', type(fn).__name__) for fn in rewards]}"
    )

    output_dir = str(Path(config.output_dir))
    max_prompt_length = max(
        1, config.max_seq_length - config.max_completion_length
    )
    has_eval = eval_dataset is not None

    trainer_args: dict[str, Any] = dict(
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        # GRPO evaluates prompts in groups of `num_generations`.
        per_device_eval_batch_size=config.num_generations,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=bool(config.gradient_checkpointing),
        bf16=config.mixed_precision == "bf16",
        fp16=config.mixed_precision == "fp16",
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        max_grad_norm=config.max_grad_norm,
        seed=config.seed,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_strategy="steps",
        report_to="wandb",
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        max_prompt_length=max_prompt_length,
        beta=config.beta,
        temperature=config.generation_temperature,
        top_p=config.generation_top_p,
        top_k=config.generation_top_k,
        min_p=config.generation_min_p if config.generation_min_p > 0 else None,
        reward_weights=config.reward_weights,
        remove_unused_columns=False,
        eval_strategy="steps" if has_eval else "no",
        # TRL prints sampled prompt/completion pairs at logging steps and
        # includes them in the configured reporter (W&B here).
        log_completions=True,
        num_completions_to_print=2,
    )
    if has_eval:
        trainer_args["eval_steps"] = config.save_steps

    config_kwargs, ctor_kwargs = partition_trainer_kwargs(
        GRPOTrainer, config.trainer_kwargs
    )
    trainer_args.update(config_kwargs)
    if config.trainer_kwargs:
        print(f"[grpo] Trainer kwargs: {sorted(config.trainer_kwargs)}")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=rewards,
        args=TRLGRPOConfig(**trainer_args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **ctor_kwargs,
    )

    if config.resume_from_checkpoint:
        print(
            f"[grpo] Resuming training from checkpoint: "
            f"{config.resume_from_checkpoint}"
        )
    else:
        print("[grpo] Starting training")
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    save_dir = output_dir + "/best_model"
    print(f"[grpo] Saving LoRA adapter to {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("[grpo] Done.")
