"""Schemas for a model recipe and the training methods it supports.

A recipe is the supported-model unit: checkpoint, LoRA targets, sampling
and chat-template markers, plus one hyperparameter block per method
(SFT, GRPO, …). Experiment-specific values (dataset paths, output
directory, resume checkpoint) are passed when the method config is built.

Concrete recipes live in `train/configs/` and are loaded with
`train.configs.load_model`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


LossMasking = Literal["unsloth_responses_only", "assistant_turns"]


@dataclass(frozen=True)
class LoRASpec:
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class GenerationSpec:
    """Sampling used whenever the model generates (eval callback, GRPO rollouts)."""

    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    min_p: float


@dataclass(frozen=True)
class ChatTemplateSpec:
    """Markers and tokenizer kwargs that depend on the chat template, not the method."""

    instruction_part: str
    response_part: str
    turn_end_marker: str
    forbidden_in_loss: tuple[str, ...]
    # Suffisso appeso al prompt di eval per forzare il formato di generazione
    # (ruolo assistant, thinking disattivato, ecc.).
    generation_prompt_suffix: str
    # Kwargs inoltrati a `tokenizer.apply_chat_template` (es. enable_thinking).
    template_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SFTHyperparams:
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    lr_scheduler_type: str
    optim: str
    max_grad_norm: float
    seed: int
    logging_steps: int
    save_steps: int
    eval_steps: int
    # "bf16" for Ampere+ (3090/4090/A100). "fp16" on older GPUs.
    mixed_precision: str
    gradient_checkpointing: str | bool
    # "unsloth_responses_only": Unsloth train_on_responses_only.
    # "assistant_turns": maschera custom su ogni turno assistant.
    loss_masking: LossMasking
    last_response_only: bool


@dataclass(frozen=True)
class GRPOHyperparams:
    """Method block for GRPO. The trainer is not wired yet; the values are."""

    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    # Completions sampled per prompt (group size).
    num_generations: int
    max_completion_length: int
    # Coefficiente KL verso il modello di riferimento.
    beta: float
    warmup_ratio: float
    weight_decay: float
    lr_scheduler_type: str
    optim: str
    max_grad_norm: float
    seed: int
    logging_steps: int
    save_steps: int
    mixed_precision: str
    gradient_checkpointing: str | bool


@dataclass(frozen=True)
class ModelRecipe:
    """One supported checkpoint and the method configs that train it."""

    id: str
    model_name: str
    max_seq_length: int
    load_in_4bit: bool
    lora: LoRASpec
    generation: GenerationSpec
    chat: ChatTemplateSpec
    sft_defaults: SFTHyperparams
    grpo_defaults: GRPOHyperparams

    def sft(
        self,
        *,
        train_dataset_path: str,
        output_dir: str,
        test_dataset_path: str | None = None,
        eval_dataset_path: str | None = None,
        resume_from_checkpoint: str | None = None,
        shuffle: bool = False,
        **overrides: Any,
    ) -> SFTConfig:
        """Build the flat config consumed by `run_sft`.

        `overrides` replace recipe defaults (learning rate, LoRA rank, …)
        for a single run. Unknown names raise `TypeError`.
        """
        params: dict[str, Any] = dict(
            train_dataset_path=train_dataset_path,
            test_dataset_path=test_dataset_path,
            eval_dataset_path=eval_dataset_path,
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            shuffle=shuffle,
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
            lora_r=self.lora.r,
            lora_alpha=self.lora.alpha,
            lora_dropout=self.lora.dropout,
            lora_target_modules=list(self.lora.target_modules),
            learning_rate=self.sft_defaults.learning_rate,
            num_train_epochs=self.sft_defaults.num_train_epochs,
            per_device_train_batch_size=self.sft_defaults.per_device_train_batch_size,
            gradient_accumulation_steps=self.sft_defaults.gradient_accumulation_steps,
            warmup_ratio=self.sft_defaults.warmup_ratio,
            weight_decay=self.sft_defaults.weight_decay,
            lr_scheduler_type=self.sft_defaults.lr_scheduler_type,
            optim=self.sft_defaults.optim,
            max_grad_norm=self.sft_defaults.max_grad_norm,
            seed=self.sft_defaults.seed,
            logging_steps=self.sft_defaults.logging_steps,
            save_steps=self.sft_defaults.save_steps,
            eval_steps=self.sft_defaults.eval_steps,
            mixed_precision=self.sft_defaults.mixed_precision,
            gradient_checkpointing=self.sft_defaults.gradient_checkpointing,
            loss_masking=self.sft_defaults.loss_masking,
            last_response_only=self.sft_defaults.last_response_only,
            generation_do_sample=self.generation.do_sample,
            generation_temperature=self.generation.temperature,
            generation_top_p=self.generation.top_p,
            generation_top_k=self.generation.top_k,
            generation_min_p=self.generation.min_p,
            generation_prompt_suffix=self.chat.generation_prompt_suffix,
            instruction_part=self.chat.instruction_part,
            response_part=self.chat.response_part,
            turn_end_marker=self.chat.turn_end_marker,
            forbidden_in_loss=self.chat.forbidden_in_loss,
            chat_template_kwargs=dict(self.chat.template_kwargs),
        )
        _apply_overrides(params, overrides, label="SFT")
        return SFTConfig(**params)

    def grpo(
        self,
        *,
        dataset_path: str,
        output_dir: str,
        resume_from_checkpoint: str | None = None,
        **overrides: Any,
    ) -> GRPOConfig:
        """Build the flat config a GRPO run will consume."""
        params: dict[str, Any] = dict(
            dataset_path=dataset_path,
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
            lora_r=self.lora.r,
            lora_alpha=self.lora.alpha,
            lora_dropout=self.lora.dropout,
            lora_target_modules=list(self.lora.target_modules),
            learning_rate=self.grpo_defaults.learning_rate,
            num_train_epochs=self.grpo_defaults.num_train_epochs,
            per_device_train_batch_size=self.grpo_defaults.per_device_train_batch_size,
            gradient_accumulation_steps=self.grpo_defaults.gradient_accumulation_steps,
            num_generations=self.grpo_defaults.num_generations,
            max_completion_length=self.grpo_defaults.max_completion_length,
            beta=self.grpo_defaults.beta,
            warmup_ratio=self.grpo_defaults.warmup_ratio,
            weight_decay=self.grpo_defaults.weight_decay,
            lr_scheduler_type=self.grpo_defaults.lr_scheduler_type,
            optim=self.grpo_defaults.optim,
            max_grad_norm=self.grpo_defaults.max_grad_norm,
            seed=self.grpo_defaults.seed,
            logging_steps=self.grpo_defaults.logging_steps,
            save_steps=self.grpo_defaults.save_steps,
            mixed_precision=self.grpo_defaults.mixed_precision,
            gradient_checkpointing=self.grpo_defaults.gradient_checkpointing,
            generation_do_sample=self.generation.do_sample,
            generation_temperature=self.generation.temperature,
            generation_top_p=self.generation.top_p,
            generation_top_k=self.generation.top_k,
            generation_min_p=self.generation.min_p,
            generation_prompt_suffix=self.chat.generation_prompt_suffix,
            instruction_part=self.chat.instruction_part,
            response_part=self.chat.response_part,
            turn_end_marker=self.chat.turn_end_marker,
            forbidden_in_loss=self.chat.forbidden_in_loss,
            chat_template_kwargs=dict(self.chat.template_kwargs),
        )
        _apply_overrides(params, overrides, label="GRPO")
        return GRPOConfig(**params)

    def build(
        self,
        method: Literal["sft", "grpo"],
        *,
        output_dir: str,
        train_dataset_path: str | None = None,
        test_dataset_path: str | None = None,
        eval_dataset_path: str | None = None,
        dataset_path: str | None = None,
        resume_from_checkpoint: str | None = None,
        shuffle: bool = False,
        **overrides: Any,
    ) -> SFTConfig | GRPOConfig:
        if method == "sft":
            if train_dataset_path is None:
                raise TypeError(
                    "build(method='sft') richiede train_dataset_path."
                )
            return self.sft(
                train_dataset_path=train_dataset_path,
                test_dataset_path=test_dataset_path,
                eval_dataset_path=eval_dataset_path,
                output_dir=output_dir,
                resume_from_checkpoint=resume_from_checkpoint,
                shuffle=shuffle,
                **overrides,
            )
        if method == "grpo":
            if dataset_path is None:
                raise TypeError("build(method='grpo') richiede dataset_path.")
            return self.grpo(
                dataset_path=dataset_path,
                output_dir=output_dir,
                resume_from_checkpoint=resume_from_checkpoint,
                **overrides,
            )
        raise ValueError(
            f"Metodo sconosciuto {method!r}. Disponibili: 'sft', 'grpo'."
        )


def _apply_overrides(params: dict[str, Any], overrides: Mapping[str, Any], *, label: str) -> None:
    unknown = sorted(set(overrides) - set(params))
    if unknown:
        known = ", ".join(sorted(params))
        raise TypeError(
            f"Override {label} sconosciuti: {unknown}. Campi validi: {known}."
        )
    params.update(overrides)


@dataclass
class SFTConfig:
    """Resolved supervised fine-tuning run. Built by `ModelRecipe.sft`."""

    # I/O
    train_dataset_path: str
    test_dataset_path: str | None
    eval_dataset_path: str | None
    output_dir: str
    resume_from_checkpoint: str | None
    # Shuffle the train split before formatting / training.
    shuffle: bool

    # Model
    model_name: str
    max_seq_length: int
    load_in_4bit: bool

    # LoRA / PEFT
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]

    # Training hyperparameters
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    lr_scheduler_type: str
    optim: str
    max_grad_norm: float
    seed: int
    logging_steps: int
    save_steps: int
    eval_steps: int
    mixed_precision: str
    gradient_checkpointing: str | bool
    loss_masking: LossMasking
    last_response_only: bool

    # Sampling
    generation_do_sample: bool
    generation_temperature: float
    generation_top_p: float
    generation_top_k: int
    generation_min_p: float
    generation_prompt_suffix: str

    # Chat template
    instruction_part: str
    response_part: str
    turn_end_marker: str
    forbidden_in_loss: tuple[str, ...]
    chat_template_kwargs: dict[str, Any]


@dataclass
class GRPOConfig:
    """Resolved GRPO run. Built by `ModelRecipe.grpo`.

    Holds everything the trainer will need. `run_grpo` is not implemented yet.
    """

    dataset_path: str
    output_dir: str
    resume_from_checkpoint: str | None

    model_name: str
    max_seq_length: int
    load_in_4bit: bool

    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]

    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_generations: int
    max_completion_length: int
    beta: float
    warmup_ratio: float
    weight_decay: float
    lr_scheduler_type: str
    optim: str
    max_grad_norm: float
    seed: int
    logging_steps: int
    save_steps: int
    mixed_precision: str
    gradient_checkpointing: str | bool

    generation_do_sample: bool
    generation_temperature: float
    generation_top_p: float
    generation_top_k: int
    generation_min_p: float
    generation_prompt_suffix: str

    instruction_part: str
    response_part: str
    turn_end_marker: str
    forbidden_in_loss: tuple[str, ...]
    chat_template_kwargs: dict[str, Any]
