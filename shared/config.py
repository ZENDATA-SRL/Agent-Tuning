"""Schemas for a model recipe shared by train and benchmark.

Training loops consume ``SFTConfig`` / ``GRPOConfig``. Serving (vLLM) reads
``VLLMSpec`` and chat/generation knobs from the same ``ModelRecipe``.
"""
from __future__ import annotations

import inspect
import uuid
from dataclasses import asdict, dataclass, field, replace
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal


def _sampling_fields(
    setup: DecodingSetup, spec: GenerationSpec
) -> dict[str, Any]:
    """Flat sampling columns shared by ``SFTConfig`` and ``GRPOConfig``."""
    return {
        "generation_do_sample": spec.do_sample,
        "generation_temperature": spec.temperature,
        "generation_top_p": spec.top_p,
        "generation_top_k": spec.top_k,
        "generation_min_p": spec.min_p,
        "generation_prompt_suffix": setup.generation_prompt_suffix,
        "chat_template_kwargs": dict(setup.template_kwargs),
    }


def _resolve_run(output_dir: str, run_id: str | None) -> tuple[str, str]:
    """Return ``(run_id, output_dir/run_id)`` with a generated id when omitted."""
    rid = run_id or str(uuid.uuid4())
    base = output_dir.rstrip("/")
    return rid, f"{base}-{rid}"


def _normalize_datasets(
    datasets: Sequence[tuple[str, float]],
) -> tuple[tuple[str, float], ...]:
    """Freeze ``(path, fraction)`` pairs. Each fraction is of that file."""
    rows = tuple((str(path), float(fraction)) for path, fraction in datasets)
    if not rows:
        raise ValueError("datasets non può essere vuoto.")
    return rows


LossMasking = Literal["unsloth_responses_only", "assistant_turns"]

# Sentinel so ``tool_trace_fraction=None`` means "natural mix", not "use recipe default".
_UNSET: Any = object()


@dataclass(frozen=True)
class LoRASpec:
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class GenerationSpec:
    """Sampling numbers for one decoding mode (thinking or not)."""

    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    min_p: float


DecodingMode = Literal["thinking", "no_thinking"]


@dataclass(frozen=True)
class GenerationOverride:
    """Fields replaced on top of a mode's ``GenerationSpec``.

    ``None`` keeps the mode value. GRPO uses this to widen rollouts without
    copying the official sampler.
    """

    do_sample: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None


def apply_generation_override(
    spec: GenerationSpec,
    override: GenerationOverride | None,
) -> GenerationSpec:
    """Return ``spec`` with the non-null fields of ``override`` applied."""
    if override is None:
        return spec
    updates = {
        key: value
        for key, value in asdict(override).items()
        if value is not None
    }
    if not updates:
        return spec
    return replace(spec, **updates)


@dataclass(frozen=True)
class DecodingSetup:
    """Sampler plus the prompt pieces that change with thinking mode.

    Suffix and template kwargs travel with the sampler so a thinking run
    cannot mix Qwen's non-thinking numbers with the wrong ``<think>`` block.
    """

    generation: GenerationSpec
    generation_prompt_suffix: str
    template_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallFormat:
    """How a chat template serialises tool calls in generated text.

    Owned by each model recipe so training rewards stay format-agnostic.
    ``parse`` returns ``[{name, arguments}, ...]``; ``has_markup`` is true
    when the text looks like a tool-call attempt (even if not parseable).
    """

    parse: Callable[[str], list[dict[str, Any]]]
    has_markup: Callable[[str], bool]


@dataclass(frozen=True)
class ChatTemplateSpec:
    """Markers that depend on the chat template, not on thinking mode or method.

    The generation suffix and ``enable_thinking`` live on ``DecodingSetup``.
    """

    instruction_part: str
    response_part: str
    turn_end_marker: str
    forbidden_in_loss: tuple[str, ...]
    # False → parallel tool calls are split into sequential single-call turns
    # before rendering (required by templates like Llama 3.1).
    support_multi_tool_calls: bool
    # How this template encodes tool calls in assistant text (GRPO rewards).
    tool_calls: ToolCallFormat


@dataclass(frozen=True)
class VLLMSpec:
    """Serving knobs for ``vllm serve`` (benchmark / local OpenAI server)."""

    # Value for ``--tool-call-parser`` (hermes, llama3_json, gemma4, …).
    tool_call_parser: str
    # Static dict or ``(model_id) -> dict`` for ``vllm serve --hf-overrides``.
    # ``None`` → flag omitted. Defined on the model recipe when needed.
    hf_overrides: dict[str, Any] | Callable[[str], dict[str, Any]] | None = None
    # When set, override the server default for ``--max-model-len``.
    max_model_len: int | None = None
    # When set, override the server default for ``--gpu-memory-utilization``.
    gpu_memory_utilization: float | None = None


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
    # Fraction of training traces that must end in a tool call, in [0.0, 1.0].
    # None → keep the natural mix. 1.0 → tool-only; 0.0 → text-only.
    tool_trace_fraction: float | None


@dataclass(frozen=True)
class GRPOHyperparams:
    """Method block for GRPO (online RL with group-relative advantages)."""

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
    # Fraction of training traces that must end in a tool call, in [0.0, 1.0].
    # None → keep the natural mix. 1.0 → tool-only; 0.0 → text-only.
    tool_trace_fraction: float | None
    # None → rollouts use the decoding mode's GenerationSpec unchanged.
    generation_override: GenerationOverride | None = None


# Default method blocks. Recipes may ``replace()`` individual fields.
BASE_SFT = SFTHyperparams(
    learning_rate=2e-4,
    num_train_epochs=2.0,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_ratio=0.03,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    max_grad_norm=0.3,
    seed=3407,
    logging_steps=5,
    save_steps=40,
    eval_steps=40,
    mixed_precision="bf16",
    gradient_checkpointing="unsloth",
    loss_masking="unsloth_responses_only",
    last_response_only=True,
    tool_trace_fraction=None,
)

BASE_GRPO = GRPOHyperparams(
    learning_rate=5e-6,
    num_train_epochs=1.0,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_generations=4,
    max_completion_length=1024,
    beta=0.04,
    warmup_ratio=0.03,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    max_grad_norm=0.3,
    seed=3407,
    logging_steps=5,
    save_steps=40,
    mixed_precision="bf16",
    gradient_checkpointing="unsloth",
    tool_trace_fraction=None,
)


@dataclass(frozen=True)
class ModelRecipe:
    """One supported checkpoint: train defaults plus serving (vLLM) knobs."""

    id: str
    model_name: str
    max_seq_length: int
    load_in_4bit: bool
    lora: LoRASpec
    # Keyed by thinking mode. Each model declares only the modes it supports.
    decoding: Mapping[DecodingMode, DecodingSetup]
    chat: ChatTemplateSpec
    vllm: VLLMSpec
    sft_defaults: SFTHyperparams = BASE_SFT
    grpo_defaults: GRPOHyperparams = BASE_GRPO

    def decoding_setup(self, mode: DecodingMode) -> DecodingSetup:
        """Return the mode bundle. Raises ``KeyError`` if the recipe lacks ``mode``."""
        try:
            return self.decoding[mode]
        except KeyError:
            known = ", ".join(sorted(self.decoding)) or "(nessuna)"
            raise KeyError(
                f"Modalità {mode!r} assente per {self.id}. Disponibili: {known}."
            ) from None

    def resolve_generation(
        self,
        method: Literal["sft", "grpo"],
        mode: DecodingMode,
    ) -> GenerationSpec:
        """Sampling for ``method`` in ``mode``.

        SFT uses the mode profile. GRPO applies
        ``grpo_defaults.generation_override`` on top and leaves unspecified
        fields at the mode values.
        """
        spec = self.decoding_setup(mode).generation
        if method == "sft":
            return spec
        if method == "grpo":
            return apply_generation_override(
                spec, self.grpo_defaults.generation_override
            )
        raise ValueError(
            f"Metodo sconosciuto {method!r}. Disponibili: 'sft', 'grpo'."
        )

    def sft(
        self,
        *,
        datasets: Sequence[tuple[str, float]],
        output_dir: str,
        resume_from_checkpoint: str | None = None,
        shuffle: bool = False,
        max_traces: int | None = None,
        tool_trace_fraction: float | None | Any = _UNSET,
        decoding_mode: DecodingMode = "no_thinking",
        run_id: str | None = None,
        **overrides: Any,
    ) -> SFTConfig:
        """Build the flat config consumed by `run_sft`.

        `decoding_mode` selects the recipe's ``DecodingSetup``. SFT uses
        that sampler as-is. `overrides` replace recipe defaults (learning
        rate, LoRA rank, …) for a single run. Names that are not fields of
        `SFTConfig` are stored in `trainer_kwargs` and forwarded to the trainer.
        """
        run_id, output_dir = _resolve_run(output_dir, run_id)
        fraction = (
            self.sft_defaults.tool_trace_fraction
            if tool_trace_fraction is _UNSET
            else tool_trace_fraction
        )
        setup = self.decoding_setup(decoding_mode)
        spec = self.resolve_generation("sft", decoding_mode)
        params: dict[str, Any] = dict(
            datasets=_normalize_datasets(datasets),
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            shuffle=shuffle,
            max_traces=max_traces,
            tool_trace_fraction=fraction,
            temp_id=run_id,
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
            **_sampling_fields(setup, spec),
            instruction_part=self.chat.instruction_part,
            response_part=self.chat.response_part,
            turn_end_marker=self.chat.turn_end_marker,
            forbidden_in_loss=self.chat.forbidden_in_loss,
            support_multi_tool_calls=self.chat.support_multi_tool_calls,
        )
        params["trainer_kwargs"] = _collect_trainer_kwargs(params, overrides)
        return SFTConfig(**params)

    def grpo(
        self,
        *,
        datasets: Sequence[tuple[str, float]],
        output_dir: str,
        resume_from_checkpoint: str | None = None,
        lora_adapter_path: str | None = None,
        shuffle: bool = False,
        max_traces: int | None = None,
        tool_trace_fraction: float | None | Any = _UNSET,
        reward_weights: list[float] | None = None,
        decoding_mode: DecodingMode = "no_thinking",
        run_id: str | None = None,
        **overrides: Any,
    ) -> GRPOConfig:
        """Build the flat config consumed by `run_grpo`.

        `lora_adapter_path` points at a saved LoRA directory (e.g. an SFT
        `best_eval_model`) to continue from. `None` attaches a fresh LoRA.
        `decoding_mode` selects the recipe's ``DecodingSetup``; GRPO then
        applies ``generation_override`` on top. Reward callables are passed
        separately to `run_grpo`.
        Names that are not fields of `GRPOConfig` are stored in
        `trainer_kwargs` and forwarded to the trainer.
        """
        run_id, output_dir = _resolve_run(output_dir, run_id)
        fraction = (
            self.grpo_defaults.tool_trace_fraction
            if tool_trace_fraction is _UNSET
            else tool_trace_fraction
        )
        setup = self.decoding_setup(decoding_mode)
        spec = self.resolve_generation("grpo", decoding_mode)
        params: dict[str, Any] = dict(
            datasets=_normalize_datasets(datasets),
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            lora_adapter_path=lora_adapter_path,
            shuffle=shuffle,
            max_traces=max_traces,
            tool_trace_fraction=fraction,
            reward_weights=reward_weights,
            temp_id=run_id,
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
            **_sampling_fields(setup, spec),
            instruction_part=self.chat.instruction_part,
            response_part=self.chat.response_part,
            turn_end_marker=self.chat.turn_end_marker,
            forbidden_in_loss=self.chat.forbidden_in_loss,
            support_multi_tool_calls=self.chat.support_multi_tool_calls,
        )
        params["trainer_kwargs"] = _collect_trainer_kwargs(params, overrides)
        return GRPOConfig(**params)

    def build(
        self,
        method: Literal["sft", "grpo"],
        *,
        output_dir: str,
        datasets: Sequence[tuple[str, float]] | None = None,
        resume_from_checkpoint: str | None = None,
        shuffle: bool = False,
        max_traces: int | None = None,
        tool_trace_fraction: float | None | Any = _UNSET,
        decoding_mode: DecodingMode = "no_thinking",
        temp_id: str | None = None,
        **overrides: Any,
    ) -> SFTConfig | GRPOConfig:
        if datasets is None:
            raise TypeError(f"build(method={method!r}) richiede datasets.")
        fraction_kw: dict[str, Any] = {}
        if tool_trace_fraction is not _UNSET:
            fraction_kw["tool_trace_fraction"] = tool_trace_fraction
        if method == "sft":
            return self.sft(
                datasets=datasets,
                output_dir=output_dir,
                resume_from_checkpoint=resume_from_checkpoint,
                shuffle=shuffle,
                max_traces=max_traces,
                decoding_mode=decoding_mode,
                run_id=temp_id,
                **fraction_kw,
                **overrides,
            )
        if method == "grpo":
            return self.grpo(
                datasets=datasets,
                output_dir=output_dir,
                resume_from_checkpoint=resume_from_checkpoint,
                shuffle=shuffle,
                max_traces=max_traces,
                decoding_mode=decoding_mode,
                run_id=temp_id,
                **fraction_kw,
                **overrides,
            )
        raise ValueError(
            f"Metodo sconosciuto {method!r}. Disponibili: 'sft', 'grpo'."
        )


# Chiavi già fissate dal loop di training. Non si inoltrano al trainer.
_OWNED_TRAINER_KEYS = frozenset(
    {
        "self",
        "model",
        "args",
        "train_dataset",
        "eval_dataset",
        "tokenizer",
        "processing_class",
        "reward_funcs",
    }
)


def _collect_trainer_kwargs(
    params: dict[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply known overrides in place. Return names that are not config fields."""
    extra: dict[str, Any] = {}
    for key, value in overrides.items():
        if key == "trainer_kwargs":
            raise TypeError(
                "trainer_kwargs è riservato: passa i parametri extra come kwargs."
            )
        if key in params:
            params[key] = value
        else:
            extra[key] = value
    return extra


def partition_trainer_kwargs(
    trainer_cls: type,
    trainer_kwargs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split extras into `(trl_config, trainer_ctor)`.

    A name that matches the trainer constructor (and is not owned by the
    training loop) goes to the constructor. Everything else is a TRL
    config field and overrides the args built by the loop.
    """
    ctor_names = set(inspect.signature(trainer_cls.__init__).parameters)
    config_kwargs: dict[str, Any] = {}
    ctor_kwargs: dict[str, Any] = {}
    for key, value in trainer_kwargs.items():
        if key in _OWNED_TRAINER_KEYS:
            raise TypeError(
                f"{key!r} è gestito dal loop di training e non si può sovrascrivere."
            )
        if key in ctor_names:
            ctor_kwargs[key] = value
        else:
            config_kwargs[key] = value
    return config_kwargs, ctor_kwargs


@dataclass
class SFTConfig:
    """Resolved supervised fine-tuning run. Built by `ModelRecipe.sft`."""

    # I/O. Each pair is (path, fraction of that file) before the 10/10/80 split.
    datasets: tuple[tuple[str, float], ...]
    output_dir: str
    resume_from_checkpoint: str | None
    # Shuffle the train split before formatting / training.
    shuffle: bool
    # Cap the train split after shuffle. None loads every trace.
    max_traces: int | None
    # Target share of tool-ending traces in [0.0, 1.0], or None for natural mix.
    tool_trace_fraction: float | None
    # Folder name under data/merged for the train/test/eval JSONL files.
    temp_id: str | None

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
    support_multi_tool_calls: bool
    chat_template_kwargs: dict[str, Any]
    # Kwargs assenti dallo schema: inoltrati a SFTTrainer / TRL SFTConfig.
    trainer_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class GRPOConfig:
    """Resolved GRPO run. Built by `ModelRecipe.grpo`.

    Reward callables are not stored here: pass them to `run_grpo`.
    Optional `reward_weights` must match the number of reward functions.
    """

    # Each pair is (path, fraction of that file) before the 10/10/80 split.
    datasets: tuple[tuple[str, float], ...]
    output_dir: str
    resume_from_checkpoint: str | None
    # Directory with adapter_config.json (e.g. SFT best_eval_model). None = fresh LoRA.
    lora_adapter_path: str | None
    shuffle: bool
    # Cap the train split after shuffle. None loads every trace.
    max_traces: int | None
    # Target share of tool-ending traces in [0.0, 1.0], or None for natural mix.
    tool_trace_fraction: float | None
    # Folder name under data/merged for the train/test/eval JSONL files.
    temp_id: str | None
    reward_weights: list[float] | None

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
    support_multi_tool_calls: bool
    chat_template_kwargs: dict[str, Any]
    # Kwargs assenti dallo schema: inoltrati a GRPOTrainer / TRL GRPOConfig.
    trainer_kwargs: dict[str, Any] = field(default_factory=dict)
