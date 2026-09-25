"""Qwen3-8B, checkpoint Unsloth 4-bit. SFT and GRPO defaults."""
from __future__ import annotations

from train.config import (
    ChatTemplateSpec,
    GenerationSpec,
    GRPOHyperparams,
    LoRASpec,
    ModelRecipe,
    SFTHyperparams,
)

# Un checkpoint "*-unsloth-bnb-4bit" evita di scaricare ~16 GB di pesi fp16
# e di quantizzarli a ogni avvio.
RECIPE = ModelRecipe(
    id="qwen3_8b_unsloth_bnb_4bit",
    model_name="unsloth/Qwen3-8B-unsloth-bnb-4bit",
    max_seq_length=16384,
    load_in_4bit=True,
    lora=LoRASpec(
        r=16,
        alpha=32,
        dropout=0.05,
        # Proiezioni attention e MLP di Qwen3.
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    ),
    # Sampling consigliato da Qwen3 in modalità non-thinking.
    generation=GenerationSpec(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
    ),
    chat=ChatTemplateSpec(
        # ChatML. I marker devono comparire verbatim nel testo renderizzato.
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        turn_end_marker="<|im_end|>",
        forbidden_in_loss=(
            "<tool_response>",
            "<|im_start|>user",
            "<|im_start|>system",
        ),
        # enable_thinking=False è il blocco <think> vuoto già chiuso.
        # "<|im_start|>assistant\n" da solo lascia il thinking acceso.
        generation_prompt_suffix=(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ),
        template_kwargs={"enable_thinking": False},
    ),
    sft_defaults=SFTHyperparams(
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
    ),
    # Punti di partenza per GRPO su LoRA 4-bit, non un run già tarato.
    # Learning rate più basso dell'SFT: il vantaggio policy è rumoroso.
    grpo_defaults=GRPOHyperparams(
        learning_rate=5e-6,
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=16384,
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
    ),
)
