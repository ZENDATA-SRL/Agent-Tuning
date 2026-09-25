"""Gemma4-E2B, checkpoint Unsloth 4-bit. SFT and GRPO defaults."""
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
    id="gemma4_e2b_unsloth_bnb_4bit",
    model_name="unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
    max_seq_length=16384,
    load_in_4bit=True,
    lora=LoRASpec(
        r=16,
        alpha=32,
        dropout=0.00,
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
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        min_p=0.0,
    ),
    chat=ChatTemplateSpec(
        # Il template emette "<|turn>" + role + "\n", con role "user" o "system".
        instruction_part="<|turn>user\n",
        # Gli assistant vengono rinominati "model" dal template.
        response_part="<|turn>model\n",
        turn_end_marker="<turn|>",
        forbidden_in_loss=(
            "<|turn>user",
            "<|turn>system",
            # Chiusura del blocco tool response: il modello non la genera mai,
            # quindi se compare nella loss vuol dire che l'output del tool è entrato.
            "<tool_response|>",
        ),
        # Con enable_thinking=False non c'è nessun canale di thinking vuoto da chiudere,
        # a differenza di Qwen3.
        generation_prompt_suffix="<|turn>model\n",
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
