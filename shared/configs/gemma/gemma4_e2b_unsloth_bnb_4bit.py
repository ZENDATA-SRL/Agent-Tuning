"""Gemma4-E2B, checkpoint Unsloth 4-bit. SFT, GRPO, and vLLM defaults."""
from __future__ import annotations

import json
from pathlib import Path

from shared.config import (
    ChatTemplateSpec,
    DecodingSetup,
    GenerationSpec,
    GRPOHyperparams,
    LoRASpec,
    ModelRecipe,
    SFTHyperparams,
    VLLMSpec,
)
from shared.configs.gemma.tool_calls import GEMMA4_TOOL_CALLS


def gemma4_bnb_hf_overrides(model_id: str) -> dict:
    """Rewrite Unsloth skip-module names into the prefixes vLLM actually checks.

    Unsloth lists mixed-precision layers as Hugging Face paths such as
    ``model.language_model.layers.0.mlp``. vLLM looks up
    ``language_model.model.layers.0.mlp``. Without the rewrite, bf16 weights
    are loaded into packed 4-bit parameters and the server exits on startup.

    Args:
        model_id: Hugging Face id of the Unsloth bitsandbytes checkpoint.

    Returns:
        A dict suitable for vLLM ``--hf-overrides``.
    """
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(model_id, "config.json")
    quantization = json.loads(Path(config_path).read_text(encoding="utf-8"))[
        "quantization_config"
    ]
    prefix = "model.language_model."
    quantization["llm_int8_skip_modules"] = [
        "language_model.model." + name[len(prefix):]
        if name.startswith(prefix)
        else name
        for name in quantization["llm_int8_skip_modules"]
    ]
    return {"quantization_config": quantization}


# Un checkpoint "*-unsloth-bnb-4bit" evita di scaricare ~16 GB di pesi fp16
# e di quantizzarli a ogni avvio.
RECIPE = ModelRecipe(
    id="unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
    model_name="unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
    max_seq_length=16384,
    load_in_4bit=True,
    lora=LoRASpec(
        r=16,
        alpha=32,
        dropout=0.00,
        # Proiezioni attention e MLP di Gemma 4.
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
    decoding={
        "no_thinking": DecodingSetup(
            generation=GenerationSpec(
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                min_p=0.0,
            ),
            # Con enable_thinking=False non c'è un canale di thinking vuoto da chiudere,
            # a differenza di Qwen3.
            generation_prompt_suffix="<|turn>model\n",
            template_kwargs={"enable_thinking": False},
        ),
    },
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
        support_multi_tool_calls=True,
        tool_calls=GEMMA4_TOOL_CALLS,
    ),
    vllm=VLLMSpec(
        tool_call_parser="gemma4",
        # Rewrite Unsloth skip-module names for bitsandbytes + vLLM.
        hf_overrides=gemma4_bnb_hf_overrides,
    ),
    sft_defaults=SFTHyperparams(
        learning_rate=2e-4,
        num_train_epochs=2.0,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
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
    ),
    # Punti di partenza per GRPO su LoRA 4-bit, non un run già tarato.
    # Learning rate più basso dell'SFT: il vantaggio policy è rumoroso.
    grpo_defaults=GRPOHyperparams(
        learning_rate=5e-6,
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
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
    ),
)
