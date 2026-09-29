"""Qwen3-8B, checkpoint Unsloth 4-bit. SFT, GRPO, and vLLM defaults."""
from __future__ import annotations

from shared.config import (
    ChatTemplateSpec,
    GenerationSpec,
    LoRASpec,
    ModelRecipe,
    VLLMSpec,
)

# Un checkpoint "*-unsloth-bnb-4bit" evita di scaricare ~16 GB di pesi fp16
# e di quantizzarli a ogni avvio.
RECIPE = ModelRecipe(
    id="unsloth/Qwen3-8B-unsloth-bnb-4bit",
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
        support_multi_tool_calls=True,
        template_kwargs={"enable_thinking": False},
    ),
    vllm=VLLMSpec(tool_call_parser="hermes"),
    # sft_defaults / grpo_defaults: BASE_SFT / BASE_GRPO.
)
