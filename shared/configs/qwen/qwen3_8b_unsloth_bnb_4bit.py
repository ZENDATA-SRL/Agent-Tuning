"""Qwen3-8B, checkpoint Unsloth 4-bit. SFT, GRPO, and vLLM defaults."""
from __future__ import annotations

from dataclasses import replace

from shared.config import (
    BASE_GRPO,
    ChatTemplateSpec,
    DecodingSetup,
    GenerationOverride,
    GenerationSpec,
    LoRASpec,
    ModelRecipe,
    VLLMSpec,
)
from shared.configs.qwen.tool_calls import HERMES_TOOL_CALLS

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
    # Parametri pubblicati da Qwen3. Lo script sceglie la chiave.
    decoding={
        "no_thinking": DecodingSetup(
            generation=GenerationSpec(
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
            ),
            # Il blocco <think> vuoto già chiuso tiene il thinking spento.
            generation_prompt_suffix=(
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            ),
            template_kwargs={"enable_thinking": False},
        ),
        "thinking": DecodingSetup(
            generation=GenerationSpec(
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
            ),
            # Senza </think> il template lascia il canale di thinking acceso.
            generation_prompt_suffix="<|im_start|>assistant\n",
            template_kwargs={"enable_thinking": True},
        ),
    },
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
        support_multi_tool_calls=True,
        tool_calls=HERMES_TOOL_CALLS,
    ),
    vllm=VLLMSpec(tool_call_parser="hermes"),
    # SFT usa il GenerationSpec della modalità. GRPO allenta solo temperatura e top_p.
    grpo_defaults=replace(
        BASE_GRPO,
        generation_override=GenerationOverride(
            temperature=1.0,
            top_p=0.95,
            top_k=0, # all token distribution is used
        ),
    ),
)
