"""Llama-3.1-8B-Instruct, checkpoint Unsloth 4-bit. SFT, GRPO, and vLLM defaults."""
from __future__ import annotations

from shared.config import (
    ChatTemplateSpec,
    DecodingSetup,
    GenerationSpec,
    LoRASpec,
    ModelRecipe,
    VLLMSpec,
)
from shared.configs.llama.tool_calls import LLAMA3_JSON_TOOL_CALLS


RECIPE = ModelRecipe(
    id="unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
    model_name="unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
    max_seq_length=16384,
    load_in_4bit=True,
    lora=LoRASpec(
        r=16,
        alpha=32,
        # Dropout diverso da 0 disattiva il patching veloce di Unsloth sulle matrici LoRA.
        dropout=0.0,
        # Proiezioni attention e MLP di Llama 3.1.
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
    # generation_config del checkpoint: temperature 0.6, top_p 0.9.
    # top_k non è nel file, quindi resta il default di transformers.
    # Nessun canale di thinking.
    decoding={
        "no_thinking": DecodingSetup(
            generation=GenerationSpec(
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                top_k=50,
                min_p=0.0,
            ),
            generation_prompt_suffix=(
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
        ),
    },
    chat=ChatTemplateSpec(
        # Header Llama 3.1, due newline inclusi: Unsloth li cerca verbatim.
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        # Con `tools=` (non `builtin_tools`) il turno assistant chiude con eot_id.
        turn_end_marker="<|eot_id|>",
        forbidden_in_loss=(
            "<|start_header_id|>user",
            "<|start_header_id|>system",
            # I risultati dei tool sono un turno `ipython`, non assistant.
            "<|start_header_id|>ipython",
        ),
        # Il Jinja di Llama 3.1 alza eccezione su tool_calls parallele.
        support_multi_tool_calls=False,
        tool_calls=LLAMA3_JSON_TOOL_CALLS,
    ),
    vllm=VLLMSpec(tool_call_parser="llama3_json"),
    # sft_defaults / grpo_defaults: BASE_SFT / BASE_GRPO.
)
