"""Llama-3.2-3B-Instruct, checkpoint Unsloth 4-bit. SFT, GRPO, and vLLM defaults."""
from __future__ import annotations

from dataclasses import replace

from shared.config import (
    BASE_GRPO,
    BASE_SFT,
    ChatTemplateSpec,
    GenerationSpec,
    LoRASpec,
    ModelRecipe,
    VLLMSpec,
)


RECIPE = ModelRecipe(
    id="unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    model_name="unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    max_seq_length=16384,
    load_in_4bit=True,
    lora=LoRASpec(
        r=16,
        alpha=32,
        # Dropout diverso da 0 disattiva il patching veloce di Unsloth sulle matrici LoRA.
        dropout=0.0,
        # Proiezioni attention e MLP di Llama 3.2.
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
    generation=GenerationSpec(
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        top_k=50,
        min_p=0.0,
    ),
    chat=ChatTemplateSpec(
        # Header Llama 3.2, due newline inclusi: Unsloth li cerca verbatim.
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        # Il template chiude i turni con eot_id (nessun eom_id / builtin_tools).
        turn_end_marker="<|eot_id|>",
        forbidden_in_loss=(
            "<|start_header_id|>user",
            "<|start_header_id|>system",
            # I risultati dei tool sono un turno `ipython` (accetta anche role tool).
            "<|start_header_id|>ipython",
        ),
        # Nessun canale di thinking: il prompt di generazione è solo l'header assistant.
        generation_prompt_suffix="<|start_header_id|>assistant<|end_header_id|>\n\n",
        # Jinja: "This model only supports single tool-calls at once!".
        support_multi_tool_calls=False,
        # Default del template: tools_in_user_message=true, date via strftime_now.
        template_kwargs={},
    ),
    vllm=VLLMSpec(tool_call_parser="llama3_json"),
    # 3B: batch più alto e checkpoint più frequenti.
    sft_defaults=replace(
        BASE_SFT,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        save_steps=20,
        eval_steps=20,
    ),
    grpo_defaults=BASE_GRPO,
)
