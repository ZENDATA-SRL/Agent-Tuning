"""Launchable entrypoint: merge a saved LoRA adapter and export to GGUF.

The adapter directory must contain `adapter_config.json` (referencing the
base model) and the LoRA weights. We reload the base model in 4-bit, the
adapter is automatically picked up by Unsloth, and the merged 16-bit model
is converted to a single quantised .gguf file ready for Ollama / llama.cpp.

Edit the constants below, then run via the "Export: Qwen3 LoRA → GGUF"
configuration in `.vscode/launch.json`.

Quantisation methods: q4_k_m (recommended), q5_k_m, q8_0, f16.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── Edit these ────────────────────────────────────────────────────────────
ADAPTER_DIR = "outputs/sft-qwen3-8b-v1"
OUTPUT_DIR = "outputs/sft-qwen3-8b-gguf"
QUANT_METHOD = "q4_k_m"
MAX_SEQ_LENGTH = 4096
# ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    from unsloth import FastLanguageModel  # type: ignore

    adapter = Path(ADAPTER_DIR).resolve()
    output = Path(OUTPUT_DIR).resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not adapter.exists():
        raise SystemExit(f"Adapter directory not found: {adapter}")

    print(f"[export-gguf] Loading adapter: {adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    print(f"[export-gguf] Saving GGUF (quant={QUANT_METHOD}) → {output}")
    model.save_pretrained_gguf(str(output), tokenizer, quantization_method=QUANT_METHOD)
    print("[export-gguf] Done.")


if __name__ == "__main__":
    main()
