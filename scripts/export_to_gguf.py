"""Export a LoRA adapter to a single GGUF file for Ollama / llama.cpp.

Loads the adapter (which references the base model in the HF cache),
merges the LoRA weights, and writes one quantised .gguf.

Usage:
    python scripts/export_to_gguf.py \\
        --adapter outputs/sft-gemma4-e2b-real-v1 \\
        --output  outputs/sft-gemma4-e2b-gguf

    python scripts/export_to_gguf.py \\
        --adapter outputs/sft-gemma4-e2b-real-v1 \\
        --output  outputs/sft-gemma4-e2b-gguf \\
        --quant   q8_0
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and export to GGUF.")
    parser.add_argument("--adapter", required=True, help="Path to the LoRA adapter directory.")
    parser.add_argument("--output", required=True, help="Destination directory for the .gguf file.")
    parser.add_argument("--quant", default="q4_k_m", help="Quantisation method (e.g. q4_k_m, q8_0, f16).")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    args = parser.parse_args()

    from unsloth import FastLanguageModel  # type: ignore

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    print(f"[export-gguf] Loading adapter: {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    print(f"[export-gguf] Saving GGUF (quant={args.quant}) → {output}")
    model.save_pretrained_gguf(str(output), tokenizer, quantization_method=args.quant)
    print("[export-gguf] Done.")


if __name__ == "__main__":
    main()
