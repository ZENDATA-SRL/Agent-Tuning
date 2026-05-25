"""Export a LoRA checkpoint to a merged 16-bit model ready for vLLM.

Run this script on the machine that holds the checkpoint (typically the
remote GPU server) BEFORE launching `vllm serve`.

Usage:
    python scripts/export_to_vllm.py \\
        --adapter outputs/sft-gemma4-e2b-real-v1 \\
        --output  outputs/sft-gemma4-e2b-merged-16bit

    # Specific intermediate checkpoint:
    python scripts/export_to_vllm.py \\
        --adapter outputs/sft-gemma4-e2b-real-v1/checkpoint-160 \\
        --output  outputs/sft-gemma4-e2b-ck160-merged-16bit

    # Override base model (default: read from adapter_config.json):
    python scripts/export_to_vllm.py \\
        --adapter outputs/sft-gemma4-e2b-real-v1 \\
        --output  outputs/sft-gemma4-e2b-merged-16bit \\
        --base-model unsloth/gemma-4-E2B-it-unsloth-bnb-4bit

After export, serve with:
    vllm serve outputs/sft-gemma4-e2b-merged-16bit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_base_model(adapter_dir: Path) -> str:
    """Read base_model_name_or_path from adapter_config.json."""
    cfg = adapter_dir / "adapter_config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    with open(cfg) as f:
        data = json.load(f)
    base = data.get("base_model_name_or_path")
    if not base:
        raise KeyError("base_model_name_or_path missing from adapter_config.json")
    return base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into the base model and save as 16-bit for vLLM."
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the saved LoRA adapter directory (or checkpoint-N subfolder).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination directory for the merged 16-bit model.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "HuggingFace model ID or local path for the base model. "
            "Defaults to base_model_name_or_path read from adapter_config.json."
        ),
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Maximum sequence length (passed to FastModel.from_pretrained).",
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter).resolve()
    output_path = Path(args.output).resolve()

    if not adapter_path.exists():
        raise SystemExit(f"Adapter path not found: {adapter_path}")

    base_model = args.base_model or _read_base_model(adapter_path)
    print(f"[export] Base model : {base_model}")
    print(f"[export] Adapter    : {adapter_path}")
    print(f"[export] Output     : {output_path}")

    # Unsloth must be imported AFTER torch is fully initialised.
    from unsloth import FastModel  # type: ignore

    print("[export] Loading base model + LoRA adapter (4-bit) ...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    print("[export] Merging LoRA weights and saving as 16-bit ...")
    model.save_pretrained_merged(
        str(output_path),
        tokenizer,
        save_method="merged_16bit",
    )

    print(f"[export] Done. Merged model saved to: {output_path}")
    print()
    print("Start vLLM with:")
    print(f"    vllm serve {output_path}")


if __name__ == "__main__":
    main()
