"""Export a LoRA checkpoint to a merged 16-bit model ready for vLLM.

The LoRA adapter is merged into the base model and saved as bfloat16.
vLLM then loads the 16-bit weights and applies bitsandbytes 4-bit quantisation
at serving time via --quantization bitsandbytes, achieving the same VRAM
footprint as training.

WHY NOT PRE-QUANTISED BNB NF4?
  unsloth's merged_4bit_forced saves weights in peft NF4 packed format
  (shape [out, in/2]).  vLLM's bitsandbytes_loader expects FP16 weights
  (shape [out, in]) and quantises them itself.  Loading pre-packed NF4
  weights causes an AssertionError in vllm/model_executor/layers/linear.py.

WHY NOT AWQ / llm-compressor?
  AutoAWQ is deprecated and does not support Gemma4.
  llm-compressor requires transformers<=4.57.6; Gemma4 needs >=5.5.x.

Run this script on the machine that holds the checkpoint BEFORE launching vllm.

Usage:
    python scripts/export_to_vllm.py \\
        --adapter outputs/sft-gemma4-e2b-real-v1 \\
        --output  outputs/sft-gemma4-e2b-merged

    # Specific intermediate checkpoint:
    python scripts/export_to_vllm.py \\
        --adapter outputs/sft-gemma4-e2b-real-v1/checkpoint-160 \\
        --output  outputs/sft-gemma4-e2b-ck160-merged

After export, serve with (in the vLLM venv):
    vllm serve outputs/sft-gemma4-e2b-merged \\
        --quantization bitsandbytes \\
        --load-format bitsandbytes \\
        --gpu-memory-utilization 0.90 \\
        --max-model-len 4096 \\
        --enable-auto-tool-choice \\
        --tool-call-parser pythonic \\
        --host 0.0.0.0 --port 8000
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


def _patch_disable_weight_conversion() -> None:
    """Disable transformers 5.x weight_conversion/revert_weight_conversion for Gemma4.

    Root cause of empty/garbage vLLM output:
      - transformers 5.x applies weight_conversion when loading Gemma4 (converts
        checkpoint weights → internal representation).
      - On save, revert_weight_conversion should convert back.  For Gemma4 in 5.5.x
        several ops lack reverse_op, raising NotImplementedError.
      - Catching NotImplementedError and returning state_dict unchanged saves weights
        in *internal* format.  When vLLM loads these with transformers 5.9+ it applies
        weight_conversion again → double-converted → garbage / <pad> tokens.

    Fix: make both functions no-ops so weights are always in checkpoint format.
      - Load:  no conversion applied, _weight_conversion_applied stays False.
      - Save:  revert_weight_conversion is not called (flag is False); weights are
               saved in the original checkpoint format.
      - vLLM:  loads checkpoint-format weights and applies weight_conversion once
               (correctly) → correct internal representation → model works.
    """
    try:
        import transformers.core_model_loading as _cml

        def _noop(model, state_dict):  # noqa: ARG001
            return state_dict

        _cml.weight_conversion = _noop
        _cml.revert_weight_conversion = _noop
    except Exception:
        pass

    try:
        import transformers.modeling_utils as _mu

        def _noop_mu(model, state_dict):  # noqa: ARG001
            return state_dict

        if hasattr(_mu, "weight_conversion"):
            _mu.weight_conversion = _noop_mu
        if hasattr(_mu, "revert_weight_conversion"):
            _mu.revert_weight_conversion = _noop_mu
    except Exception:
        pass


def _patch_unsloth_zoo_gemma4() -> None:
    """Fix incompatibility between unsloth_zoo's Gemma4 proxy and transformers>=5.9.

    unsloth_zoo raises AttributeError on num_kv_shared_layers==0 to make hasattr()
    return False (avoiding the layer_types[:-0] bug).  transformers 5.9+ calls
    getattr() directly without a default, crashing.  The fix returns 0 instead.
    """
    try:
        from unsloth_zoo.temporary_patches import gemma4 as _g4  # type: ignore
        proxy_cls = _g4._Gemma4KVSharedSafeProxy
        original_getattr = proxy_cls.__getattr__

        def _patched_getattr(self, name: str):  # type: ignore[override]
            if name == "num_kv_shared_layers":
                return 0
            return original_getattr(self, name)

        proxy_cls.__getattr__ = _patched_getattr
    except Exception:
        pass


def export_merged(
    adapter_path: Path,
    output_path: Path,
    max_seq_length: int,
) -> None:
    from unsloth import FastLanguageModel  # type: ignore

    _patch_unsloth_zoo_gemma4()
    _patch_disable_weight_conversion()

    print("[export] Loading base model + LoRA adapter (4-bit) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[export] Merging LoRA → 16-bit and saving to {output_path} ...")
    print("[export] Note: weight_conversion disabled — weights saved in checkpoint format.")
    model.save_pretrained_merged(
        str(output_path),
        tokenizer,
        save_method="merged_16bit",
    )

    print(f"[export] Done. Merged 16-bit model saved to: {output_path}")
    print()
    print("Start vLLM with (BNB 4-bit quantised at load time, ~same VRAM as training):")
    print(f"    vllm serve {output_path} \\")
    print( "        --quantization bitsandbytes \\")
    print( "        --load-format bitsandbytes \\")
    print( "        --gpu-memory-utilization 0.90 \\")
    print( "        --max-model-len 4096 \\")
    print( "        --enable-auto-tool-choice \\")
    print( "        --tool-call-parser pythonic \\")
    print( "        --host 0.0.0.0 --port 8000")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into a 16-bit model for vLLM serving (BNB 4-bit applied at load time)."
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
        help="Maximum sequence length (passed to FastLanguageModel.from_pretrained).",
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

    export_merged(
        adapter_path=adapter_path,
        output_path=output_path,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
