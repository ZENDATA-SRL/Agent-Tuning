"""CLI entry point for the SFT training pipeline.

Usage examples:

  # Default Gemma 4 E4B run on a curated trace dataset:
  python -m train \\
    --dataset data/traces-2026-05-18.jsonl \\
    --output-dir outputs/sft-gemma4-e4b-v1

  # Override model. The chat template, special tokens, and loss-masking
  # markers are picked automatically by `train.templates.get_template_adapter`
  # based on the model id — register a new adapter there to support a new
  # family.
  python -m train \\
    --dataset data/traces-2026-05-18.jsonl \\
    --output-dir outputs/sft-qwen2.5-7b \\
    --model-name unsloth/Qwen2.5-7B-Instruct-bnb-4bit

  # Smoke test — disable 4-bit only if you have the VRAM:
  python -m train \\
    --dataset data/traces-2026-05-18.jsonl \\
    --output-dir outputs/smoke \\
    --num-epochs 0.1 \\
    --logging-steps 1

  # Run on a remote GPU host via SSH (connection details in `.env`):
  python -m train --remote \\
    --dataset data/traces-2026-05-18.jsonl \\
    --output-dir outputs/sft-gemma4-e4b-v1

  # Track training in Weights & Biases (works locally and over SSH):
  python -m train --wandb \\
    --wandb-project agent-tuning --wandb-tags sft,gemma-4 \\
    --dataset data/traces-2026-05-18.jsonl \\
    --output-dir outputs/sft-gemma4-e4b-v1
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from train.config import RemoteConfig, SFTConfig
from train.sft import run_sft

load_dotenv()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Supervised fine-tune an LLM on an agent trace dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # I/O
    p.add_argument("--dataset", required=True, help="Path to the .jsonl dataset file.")
    p.add_argument(
        "--output-dir",
        required=True,
        help="Where to save the LoRA adapter and tokenizer.",
    )

    # Model
    p.add_argument(
        "--model-name",
        default=os.getenv("TRAIN_MODEL_NAME", SFTConfig.model_name),
        help="HuggingFace model id (or local path). Falls back to the "
             "TRAIN_MODEL_NAME env var, then to the SFTConfig default.",
    )
    p.add_argument("--max-seq-length", type=int, default=SFTConfig.max_seq_length)
    p.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (use bf16 LoRA instead — needs more VRAM).",
    )

    # LoRA
    p.add_argument("--lora-r", type=int, default=SFTConfig.lora_r)
    p.add_argument("--lora-alpha", type=int, default=SFTConfig.lora_alpha)
    p.add_argument("--lora-dropout", type=float, default=SFTConfig.lora_dropout)

    # Training
    p.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float,
                   default=SFTConfig.learning_rate)
    p.add_argument("--num-epochs", type=float, default=SFTConfig.num_train_epochs)
    p.add_argument("--batch-size", type=int,
                   default=SFTConfig.per_device_train_batch_size)
    p.add_argument("--grad-accum", type=int,
                   default=SFTConfig.gradient_accumulation_steps)
    p.add_argument("--warmup-ratio", type=float, default=SFTConfig.warmup_ratio)
    p.add_argument("--weight-decay", type=float, default=SFTConfig.weight_decay)
    p.add_argument("--lr-scheduler", default=SFTConfig.lr_scheduler_type)
    p.add_argument("--optim", default=SFTConfig.optim)
    p.add_argument("--max-grad-norm", type=float, default=SFTConfig.max_grad_norm)
    p.add_argument("--seed", type=int, default=SFTConfig.seed)
    p.add_argument("--logging-steps", type=int, default=SFTConfig.logging_steps)
    p.add_argument("--save-steps", type=int, default=SFTConfig.save_steps)
    p.add_argument(
        "--mixed-precision",
        default=SFTConfig.mixed_precision,
        choices=["bf16", "fp16", "no"],
        help="Mixed precision mode. bf16 recommended for Ampere+ GPUs (3090/4090/A100).",
    )
    p.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (faster but uses more VRAM).",
    )

    # Loss masking
    p.add_argument(
        "--no-tool-calls-only",
        action="store_true",
        help="Disable selective tool-call masking and revert to training on all "
             "assistant tokens (vanilla train_on_responses_only behaviour).",
    )
    p.add_argument(
        "--loss-include-final-response",
        action="store_true",
        help="When --no-tool-calls-only is NOT set, also include the last "
             "assistant turn (the final answer) in the loss in addition to "
             "tool-call tokens.",
    )
    p.add_argument(
        "--log-generate-samples",
        action="store_true",
        help="At each logging step, run model.generate() on a few training "
             "examples and print/log the model's current output next to the "
             "expected response.  Disabled by default to avoid OOM: enabling "
             "this requires enough free VRAM on top of the training footprint "
             "to hold the generation KV-cache.",
    )

    # Reporting
    p.add_argument("--report-to", default=SFTConfig.report_to,
                   help='Trainer reporter ("none", "wandb", "tensorboard", ...). '
                        'Auto-set to "wandb" if any --wandb* flag (or WANDB_* env '
                        'var) is provided.')

    # Weights & Biases. Setting any of these is treated as an implicit
    # `--report-to wandb`, so the most common case (`--wandb-project foo`)
    # works without needing both flags.
    p.add_argument(
        "--wandb",
        action="store_true",
        help='Shortcut for `--report-to wandb`.',
    )
    p.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT"),
                   help="W&B project name. Default: $WANDB_PROJECT or 'agent-tuning'.")
    p.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"),
                   help="W&B entity (user or team). Default: $WANDB_ENTITY.")
    p.add_argument("--wandb-run-name", default=None,
                   help="W&B run name. Default: <output_dir>-<model>-<utc_timestamp>.")
    p.add_argument("--wandb-tags", default=os.getenv("WANDB_TAGS"),
                   help="Comma-separated W&B tags (e.g. 'sft,gemma-4,smoke').")

    # Remote execution
    p.add_argument(
        "--remote",
        action="store_true",
        help="Run training on a remote GPU host via SSH + rsync. "
             "Connection details are read from the SSH_REMOTE_* env vars "
             "(see .env.example).",
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Auto-promote to "wandb" reporting if the user passed any wandb-specific
    # flag, or set WANDB_API_KEY in the environment as a clear opt-in signal.
    # This keeps the common case ergonomic: `--wandb-project foo` is enough.
    # Computed here (before the remote branch) so the resolved value can be
    # injected into the forwarded argv when dispatching to a remote host.
    wandb_requested = (
        args.wandb
        or args.wandb_project
        or args.wandb_entity
        or args.wandb_run_name
        or args.wandb_tags
        or "wandb" in (args.report_to or "").lower()
        or bool(os.getenv("WANDB_API_KEY"))
    )
    report_to = "wandb" if wandb_requested else args.report_to

    if args.remote:
        remote = RemoteConfig.from_env()
        if remote is None:
            sys.exit(
                "[train] --remote requested but SSH_REMOTE_HOST / SSH_REMOTE_USER "
                "are not set. Configure them in .env (see .env.example)."
            )
        # Imported lazily so a local-only run does not require `train.remote`
        # to import cleanly (it only depends on stdlib, so this is mostly
        # defensive).
        from train.remote import run_remote

        # Inject the resolved --report-to into the forwarded argv so the
        # remote process gets "wandb" when WANDB_API_KEY is in the local .env
        # (which is not synced to the remote). Without this the remote sees
        # report_to="none" and wandb.init() is never called.
        forwarded = sys.argv[1:]
        if report_to and "--report-to" not in forwarded:
            forwarded = forwarded + ["--report-to", report_to]

        run_remote(
            remote,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            forwarded_argv=forwarded,
        )
        return

    config = SFTConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        optim=args.optim,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        loss_on_tool_calls_only=not args.no_tool_calls_only,
        loss_include_final_response=args.loss_include_final_response,
        log_generate_samples=args.log_generate_samples,
        report_to=report_to,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )

    run_sft(config)


if __name__ == "__main__":
    main()
