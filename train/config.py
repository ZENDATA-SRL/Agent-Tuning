from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class SFTConfig:
    """
    Configuration for a Supervised Fine-Tuning run with Unsloth + QLoRA.

    Defaults are tuned for `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` on a
    single consumer GPU (~10GB VRAM with 4-bit quantization). Most fields
    are overridable via the CLI in `train/__main__.py` so the same config
    can drive different models (Qwen2.5, Llama-3.1, ...) by swapping
    `model_name`. Anything that is *model-family specific* (chat template
    choice, loss-masking marker strings, tool-call serialization rules)
    lives in `train/templates.py` instead — see `ChatTemplateAdapter`.
    """

    # ── I/O ───────────────────────────────────────────────────────────────
    dataset_path: str
    output_dir: str

    # ── Model ─────────────────────────────────────────────────────────────
    # Default points to Unsloth's pre-quantized 4-bit checkpoint
    # (`-unsloth-bnb-4bit` suffix). Using the stock `unsloth/gemma-4-E4B-it`
    # repo with `load_in_4bit=True` would force Unsloth to download the
    # full-precision weights (~16 GB) and quantize them at load time on
    # every fresh machine — slow and disk-hungry. The pre-quantized repo
    # ships ready-to-load 4-bit safetensors (~5.5 GB).
    model_name: str = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
    max_seq_length: int = 8192
    load_in_4bit: bool = True

    # ── LoRA / PEFT ───────────────────────────────────────────────────────
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    finetune_vision_layers: bool = False  # text-only fine-tuning

    # ── Training hyperparameters ──────────────────────────────────────────
    learning_rate: float = 2e-4
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    max_grad_norm: float = 0.3
    seed: int = 3407
    logging_steps: int = 10
    save_steps: int = 40
    # Mixed precision: "bf16" (recommended for Ampere+ GPUs, e.g. A100/3090/4090),
    # "fp16" (older GPUs), or "no" to stay in fp32.
    mixed_precision: str = "bf16"
    # Recompute activations during the backward pass instead of storing them.
    # Trades ~20% speed for a large VRAM reduction — essential at seq_len ≥ 4096.
    gradient_checkpointing: bool = True

    # ── Loss masking ──────────────────────────────────────────────────────
    # When True (default), the loss is computed ONLY on tool-call tokens inside
    # each assistant turn (i.e. the spans delimited by the adapter's
    # `tool_call_start` / `tool_call_end` markers). All other assistant text
    # (e.g. preamble or chain-of-thought before the call) is masked out.
    #
    # Set to False to revert to vanilla `train_on_responses_only` behaviour
    # (all assistant tokens contribute to the loss).
    loss_on_tool_calls_only: bool = True

    # When `loss_on_tool_calls_only=True`, setting this flag also includes the
    # *last* assistant turn in the loss (the final natural-language answer that
    # closes the agentic loop). Ignored when `loss_on_tool_calls_only=False`.
    loss_include_final_response: bool = False

    # ── Sample logging ────────────────────────────────────────────────────
    # When True, the SampleLogCallback runs model.generate() at each
    # logging step to print the model's current output next to the expected
    # completion.  Disabled by default because it stacks the generation
    # KV-cache on top of the training memory footprint and easily causes
    # OOM on GPUs with ≤ 24 GB VRAM.  Enable only when you have confirmed
    # enough free headroom after loading model + optimizer.
    log_generate_samples: bool = False

    # ── Reporting ─────────────────────────────────────────────────────────
    # `report_to` is forwarded directly to TRL's `SFTConfig.report_to`. Any
    # value other than "none" enables one or more reporters (W&B, TensorBoard,
    # MLflow, ...). For W&B specifically, `train/wandb_setup.py` reads the
    # `wandb_*` fields below to call `wandb.init(...)` BEFORE TRL builds the
    # trainer — this lets us seed the run with our own name/tags/config.
    report_to: str = "none"  # "none" | "wandb" | "tensorboard" | ...

    # ── Weights & Biases (only used if `report_to` includes "wandb") ──────
    # All fields are optional; sensible defaults are derived at runtime by
    # `train.wandb_setup.maybe_init_wandb` when not provided. Resolution
    # order for each: explicit CLI flag > SFTConfig field > WANDB_* env var
    # > a sensible auto-default.
    wandb_project: str | None = None  # default: "agent-tuning"
    wandb_entity:  str | None = None  # default: WANDB_ENTITY env var (your account)
    # If None, the run name is auto-derived as
    #   "<basename(output_dir)>-<short_model_label>-<utc_timestamp>"
    # so multiple runs into the same output_dir each get a unique name.
    wandb_run_name: str | None = None
    # Comma-separated tags (e.g. "sft,gemma-4,smoke"). Useful for dashboard
    # filtering across many runs.
    wandb_tags: str | None = None

    def label(self) -> str:
        """Human-readable identifier used in reports and output filenames."""
        return self.model_name.replace("/", "_")


@dataclass
class RemoteConfig:
    """SSH connection profile for executing the training on a remote GPU host.

    All fields are populated from environment variables (typically loaded
    from `.env` via `python-dotenv`) by `RemoteConfig.from_env()`. Keep the
    secret material (private key path, password) outside the repo.

    The remote workflow is:
      1. rsync the project tree (excluding caches) → remote_workdir
      2. rsync dataset_path                        → remote_workdir/<basename>
      3. ssh into the host and run the training command in remote_workdir
      4. rsync remote_workdir/<output_dir>         ← back to local
    """

    host: str
    user: str
    port: int = 22
    # Path to the SSH private key (e.g. ~/.ssh/id_ed25519). When None the
    # user's default ssh-agent / config is used.
    key_path: str | None = None
    # Plaintext SSH password. Only populated when key-based auth is not
    # available; requires the `sshpass` binary on the local machine. Stored
    # in `.env` (which is gitignored). NEVER log or print this field.
    password: str | None = None
    # Absolute path on the remote host where the project will be synced.
    remote_workdir: str = "~/agent-tuning"
    # Python interpreter on the remote host. Set to the venv's bin/python
    # to avoid activating the venv interactively over ssh.
    python_bin: str = "python"
    # Python interpreter (or full path) used on the remote to BOOTSTRAP a
    # fresh virtualenv when the venv at `python_bin` does not yet exist.
    # Must be in PATH on the remote. Defaults to "python3.12" because
    # Unsloth officially supports 3.10–3.12.
    bootstrap_python: str = "python3.12"
    # Optional shell snippet executed before `python_bin` on the remote
    # (e.g. "source ~/agent-tuning/.venv/bin/activate" or
    # "conda activate train"). Concatenated with `&&` to the python call.
    pre_command: str | None = None
    # If True, also rsync the local dataset file to the remote. Disable
    # when the dataset already lives on the remote (e.g. on a shared FS)
    # to avoid a slow upload on every run.
    sync_dataset: bool = True
    # If True, rsync the remote `output_dir` back to the local path after
    # the training finishes. Disable for very large checkpoints when you
    # plan to fetch them out-of-band.
    fetch_output: bool = True

    @classmethod
    def from_env(cls) -> "RemoteConfig | None":
        """Build a `RemoteConfig` from environment variables.

        Returns None if `SSH_REMOTE_HOST` is not set, allowing the caller
        to gracefully fall back to local execution.

        Recognised variables (see `.env.example`):
          SSH_REMOTE_HOST           - required, e.g. "gpu-01.example.com"
          SSH_REMOTE_USER           - required, e.g. "ubuntu"
          SSH_REMOTE_PORT           - optional, default 22
          SSH_REMOTE_KEY_PATH       - optional, path to private key
          SSH_REMOTE_PASSWORD       - optional, plaintext password (uses sshpass)
          SSH_REMOTE_WORKDIR        - optional, default "~/agent-tuning"
          SSH_REMOTE_PYTHON         - optional, default "python"
          SSH_REMOTE_BOOTSTRAP_PYTHON - optional, default "python3.12"
                                        Used to create the venv if missing.
          SSH_REMOTE_PRE_COMMAND    - optional, shell snippet to prepend
          SSH_REMOTE_SYNC_DATASET   - optional, "1"/"0", default "1"
          SSH_REMOTE_FETCH_OUTPUT   - optional, "1"/"0", default "1"
        """
        host = os.getenv("SSH_REMOTE_HOST")
        user = os.getenv("SSH_REMOTE_USER")
        if not host or not user:
            return None

        def _flag(name: str, default: bool) -> bool:
            v = os.getenv(name)
            if v is None:
                return default
            return v.strip().lower() in ("1", "true", "yes", "on")

        return cls(
            host=host,
            user=user,
            port=int(os.getenv("SSH_REMOTE_PORT", "22")),
            key_path=os.getenv("SSH_REMOTE_KEY_PATH") or None,
            password=os.getenv("SSH_REMOTE_PASSWORD") or None,
            remote_workdir=os.getenv("SSH_REMOTE_WORKDIR", "~/agent-tuning"),
            python_bin=os.getenv("SSH_REMOTE_PYTHON", "python"),
            bootstrap_python=os.getenv("SSH_REMOTE_BOOTSTRAP_PYTHON", "python3.12"),
            pre_command=os.getenv("SSH_REMOTE_PRE_COMMAND") or None,
            sync_dataset=_flag("SSH_REMOTE_SYNC_DATASET", True),
            fetch_output=_flag("SSH_REMOTE_FETCH_OUTPUT", True),
        )

    def ssh_target(self) -> str:
        return f"{self.user}@{self.host}"
