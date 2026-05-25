"""Weights & Biases bootstrap for the SFT pipeline.

Why a dedicated module instead of just `report_to="wandb"` in TRL?
------------------------------------------------------------------
TRL's `SFTConfig(report_to="wandb")` does call `wandb.init()` for you, but
*lazily* and with a default project name (the script's basename) and no
control over the run name, tags, or config dict. For research runs we want:

  - deterministic, human-readable run names that link back to the
    `output_dir` so the local checkpoint and the remote W&B run can be
    cross-referenced at a glance,
  - the *full* `SFTConfig` (and a snapshot of the actually-loaded model id
    and dataset path) logged as `wandb.config`, so a year from now you
    can replay a run from its W&B page alone,
  - a couple of system metrics (GPU name, VRAM, transformers / unsloth /
    trl versions) attached up-front, before training starts,
  - a clean `finish()` in a `finally:` so exits-on-error still close
    the run instead of leaving it dangling.

Calling `wandb.init(...)` explicitly here BEFORE `SFTTrainer` is built
makes TRL's W&B integration pick up our pre-existing run instead of
creating a fresh one, so we get all of the above + TRL's per-step
loss/lr/grad-norm logging "for free".
"""
from __future__ import annotations

import os
import platform
import socket
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from train.config import SFTConfig


def _wants_wandb(config: SFTConfig) -> bool:
    """Whether the current run should initialise W&B.

    True when `report_to` includes "wandb" (the canonical TRL trigger), OR
    when *any* of the `wandb_*` fields was explicitly set on the config, OR
    when `WANDB_API_KEY` is present in the environment — treating all of
    these as implicit opt-ins keeps the CLI ergonomic (`--wandb-project=foo`
    or a populated `.env` should both "just work").
    """
    if "wandb" in (config.report_to or "").lower():
        return True
    if os.getenv("WANDB_API_KEY"):
        return True
    return any([
        config.wandb_project,
        config.wandb_entity,
        config.wandb_run_name,
        config.wandb_tags,
    ])


def _auto_run_name(config: SFTConfig) -> str:
    """Stable, human-readable run name derived from output_dir + model + UTC ts."""
    out_basename = Path(config.output_dir).name or "sft-run"
    short_model = config.model_name.split("/")[-1]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{out_basename}-{short_model}-{ts}"


def _system_info() -> dict[str, Any]:
    """Collect a small footprint of system facts to log as `wandb.config`.

    We avoid heavy imports here so that callers who don't actually use
    W&B don't pay the cost of importing torch/cuda just to call this.
    """
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    for pkg in ("torch", "transformers", "trl", "peft", "accelerate", "unsloth", "datasets"):
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            info[f"{pkg}_version"] = "not_installed"

    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            info["gpu_name"] = torch.cuda.get_device_name(dev)
            props = torch.cuda.get_device_properties(dev)
            info["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
        else:
            info["gpu_name"] = None
    except Exception as exc:  # noqa: BLE001
        info["gpu_probe_error"] = str(exc)

    return info


def _parse_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def maybe_init_wandb(config: SFTConfig) -> "Any | None":
    """Initialise a W&B run if the config asks for it; return the run handle.

    Side-effects:
      - Sets `WANDB_PROJECT` / `WANDB_NAME` / `WANDB_TAGS` env vars *as well*,
        so that any code path that bypasses our handle (e.g. TRL's lazy
        `wandb.init()` fallback if our import chain breaks) still picks up
        the right values.
      - Calls `wandb.init(...)` exactly once. Re-entry into this function is
        guarded so accidental double-init in pytest / notebooks is a no-op.
      - On any failure (missing API key, offline mode, network blip) the
        function logs a warning and returns None instead of raising — we'd
        rather train without telemetry than lose a 4-hour run to a typo.
    """
    if not _wants_wandb(config):
        return None

    try:
        import wandb  # type: ignore
    except ImportError:
        print(
            "[train.wandb] WARNING: wandb is requested but not installed. "
            "Install it with `pip install wandb` and re-run. Continuing "
            "without telemetry.",
            file=sys.stderr,
        )
        return None

    if wandb.run is not None:
        # Honor an externally-started run (e.g. when called from a sweep
        # agent) instead of starting a second one.
        return wandb.run

    project = config.wandb_project or os.getenv("WANDB_PROJECT") or "agent-tuning"
    entity = config.wandb_entity or os.getenv("WANDB_ENTITY") or None
    run_name = config.wandb_run_name or _auto_run_name(config)
    tags = _parse_tags(config.wandb_tags) or _parse_tags(os.getenv("WANDB_TAGS"))

    os.environ.setdefault("WANDB_PROJECT", project)
    os.environ["WANDB_NAME"] = run_name
    if tags:
        os.environ["WANDB_TAGS"] = ",".join(tags)
    # Prevent wandb from reading an empty WANDB_ENTITY from the environment and
    # sending it to the API, which returns a 400 "entityName required" error.
    if not entity:
        os.environ.pop("WANDB_ENTITY", None)
    else:
        os.environ["WANDB_ENTITY"] = entity

    full_config: dict[str, Any] = {}
    full_config.update(asdict(config))
    full_config["dataset_path_resolved"] = str(Path(config.dataset_path).resolve())
    full_config["output_dir_resolved"] = str(Path(config.output_dir).resolve())
    full_config["argv"] = sys.argv
    full_config.update({f"system/{k}": v for k, v in _system_info().items()})

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=tags or None,
            config=full_config,
            dir=str(Path(config.output_dir)),
            # `resume="allow"` triggers a resume-status API lookup even for
            # brand-new runs, which requires an entity and fails with a 400
            # when the account has no default entity configured.
            # We only need resume semantics if the caller explicitly passes a
            # wandb run id to continue; since run names here are always
            # timestamped-unique, omit resume entirely for fresh starts.
            resume=None,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[train.wandb] WARNING: wandb.init failed ({type(exc).__name__}: {exc}). "
            f"Continuing without telemetry.",
            file=sys.stderr,
        )
        # Disable wandb process-wide so that TRL's WandbCallback (triggered by
        # `report_to="wandb"` in SFTConfig) does not attempt a second
        # `wandb.init()` call during trainer.train() and crash fatally.
        os.environ["WANDB_MODE"] = "disabled"
        return None

    print(f"[train.wandb] Run initialised: {run.url if hasattr(run, 'url') else run_name}")
    return run


def log_trainable_parameters(model: Any) -> None:
    """Log the count / fraction of trainable parameters under `model_stats/*`.

    Only called for PEFT-wrapped models. Quietly no-ops if W&B is not
    active or the model object doesn't expose the expected attributes.
    """
    try:
        import wandb  # type: ignore
    except ImportError:
        return
    if wandb.run is None:
        return

    try:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        wandb.run.summary["model_stats/trainable_parameters"] = trainable
        wandb.run.summary["model_stats/total_parameters"] = total
        wandb.run.summary["model_stats/trainable_fraction"] = trainable / total if total else 0.0
    except Exception:  # noqa: BLE001
        pass


def finish_wandb(run: "Any | None") -> None:
    """Best-effort `wandb.finish()`; safe to call with None."""
    if run is None:
        return
    try:
        import wandb  # type: ignore
        wandb.finish()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Sample-logging callback
# ---------------------------------------------------------------------------

class SampleLogCallback:
    """TrainerCallback that logs expected vs. generated responses each logging step.

    At every ``logging_steps`` the callback picks ``num_samples`` examples,
    prints the **expected** completion to stdout, and — only when
    ``generate_samples=True`` — also runs a forward pass to produce the
    model's current output for comparison.

    **Why ``generate_samples`` defaults to False:**
    Calling ``model.generate()`` mid-training stacks the KV-cache and the
    generation buffers on top of the already-resident optimizer states,
    parameters, and (with gradient-checkpointing off) stored activations.
    On a 24 GB consumer GPU this reliably causes OOM.  Set
    ``generate_samples=True`` only if you have spare VRAM headroom (a rough
    rule of thumb: ≥ 2× the model's idle-inference footprint on top of the
    training footprint).

    Args:
        model: The (PEFT-wrapped) model being trained.
        tokenizer: The model's tokenizer / processor.
        raw_texts: Pre-rendered chat strings (one per training example), used
            to re-extract the prompt prefix (everything before the last
            assistant turn) and the expected completion.
        adapter: ``ChatTemplateAdapter`` that knows the ``response_part``
            marker string so we can split prompt from completion.
        num_samples: How many examples to log at each logging step.
        generate_samples: If True, run ``model.generate()`` to produce the
            agent's current response alongside the expected one.  Disabled
            by default to avoid VRAM spikes during training.
        max_new_tokens: Maximum tokens the model may generate per sample
            (only relevant when ``generate_samples=True``).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        raw_texts: list[str],
        adapter: Any,
        *,
        num_samples: int = 2,
        generate_samples: bool = False,
        max_new_tokens: int = 256,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.raw_texts = raw_texts
        self.adapter = adapter
        self.num_samples = min(num_samples, len(raw_texts))
        self.generate_samples = generate_samples
        self.max_new_tokens = max_new_tokens
        self._wandb_columns = ["step", "sample_idx", "expected", "generated"]
        self._pending_rows: list[list] = []

    # ------------------------------------------------------------------
    # TrainerCallback protocol (called by Hugging Face Trainer)
    # ------------------------------------------------------------------

    def on_log(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Called by the Trainer at each logging step."""
        import random

        step = state.global_step
        indices = random.sample(range(len(self.raw_texts)), self.num_samples)

        print("\n" + "=" * 70)
        print(f"[train.sample_log] Step {step} — expected responses")
        print("=" * 70)

        for idx in indices:
            full_text = self.raw_texts[idx]
            _prompt, expected = self._split_prompt_expected(full_text)

            generated = self._generate(full_text) if self.generate_samples else None

            print(f"\n--- Sample {idx} ---")
            print(f"[EXPECTED]\n{expected.strip()}")
            if generated is not None:
                print(f"[GENERATED]\n{generated.strip()}")

            self._log_to_wandb(step, idx, expected.strip(), generated or "")

        print("=" * 70 + "\n")
        self._flush_wandb_table(step)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate(self, full_text: str) -> str:
        """Run greedy generation for the last assistant turn.

        Switches the model to eval mode, clears the CUDA allocator cache
        before and after, then restores training mode.  This minimises
        the peak VRAM delta but still requires enough free VRAM to hold
        the KV-cache for ``max_new_tokens`` tokens.
        """
        import torch  # type: ignore

        prompt, _ = self._split_prompt_expected(full_text)
        _inner_tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        enc = _inner_tok(text=prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.model.device)

        was_training = self.model.training
        try:
            self.model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=_inner_tok.pad_token_id or _inner_tok.eos_token_id,
                )
        finally:
            if was_training:
                self.model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        generated_ids = output_ids[0][input_ids.shape[-1]:]
        return _inner_tok.decode(generated_ids, skip_special_tokens=False)

    def _split_prompt_expected(self, full_text: str) -> tuple[str, str]:
        """Return (prompt_prefix, expected_completion) by splitting at the
        last occurrence of the response marker in *full_text*."""
        marker = self.adapter.response_part  # e.g. "<|turn>model\n"
        last_pos = full_text.rfind(marker)
        if last_pos == -1:
            return full_text, ""
        split_pos = last_pos + len(marker)
        return full_text[:split_pos], full_text[split_pos:]

    def _log_to_wandb(
        self, step: int, idx: int, expected: str, generated: str
    ) -> None:
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        if wandb.run is None:
            return

        # Accumulate rows for this step in a plain list; the table is
        # assembled and logged once per on_log call (see on_log) rather
        # than kept as a growing instance attribute.  A persistent
        # wandb.Table that receives add_data() every step causes wandb to
        # hold a cumulative copy in memory for diffing, which grows
        # without bound and steadily increases RAM (and VRAM).
        if not hasattr(self, "_pending_rows"):
            self._pending_rows: list[list] = []
        self._pending_rows.append([step, idx, expected, generated])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flush_wandb_table(self, step: int) -> None:
        """Log all pending rows as a *fresh* wandb.Table and clear the buffer."""
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        if wandb.run is None or not getattr(self, "_pending_rows", None):
            return
        table = wandb.Table(columns=self._wandb_columns, data=self._pending_rows)
        wandb.log({"samples/expected_vs_generated": table}, step=step)
        self._pending_rows = []


def make_sample_log_callback(
    model: Any,
    tokenizer: Any,
    raw_texts: list[str],
    adapter: Any,
    *,
    num_samples: int = 2,
    generate_samples: bool = False,
    max_new_tokens: int = 256,
) -> "Any":
    """Instantiate and return a ``SampleLogCallback`` wrapped as a TRL/HF
    ``TrainerCallback``, so it can be passed directly to ``SFTTrainer``.

    The indirection exists so that ``train/sft.py`` does not need to import
    ``transformers`` at module-load time (we only need it when actually
    building the trainer).

    Set ``generate_samples=True`` only when you have enough free VRAM after
    loading the model + optimizer (see ``SampleLogCallback`` docstring).
    """
    from transformers import TrainerCallback  # type: ignore

    inner = SampleLogCallback(
        model,
        tokenizer,
        raw_texts,
        adapter,
        num_samples=num_samples,
        generate_samples=generate_samples,
        max_new_tokens=max_new_tokens,
    )

    class _Wrapper(TrainerCallback):
        def on_log(self, args, state, control, **kwargs):  # noqa: ANN001
            inner.on_log(args, state, control, **kwargs)

    return _Wrapper()
