"""Preflight and runtime guards against CUDA out-of-memory during SFT.

The collator right-pads every row to the longest sequence in the batch, and
the default sampler can place the longest rows together. Peak memory is
therefore the longest batch, not the average one. Eval is checked separately
because its forward materializes logits, which training with fused
cross-entropy does not.

`batch_keep_percentile` (train only) may drop the longest traces at the
current batch size before that size is lowered again. Rows that do not fit
even alone are still removed afterwards: those traces cannot be trained.
"""
from __future__ import annotations

import gc
import math
from typing import Any

import torch
from datasets import Dataset
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import IntervalStrategy, SaveStrategy
from trl import SFTTrainer  # type: ignore

# Reporting callbacks would record every probe as part of the real run.
_QUIET_CALLBACKS = {"WandbCallback", "PrinterCallback", "ProgressCallback"}


def _is_oom(exc: BaseException) -> bool:
    """Return whether `exc` is a CUDA out-of-memory failure."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _is_token_row(value: Any) -> bool:
    """Return whether `value` is a 1-D sequence of token ids."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        return False
    return bool(value) and isinstance(value[0], int)


def _release_cuda(trainer: SFTTrainer) -> None:
    """Drop everything a probe step allocated and return the cache to the driver.

    The probe calls `train` or `evaluate`, which builds an optimizer, a
    scheduler and a dataloader. Those references are dropped here, including
    the copies Accelerate keeps, so the next attempt starts from a free card.
    The prepared model is left in place.
    """
    trainer.model.zero_grad(set_to_none=True)
    trainer.optimizer = None
    trainer.lr_scheduler = None
    trainer._created_lr_scheduler = False
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None:
        accelerator._optimizers = []
        accelerator._schedulers = []
        accelerator._dataloaders = []
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def _sequence_lengths(dataset: Any) -> list[int]:
    """Return the token length of every row in a tokenized dataset."""
    columns = getattr(dataset, "column_names", None)
    if not columns or "input_ids" not in columns:
        raise RuntimeError(
            "Il dataset non ha la colonna 'input_ids': impossibile misurare "
            "la lunghezza delle sequenze."
        )
    return [len(ids) for ids in dataset["input_ids"]]


def _longest_indices(lengths: list[int], count: int) -> list[int]:
    """Return the indices of the `count` longest sequences."""
    count = min(count, len(lengths))
    order = sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True)
    return order[:count]


def _row(dataset: Any, index: int) -> dict:
    """Return one dataset row as a dict."""
    row = dataset[index]
    if not isinstance(row, dict):
        raise RuntimeError(
            f"Riga {index} del dataset non è un dict: {type(row).__name__}."
        )
    return row


def _truncate_sequence_fields(example: dict, max_length: int) -> dict:
    """Copy `example`, cutting token-id fields to `max_length`."""
    trimmed: dict = {}
    for key, value in example.items():
        if _is_token_row(value) and len(value) > max_length:
            trimmed[key] = list(value[:max_length])
        else:
            trimmed[key] = value
    return trimmed


def _format_lengths(lengths: list[int], indices: list[int]) -> str:
    """Format the chosen sequence lengths for a log line."""
    return ", ".join(str(lengths[index]) for index in indices)


def _capture_run(trainer: SFTTrainer) -> dict[str, Any]:
    """Snapshot the trainer fields a one-step probe overwrites."""
    args = trainer.args
    return {
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "eval_strategy": args.eval_strategy,
        "save_strategy": args.save_strategy,
        "load_best_model_at_end": args.load_best_model_at_end,
        "logging_steps": args.logging_steps,
        "train_dataset": trainer.train_dataset,
        "eval_dataset": trainer.eval_dataset,
        "train_batch_size": trainer._train_batch_size,
        "state": trainer.state,
        "control": trainer.control,
        "is_in_train": trainer.is_in_train,
        "current_flos": getattr(trainer, "current_flos", 0),
        "callbacks": list(trainer.callback_handler.callbacks),
    }


def _restore_run(trainer: SFTTrainer, saved: dict[str, Any]) -> None:
    """Put back the fields captured by `_capture_run`."""
    args = trainer.args
    args.max_steps = saved["max_steps"]
    args.per_device_train_batch_size = saved["per_device_train_batch_size"]
    args.per_device_eval_batch_size = saved["per_device_eval_batch_size"]
    args.gradient_accumulation_steps = saved["gradient_accumulation_steps"]
    args.eval_strategy = saved["eval_strategy"]
    args.save_strategy = saved["save_strategy"]
    args.load_best_model_at_end = saved["load_best_model_at_end"]
    args.logging_steps = saved["logging_steps"]
    trainer.train_dataset = saved["train_dataset"]
    trainer.eval_dataset = saved["eval_dataset"]
    trainer._train_batch_size = saved["train_batch_size"]
    trainer.state = saved["state"]
    trainer.control = saved["control"]
    trainer.is_in_train = saved["is_in_train"]
    trainer.current_flos = saved["current_flos"]
    trainer.callback_handler.callbacks = saved["callbacks"]


def _trainable_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy trainable parameters to CPU so a probe step can be undone."""
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _restore_weights(
    model: torch.nn.Module, saved: dict[str, torch.Tensor]
) -> None:
    """Write `saved` back over the trainable parameters."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            snapshot = saved.get(name)
            if snapshot is None:
                continue
            param.copy_(snapshot.to(device=param.device, dtype=param.dtype))


def _arm_probe(
    trainer: SFTTrainer, examples: list[dict], *, eval_mode: bool
) -> None:
    """Point the trainer at `examples` for exactly one real step.

    Train goes through `trainer.train`, so checkpointing, accumulation and
    `num_items_in_batch` are whatever the real loop sets. Eval goes through
    `trainer.evaluate`. Saving, evaluation and report callbacks stay off.
    """
    dataset = Dataset.from_list([dict(example) for example in examples])
    batch_size = len(examples)
    args = trainer.args
    args.logging_steps = 1_000_000
    args.save_strategy = SaveStrategy.NO
    args.load_best_model_at_end = False
    trainer.state = TrainerState()
    trainer.control = TrainerControl()
    trainer.is_in_train = False
    trainer.callback_handler.callbacks = [
        callback
        for callback in trainer.callback_handler.callbacks
        if type(callback).__name__ not in _QUIET_CALLBACKS
    ]
    if eval_mode:
        trainer.eval_dataset = dataset
        args.per_device_eval_batch_size = batch_size
        return
    trainer.train_dataset = dataset
    args.max_steps = 1
    args.per_device_train_batch_size = batch_size
    args.gradient_accumulation_steps = 1
    args.eval_strategy = IntervalStrategy.NO
    trainer._train_batch_size = batch_size


def _fits(trainer: SFTTrainer, examples: list[dict], *, eval_mode: bool) -> bool:
    """Run one real step on `examples`. Return False when it runs out of memory.

    A step that fits is rolled back: the optimizer update is discarded and the
    trainable weights are restored. An out-of-memory step drops the optimizer,
    the scheduler and the dataloader before the next attempt.
    """
    saved_run = _capture_run(trainer)
    saved_weights = _trainable_weights(trainer.model)
    trainer._oom_probing = True  # type: ignore[attr-defined]
    try:
        _arm_probe(trainer, examples, eval_mode=eval_mode)
        if eval_mode:
            trainer.evaluate()
        else:
            trainer.train()
        return True
    except Exception as exc:
        if not _is_oom(exc):
            raise
        return False
    finally:
        trainer._oom_probing = False  # type: ignore[attr-defined]
        # Free the probe's activations before copying weights back. A card
        # that just ran out of memory cannot allocate the restore buffer.
        _release_cuda(trainer)
        _restore_weights(trainer.model, saved_weights)
        _restore_run(trainer, saved_run)


def _max_fitting_length(
    trainer: SFTTrainer,
    example: dict,
    *,
    eval_mode: bool,
    kind: str,
) -> int:
    """Binary-search the longest prefix of `example` that fits as one row.

    The full sequence has already failed. Truncation is only a measurement:
    callers drop over-long rows instead of training on a cut trace.
    """
    full = len(example["input_ids"])
    best = 0
    low = 1
    high = full - 1
    while low <= high:
        mid = (low + high) // 2
        print(f"[sft] Probing {kind} sequence of {mid} tokens...")
        probe = _truncate_sequence_fields(example, mid)
        if _fits(trainer, [probe], eval_mode=eval_mode):
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def _filter_dataset(dataset: Any, max_length: int) -> tuple[Any, int]:
    """Drop rows longer than `max_length`. Return the dataset and the drop count."""
    lengths = _sequence_lengths(dataset)
    keep = [index for index, length in enumerate(lengths) if length <= max_length]
    dropped = len(lengths) - len(keep)
    if dropped == 0:
        return dataset, 0
    return dataset.select(keep), dropped


def _set_train_batch_size(trainer: SFTTrainer, batch_size: int, effective: int) -> None:
    """Shrink the train batch and raise accumulation to keep the same effective size."""
    args = trainer.args
    previous = args.per_device_train_batch_size
    if batch_size == previous:
        return
    args.per_device_train_batch_size = batch_size
    args.gradient_accumulation_steps = max(1, math.ceil(effective / batch_size))
    # get_train_dataloader reads this cached value, not the property.
    trainer._train_batch_size = args.train_batch_size
    print(
        "[sft] Train batch reduced from "
        f"{previous} to {batch_size}; "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps} "
        f"(effective batch {batch_size * args.gradient_accumulation_steps})."
    )


def _set_eval_batch_size(trainer: SFTTrainer, batch_size: int) -> None:
    """Shrink the eval batch. Eval has no gradient accumulation to compensate."""
    previous = trainer.args.per_device_eval_batch_size
    if batch_size == previous:
        return
    trainer.args.per_device_eval_batch_size = batch_size
    print(f"[sft] Eval batch reduced from {previous} to {batch_size}.")


def _oom_limit_error(kind: str) -> RuntimeError:
    """Error used when even a one-token row does not fit."""
    return RuntimeError(
        f"CUDA OOM su un singolo token in {kind}. Un altro processo sta "
        "occupando la GPU: liberala e rilancia. Un server vLLM con "
        "gpu-memory-utilization alto sulla stessa scheda produce questo errore."
    )


def _validate_keep_percentile(percentile: float) -> None:
    """Raise when `percentile` is outside [0.0, 1.0]."""
    if isinstance(percentile, bool) or not isinstance(percentile, (int, float)):
        raise ValueError(
            "batch_keep_percentile deve essere un numero in [0.0, 1.0], "
            f"ricevuto {percentile!r}."
        )
    if not 0.0 <= float(percentile) <= 1.0:
        raise ValueError(
            "batch_keep_percentile deve essere in [0.0, 1.0], "
            f"ricevuto {percentile}."
        )


def _length_at_keep_percentile(lengths: list[int], percentile: float) -> int:
    """Return the longest token count still inside the shortest `percentile`.

    `percentile` is the share of shortest rows to keep, in [0.0, 1.0].
    1.0 is the dataset maximum. 0.0 is the shortest row. The cutoff rank is
    `ceil(percentile * n)`, so a row at that length is kept. Ties at the
    cutoff stay too, and the kept share can be a little larger than requested.
    """
    order = sorted(lengths)
    if percentile <= 0.0:
        return order[0]
    if percentile >= 1.0:
        return order[-1]
    rank = math.ceil(float(percentile) * len(order))
    rank = min(max(rank, 1), len(order))
    return order[rank - 1]


def _candidate_batch_sizes(
    batch_size: int,
    count: int,
    effective: int | None,
) -> list[int]:
    """Batch sizes to probe, largest first.

    Train sizes are the divisors of the original effective batch, so lowering
    the micro-batch can be matched by gradient accumulation without changing
    how many examples contribute to one update. Eval tries every smaller size.
    """
    start = min(batch_size, count)
    if effective is None:
        sizes = list(range(start, 0, -1))
    else:
        sizes = [size for size in range(start, 0, -1) if effective % size == 0]
        if 1 not in sizes:
            sizes.append(1)
    return sizes


def _probe_longest(
    trainer: SFTTrainer,
    dataset: Any,
    lengths: list[int],
    size: int,
    *,
    eval_mode: bool,
    kind: str,
) -> bool:
    """Run one step on the `size` longest rows. Return whether it fits."""
    indices = _longest_indices(lengths, size)
    shown = _format_lengths(lengths, indices)
    step = "forward" if eval_mode else "forward/backward"
    print(
        f"[sft] Memory check ({kind}): batch of {len(indices)}, "
        f"lengths [{shown}], one {step}."
    )
    examples = [_row(dataset, index) for index in indices]
    if _fits(trainer, examples, eval_mode=eval_mode):
        print(f"[sft] Memory check passed ({kind}, batch size {len(indices)}).")
        return True
    print(f"[sft] {kind.capitalize()} batch does not fit.")
    return False


def _kept_by_percentile(
    dataset: Any,
    lengths: list[int],
    percentile: float,
    size: int,
    kind: str,
) -> tuple[Any, list[int], int, int] | None:
    """Drop rows above `percentile` when that still fills a batch of `size`.

    Returns None when the cut removes nothing or leaves fewer than `size`
    rows. The caller then keeps the original dataset and tries a smaller
    batch before cutting again.
    """
    limit = _length_at_keep_percentile(lengths, percentile)
    capped, dropped = _filter_dataset(dataset, limit)
    if dropped == 0:
        return None
    if len(capped) < size:
        print(
            f"[sft] batch_keep_percentile={percentile:g} leaves "
            f"{len(capped)} {kind} examples, fewer than batch size {size}. "
            "Keeping every example and reducing the batch size."
        )
        return None
    return capped, _sequence_lengths(capped), dropped, limit


def _fit_split(
    trainer: SFTTrainer,
    dataset: Any,
    *,
    batch_size: int,
    eval_mode: bool,
    kind: str,
    effective: int | None = None,
    keep_percentile: float = 1.0,
) -> tuple[Any, int]:
    """Return a dataset and batch size whose worst batch fits.

    For each candidate batch size, largest first, probes the longest rows of
    the full dataset. When that fails and `keep_percentile` is below 1, drops
    the longest traces up to that share and retries the same batch size. The
    cut is kept only when the retry fits. Otherwise the next smaller batch is
    tried on the original dataset, and the percentile cut is offered again
    before the size drops further. Only when one full row still does not fit
    are longer rows removed without the percentile cap. An empty dataset
    means nothing of this split fits; the caller decides whether that aborts
    the run.
    """
    lengths = _sequence_lengths(dataset)
    sizes = _candidate_batch_sizes(batch_size, len(lengths), effective)
    for size in sizes:
        if _probe_longest(
            trainer, dataset, lengths, size, eval_mode=eval_mode, kind=kind
        ):
            return dataset, size
        if keep_percentile >= 1.0:
            continue
        kept = _kept_by_percentile(
            dataset, lengths, keep_percentile, size, kind
        )
        if kept is None:
            continue
        capped, capped_lengths, dropped, limit = kept
        print(
            f"[sft] Trying batch size {size} after dropping examples above "
            f"batch_keep_percentile={keep_percentile:g} "
            f"({limit} tokens, {dropped} examples)."
        )
        if _probe_longest(
            trainer,
            capped,
            capped_lengths,
            size,
            eval_mode=eval_mode,
            kind=kind,
        ):
            print(
                f"[sft] Dropping {dropped} {kind} examples longer than "
                f"{limit} tokens (batch_keep_percentile={keep_percentile:g}) "
                f"to keep batch size {size}."
            )
            return capped, size
        smaller = (
            "reducing the batch size."
            if size > 1
            else "dropping rows that do not fit even alone."
        )
        print(
            f"[sft] Batch size {size} still does not fit after "
            f"batch_keep_percentile={keep_percentile:g}. "
            f"Keeping every example and {smaller}"
        )

    longest = _row(dataset, _longest_indices(lengths, 1)[0])
    limit = _max_fitting_length(
        trainer, longest, eval_mode=eval_mode, kind=kind
    )
    if limit < 1:
        raise _oom_limit_error(kind)
    filtered, dropped = _filter_dataset(dataset, limit)
    print(
        f"[sft] Dropping {dropped} {kind} examples longer than {limit} tokens. "
        "Traces are dropped whole so the supervised turn is not cut off."
    )
    return filtered, 1


def _disable_eval(trainer: SFTTrainer) -> None:
    """Turn evaluation off when no eval row fits."""
    trainer.eval_dataset = None
    trainer.args.eval_strategy = IntervalStrategy.NO
    trainer.args.load_best_model_at_end = False
    print("[sft] Eval disabled: no example fits in memory.")


def _clear_probe_metrics(trainer: SFTTrainer) -> None:
    """Forget loss and token stats produced by the probe steps."""
    metrics = getattr(trainer, "_metrics", None)
    if isinstance(metrics, dict):
        for bucket in metrics.values():
            bucket.clear()
    if hasattr(trainer, "_total_train_tokens"):
        trainer._total_train_tokens = 0


def ensure_batches_fit(
    trainer: SFTTrainer,
    *,
    batch_keep_percentile: float = 1.0,
) -> None:
    """Probe the longest batches and shrink the run until they fit.

    Call this after loss masking is installed and before `trainer.train`.
    Each attempt is one real `train` or `evaluate` step on the longest rows.
    A step that runs out of memory is discarded, the card is freed, and the
    next smaller batch is tried.

    On the train split, `batch_keep_percentile` in [0.0, 1.0] is the share of
    shortest traces that may be kept at the current batch size before that
    size is lowered again. 1.0 drops nothing for that reason. Eval is
    unchanged: its batch is already 1, so the same tradeoff does not apply.

    Args:
        trainer: SFT trainer whose datasets are already tokenized.
        batch_keep_percentile: Shortest share of train rows to keep when
            retrying the current train batch size before shrinking it.
    """
    _validate_keep_percentile(batch_keep_percentile)
    effective = (
        trainer.args.per_device_train_batch_size
        * trainer.args.gradient_accumulation_steps
    )
    train_dataset, train_batch = _fit_split(
        trainer,
        trainer.train_dataset,
        batch_size=trainer.args.per_device_train_batch_size,
        eval_mode=False,
        kind="train",
        effective=effective,
        keep_percentile=batch_keep_percentile,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("Nessun esempio di train entra in memoria.")
    trainer.train_dataset = train_dataset
    _set_train_batch_size(trainer, train_batch, effective)

    eval_dataset = trainer.eval_dataset
    if eval_dataset is not None and len(eval_dataset) > 0:
        filtered, eval_batch = _fit_split(
            trainer,
            eval_dataset,
            batch_size=trainer.args.per_device_eval_batch_size,
            eval_mode=True,
            kind="eval",
        )
        if len(filtered) == 0:
            _disable_eval(trainer)
        else:
            trainer.eval_dataset = filtered
            _set_eval_batch_size(trainer, eval_batch)

    trainer.model.train()
    _clear_probe_metrics(trainer)
    _release_cuda(trainer)


class OOMTolerantSFTTrainer(SFTTrainer):
    """SFT trainer that skips a batch when CUDA runs out of memory.

    The preflight in `ensure_batches_fit` covers the longest batch. This
    catch remains for fragmentation and for memory another process allocates
    after the probe.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._oom_probing = False
        self.skipped_train_batches = 0
        self.skipped_eval_batches = 0

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Run one train step, or return NaN after dropping a batch that OOMs.

        NaN is ignored by the trainer's inf/NaN log filter, so a skipped batch
        does not pull the reported loss to zero. Gradients are cleared because
        a backward that dies midway would otherwise be stepped.
        """
        try:
            return super().training_step(
                model, inputs, num_items_in_batch=num_items_in_batch
            )
        except Exception as exc:
            if self._oom_probing or not _is_oom(exc):
                raise
            self.skipped_train_batches += 1
            shape = _batch_shape(inputs)
            print(
                "[sft] CUDA OOM on a train batch "
                f"{shape}; skipping it "
                f"({self.skipped_train_batches} skipped so far)."
            )
            _release_cuda(self)
            return torch.tensor(float("nan"), device=self.args.device)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Run one eval step, or skip the batch when it does not fit.

        A `None` loss is left out of the eval average, so a skipped row does
        not become a zero or a NaN in `eval_loss`.
        """
        try:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )
        except Exception as exc:
            if self._oom_probing or not _is_oom(exc):
                raise
            self.skipped_eval_batches += 1
            shape = _batch_shape(inputs)
            print(
                "[sft] CUDA OOM on an eval batch "
                f"{shape}; skipping it "
                f"({self.skipped_eval_batches} skipped so far)."
            )
            _release_cuda(self)
            return (None, None, None)


def _batch_shape(inputs: dict[str, Any]) -> str:
    """Describe the token tensor of a batch for the skip log."""
    ids = inputs.get("input_ids")
    if torch.is_tensor(ids):
        return "x".join(str(dim) for dim in ids.shape)
    return "unknown shape"
