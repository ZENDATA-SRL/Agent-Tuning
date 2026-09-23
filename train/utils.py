from typing import List, Tuple, Any
import torch

IGNORE_INDEX = -100

def _extract_labels(trainer, index: int) -> Tuple[List[int], List[int]]:
    """Return (input_ids, labels) for one training example.

    Labels are usually produced by the collator at batch time, not stored
    in the dataset, so we run the sample through the collator to see what
    the model will actually receive.
    """
    sample = trainer.train_dataset[index]

    if "labels" in sample:
        ids = list(sample["input_ids"])
        labels = list(sample["labels"])
    else:
        batch = trainer.data_collator([sample])
        if "labels" not in batch:
            raise RuntimeError(
                "Il collator non produce 'labels': la loss verrebbe calcolata "
                "su tutta la sequenza. Configura il masking prima di procedere."
            )
        ids = batch["input_ids"][0].tolist()
        labels = batch["labels"][0].tolist()

    if len(ids) != len(labels):
        raise RuntimeError(
            f"Disallineamento input_ids ({len(ids)}) / labels ({len(labels)})."
        )
    return ids, labels

def _contiguous_segments(labels: List[int]) -> List[List[int]]:
    """Group loss-bearing tokens into contiguous runs (one per assistant turn).

    Rebuilding segments from the label positions avoids decoding through a
    pad token, which would be ambiguous whenever pad_token == eos_token.
    """
    segments: List[List[int]] = []
    current: List[int] = []
    previous = None

    for position, token in enumerate(labels):
        if token == IGNORE_INDEX:
            continue
        if previous is not None and position != previous + 1:
            segments.append(current)
            current = []
        current.append(token)
        previous = position

    if current:
        segments.append(current)
    return segments


def assert_masking_ok(
    trainer,
    tokenizer,
    *,
    forbidden_in_loss: tuple[str, ...],
    turn_end_marker: str,
    num_examples: int = 5,
    min_ratio: float = 0.00,
    max_ratio: float = 0.90,
    verbose: bool = True,
) -> None:
    """Validate the loss mask on the first examples; raise if it is wrong.

    Checks, in order of severity:
      1. at least one token carries loss;
      2. the share of loss-bearing tokens is plausible;
      3. no user/system/tool text leaked into the loss (the failure that
         teaches the model to hallucinate tool results);
      4. every assistant turn ends on the configured end marker, so the model
         learns to stop.
    """
    total_examples = len(trainer.train_dataset)
    checked = min(num_examples, total_examples)

    for index in range(checked):
        ids, labels = _extract_labels(trainer, index)
        segments = _contiguous_segments(labels)

        if not segments:
            raise RuntimeError(
                f"[esempio {index}] nessun token in loss: maschera troppo stretta."
            )

        kept = sum(len(segment) for segment in segments)
        ratio = kept / len(ids)
        if not min_ratio < ratio < max_ratio:
            raise RuntimeError(
                f"[esempio {index}] {ratio:.1%} dei token in loss "
                f"(atteso tra {min_ratio:.0%} e {max_ratio:.0%}): maschera sospetta."
            )

        texts = [tokenizer.decode(segment) for segment in segments]
        joined = "\n".join(texts)

        for marker in forbidden_in_loss:
            if marker in joined:
                raise RuntimeError(
                    f"[esempio {index}] '{marker}' finisce tra i token in loss: "
                    "il modello imparerebbe a generare input che deve solo leggere."
                )

        for turn_index, text in enumerate(texts):
            if not text.rstrip().endswith(turn_end_marker):
                raise RuntimeError(
                    f"[esempio {index}] il turno {turn_index} non termina con "
                    f"{turn_end_marker}: il modello non imparerebbe a chiudere il turno."
                )

        if verbose:
            print("\n" + "=" * 70)
            print(
                f"[sft] esempio {index} — {len(segments)} turni assistant, "
                f"{kept}/{len(ids)} token in loss ({ratio:.1%})"
            )
            print("=" * 70)
            for turn_index, text in enumerate(texts):
                print(f"--- turno {turn_index} ---")
                print(text)

    print(f"\n[sft] masking validato su {checked}/{total_examples} esempi.\n")








class GenerationCallback:
    """Sample a real response from the in-training model after each eval.

    Uses the same model instance already loaded for SFT (no second load).
    SFT itself is teacher-forcing and never generates; this is the only way
    to inspect what the live weights would actually answer.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        eval_dataset: Any,
        *,
        response_part: str,
        generation_prompt_suffix: str,
        max_new_tokens: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.response_part = response_part
        self.generation_prompt_suffix = generation_prompt_suffix
        self.max_new_tokens = max_new_tokens

    def __getattr__(self, name: str):
        """Make unused Trainer callback events no-ops."""
        if name.startswith("on_"):
            return lambda *args, **kwargs: args[2] if len(args) > 2 else None
        raise AttributeError(name)

    @staticmethod
    def _unwrap(model: Any) -> Any:
        # Accelerate / DDP wrappers still point at the same GPU weights.
        while hasattr(model, "module"):
            model = model.module
        return model

    def _print_generation(self, state):
        if len(self.eval_dataset) == 0:
            return

        model = self._unwrap(self.model)
        text = self.eval_dataset[0]["text"]
        assistant_start = text.rfind(self.response_part)
        prompt = text[:assistant_start] if assistant_start >= 0 else text
        prompt = prompt.rstrip() + "\n" + self.generation_prompt_suffix

        device = next(model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        was_training = model.training
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
        finally:
            if was_training:
                model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        prompt_length = inputs["input_ids"].shape[1]
        output = self.tokenizer.decode(
            generated[0, prompt_length:],
            skip_special_tokens=False,
        ).strip()
        print(
            f"\n[sft] step {state.global_step} - live model generation "
            f"(same weights as training):\n{output}\n",
            flush=True,
        )

    def on_evaluate(self, args, state, control, **kwargs):
        # Only run after eval: logging_steps can be 1 and generate() would OOM
        # if called every training step while grads/optimizer still occupy VRAM.
        self._print_generation(state)
        return control