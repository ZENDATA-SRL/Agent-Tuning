"""Supervised fine-tuning loop on agent traces.

Minimal, single-purpose: load the model from `SFTConfig`, attach LoRA
adapters, render the JSONL traces with the tokenizer's chat template,
mask the loss to assistant turns only, train, save the LoRA adapter.

Model-specific choices (checkpoint, LoRA targets, sampling, chat-template
markers) come from the config, not from this module.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Tuple

import torch  # type: ignore[import-not-found]

from train.config import SFTConfig
from train.data import format_dataset, load_jsonl

def load_dataset(dataset_path: str) -> tuple[list[dict], list[dict], list[dict]]:
    print(f"[sft] Loading TRAIN dataset: {dataset_path}/train_only_tool_call.jsonl")
    traces_train = load_jsonl(dataset_path + "/train_only_tool_call.jsonl")
    print(f"[sft] {len(traces_train)} train traces loaded")
    print(f"[sft] Loading TEST dataset: {dataset_path}/test_only_tool_call.jsonl")
    traces_test = load_jsonl(dataset_path + "/test_only_tool_call.jsonl")
    print(f"[sft] {len(traces_test)} test traces loaded")
    print(f"[sft] Loading EVAL dataset: {dataset_path}/eval_only_tool_call.jsonl")
    traces_eval = load_jsonl(dataset_path + "/eval_only_tool_call.jsonl")
    print(f"[sft] {len(traces_eval)} eval traces loaded")
    return traces_train, traces_test, traces_eval

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
    saw_tool_call = False

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

        if "<tool_call>" in joined:
            saw_tool_call = True

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

    if not saw_tool_call:
        print(
            f"\n[sft] ATTENZIONE: nessuna <tool_call> tra i token in loss nei primi "
            f"{checked} esempi. Verifica la composizione del dataset."
        )

    print(f"\n[sft] masking validato su {checked}/{total_examples} esempi.\n")


def custom_masking(
    trainer: Any,
    tokenizer: Any,
    *,
    response_part: str,
    turn_end_marker: str,
) -> Any:
    """Mask every token except the contents of assistant turns.

    The dataset contains already-rendered chat-template text.  Applying the
    mask after the trainer's collator has tokenized the batch makes the
    behavior independent of tokenizer boundaries (a marker may be split over
    multiple tokens) and also masks ``tool`` messages.

    The assistant role marker itself is ignored; the closing end-marker
    token is kept so the model learns when to stop generating.
    """
    start_ids = tokenizer.encode(response_part, add_special_tokens=False)
    end_ids = tokenizer.encode(turn_end_marker, add_special_tokens=False)
    if not start_ids or not end_ids:
        raise ValueError("Impossibile tokenizzare i marker dei turni assistant.")

    original_collator = trainer.data_collator

    def collate_with_custom_masking(features: list[dict]) -> dict:
        batch = original_collator(features)
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        if labels is None:
            labels = input_ids.clone()

        masked_labels = torch.full_like(labels, IGNORE_INDEX)
        for row in range(input_ids.shape[0]):
            position = 0
            while position <= input_ids.shape[1] - len(start_ids):
                marker_end = position + len(start_ids)
                if input_ids[row, position:marker_end].tolist() != start_ids:
                    position += 1
                    continue

                content_start = marker_end
                end = content_start
                while end <= input_ids.shape[1] - len(end_ids):
                    if input_ids[row, end : end + len(end_ids)].tolist() == end_ids:
                        content_end = end + len(end_ids)
                        masked_labels[row, content_start:content_end] = labels[
                            row, content_start:content_end
                        ]
                        position = content_end
                        break
                    end += 1
                else:
                    raise RuntimeError(
                        f"Trovato un marker assistant senza {turn_end_marker} "
                        "nella sequenza tokenizzata."
                    )
            # Padding and every non-assistant message remain IGNORE_INDEX.

        batch["labels"] = masked_labels
        return batch

    trainer.data_collator = collate_with_custom_masking
    return trainer


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


def run_sft(config: SFTConfig) -> None:
    """Run a full SFT loop and persist the LoRA adapter to `config.output_dir`."""
    # Reduce CUDA allocator fragmentation when VRAM is nearly full.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    # Pop any malformed WANDB_* values so they can't propagate through
    # subprocesses or unexpected callbacks. Empty WANDB_MODE is especially
    # bad: wandb Settings requires one of online/offline/shared/disabled/…
    for _k in (
        "WANDB_TAGS",
        "WANDB_PROJECT",
        "WANDB_NAME",
        "WANDB_ENTITY",
        "WANDB_MODE",
    ):
        if os.environ.get(_k, "").strip() == "":
            os.environ.pop(_k, None)

    # Unsloth must be imported before transformers/trl to install its patches.
    from unsloth import FastLanguageModel  # type: ignore
    from unsloth.chat_templates import train_on_responses_only  # type: ignore
    from trl import SFTConfig as TRLSFTConfig  # type: ignore
    from trl import SFTTrainer  # type: ignore

    print(f"[sft] Loading model: {config.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        full_finetuning=False,
    )

    print("[sft] Attaching LoRA adapters")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        use_gradient_checkpointing=config.gradient_checkpointing,
        random_state=config.seed,
        max_seq_length=config.max_seq_length,
    )

    model.generation_config.do_sample = config.generation_do_sample
    model.generation_config.temperature = config.generation_temperature
    model.generation_config.top_p = config.generation_top_p
    model.generation_config.top_k = config.generation_top_k
    model.generation_config.min_p = config.generation_min_p

    train_dataset, test_dataset, eval_dataset = load_dataset(config.dataset_path)

    print("[sft] Rendering dataset with chat template")
    train_dataset = format_dataset(
        train_dataset, tokenizer, chat_template_kwargs=config.chat_template_kwargs
    )
    test_dataset = format_dataset(
        test_dataset, tokenizer, chat_template_kwargs=config.chat_template_kwargs
    )
    eval_dataset = format_dataset(
        eval_dataset, tokenizer, chat_template_kwargs=config.chat_template_kwargs
    )

    output_dir = str(Path(config.output_dir))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=TRLSFTConfig(
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_checkpointing=bool(config.gradient_checkpointing),
            bf16=config.mixed_precision == "bf16",
            fp16=config.mixed_precision == "fp16",
            warmup_ratio=config.warmup_ratio,
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            save_steps=config.save_steps,
            save_strategy="steps",

            eval_steps=config.eval_steps,
            eval_strategy="steps",

            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,


            optim=config.optim,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            max_grad_norm=config.max_grad_norm,
            seed=config.seed,
            output_dir=output_dir,
            report_to="wandb",
            logging_steps=config.logging_steps,
        ),
        #callbacks=[GenerationCallback(
        #    model,
        #    tokenizer,
        #    eval_dataset,
        #    response_part=config.response_part,
        #    generation_prompt_suffix=config.generation_prompt_suffix,
        #)],
    )

    if config.loss_masking == "unsloth_responses_only":
        print("[sft] Wiring loss masking (train_on_responses_only)")
        trainer = train_on_responses_only(
            trainer,
            instruction_part=config.instruction_part,
            response_part=config.response_part,
            last_response_only=config.last_response_only,
        )
    elif config.loss_masking == "assistant_turns":
        print("[sft] Wiring custom assistant-token masking")
        trainer = custom_masking(
            trainer,
            tokenizer,
            response_part=config.response_part,
            turn_end_marker=config.turn_end_marker,
        )
    else:
        raise ValueError(
            f"loss_masking sconosciuto: {config.loss_masking!r}."
        )

    assert_masking_ok(
        trainer,
        tokenizer,
        forbidden_in_loss=config.forbidden_in_loss,
        turn_end_marker=config.turn_end_marker,
    )

    if config.resume_from_checkpoint:
        print(
            f"[sft] Resuming training from checkpoint: "
            f"{config.resume_from_checkpoint}"
        )
    else:
        print("[sft] Starting training")
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    print(f"[sft] Saving best model LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir + "/best_eval_model")
    tokenizer.save_pretrained(output_dir + "/best_eval_model")
    print("[sft] Done.")
