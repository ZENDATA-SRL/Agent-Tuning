from __future__ import annotations

import threading

from backend.adapters import BASE_MODEL_ID, BASE_MODEL_NAME, resolve_adapter

_lock = threading.Lock()
_model = None
_tokenizer = None
_loaded_id: str | None = None
_max_seq_length = 16384


def _load(model_name: str, loaded_id: str):
    global _model, _tokenizer, _loaded_id
    from unsloth import FastLanguageModel

    print(f"[model] Loading {loaded_id} from {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_name),
        max_seq_length=_max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    _model = model
    _tokenizer = tokenizer
    _loaded_id = loaded_id


def ensure_loaded(adapter_id: str):
    resolved = resolve_adapter(adapter_id)
    model_name = BASE_MODEL_NAME if adapter_id == BASE_MODEL_ID else str(resolved)
    with _lock:
        if _loaded_id == adapter_id and _model is not None:
            return _model, _tokenizer
        _load(model_name, adapter_id)
        return _model, _tokenizer


def generate(adapter_id: str, prompt: str, max_new_tokens: int = 512) -> str:
    model, tokenizer = ensure_loaded(adapter_id)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    eos = tokenizer.eos_token_id
    stop_ids = [eos] if eos is not None else None
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        stop_ids = list({*(stop_ids or []), im_end})

    with _lock:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or eos,
            eos_token_id=stop_ids,
        )
    new_tokens = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)
