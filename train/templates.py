"""Per-model chat template adapters.

The wire format that an LLM expects at training time is *not* a property of
the dataset, it is a property of the model. Every family ships its own
combination of:

  - special tokens delimiting turns (e.g. `<|turn>user\\n` for Gemma 4,
    `<|im_start|>user\\n` for Qwen 2.5, `<|start_header_id|>user...` for Llama 3),
  - a Jinja `chat_template` that may or may not understand `tools=` and
    `role:"tool"` messages,
  - whether Unsloth ships a *replacement* template via `get_chat_template`
    that is preferable to (or worse than) the one bundled with the repo.

Hard-coding any of this in `train/data.py` and `train/sft.py` couples the
trainer to a single model family. Instead, each supported family is wrapped
in a `ChatTemplateAdapter` that:

  1. decides whether to keep the tokenizer's *native* template or to swap it
     for an Unsloth-supplied one,
  2. applies the template to a list of OpenAI-canonical messages (with
     optional `tools=`),
  3. exposes the exact `instruction_part` / `response_part` marker strings
     that Unsloth's `train_on_responses_only` needs to mask the loss.

The active adapter is picked by `get_template_adapter(model_name, tokenizer)`
based on the HF model id. New families are added by writing a new subclass
and registering it in `_ADAPTER_REGISTRY`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ChatTemplateAdapter(ABC):
    """Strategy for serializing a conversation into a model-specific string.

    Subclasses encapsulate *all* knowledge about a model family's wire format
    so that `train.data.format_dataset` and `train.sft.run_sft` stay generic.
    """

    name: str = "abstract"

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    @abstractmethod
    def prepare_tokenizer(self) -> Any:
        """Mutate (or replace) the tokenizer so its `apply_chat_template` is
        the one we actually want to train on. Returns the prepared tokenizer.

        Implementations may either:
          - return the tokenizer unchanged (use the repo's native template), or
          - call `unsloth.chat_templates.get_chat_template(...)` to swap in
            an Unsloth-supplied template.
        """

    @abstractmethod
    def apply_template(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        add_generation_prompt: bool = False,
    ) -> str:
        """Render `messages` (and optional `tools`) into the wire-format string
        consumed by the model at training time."""

    @property
    @abstractmethod
    def instruction_part(self) -> str:
        """Marker emitted by the template at the start of every USER turn.

        Passed verbatim to `unsloth.chat_templates.train_on_responses_only`
        as the boundary between masked and unmasked tokens.
        """

    @property
    @abstractmethod
    def response_part(self) -> str:
        """Marker emitted by the template at the start of every ASSISTANT turn.

        Tokens between this marker and the next `instruction_part` (or EOS)
        are the ones the loss is computed on.
        """

    @property
    @abstractmethod
    def tool_call_start(self) -> str:
        """Opening marker that the template emits before each tool-call block.

        Used by `train.data.mask_labels_selective` to locate tool-call spans
        inside a rendered assistant turn when `loss_on_tool_calls_only=True`.
        """

    @property
    @abstractmethod
    def tool_call_end(self) -> str:
        """Closing marker that the template emits after each tool-call block."""

    @property
    @abstractmethod
    def turn_end(self) -> str:
        """Token / string that closes an assistant turn in the rendered text.

        Used to find the boundary of the *last* assistant turn so it can be
        optionally included in the loss via `loss_include_final_response`.
        """


class Gemma4NativeAdapter(ChatTemplateAdapter):
    """Adapter for the Gemma 4 family using the *native* repo chat template.

    Why not `get_chat_template(tokenizer, "gemma-4")`?
      The Unsloth-supplied "gemma-4" template is a simplified user/assistant
      alternation template that does NOT support `tools=` nor `role:"tool"`
      messages. Feeding it an agentic trace
      (system → user → assistant(tool_calls) → tool → assistant) raises
      `TemplateError: Conversation roles must alternate user/assistant/...`.

      The template shipped with `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`
      (and the upstream Google checkpoint) is the *full* Gemma 4 template
      that natively understands:
        - `tools=[...]` rendered as `<|tool>declaration:name{...}` blocks,
        - `role:"tool"` messages rendered as `<|tool_response>...` blocks
          forward-scanned after an assistant `tool_calls` turn,
        - thinking channels via `<|channel>thought\\n...`.
      That is exactly the format we want to teach.

    Loss-masking markers come straight from the template source:
        instruction = "<|turn>user\\n"
        response    = "<|turn>model\\n"
    Both appear verbatim in the rendered text so `train_on_responses_only`
    can split on them.
    """

    name = "gemma-4-native"

    def prepare_tokenizer(self) -> Any:
        return self.tokenizer

    def apply_template(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        add_generation_prompt: bool = False,
    ) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        # Strip a leading `<bos>` if the template prepended one — the SFTTrainer's
        # collator re-adds BOS, and a duplicated BOS is the silent-bug pattern
        # documented in the official Unsloth Gemma-4 notebook.
        if isinstance(text, str) and text.startswith("<bos>"):
            text = text[len("<bos>"):]
        return text

    @property
    def instruction_part(self) -> str:
        return "<|turn>user\n"

    @property
    def response_part(self) -> str:
        return "<|turn>model\n"

    @property
    def tool_call_start(self) -> str:
        return "<|tool_call>"

    @property
    def tool_call_end(self) -> str:
        return "<tool_call|>"

    @property
    def turn_end(self) -> str:
        return "<turn|>\n"


_ADAPTER_REGISTRY: tuple[tuple[tuple[str, ...], type[ChatTemplateAdapter]], ...] = (
    # Match by substring against the lowercased HF model id. Longer/more
    # specific prefixes should come first.
    (("gemma-4", "gemma4"), Gemma4NativeAdapter),
)


def get_template_adapter(model_name: str, tokenizer: Any) -> ChatTemplateAdapter:
    """Pick the right adapter for `model_name`.

    Raises NotImplementedError if no adapter is registered: we'd rather fail
    loudly than silently fall back to a generic template that may quietly
    mangle tool calls.
    """
    key = model_name.lower()
    for needles, cls in _ADAPTER_REGISTRY:
        if any(n in key for n in needles):
            return cls(tokenizer)

    raise NotImplementedError(
        f"No ChatTemplateAdapter registered for model '{model_name}'. "
        f"Add one in train/templates.py and register it in _ADAPTER_REGISTRY."
    )
