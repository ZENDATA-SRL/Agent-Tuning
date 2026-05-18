from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A single tool call produced by the model."""

    name: str
    args: dict = field(default_factory=dict)
    call_id: str = ""


@dataclass
class GenerationResult:
    """Everything returned from one model inference call."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    # Timing (milliseconds)
    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def throughput_tok_s(self) -> float:
        if self.total_latency_ms <= 0:
            return 0.0
        return self.output_tokens / (self.total_latency_ms / 1000.0)


class InferenceClient(ABC):
    """Abstract base for all inference backends."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> GenerationResult:
        """
        Run one inference call.

        Args:
            messages: OpenAI-style message list
                      [{"role": "system"|"user"|"assistant"|"tool", "content": ...}, ...]
            tools: Optional list of tool definitions in OpenAI function-calling schema.

        Returns:
            GenerationResult with text, tool_calls, and timing/token data.
        """
