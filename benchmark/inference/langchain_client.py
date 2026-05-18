from __future__ import annotations

import time

from benchmark.config import InferenceProfile
from benchmark.inference.base import GenerationResult, InferenceClient, ToolCall


def _lc_provider(profile: InferenceProfile) -> str:
    """Map our backend name to the init_chat_model provider string."""
    mapping = {
        "openai": "openai",
        "vllm": "openai",          # vLLM exposes an OpenAI-compatible endpoint
        "ollama": "ollama",
        "bedrock": "bedrock_converse",
    }
    return mapping[profile.backend]


def _build_lc_model(profile: InferenceProfile):
    """
    Instantiate the appropriate LangChain chat model for the given profile.

    - openai/vllm  → ChatOpenAI  (langchain-openai)
    - azure        → AzureChatOpenAI (langchain-openai)
    - ollama       → ChatOllama  (langchain-ollama)
    - bedrock      → ChatBedrockConverse (langchain-aws)

    All share the same BaseChatModel interface so the rest of the
    benchmarking code stays backend-agnostic.
    """
    kwargs: dict = {
        "temperature": profile.temperature,
        "max_tokens": profile.max_tokens,
        **profile.extra,
    }

    if profile.backend in ("openai", "vllm"):
        from langchain_openai import ChatOpenAI

        if profile.api_key:
            kwargs["api_key"] = profile.api_key
        if profile.base_url:
            kwargs["base_url"] = profile.base_url

        # stream_usage=True ensures token counts are returned even during streaming
        return ChatOpenAI(model=profile.model, stream_usage=True, **kwargs)

    if profile.backend == "azure":
        from langchain_openai import AzureChatOpenAI
        import os

        return AzureChatOpenAI(
            azure_deployment=profile.azure_deployment or profile.model,
            azure_endpoint=profile.base_url or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=profile.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=profile.azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            **kwargs,
        )

    if profile.backend == "ollama":
        from langchain_ollama import ChatOllama

        if profile.base_url:
            kwargs["base_url"] = profile.base_url

        return ChatOllama(model=profile.model, **kwargs)

    if profile.backend == "bedrock":
        from langchain_aws import ChatBedrockConverse

        bedrock_kwargs: dict = {"model_id": profile.model, **kwargs}
        if profile.region:
            bedrock_kwargs["region_name"] = profile.region
        if profile.aws_access_key_id:
            bedrock_kwargs["aws_access_key_id"] = profile.aws_access_key_id
        if profile.aws_secret_access_key:
            bedrock_kwargs["aws_secret_access_key"] = profile.aws_secret_access_key

        return ChatBedrockConverse(**bedrock_kwargs)

    raise ValueError(f"Unknown backend: {profile.backend!r}")


def _parse_tool_calls(ai_message) -> list[ToolCall]:
    """
    Extract tool calls from a LangChain AIMessage.
    ai_message.tool_calls is a list of dicts:
      {"name": str, "args": dict, "id": str, "type": "tool_call"}
    """
    result = []
    for tc in getattr(ai_message, "tool_calls", []):
        result.append(ToolCall(
            name=tc.get("name", ""),
            args=tc.get("args", {}),
            call_id=tc.get("id", ""),
        ))
    return result


def _parse_usage(ai_message) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) from AIMessage.usage_metadata."""
    meta = getattr(ai_message, "usage_metadata", None) or {}
    return (
        meta.get("input_tokens", 0),
        meta.get("output_tokens", 0),
    )


class LangChainClient(InferenceClient):
    """
    Thin InferenceClient wrapper around any LangChain BaseChatModel.

    Timing is measured wall-clock around the synchronous invoke() call.
    TTFT is approximated by measuring the first streamed chunk when the
    model supports streaming (stream() method); otherwise it equals total
    latency (non-streaming models like some Bedrock models).
    """

    def __init__(self, profile: InferenceProfile) -> None:
        self._model = _build_lc_model(profile)
        self._profile = profile

    def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> GenerationResult:
        model = self._model
        if tools:
            # bind_tools accepts OpenAI-style function-calling dicts directly
            model = model.bind_tools(tools)

        # --- Streaming path for TTFT measurement ---
        first_chunk_time: float | None = None
        start = time.perf_counter()

        try:
            chunks = []
            for chunk in model.stream(messages):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                chunks.append(chunk)

            end = time.perf_counter()
            ttft_ms = (first_chunk_time - start) * 1000 if first_chunk_time else 0.0
            total_ms = (end - start) * 1000

            # Reduce chunks into a single AIMessage
            from langchain_core.messages import AIMessageChunk
            if chunks:
                final = chunks[0]
                for c in chunks[1:]:
                    final = final + c
            else:
                from langchain_core.messages import AIMessage
                final = AIMessage(content="")

        except Exception:
            # Fall back to invoke() if streaming fails (e.g. unsupported backend)
            start = time.perf_counter()
            final = model.invoke(messages)
            end = time.perf_counter()
            ttft_ms = (end - start) * 1000
            total_ms = ttft_ms

        text = final.content if isinstance(final.content, str) else ""
        tool_calls = _parse_tool_calls(final)
        input_tokens, output_tokens = _parse_usage(final)

        return GenerationResult(
            text=text,
            tool_calls=tool_calls,
            ttft_ms=round(ttft_ms, 2),
            total_latency_ms=round(total_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
