from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class InferenceProfile:
    """
    Configuration for a single model + inference backend combination.

    Backends:
      openai   → langchain-openai ChatOpenAI (cloud or any OpenAI-compat endpoint)
      vllm     → langchain-openai ChatOpenAI with a custom base_url pointing at vLLM
      azure    → langchain-openai AzureChatOpenAI (Azure OpenAI Service)
      ollama   → langchain-ollama ChatOllama
      bedrock  → langchain-aws ChatBedrockConverse
    """

    backend: Literal["openai", "vllm", "ollama", "bedrock", "azure"]
    model: str

    # OpenAI / vLLM
    base_url: str | None = None   # override endpoint (required for vLLM, optional for Ollama)
    api_key: str | None = None    # falls back to OPENAI_API_KEY env var

    # AWS Bedrock
    region: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # Generation parameters
    temperature: float = 0.0
    max_tokens: int = 1024

    # Extra Azure-specific fields
    azure_deployment: str | None = None   # Azure deployment name (often same as model)
    azure_api_version: str | None = None  # e.g. "2025-01-01-preview"

    # Extra kwargs forwarded verbatim to the LangChain model constructor
    extra: dict = field(default_factory=dict)

    # False → parallel reference tool_calls are split into sequential single-call
    # turns before replay (required for models like Llama 3.1).
    support_multi_tool_calls: bool = True

    def label(self) -> str:
        """Human-readable identifier used in reports and output filenames."""
        return f"{self.backend}/{self.model}"


def resolve_support_multi_tool_calls(
    model: str,
    override: bool | None = None,
) -> bool:
    """Resolve multi-tool support from an override or the matching shared recipe.

    Falls back to a name heuristic for Llama 3.x when recipes are not
    importable.
    """
    if override is not None:
        return override
    try:
        from shared.configs import load_model

        return load_model(model).chat.support_multi_tool_calls
    except Exception:
        lowered = model.lower()
        if "llama-3.1" in lowered or "llama3.1" in lowered:
            return False
        if "llama-3.2" in lowered or "llama3.2" in lowered:
            return False
        return True
