from benchmark.config import InferenceProfile
from benchmark.inference.base import GenerationResult, InferenceClient, ToolCall
from benchmark.inference.langchain_client import LangChainClient


def build_client(profile: InferenceProfile) -> InferenceClient:
    """Factory: return a LangChainClient for any supported backend."""
    if profile.backend in ("openai", "vllm", "azure", "ollama", "bedrock"):
        return LangChainClient(profile)
    raise ValueError(f"Unknown backend: {profile.backend!r}")


__all__ = [
    "InferenceClient",
    "GenerationResult",
    "ToolCall",
    "LangChainClient",
    "build_client",
]
