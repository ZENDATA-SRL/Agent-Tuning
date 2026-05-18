"""
CLI entry point for the benchmarking module.

Usage examples:

  # OpenAI
  python -m benchmark \\
    --dataset data/traces-2026-05-18.jsonl \\
    --backend openai --model gpt-4o-mini \\
    --output results/gpt4o_mini.json

  # vLLM (OpenAI-compatible local endpoint)
  python -m benchmark \\
    --dataset data/traces-2026-05-18.jsonl \\
    --backend vllm --model meta-llama/Llama-3-8b-instruct \\
    --base-url http://localhost:8000/v1

  # Ollama (local, default http://localhost:11434)
  python -m benchmark \\
    --dataset data/traces-2026-05-18.jsonl \\
    --backend ollama --model llama3.1:8b

  # Ollama with custom host
  python -m benchmark \\
    --dataset data/traces-2026-05-18.jsonl \\
    --backend ollama --model llama3.1:8b \\
    --base-url http://192.168.1.10:11434

  # AWS Bedrock
  python -m benchmark \\
    --dataset data/traces-2026-05-18.jsonl \\
    --backend bedrock --model anthropic.claude-3-5-sonnet-20240620-v1:0 \\
    --region eu-west-1

  # With LLM-as-judge scoring (uses same model as judge by default)
  python -m benchmark \\
    --dataset data/traces-2026-05-18.jsonl \\
    --backend openai --model gpt-4o \\
    --use-judge

  # Smoke test — first 5 traces only
  python -m benchmark \\
    --dataset data/traces-2026-05-18.jsonl \\
    --backend ollama --model llama3.1:8b \\
    --max-traces 5
"""
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from benchmark.config import InferenceProfile
from benchmark.runner import run_benchmark

load_dotenv()



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark an LLM against an agent trace dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", required=True, help="Path to the .jsonl dataset file.")
    p.add_argument(
        "--backend",
        required=True,
        choices=["openai", "vllm", "ollama", "bedrock", "azure"],
        help=(
            "Inference backend: "
            "openai=ChatOpenAI (cloud), "
            "vllm=ChatOpenAI (local vLLM, requires --base-url), "
            "azure=AzureChatOpenAI, "
            "ollama=ChatOllama, "
            "bedrock=ChatBedrockConverse."
        ),
    )
    p.add_argument("--model", required=True, help="Model name / ID.")
    p.add_argument(
        "--base-url",
        default=None,
        help="Base URL override. Required for vLLM (e.g. http://localhost:8000/v1). "
             "Optional for Ollama (default: http://localhost:11434).",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API key. Falls back to OPENAI_API_KEY env var for openai/vllm backends.",
    )
    p.add_argument("--region", default=None, help="AWS region (Bedrock only).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--output", default=None, help="Output JSON report path.")
    p.add_argument(
        "--use-judge",
        action="store_true",
        help="Enable LLM-as-judge scoring for answer quality and rule compliance.",
    )
    p.add_argument("--judge-backend", default=None, choices=["openai", "vllm", "ollama", "bedrock", "azure"], help="Inference backend for the judge model.")
    p.add_argument("--judge-model", default=None, help="Model name / ID for the judge model.")
    p.add_argument("--judge-base-url", default=None, help="Base URL override for the judge model.")
    p.add_argument("--judge-api-key", default=None, help="API key for the judge model.")
    p.add_argument("--judge-region", default=None, help="AWS region for the judge model.")
    p.add_argument("--judge-deployment", default=None, help="Azure deployment name for the judge (azure backend only).")
    p.add_argument("--judge-api-version", default=None, help="Azure API version for the judge (azure backend only).")
    p.add_argument("--judge-temperature", type=float, default=0.0)
    p.add_argument("--judge-max-tokens", type=int, default=512)
    p.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Process only the first N traces (useful for smoke tests).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    profile = InferenceProfile(
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        region=args.region,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Build judge profile only when --use-judge is requested and a separate
    # judge backend/model is explicitly configured. Otherwise runner falls
    # back to using the same model as the one under test.
    judge_profile: InferenceProfile | None = None
    if args.use_judge and args.judge_backend and args.judge_model:
        judge_api_key = args.judge_api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        judge_base_url = args.judge_base_url or (
            os.getenv("AZURE_OPENAI_ENDPOINT") if args.judge_backend == "azure" else None
        )
        judge_profile = InferenceProfile(
            backend=args.judge_backend,
            model=args.judge_model,
            base_url=judge_base_url,
            api_key=judge_api_key,
            region=args.judge_region,
            azure_deployment=args.judge_deployment or os.getenv("AZURE_OPENAI_MODEL"),
            azure_api_version=args.judge_api_version or os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
        )

    run_benchmark(
        dataset_path=args.dataset,
        profile=profile,
        output_path=args.output,
        use_judge=args.use_judge,
        judge_profile=judge_profile,
        max_traces=args.max_traces,
    )


if __name__ == "__main__":
    main()
