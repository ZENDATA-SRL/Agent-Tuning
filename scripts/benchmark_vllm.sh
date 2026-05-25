#!/usr/bin/env bash
# ============================================================
#  benchmark_vllm.sh — Run the Agent-Tuning benchmark against
#                       a vLLM server (local or remote)
#
#  Usage:
#    ./scripts/benchmark_vllm.sh [model-name] [vllm-base-url] [options]
#
#  Positional args (both optional):
#    model-name     Name passed to --model (default: sft-model)
#    vllm-base-url  OpenAI-compatible endpoint  (default: http://10.100.10.6:8000/v1)
#
#  Options:
#    --smoke        Run on first 10 traces only
#    --no-judge     Skip the LLM-as-judge metrics
#    --traces N     Run on first N traces
#
#  Examples:
#    # Full run against default remote server
#    ./scripts/benchmark_vllm.sh
#
#    # Smoke test (10 traces)
#    ./scripts/benchmark_vllm.sh sft-model http://10.100.10.6:8000/v1 --smoke
#
#    # Different checkpoint exported as a separate vLLM instance
#    ./scripts/benchmark_vllm.sh sft-ck160 http://10.100.10.6:8001/v1
#
#    # Local vLLM instance
#    ./scripts/benchmark_vllm.sh my-model http://localhost:8000/v1 --no-judge
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Load .env ────────────────────────────────────────────────────────────────
if [[ -f ".env" ]]; then
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi

# ── Defaults ─────────────────────────────────────────────────────────────────
VLLM_MODEL="${1:-sft-model}"
VLLM_BASE_URL="${2:-http://${SSH_REMOTE_HOST:-10.100.10.6}:8000/v1}"
USE_JUDGE=true
MAX_TRACES=""

# ── Parse optional flags (everything after the first two positional args) ────
shift 2 2>/dev/null || shift "$#" 2>/dev/null || true
for arg in "$@"; do
    case "$arg" in
        --smoke)       MAX_TRACES=10 ;;
        --no-judge)    USE_JUDGE=false ;;
        --traces)      shift; MAX_TRACES="$1" ;;
        --traces=*)    MAX_TRACES="${arg#--traces=}" ;;
        --help|-h)
            sed -n '2,20p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
    esac
done

# ── Validate vLLM server is reachable ────────────────────────────────────────
VLLM_HEALTH="${VLLM_BASE_URL%/v1}/health"
echo "▶ Checking vLLM at ${VLLM_BASE_URL} ..."
if ! curl -sf "${VLLM_HEALTH}" > /dev/null 2>&1; then
    # /health may not exist on all versions; try /v1/models as fallback
    if ! curl -sf "${VLLM_BASE_URL}/models" > /dev/null 2>&1; then
        echo "✗ vLLM not reachable at ${VLLM_BASE_URL}"
        echo ""
        echo "  Start vLLM on the remote server with:"
        echo "    vllm serve <merged-model-path> --host 0.0.0.0 --port 8000"
        echo ""
        echo "  Or export a checkpoint first:"
        echo "    python scripts/export_to_vllm.py \\"
        echo "        --adapter outputs/sft-gemma4-e2b-real-v1 \\"
        echo "        --output  outputs/sft-gemma4-e2b-merged-16bit"
        exit 1
    fi
fi
echo "  ✓ vLLM is up"

# ── Build output path ─────────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAFE_MODEL=$(echo "$VLLM_MODEL" | tr ':/' '__')
OUTPUT_DIR="results"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="${OUTPUT_DIR}/benchmark_vllm_${SAFE_MODEL}_${TIMESTAMP}.json"

# ── Build command ─────────────────────────────────────────────────────────────
VENV_PYTHON="${PROJECT_ROOT}/venv/bin/python"
CMD=(
    "$VENV_PYTHON" -m benchmark
    --dataset     "data/traces-2026-05-18.jsonl"
    --backend     vllm
    --model       "$VLLM_MODEL"
    --base-url    "$VLLM_BASE_URL"
    --temperature 0.0
    --max-tokens  1024
    --output      "$OUTPUT_FILE"
)

if [[ -n "$MAX_TRACES" ]]; then
    CMD+=(--max-traces "$MAX_TRACES")
fi

if [[ "$USE_JUDGE" == "true" ]]; then
    if [[ -z "${AZURE_OPENAI_ENDPOINT:-}" || -z "${AZURE_OPENAI_API_KEY:-}" || -z "${AZURE_OPENAI_MODEL:-}" ]]; then
        echo "⚠  Azure OpenAI credentials missing — running without judge."
        echo "   Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL in .env to enable."
    else
        CMD+=(
            --use-judge
            --judge-backend    azure
            --judge-model      "${AZURE_OPENAI_MODEL}"
            --judge-base-url   "${AZURE_OPENAI_ENDPOINT}"
            --judge-deployment "${AZURE_OPENAI_MODEL}"
            --judge-api-version "${AZURE_OPENAI_API_VERSION:-2025-01-01-preview}"
        )
        echo "  ✓ Judge: Azure OpenAI (${AZURE_OPENAI_MODEL})"
    fi
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Generator : vllm / ${VLLM_MODEL}"
echo "  Endpoint  : ${VLLM_BASE_URL}"
echo "  Dataset   : data/traces-2026-05-18.jsonl"
[[ -n "$MAX_TRACES" ]] && echo "  Max traces: ${MAX_TRACES} (smoke test)"
echo "  Output    : ${OUTPUT_FILE}"
echo "============================================================"
echo ""

"${CMD[@]}"

echo ""
echo "✓ Done. Report written to: ${OUTPUT_FILE}"
