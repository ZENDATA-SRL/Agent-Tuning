#!/usr/bin/env bash
# ============================================================
#  benchmark.sh — Run the Agent-Tuning benchmark
#
#  Generator  : Ollama (local)
#  Judge      : Azure OpenAI (uses .env credentials)
#
#  Usage:
#    ./scripts/benchmark.sh [ollama-model] [--smoke] [--no-judge]
#
#  Examples:
#    ./scripts/benchmark.sh                          # llama3.1:8b, full run, with judge
#    ./scripts/benchmark.sh qwen2.5:7b               # different model
#    ./scripts/benchmark.sh llama3.1:8b --smoke      # first 10 traces only
#    ./scripts/benchmark.sh llama3.1:8b --no-judge   # skip LLM judge (faster)
# ============================================================

set -euo pipefail

# ── Resolve project root (directory containing this script's parent) ────────
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
OLLAMA_MODEL="${1:-llama3.2:3b}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
USE_JUDGE=true
MAX_TRACES=""

# ── Parse optional flags ─────────────────────────────────────────────────────
shift || true          # shift past positional model arg if present
for arg in "$@"; do
    case "$arg" in
        --smoke)      MAX_TRACES=10 ;;
        --no-judge)   USE_JUDGE=false ;;
        --help|-h)
            sed -n '2,12p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
    esac
done

# ── Validate Ollama is reachable ─────────────────────────────────────────────
echo "▶ Checking Ollama at ${OLLAMA_BASE_URL} ..."
if ! curl -sf "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; then
    echo "✗ Ollama not reachable at ${OLLAMA_BASE_URL}"
    echo "  Start it with:  ollama serve"
    exit 1
fi
echo "  ✓ Ollama is up"

# ── Validate model is pulled ─────────────────────────────────────────────────
echo "▶ Checking model '${OLLAMA_MODEL}' is available ..."
AVAILABLE=$(curl -sf "${OLLAMA_BASE_URL}/api/tags" | python3 -c \
    "import json,sys; tags=json.load(sys.stdin); print(' '.join(m['name'] for m in tags.get('models', [])))")
if echo "$AVAILABLE" | grep -qw "$OLLAMA_MODEL"; then
    echo "  ✓ Model found"
else
    echo "  ⚠ Model '${OLLAMA_MODEL}' not found locally. Pulling now..."
    ollama pull "$OLLAMA_MODEL"
fi

# ── Build output path ─────────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAFE_MODEL=$(echo "$OLLAMA_MODEL" | tr ':/' '__')
OUTPUT_DIR="results"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="${OUTPUT_DIR}/benchmark_ollama_${SAFE_MODEL}_${TIMESTAMP}.json"

# ── Build command ─────────────────────────────────────────────────────────────
VENV_PYTHON="${PROJECT_ROOT}/venv/bin/python"
CMD=(
    "$VENV_PYTHON" -m benchmark
    --dataset  "data/traces-2026-05-18.jsonl"
    --backend  ollama
    --model    "$OLLAMA_MODEL"
    --base-url "$OLLAMA_BASE_URL"
    --temperature 0.0
    --max-tokens  1024
    --output   "$OUTPUT_FILE"
)

if [[ -n "$MAX_TRACES" ]]; then
    CMD+=(--max-traces "$MAX_TRACES")
fi

if [[ "$USE_JUDGE" == "true" ]]; then
    # Validate Azure credentials are present
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
echo "  Generator : ollama / ${OLLAMA_MODEL}"
echo "  Dataset   : data/traces-2026-05-18.jsonl"
[[ -n "$MAX_TRACES" ]] && echo "  Max traces: ${MAX_TRACES} (smoke test)"
echo "  Output    : ${OUTPUT_FILE}"
echo "============================================================"
echo ""

"${CMD[@]}"

echo ""
echo "✓ Done. Report written to: ${OUTPUT_FILE}"
