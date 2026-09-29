# Agent-Tuning

Toolkit for training and evaluating language models on agent traces, with a focus on reliable tool calling.

The repository contains two independent Python environments:

- `train/`: LoRA/QLoRA fine-tuning with Unsloth and TRL.
- `benchmark/`: trace replay and model evaluation through LangChain clients.

It also includes a small FastAPI playground for trying saved adapters.

> [!WARNING]
> Training requires a CUDA-capable GPU. The vLLM benchmark scripts start the server themselves. Ollama and hosted backends still need a server that is already running.

## Repository layout

```text
.
├── train/                 # SFT and GRPO training environment
├── benchmark/             # Replay engine, inference clients, and metrics
├── scripts/               # Runnable training and dataset utilities
├── app/backend/           # FastAPI playground backend
├── app/frontend/          # Playground UI
├── data/                  # Local datasets (ignored by git)
└── outputs/               # Checkpoints and reports (ignored by git)
```

## Dataset format

Datasets are JSONL files: one trace per line. The current schema uses `messages`; legacy files using `full_trace` are also accepted by the benchmark.

```json
{
  "trace_id": "trace-001",
  "messages": [
    {"role": "system", "content": "You are an assistant."},
    {"role": "user", "content": "Find my latest invoice."},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "name": "get_invoice", "args": {"limit": 1}}]},
    {"role": "tool", "tool_call_id": "call-1", "content": "{\"id\": \"inv-001\"}"},
    {"role": "assistant", "content": "Your latest invoice is inv-001."}
  ],
  "tools": [{"type": "function", "function": {"name": "get_invoice", "description": "Get invoices.", "parameters": {"type": "object", "properties": {}}}}],
  "final_answer": "Your latest invoice is inv-001."
}
```

Required fields are `trace_id` (or `id`) and `messages` (or `full_trace`). `tools` can be supplied per trace. `final_answer` is optional; when absent, the last assistant message is used.

## Installation

Install [uv](https://docs.astral.sh/uv/) and create the environment you need:

```bash
uv sync --project train       # GPU training
uv sync --project benchmark   # evaluation clients
cp .env.example .env
```

Keep the environments separate. Their PyTorch and inference dependencies are not intended to share one virtual environment.

## Fine-tuning

Model recipes live in `shared/configs/` (train hyperparams plus vLLM serving knobs such as `tool_call_parser`). Configure the run by editing constants at the top of the script (model id, dataset, output dir, …), then launch it — no CLI flags:

```bash
# Edit MODEL_ID / DATASET_PATH / … in scripts/train_sft.py, then:
uv run --project train python scripts/train_sft.py

# Same pattern for GRPO:
uv run --project train python scripts/train_qwen3_grpo.py
```

Each recipe creates a unique output directory and saves a LoRA adapter:

- SFT: `outputs/.../best_eval_model`
- GRPO: `outputs/.../best_model`

SFT masks the loss to assistant responses. GRPO accepts one or more reward functions; the example uses tool-call format, tool-name matching, and non-empty completion rewards. An SFT adapter can be passed to GRPO through `lora_adapter_path`.

For remote training, configure the `SSH_REMOTE_*` variables in `.env`. The training wrapper syncs the project and dataset over SSH, streams logs, and can fetch the resulting adapter. The remote host must have a compatible Python environment and CUDA stack.

### Experiment tracking

Set `WANDB_API_KEY` in `.env` and enable W&B in the training configuration. Use `WANDB_MODE=offline` when the training host cannot access W&B; the run can be synchronized later with `wandb sync`.

## Benchmarking

The benchmark replays tool-calling and final-answer turns from each trace. Reference intermediate turns are injected back into the conversation, keeping the evaluation deterministic while measuring the model's decisions.

Supported backends:

- `openai`: OpenAI or another OpenAI-compatible hosted endpoint.
- `vllm`: an OpenAI-compatible vLLM server.
- `ollama`: a local or remote Ollama server.
- `azure`: Azure OpenAI.
- `bedrock`: AWS Bedrock Converse.

The vLLM script stops other GPU processes, starts the server, replays the dataset, then stops the server. Edit constants at the top of `scripts/benchmark.py` (set `LORA_PATH` / `LORA_NAME` to `None` for a base-model run), then launch it — no CLI flags:

```bash
uv run --project benchmark python scripts/benchmark.py
```

Each scored turn prints only the reference action and the model action. The JSON report is still written next to the dataset.

To point an already running server at the client:

```bash
uv run --project benchmark python -m benchmark \
  --dataset data/traces.jsonl \
  --backend vllm \
  --model qwen-umore \
  --base-url http://localhost:8000/v1 \
  --max-traces 5 \
  --output outputs/benchmark.json
```

For a quick Ollama run:

```bash
uv run --project benchmark python -m benchmark \
  --dataset data/traces.jsonl \
  --backend ollama \
  --model llama3.1:8b
```

The report includes tool-call, answer, compliance, and latency metrics. Add `--use-judge` to score answer quality and rule compliance with an LLM judge. By default the tested model is reused as the judge; configure `--judge-backend` and `--judge-model` to use a separate one.

Use `--max-traces N` for smoke tests. If `--output` is omitted, the report is written next to the dataset with a model-specific filename.

## Playground

The backend exposes `GET /api/health`, `GET /api/adapters`, `GET /api/system-prompt`, and `POST /api/chat`.

Start it from the repository root with:

```bash
uv run --project train python -m app.backend
```

The backend discovers adapters from the configured local output directories. The frontend lives in `app/frontend/`; use the package manager and scripts defined by that application to run the UI.

## Configuration

`.env.example` documents the supported integrations:

- Langfuse dataset enrichment.
- Hugging Face/training model selection.
- Weights & Biases.
- SSH remote training.
- OpenAI, Azure, AWS, and local inference credentials.

Never commit `.env`, datasets, model weights, or generated reports.

## Development

Run commands from the repository root. Keep generated data under `data/` and checkpoints under `outputs/`; both are ignored by git. Add a model recipe under `shared/configs/` when a new chat template, checkpoint, or vLLM parser needs dedicated defaults.

## Contributing

Issues and pull requests are welcome. Please include the dataset schema, training recipe, benchmark command, and relevant metrics when reporting a change.
