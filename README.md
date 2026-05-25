# Agent-Tuning

A simple, end-to-end stack for fine-tuning an LLM to a specific agent tool-calling context while preserving general knowledge.

---

## Overview

Agent-Tuning takes a dataset of agent execution traces and guides it through a full MLOps lifecycle: benchmark your baseline, expand your dataset, prevent catastrophic forgetting, and fine-tune your model.

### Dataset Format

All pipelines in this repository expect a dataset with the following three columns:

| Column | Type | Description |
|---|---|---|
| `trace_id` | `string` | Unique identifier for the trace |
| `full_trace` | `list` | The complete agent execution trace (tool calls, observations, reasoning steps) |
| `tools` | `list` | The list of openai formatted tool definition and schemas |
| `final_answer` | `string` | The final answer produced by the agent at the end of the trace |

---

## Features

### 1. Benchmarking

Evaluate your model's performance against your trace dataset before and after fine-tuning.

- Serve your model locally via **vLLM** or **Ollama**
- Run inference over your dataset and collect performance metrics
- Metrics include accuracy, tool-call precision/recall, latency, and throughput
- Compare baseline vs. fine-tuned model side by side

### 2. Dataset Generation & Expansion

Grow your initial trace dataset using automated data generation pipelines.

- Generate new synthetic traces from your existing samples
- Support for **external LLM providers** (OpenAI, Anthropic, etc.) and **local models**
- Configurable generation strategies: paraphrasing, augmentation, tool-call variation
- Output format matches the three-column schema for seamless pipeline compatibility

### 3. General Dataset Expansion (Anti-Catastrophic Forgetting)

Blend in open-source general-purpose datasets to prevent the model from losing broad reasoning and language capabilities during fine-tuning.

- Curated selection of open-source datasets covering general instruction-following and reasoning ([Nemotron dataset](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1), [Dataset Library](https://github.com/mlabonne/llm-datasets))
- Configurable mixing ratio between domain-specific traces and general data
- Ensures fine-tuned models retain knowledge outside the agent tool-calling context

### 4. LLM Fine-Tuning

Fine-tune your LLM on the expanded, mixed dataset.

- Supports parameter-efficient fine-tuning methods (LoRA, QLoRA)
- Compatible with models served via vLLM or Ollama
- Training configuration managed through a single config file
- Checkpointing and evaluation hooks built in

---

## Pipeline

```
Your Trace Dataset
(trace_id | full_trace | final_answer)
         │
         ▼
┌─────────────────────┐
│    Benchmarking     │  ← Baseline metrics before tuning
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Dataset Generation  │  ← Expand with synthetic traces
│   & Expansion       │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  General Dataset    │  ← Mix in open-source data
│    Expansion        │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   LLM Fine-Tuning   │  ← Train on expanded dataset
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│    Benchmarking     │  ← Post-tuning metrics & comparison
└─────────────────────┘
```

---

## Getting Started

### Supervised Fine-Tuning (SFT)

The `train/` package implements an SFT pipeline on top of [Unsloth](https://docs.unsloth.ai) + QLoRA, with loss masking on the assistant tokens only. Defaults target `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` (a pre-quantized 4-bit checkpoint) on a single consumer GPU but the same CLI works for Qwen2.5, Llama-3.1, etc. — just override `--model-name`. The chat-template wire format and the loss-masking markers are picked automatically by `train/templates.py` based on the model id; add a new family by registering a `ChatTemplateAdapter` there.

#### Local run

Requires a CUDA-capable GPU on the same machine.

```bash
pip install -r requirements.txt
python -m train \
  --dataset data/traces-2026-05-18.jsonl \
  --output-dir outputs/sft-gemma4-e4b-v1
```

#### Remote run over SSH

When you don't have a local GPU, the same CLI can drive a remote host: the project tree and dataset are rsynced over, training runs there, and the LoRA adapter is rsynced back when it finishes. Logs stream live to your terminal.

1. Copy `.env.example` → `.env` and fill in the `SSH_REMOTE_*` block (host, user, path to the Python interpreter on the remote, etc.).
2. Make sure `ssh user@host` works non-interactively (key-based auth, no password prompts).
3. Add `--remote` to the same command:

```bash
python -m train --remote \
  --dataset data/traces-2026-05-18.jsonl \
  --output-dir outputs/sft-gemma4-e4b-v1
```

#### Tracking with Weights & Biases

Add `--wandb` (or any `--wandb-*` flag) to enable telemetry. Loss, learning rate, gradient norm, throughput, the full `SFTConfig`, the trainable-parameter count, and basic GPU/system info are pushed live to the W&B dashboard.

1. Put your `WANDB_API_KEY` in `.env` (see `.env.example` for the full block of supported variables: `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_TAGS`, `WANDB_MODE`).
2. Run training with the `--wandb` shortcut, optionally overriding the project/tags from the CLI:

```bash
python -m train --wandb \
  --wandb-project agent-tuning \
  --wandb-tags sft,gemma-4,real-dataset \
  --dataset data/traces-2026-05-18.jsonl \
  --output-dir outputs/sft-gemma4-e4b-v1
```

For a remote run, just combine `--remote` and `--wandb`. The `WANDB_*` variables are automatically forwarded over SSH so you don't need a second copy of the API key on the GPU host:

```bash
python -m train --remote --wandb \
  --dataset data/traces-2026-05-18.jsonl \
  --output-dir outputs/sft-gemma4-e4b-v1
```

The run name is auto-derived as `<output_dir_basename>-<model_short_name>-<utc_timestamp>` so each run is unique and easy to correlate with its local checkpoint. Override it explicitly with `--wandb-run-name`.

Set `WANDB_MODE=offline` to log to disk only (useful when the remote can't reach `api.wandb.ai`); you can `wandb sync` the run directory later from a connected machine.

The remote workflow assumes that the project's Python dependencies (Unsloth, torch, etc.) are already installed in the venv pointed to by `SSH_REMOTE_PYTHON`. The wrapper does not re-install them on every run.

---

## Contributing

> Contribution guidelines coming soon.
