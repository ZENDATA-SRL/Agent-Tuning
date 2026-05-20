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

> Documentation and setup instructions coming soon.

---

## Contributing

> Contribution guidelines coming soon.
