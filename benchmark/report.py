from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.config import InferenceProfile
    from benchmark.runner import TraceRecord

from benchmark.metrics.answer import aggregate_answer_metrics
from benchmark.metrics.compliance import aggregate_compliance_metrics
from benchmark.metrics.latency import aggregate_latency_metrics
from benchmark.metrics.tool_calls import aggregate_tool_call_metrics


def build_report(
    profile: "InferenceProfile",
    records: list["TraceRecord"],
) -> dict:
    """
    Assemble the full benchmark report dict from per-trace records.

    Structure:
    {
      "profile": {...},
      "summary": {
        "tool_calls": {...},
        "answer": {...},
        "compliance": {...},
        "latency": {...},
        "total_traces": N,
        "error_traces": N,
      },
      "per_trace": [
        {
          "trace_id": "...",
          "tool_calls": {...},
          "answer": {...},
          "compliance": {...},
          "latency": {...},
          "replay_error": null | "..."
        },
        ...
      ]
    }
    """
    per_trace_list = []
    for rec in records:
        per_trace_list.append({
            "trace_id": rec.trace_id,
            "tool_calls": asdict(rec.tool_calls),
            "answer": asdict(rec.answer),
            "compliance": asdict(rec.compliance),
            "latency": asdict(rec.latency),
            "replay_error": rec.replay_error,
        })

    error_traces = sum(1 for r in records if r.replay_error is not None)

    summary = {
        "total_traces": len(records),
        "error_traces": error_traces,
        "tool_calls": aggregate_tool_call_metrics([r.tool_calls for r in records]),
        "answer": aggregate_answer_metrics([r.answer for r in records]),
        "compliance": aggregate_compliance_metrics([r.compliance for r in records]),
        "latency": aggregate_latency_metrics([r.latency for r in records]),
    }

    return {
        "profile": {
            "backend": profile.backend,
            "model": profile.model,
            "base_url": profile.base_url,
            "temperature": profile.temperature,
            "max_tokens": profile.max_tokens,
        },
        "summary": summary,
        "per_trace": per_trace_list,
    }


def save_report(report: dict, output_path: str | Path) -> None:
    """Serialize report to JSON and print a human-readable summary to stdout."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    _print_summary(report, output_path)


def _print_summary(report: dict, output_path: Path) -> None:
    profile = report["profile"]
    s = report["summary"]
    tc = s.get("tool_calls", {})
    ans = s.get("answer", {})
    comp = s.get("compliance", {})
    lat = s.get("latency", {})

    lines = [
        "",
        "=" * 60,
        f"  BENCHMARK REPORT — {profile['backend']}/{profile['model']}",
        "=" * 60,
        f"  Traces evaluated : {s['total_traces']}  (errors: {s['error_traces']})",
        "",
        "  TOOL CALLS",
        f"    Name accuracy    : {tc.get('tool_name_accuracy_mean', 'n/a')}",
        f"    Arg accuracy     : {tc.get('arg_accuracy_mean', 'n/a')}",
        f"    Precision        : {tc.get('sequence_precision_mean', 'n/a')}",
        f"    Recall           : {tc.get('sequence_recall_mean', 'n/a')}",
        f"    Spurious calls   : {tc.get('spurious_calls_total', 'n/a')}",
        f"    Missing calls    : {tc.get('missing_calls_total', 'n/a')}",
        "",
        "  ANSWER QUALITY",
        f"    Exact match rate : {ans.get('exact_match_rate', 'n/a')}",
        f"    Cosine sim mean  : {ans.get('cosine_similarity_mean', 'n/a')}",
        f"    LLM judge mean   : {ans.get('llm_judge_score_mean', 'disabled')}",
        "",
        "  COMPLIANCE",
        f"    Language match   : {comp.get('language_match_rate', 'n/a')}",
        f"    Turn economy     : {comp.get('turn_economy_mean', 'n/a')}",
        f"    Rule compliance  : {comp.get('rule_compliance_score_mean', 'disabled')}",
        "",
        "  INFERENCE PERFORMANCE",
        f"    TTFT mean        : {lat.get('ttft_ms_mean', 'n/a')} ms",
        f"    Latency mean     : {lat.get('total_latency_ms_mean', 'n/a')} ms",
        f"    Throughput mean  : {lat.get('throughput_tok_s_mean', 'n/a')} tok/s",
        f"    Total in tokens  : {lat.get('input_tokens_total', 'n/a')}",
        f"    Total out tokens : {lat.get('output_tokens_total', 'n/a')}",
        "",
        f"  Report saved to: {output_path}",
        "=" * 60,
        "",
    ]
    print("\n".join(lines))
