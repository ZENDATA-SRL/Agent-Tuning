from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from benchmark.config import InferenceProfile
from benchmark.inference import build_client
from benchmark.inference.base import InferenceClient
from benchmark.metrics.answer import AnswerMetrics, score_answer
from benchmark.metrics.compliance import ComplianceMetrics, score_compliance
from benchmark.metrics.latency import LatencyMetrics, score_latency
from benchmark.metrics.tool_calls import ToolCallMetrics, score_tool_calls
from benchmark.replay import ReplayEngine, TraceReplayResult
from benchmark.report import build_report, save_report


@dataclass
class TraceRecord:
    """All metric results for a single trace."""

    trace_id: str
    tool_calls: ToolCallMetrics
    answer: AnswerMetrics
    compliance: ComplianceMetrics
    latency: LatencyMetrics
    replay_error: str | None


def _load_dataset(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _extract_system_prompt(full_trace: list[dict]) -> str:
    for msg in full_trace:
        if msg.get("role", "").lower() == "system":
            return msg.get("content", "") or ""
    return ""


def run_benchmark(
    dataset_path: str | Path,
    profile: InferenceProfile,
    output_path: str | Path | None = None,
    use_judge: bool = False,
    judge_profile: InferenceProfile | None = None,
    tools: list[dict] | None = None,
    max_traces: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """
    Run the full benchmarking pipeline on a JSONL dataset.

    Args:
        dataset_path:      Path to the .jsonl dataset file.
        profile:           InferenceProfile for the model under test.
        output_path:       Where to write the JSON report. Defaults to
                           <dataset_path_stem>_<profile_label>_benchmark.json
        use_judge:         Enable LLM-as-judge metrics (answer quality + rule compliance).
        judge_profile:     InferenceProfile for the judge model. Falls back to profile
                           if None (uses the same model as judge).
        tools:             Tool definitions in OpenAI schema to pass to the model.
                           If None, tools are read from the `tools` field of each dataset
                           record (added by scripts/add_tools_to_dataset.py). If a record
                           has no `tools` field either, no tools are forwarded.
        max_traces:        Limit number of traces processed (useful for smoke tests).
        progress_callback: Optional callable(current, total) for progress reporting.

    Returns:
        The report dict (also written to output_path).
    """
    dataset_path = Path(dataset_path)
    traces = _load_dataset(dataset_path)
    if max_traces:
        traces = traces[:max_traces]

    # Unique session ID groups all Langfuse traces for this benchmark run
    session_id = f"benchmark-{profile.label().replace('/', '_')}-{int(time.time())}"

    # Build inference clients
    client: InferenceClient = build_client(profile, session_id=session_id)
    judge_client: InferenceClient | None = None
    if use_judge:
        judge_client = build_client(
            judge_profile if judge_profile else profile,
            session_id=session_id,
        )

    # tools=None means "read from each record"; tools=[] means "no tools"
    global_tools: list[dict] | None = tools
    # Build a default replay engine (per-trace tools override it below)
    replay_engine = ReplayEngine(client=client, tools=global_tools or [])

    records: list[TraceRecord] = []

    for i, trace in enumerate(tqdm(traces, desc=f"Benchmarking {profile.label()}", file=sys.stdout)):
        if progress_callback:
            progress_callback(i, len(traces))

        trace_id = trace.get("trace_id", f"trace_{i}")
        full_trace: list[dict] = trace.get("full_trace", [])
        system_prompt = _extract_system_prompt(full_trace)
        reference_final = trace.get("final_answer", "")
       
        # Resolve tools: explicit argument > per-record field > empty
        trace_tools: list[dict] = (
            global_tools
            if global_tools is not None
            else trace.get("tools", [])
        )
        if trace_tools is not replay_engine._tools:
            replay_engine = ReplayEngine(client=client, tools=trace_tools)

        # Replay
        replay_result: TraceReplayResult = replay_engine.replay(trace)

        # Tool call metrics
        tc_metrics = score_tool_calls(replay_result)

        # Answer metrics
        answer_metrics = score_answer(
            reference=reference_final,
            generated=replay_result.generated_final_answer,
            use_judge=use_judge,
            judge_client=judge_client,
            system_prompt=system_prompt,
        )

        # Compliance metrics
        compliance_metrics = score_compliance(
            replay_result=replay_result,
            full_trace=full_trace,
            use_judge=use_judge,
            judge_client=judge_client,
            system_prompt=system_prompt,
        )

        # Latency metrics
        latency_metrics = score_latency(replay_result)

        records.append(TraceRecord(
            trace_id=trace_id,
            tool_calls=tc_metrics,
            answer=answer_metrics,
            compliance=compliance_metrics,
            latency=latency_metrics,
            replay_error=replay_result.error,
        ))

    report = build_report(profile=profile, records=records)

    if output_path is None:
        safe_label = profile.label().replace("/", "_")
        output_path = dataset_path.parent / f"{dataset_path.stem}_{safe_label}_benchmark.json"

    save_report(report, output_path)
    return report
