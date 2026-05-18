from __future__ import annotations

from dataclasses import dataclass

from benchmark.replay import TraceReplayResult


@dataclass
class LatencyMetrics:
    """Per-trace inference performance metrics."""

    # All values are averages across turns within this trace
    ttft_ms_mean: float
    total_latency_ms_mean: float
    total_latency_ms_sum: float
    input_tokens_total: int
    output_tokens_total: int
    throughput_tok_s_mean: float      # mean per-turn throughput
    throughput_tok_s_overall: float   # total output tokens / total wall time


def score_latency(replay_result: TraceReplayResult) -> LatencyMetrics:
    """Compute latency/throughput metrics from a TraceReplayResult."""
    turns = replay_result.turns
    if not turns:
        return LatencyMetrics(
            ttft_ms_mean=0.0,
            total_latency_ms_mean=0.0,
            total_latency_ms_sum=0.0,
            input_tokens_total=0,
            output_tokens_total=0,
            throughput_tok_s_mean=0.0,
            throughput_tok_s_overall=0.0,
        )

    ttfts = [t.generated.ttft_ms for t in turns]
    latencies = [t.generated.total_latency_ms for t in turns]
    in_tokens = [t.generated.input_tokens for t in turns]
    out_tokens = [t.generated.output_tokens for t in turns]
    throughputs = [t.generated.throughput_tok_s for t in turns]

    total_out = sum(out_tokens)
    total_lat = sum(latencies)
    overall_throughput = total_out / (total_lat / 1000.0) if total_lat > 0 else 0.0

    def _mean(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    return LatencyMetrics(
        ttft_ms_mean=_mean(ttfts),
        total_latency_ms_mean=_mean(latencies),
        total_latency_ms_sum=round(total_lat, 2),
        input_tokens_total=sum(in_tokens),
        output_tokens_total=total_out,
        throughput_tok_s_mean=_mean(throughputs),
        throughput_tok_s_overall=round(overall_throughput, 2),
    )


def aggregate_latency_metrics(per_trace: list[LatencyMetrics]) -> dict:
    if not per_trace:
        return {}

    def _mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0

    return {
        "ttft_ms_mean": _mean([m.ttft_ms_mean for m in per_trace]),
        "total_latency_ms_mean": _mean([m.total_latency_ms_mean for m in per_trace]),
        "total_latency_ms_sum": round(sum(m.total_latency_ms_sum for m in per_trace), 2),
        "input_tokens_total": sum(m.input_tokens_total for m in per_trace),
        "output_tokens_total": sum(m.output_tokens_total for m in per_trace),
        "throughput_tok_s_mean": _mean([m.throughput_tok_s_mean for m in per_trace]),
        "throughput_tok_s_overall": _mean([m.throughput_tok_s_overall for m in per_trace]),
    }
