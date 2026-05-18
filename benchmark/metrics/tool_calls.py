from __future__ import annotations

from dataclasses import dataclass

from benchmark.inference.base import ToolCall
from benchmark.replay import TraceReplayResult, TurnResult


@dataclass
class ToolCallMetrics:
    """Tool-call quality metrics for a single trace."""

    # Per-turn accuracy
    tool_name_accuracy: float        # fraction of AI turns with correct tool names
    arg_accuracy: float              # mean per-field arg accuracy across all tool calls
    sequence_precision: float        # correct tool calls / total generated tool calls
    sequence_recall: float           # correct tool calls / total reference tool calls

    # Raw counts
    total_reference_tool_calls: int
    total_generated_tool_calls: int
    correct_tool_calls: int          # calls where name AND all args match exactly
    spurious_calls: int              # generated calls when reference had none
    missing_calls: int               # reference calls where model generated none


def _args_match(ref_args: dict, gen_args: dict) -> float:
    """
    Per-field argument accuracy between reference and generated args.
    Returns a float in [0, 1].
    - If both are empty: 1.0
    - Missing generated keys count as 0.
    - Extra generated keys are ignored (not penalised but not rewarded).
    """
    if not ref_args and not gen_args:
        return 1.0
    if not ref_args:
        return 0.0

    correct = sum(
        1 for k, v in ref_args.items()
        if str(gen_args.get(k, "")).strip().lower() == str(v).strip().lower()
    )
    return correct / len(ref_args)


def _tool_calls_match(ref: ToolCall, gen: ToolCall) -> bool:
    """Strict exact match: same tool name and all reference args match."""
    return ref.name == gen.name and _args_match(ref.args, gen.args) == 1.0


def score_tool_calls(replay_result: TraceReplayResult) -> ToolCallMetrics:
    """Compute tool-call metrics from a single TraceReplayResult."""

    name_correct_turns = 0
    arg_accuracy_sum = 0.0
    arg_accuracy_count = 0

    total_ref_calls = 0
    total_gen_calls = 0
    correct_calls = 0
    spurious = 0
    missing = 0

    name_evaluated_turns = 0

    for turn in replay_result.turns:
        ref_tcs: list[ToolCall] = turn.reference_tool_calls
        gen_tcs: list[ToolCall] = turn.generated.tool_calls

        total_ref_calls += len(ref_tcs)
        total_gen_calls += len(gen_tcs)

        # Spurious: generated tool calls when reference had none
        if not ref_tcs and gen_tcs:
            spurious += len(gen_tcs)

        # Missing: reference had tool calls but model generated none
        if ref_tcs and not gen_tcs:
            missing += len(ref_tcs)

        if not ref_tcs and not gen_tcs:
            continue

        # Evaluate aligned tool calls (positional pairing)
        pairs = list(zip(ref_tcs, gen_tcs))
        if pairs:
            name_evaluated_turns += 1

        for ref_tc, gen_tc in pairs:
            # Tool name accuracy (per turn)
            if ref_tc.name == gen_tc.name:
                name_correct_turns += 1

            # Arg accuracy
            arg_acc = _args_match(ref_tc.args, gen_tc.args)
            arg_accuracy_sum += arg_acc
            arg_accuracy_count += 1

            # Strict exact match
            if _tool_calls_match(ref_tc, gen_tc):
                correct_calls += 1

    tool_name_acc = name_correct_turns / name_evaluated_turns if name_evaluated_turns else 0.0
    arg_acc = arg_accuracy_sum / arg_accuracy_count if arg_accuracy_count else 0.0
    precision = correct_calls / total_gen_calls if total_gen_calls else 0.0
    recall = correct_calls / total_ref_calls if total_ref_calls else 0.0

    return ToolCallMetrics(
        tool_name_accuracy=round(tool_name_acc, 4),
        arg_accuracy=round(arg_acc, 4),
        sequence_precision=round(precision, 4),
        sequence_recall=round(recall, 4),
        total_reference_tool_calls=total_ref_calls,
        total_generated_tool_calls=total_gen_calls,
        correct_tool_calls=correct_calls,
        spurious_calls=spurious,
        missing_calls=missing,
    )


def aggregate_tool_call_metrics(per_trace: list[ToolCallMetrics]) -> dict:
    """Aggregate per-trace ToolCallMetrics into dataset-level summary statistics."""
    if not per_trace:
        return {}

    def _mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "tool_name_accuracy_mean": _mean([m.tool_name_accuracy for m in per_trace]),
        "arg_accuracy_mean": _mean([m.arg_accuracy for m in per_trace]),
        "sequence_precision_mean": _mean([m.sequence_precision for m in per_trace]),
        "sequence_recall_mean": _mean([m.sequence_recall for m in per_trace]),
        "total_reference_tool_calls": sum(m.total_reference_tool_calls for m in per_trace),
        "total_generated_tool_calls": sum(m.total_generated_tool_calls for m in per_trace),
        "correct_tool_calls_total": sum(m.correct_tool_calls for m in per_trace),
        "spurious_calls_total": sum(m.spurious_calls for m in per_trace),
        "missing_calls_total": sum(m.missing_calls for m in per_trace),
    }
