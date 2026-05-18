from __future__ import annotations

from dataclasses import dataclass

from benchmark.replay import TraceReplayResult


@dataclass
class ComplianceMetrics:
    """Behavioural compliance metrics for a single trace."""

    language_match: bool        # last AI turn language matches last user turn language
    turn_economy: float         # generated turns / reference turns  (1.0 = identical)
    rule_compliance_score: float | None  # 1–5 from LLM judge, None when disabled


def _detect_language(text: str) -> str | None:
    try:
        from langdetect import detect, LangDetectException
        return detect(text)
    except Exception:  # noqa: BLE001
        return None


def _find_last_of_role(turns_raw: list[dict], role: str) -> str:
    """Return the content of the last message with the given role."""
    for msg in reversed(turns_raw):
        if msg.get("role", "").lower() in (role, {"user": "human"}.get(role, role)):
            return msg.get("content", "") or ""
    return ""


def score_compliance(
    replay_result: TraceReplayResult,
    full_trace: list[dict],
    use_judge: bool = False,
    judge_client=None,
    system_prompt: str = "",
) -> ComplianceMetrics:
    """
    Compute compliance metrics for one replayed trace.

    Args:
        replay_result: The TraceReplayResult from the replay engine.
        full_trace: The original raw message list (for language detection).
        use_judge: Whether to call an LLM judge for rule compliance scoring.
        judge_client: An InferenceClient to use as judge.
        system_prompt: The system prompt (used for rule compliance judging).
    """
    # Language match: compare language of last user turn vs last generated AI turn
    last_user_text = _find_last_of_role(full_trace, "human") or _find_last_of_role(full_trace, "user")
    last_gen_text = replay_result.generated_final_answer

    user_lang = _detect_language(last_user_text) if last_user_text else None
    gen_lang = _detect_language(last_gen_text) if last_gen_text else None
    language_match = (user_lang is not None and user_lang == gen_lang)

    # Turn economy: ratio of generated turns to reference turns
    ref_turns = replay_result.total_turns_reference
    gen_turns = replay_result.total_turns_generated
    turn_economy = round(gen_turns / ref_turns, 4) if ref_turns > 0 else 1.0

    # Rule compliance via LLM judge
    rule_score: float | None = None
    if use_judge and judge_client is not None:
        rule_score = _call_rule_judge(
            system_prompt=system_prompt,
            full_trace=full_trace,
            generated_final=last_gen_text,
            judge_client=judge_client,
        )

    return ComplianceMetrics(
        language_match=language_match,
        turn_economy=turn_economy,
        rule_compliance_score=rule_score,
    )


_RULE_JUDGE_SYSTEM = (
    "You are an auditor evaluating whether an AI assistant's response complies with "
    "the rules defined in its system prompt. Score compliance on a scale from 1 to 5:\n"
    "1 = violates multiple important rules\n"
    "2 = violates at least one important rule\n"
    "3 = partially compliant, minor rule violations\n"
    "4 = mostly compliant, very minor or debatable issues\n"
    "5 = fully compliant with all rules\n\n"
    "Respond with ONLY a single integer (1–5). No explanation."
)


def _call_rule_judge(
    system_prompt: str,
    full_trace: list[dict],
    generated_final: str,
    judge_client,
) -> float | None:
    # Build a compact conversation summary for the judge
    turns_summary_parts = []
    for msg in full_trace:
        role = msg.get("role", "")
        content = (msg.get("content") or "")[:200]
        if role.lower() in ("human", "user"):
            turns_summary_parts.append(f"User: {content}")
        elif role.lower() in ("ai", "assistant"):
            turns_summary_parts.append(f"Assistant: {content}")
    turns_summary = "\n".join(turns_summary_parts[-10:])  # last 10 turns max

    prompt = (
        f"SYSTEM PROMPT (rules):\n{system_prompt[:800]}\n\n"
        f"CONVERSATION (last turns):\n{turns_summary}\n\n"
        f"FINAL GENERATED RESPONSE:\n{generated_final}\n\n"
        "Rule compliance score (1–5):"
    )
    try:
        result = judge_client.generate(
            messages=[
                {"role": "system", "content": _RULE_JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            tools=None,
        )
        raw = result.text.strip()
        for ch in raw:
            if ch.isdigit() and ch in "12345":
                return float(ch)
    except Exception:  # noqa: BLE001
        pass
    return None


def aggregate_compliance_metrics(per_trace: list[ComplianceMetrics]) -> dict:
    if not per_trace:
        return {}

    def _mean(values: list) -> float:
        filtered = [v for v in values if v is not None]
        return round(sum(filtered) / len(filtered), 4) if filtered else 0.0

    rule_scores = [m.rule_compliance_score for m in per_trace if m.rule_compliance_score is not None]

    return {
        "language_match_rate": round(sum(1 for m in per_trace if m.language_match) / len(per_trace), 4),
        "turn_economy_mean": _mean([m.turn_economy for m in per_trace]),
        "rule_compliance_score_mean": round(sum(rule_scores) / len(rule_scores), 4) if rule_scores else None,
    }
