from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Lazy-loaded singletons
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("pip install sentence-transformers") from exc
        # Multilingual MiniLM — supports Italian, fast, ~120MB
        _embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embedding_model


@dataclass
class AnswerMetrics:
    """Answer quality metrics for a single trace."""

    exact_match: bool
    cosine_similarity: float
    llm_judge_score: float | None  # 1–5, None when judge is disabled


def _normalise(text: str) -> str:
    return text.strip().lower()


def score_answer(
    reference: str,
    generated: str,
    use_judge: bool = False,
    judge_client=None,          # InferenceClient or None
    system_prompt: str = "",
) -> AnswerMetrics:
    """
    Compute answer quality metrics for one (reference, generated) pair.

    Args:
        reference: The reference final_answer from the dataset.
        generated: The model's final generated text.
        use_judge: Whether to call an LLM judge for a 1-5 score.
        judge_client: An InferenceClient to use as the judge.
        system_prompt: The system prompt from the trace (context for the judge).
    """
    # Exact match
    exact = _normalise(reference) == _normalise(generated)

    # Cosine similarity via sentence-transformers
    cosine = 0.0
    try:
        import numpy as np

        model = _get_embedding_model()
        ref_emb, gen_emb = model.encode([reference, generated], normalize_embeddings=True)
        cosine = float(np.dot(ref_emb, gen_emb))
        cosine = round(max(-1.0, min(1.0, cosine)), 4)
    except Exception:  # noqa: BLE001
        cosine = 0.0

    # LLM judge
    judge_score: float | None = None
    if use_judge and judge_client is not None:
        judge_score = _call_judge(reference, generated, system_prompt, judge_client)

    return AnswerMetrics(
        exact_match=exact,
        cosine_similarity=cosine,
        llm_judge_score=judge_score,
    )


_JUDGE_SYSTEM = (
    "You are an objective evaluator assessing the quality of an AI assistant's response "
    "compared to a reference answer. Score the generated response on a scale from 1 to 5:\n"
    "1 = completely wrong or irrelevant\n"
    "2 = mostly wrong but some relevant elements\n"
    "3 = partially correct, missing key information\n"
    "4 = mostly correct with minor issues\n"
    "5 = correct, complete, and well-expressed\n\n"
    "Respond with ONLY a single integer (1–5). No explanation."
)


def _call_judge(
    reference: str,
    generated: str,
    system_prompt: str,
    judge_client,
) -> float | None:
    prompt = (
        f"TASK CONTEXT (system prompt summary):\n{system_prompt[:500]}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"GENERATED ANSWER:\n{generated}\n\n"
        "Score (1–5):"
    )
    try:
        result = judge_client.generate(
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            tools=None,
        )
        raw = result.text.strip()
        # Extract first digit found
        for ch in raw:
            if ch.isdigit() and ch in "12345":
                return float(ch)
    except Exception:  # noqa: BLE001
        pass
    return None


def aggregate_answer_metrics(per_trace: list[AnswerMetrics]) -> dict:
    if not per_trace:
        return {}

    def _mean(values: list) -> float:
        filtered = [v for v in values if v is not None]
        return round(sum(filtered) / len(filtered), 4) if filtered else 0.0

    judge_scores = [m.llm_judge_score for m in per_trace if m.llm_judge_score is not None]

    return {
        "exact_match_rate": round(sum(1 for m in per_trace if m.exact_match) / len(per_trace), 4),
        "cosine_similarity_mean": _mean([m.cosine_similarity for m in per_trace]),
        "llm_judge_score_mean": round(sum(judge_scores) / len(judge_scores), 4) if judge_scores else None,
    }
