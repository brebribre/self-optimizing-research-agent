"""Evaluation metrics for related-work generation.

DSPy optimizers need a metric `(example, prediction, trace) -> float|bool`.
For the PoC we provide a lightweight, dependency-free citation-recall metric:
the fraction of papers cited in the gold reference that also appear in the
generated section. This is a placeholder — once real data and quality goals
are defined, an LLM-as-judge metric (also expressible as a dspy.Signature)
will likely replace or complement it.
"""

from __future__ import annotations

import re

import dspy


def _title_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 3}


def citation_recall(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> float:
    """Approximate recall of gold-cited works mentioned in the generated text.

    Heuristic: a gold citation counts as "covered" if a majority of its
    title's significant tokens appear in the generated related_work text.
    Returns a score in [0, 1]. Returns 0.0 when there is no gold reference.
    """
    gold = (getattr(example, "related_work", "") or "").strip()
    generated = (getattr(prediction, "related_work", "") or "").strip().lower()
    if not gold or not generated:
        return 0.0

    # Pull quoted/Title-cased phrases from the gold section as proxy citations.
    gold_titles = re.findall(r"[A-Z][^.!?\n]{8,}", gold)
    if not gold_titles:
        return 0.0

    covered = 0
    for title in gold_titles:
        tokens = _title_tokens(title)
        if not tokens:
            continue
        hits = sum(1 for tok in tokens if tok in generated)
        if hits >= max(1, len(tokens) // 2):
            covered += 1

    return covered / len(gold_titles)
