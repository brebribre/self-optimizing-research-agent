"""Tests for citation metrics that run without an API key."""

import dspy

from research_agent.data import format_candidate_pool, load_examples
from research_agent.metrics import citation_f1
from tests.test_data import SAMPLE

POOL = [
    {"pool_id": "P01", "title": "ReAct: Synergizing Reasoning and Acting in Language Models", "abstract": "x"},
    {"pool_id": "P02", "title": "Reflexion: Language Agents with Verbal Reinforcement Learning", "abstract": "y"},
    {"pool_id": "P03", "title": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", "abstract": "z"},
]


def _example(gold_cited: list[str]) -> dspy.Example:
    return dspy.Example(
        title="t",
        abstract="a",
        cited_papers=format_candidate_pool(POOL),
        related_work=" ".join(f"[{p}]" for p in gold_cited),
        gold_cited=gold_cited,
    ).with_inputs("title", "abstract", "cited_papers")


def _pred(text: str) -> dspy.Prediction:
    return dspy.Prediction(related_work=text)


def test_perfect_match_scores_one():
    ex = _example(["P01", "P02"])
    pred = _pred("Agents interleave reasoning and acting [P01] and learn from reflection [P02].")
    assert citation_f1(ex, pred) == 1.0


def test_missing_citation_lowers_recall():
    ex = _example(["P01", "P02"])
    pred = _pred("Agents interleave reasoning and acting [P01].")
    # recall 0.5, precision 1.0 -> F1 = 2/3
    assert abs(citation_f1(ex, pred) - 2 / 3) < 1e-9


def test_spurious_citation_lowers_precision():
    ex = _example(["P01"])
    pred = _pred("Agents act [P01] and also search over thoughts [P03].")
    # precision 0.5, recall 1.0 -> F1 = 2/3
    assert abs(citation_f1(ex, pred) - 2 / 3) < 1e-9


def test_disjoint_sets_score_zero():
    ex = _example(["P02"])
    pred = _pred("Search over thoughts [P03] is effective.")
    assert citation_f1(ex, pred) == 0.0


def test_markers_trusted_over_title_mentions():
    # The prediction *names* Reflexion but only *cites* P01; with markers
    # present, title mentions must not count as citations.
    ex = _example(["P01"])
    pred = _pred(
        "ReAct [P01] interleaves reasoning and acting; unlike Reflexion, "
        "Language Agents with Verbal Reinforcement Learning, it needs no memory."
    )
    assert citation_f1(ex, pred) == 1.0


def test_title_fallback_without_markers():
    # No [Pxx] markers anywhere -> fall back to title matching.
    ex = _example(["P02"])
    pred = _pred(
        "Reflexion introduces language agents with verbal reinforcement learning."
    )
    assert citation_f1(ex, pred) == 1.0


def test_gold_cited_answer_key_beats_gold_text():
    # gold_cited is the reference even if the gold text mentions fewer works.
    ex = dspy.Example(
        title="t",
        abstract="a",
        cited_papers=format_candidate_pool(POOL),
        related_work="[P01]",  # gold text only shows P01...
        gold_cited=["P01", "P02"],  # ...but the answer key has two works
    ).with_inputs("title", "abstract", "cited_papers")
    pred = _pred("[P01] [P02]")
    assert citation_f1(ex, pred) == 1.0


def test_both_empty_scores_one():
    ex = _example([])
    assert citation_f1(ex, _pred("")) == 1.0


def test_no_candidates_scores_zero():
    ex = dspy.Example(
        title="t", abstract="a", cited_papers="", related_work="g", gold_cited=["P01"]
    ).with_inputs("title", "abstract", "cited_papers")
    assert citation_f1(ex, _pred("anything")) == 0.0


def test_sample_row_gold_text_scores_one():
    # The sample row's own id_mapped gold section must score 1.0 against its answer key.
    ex = load_examples(SAMPLE)[0]
    assert citation_f1(ex, _pred(ex.related_work)) == 1.0
