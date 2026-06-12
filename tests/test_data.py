"""Smoke tests that do not require an API key."""

from pathlib import Path

from research_agent.data import format_cited_papers, load_examples

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_format_cited_papers():
    out = format_cited_papers([{"title": "A", "abstract": "B"}])
    assert "Title: A" in out
    assert "Abstract: B" in out


def test_load_sample_dataset():
    examples = load_examples(REPO_ROOT / "data" / "sample.jsonl")
    assert len(examples) == 1
    ex = examples[0]
    assert ex.title.startswith("Efficient Retrieval")
    # inputs are marked, gold label is attached
    assert "title" in ex.inputs()
    assert ex.related_work
