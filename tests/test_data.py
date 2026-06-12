"""Smoke tests that do not require an API key."""

import json
from pathlib import Path

import pytest

from research_agent.data import format_candidate_pool, load_examples

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "data" / "sample.jsonl"


def test_format_candidate_pool():
    out = format_candidate_pool([{"pool_id": "P01", "title": "A", "abstract": "B"}])
    assert out == "[P01] Title: A | Abstract: B"


def test_load_sample_dataset():
    examples = load_examples(SAMPLE)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.example_id == "row-1"
    assert ex.title.startswith("Self-Correcting Tool Use")
    # inputs are marked, labels are attached
    assert set(ex.inputs().keys()) == {"title", "abstract", "cited_papers"}
    assert ex.cited_papers.count("\n") == 29  # 30 pool entries
    assert "[P01] Title: ReAct" in ex.cited_papers
    assert ex.gold_cited == ["P01", "P02", "P09", "P14", "P22", "P25", "P28"]
    assert "[P01]" in ex.related_work  # id_mapped gold
    assert "\\cite{" in ex.related_work_raw  # raw gold keeps LaTeX citations


def test_split_filtering():
    assert len(load_examples(SAMPLE, split="train")) == 1
    assert len(load_examples(SAMPLE, split="test")) == 0


def test_invalid_records_rejected(tmp_path):
    def write(record):
        p = tmp_path / "bad.jsonl"
        p.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return p

    base = {
        "example_id": "row-x",
        "query": {"title": "t", "abstract": "a"},
        "candidate_pool": [{"pool_id": "P01", "title": "x", "abstract": "y"}],
    }

    with pytest.raises(ValueError, match="title"):
        load_examples(write({**base, "query": {"title": "", "abstract": "a"}}))
    with pytest.raises(ValueError, match="non-empty list"):
        load_examples(write({**base, "candidate_pool": []}))
    with pytest.raises(ValueError, match="duplicate"):
        load_examples(
            write(
                {
                    **base,
                    "candidate_pool": [
                        {"pool_id": "P01", "title": "x"},
                        {"pool_id": "P01", "title": "y"},
                    ],
                }
            )
        )
    with pytest.raises(ValueError, match="not in candidate_pool"):
        load_examples(write({**base, "gold_cited": ["P99"]}))
