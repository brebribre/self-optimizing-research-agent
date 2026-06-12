"""Dataset loading utilities.

The training/eval dataset lives under `data/` (see data/README.md for the
schema). This module turns raw JSONL records into `dspy.Example` objects that
the agent and optimizers consume.

The expected schema is intentionally minimal for the PoC and will be refined
once real data is provided:

    {
      "id": "...",
      "title": "...",
      "abstract": "...",
      "cited_papers": [{"title": "...", "abstract": "..."}, ...],
      "related_work": "..."        # gold reference (optional for inference)
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import dspy

# Repo-root-relative default location for datasets.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def format_cited_papers(cited_papers: list[dict]) -> str:
    """Render the list of candidate papers into the single-string input format."""
    return "\n".join(
        f"Title: {p.get('title', '').strip()} | Abstract: {p.get('abstract', '').strip()}"
        for p in cited_papers
    )


def load_examples(jsonl_path: str | Path) -> list[dspy.Example]:
    """Load a JSONL dataset file into a list of `dspy.Example`.

    Each example's inputs are (title, abstract, cited_papers); `related_work`
    is attached as the gold label when present.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Add data to {DATA_DIR} (see data/README.md)."
        )

    examples: list[dspy.Example] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            example = dspy.Example(
                title=record["title"],
                abstract=record["abstract"],
                cited_papers=format_cited_papers(record.get("cited_papers", [])),
                related_work=record.get("related_work", ""),
            ).with_inputs("title", "abstract", "cited_papers")
            examples.append(example)
    return examples
