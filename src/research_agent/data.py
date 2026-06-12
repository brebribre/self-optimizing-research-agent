"""Dataset loading utilities.

The dataset lives under `data/` (see data/README.md for the full collection
spec). Each JSONL record pairs a query paper with a 30-paper candidate pool
(gold citations + same-subfield distractors) and an answer key:

    {
      "example_id": "row-1",
      "split": "train",                       # train | val | test
      "query": {
        "arxiv_id": "2602.01234",
        "title": "...",
        "abstract": "...",
        "submitted": "2026-02-03"
      },
      "candidate_pool": [
        {"pool_id": "P01", "arxiv_id": "...", "title": "...", "abstract": "..."},
        ...
      ],
      "gold_cited": ["P01", "P02", ...],      # pool_ids the real authors cited
      "gold_related_work": {
        "raw": "... \\cite{yao2022react} ...",
        "id_mapped": "... [P01] ..."          # citations rewritten to pool_ids
      }
    }

`gold_cited` / `gold_related_work` may be omitted for inference-only examples.
This module turns records into `dspy.Example` objects whose inputs are
(title, abstract, cited_papers); the id-mapped gold section and the
`gold_cited` answer key are attached as labels for the metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import dspy

# Repo-root-relative default location for datasets.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def format_candidate_pool(pool: list[dict]) -> str:
    """Render the candidate pool into the single-string agent input.

    One paper per line: ``[P01] Title: <title> | Abstract: <abstract>``.
    The bracketed pool_id is how the agent (and the metrics) refer to a work.
    """
    return "\n".join(
        f"[{p['pool_id']}] Title: {p.get('title', '').strip()}"
        f" | Abstract: {p.get('abstract', '').strip()}"
        for p in pool
    )


def _validate_record(record: dict, where: str) -> None:
    """Fail fast with a pointer to the offending record."""
    query = record.get("query")
    if not isinstance(query, dict) or not query.get("title") or not query.get("abstract"):
        raise ValueError(f"{where}: 'query' must contain non-empty 'title' and 'abstract'.")

    pool = record.get("candidate_pool")
    if not isinstance(pool, list) or not pool:
        raise ValueError(f"{where}: 'candidate_pool' must be a non-empty list.")
    pool_ids = [p.get("pool_id") for p in pool]
    if any(not pid for pid in pool_ids):
        raise ValueError(f"{where}: every candidate needs a 'pool_id'.")
    if len(set(pool_ids)) != len(pool_ids):
        raise ValueError(f"{where}: duplicate pool_ids in candidate_pool.")

    gold_cited = record.get("gold_cited", [])
    unknown = set(gold_cited) - set(pool_ids)
    if unknown:
        raise ValueError(f"{where}: gold_cited ids not in candidate_pool: {sorted(unknown)}.")


def load_examples(jsonl_path: str | Path, split: str | None = None) -> list[dspy.Example]:
    """Load a JSONL dataset file into a list of `dspy.Example`.

    Each example's inputs are (title, abstract, cited_papers). The id-mapped
    gold `related_work` and the `gold_cited` pool ids are attached as labels.
    Pass `split` ("train" / "val" / "test") to keep only matching records.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Add data to {DATA_DIR} (see data/README.md)."
        )

    examples: list[dspy.Example] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            where = f"{path.name}:{lineno} ({record.get('example_id', 'no example_id')})"
            _validate_record(record, where)

            if split is not None and record.get("split") != split:
                continue

            gold = record.get("gold_related_work") or {}
            example = dspy.Example(
                example_id=record.get("example_id", f"{path.stem}-{lineno}"),
                split=record.get("split", ""),
                title=record["query"]["title"],
                abstract=record["query"]["abstract"],
                cited_papers=format_candidate_pool(record["candidate_pool"]),
                related_work=gold.get("id_mapped", ""),
                related_work_raw=gold.get("raw", ""),
                gold_cited=list(record.get("gold_cited", [])),
            ).with_inputs("title", "abstract", "cited_papers")
            examples.append(example)
    return examples
