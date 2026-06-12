# Dataset

Related-work generation dataset: each row pairs a **query paper** with a
30-paper **candidate pool** (gold citations + distractors) and the **answer
key** of what the real authors cited. `data/sample.jsonl` holds one example
row in this schema.

## Design

- **Subfield:** LLM-based agents (cs.CL / cs.AI). High recent volume, and
  same-subfield distractors keep the task from being trivially solvable by
  topical mismatch alone.
- **Cutoff:** query papers submitted after **September 2025**, so they postdate
  LLM training data. ~9 months of papers ≈ 90 rows.
- **Pool size:** 30 candidates per row. If a paper cites k works (5 ≤ k ≤ 15),
  the remaining 30 − k are distractors, so at least half the pool is always
  distractors.

## Record schema (JSONL, one object per line)

```json
{
  "example_id": "row-1",
  "split": "train",
  "query": {
    "arxiv_id": "2602.01234",
    "title": "Self-Correcting Tool Use in LLM Agents via Execution Feedback",
    "abstract": "Large language model agents frequently fail multi-step tasks...",
    "submitted": "2026-02-03"
  },
  "candidate_pool": [
    {
      "pool_id": "P01",
      "arxiv_id": "2210.03629",
      "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
      "abstract": "While large language models..."
    }
  ],
  "gold_cited": ["P01", "P02", "P09", "P14", "P22", "P25", "P28"],
  "gold_related_work": {
    "raw": "Early agent frameworks interleave reasoning and acting \\cite{yao2022react}...",
    "id_mapped": "Early agent frameworks interleave reasoning and acting [P01]..."
  }
}
```

- `query` + `candidate_pool` are the **inputs**; the pool is rendered for the
  agent as one line per paper: `[P01] Title: <title> | Abstract: <abstract>`.
- `gold_cited` is the **answer key**: the pool_ids the real authors cited.
- `gold_related_work.id_mapped` is the gold section with `\cite{...}` rewritten
  to `[pool_id]` markers; `raw` preserves the original LaTeX form.
- `gold_cited` / `gold_related_work` may be omitted for inference-only rows.

## Inclusion requirements (query papers)

- Explicit Related Work section and an abstract.
- **5–15 citations** in the related-work section: <5 is trivial, >15 tends to
  be a survey and would leave fewer than half the pool as distractors.
- Written in English.
- **LaTeX source available** so `\cite` commands can be parsed reliably
  (PDF citation extraction is too error-prone).

## Distractor selection

Distractors are drawn from the union of all *other* rows' gold-cited papers
(same subfield, already satisfy the inclusion pipeline, keeps the corpus
manageable). For each query, pick the pooled gold papers **most similar to the
query's abstract by SPECTER2 embeddings**, excluding the query's own cited
papers and anything published after the query's submission date.

## Splits

40 / 15 / 40 — train / val / test (experimentation / validation / final
evaluation). Stored per-row in `split`; filter at load time:

```python
from research_agent.data import load_examples
train = load_examples("data/dataset.jsonl", split="train")
```

## Layout

```
data/
├── raw/           # original, unmodified source data (gitignored if large)
├── processed/     # cleaned JSONL ready for load_examples()
└── sample.jsonl   # one example row in the schema above
```
