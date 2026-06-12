# Dataset

This directory holds the data used to train/optimize the DSPy related-work agent.

> **Status:** placeholder. The schema below is a starting point for the PoC and
> will be refined once the real dataset is provided.

## Layout

```
data/
├── raw/         # original, unmodified source data (gitignored if large)
├── processed/   # cleaned JSONL ready for load_examples()
├── train.jsonl  # training split (to be added)
└── dev.jsonl    # validation split (to be added)
```

## Record schema (JSONL, one object per line)

```json
{
  "id": "unique-id",
  "title": "Title of the paper being written",
  "abstract": "Abstract of that paper",
  "cited_papers": [
    {"title": "Cited paper title", "abstract": "Cited paper abstract"}
  ],
  "related_work": "Gold-reference Related Work section (target output)"
}
```

- `title`, `abstract`, `cited_papers` are the **inputs**.
- `related_work` is the **gold label** used for optimization/evaluation.
  It can be omitted for inference-only examples.

Load with `research_agent.data.load_examples("data/train.jsonl")`.
