"""Command-line entry point: run the agent on a single example.

Usage:
    uv run python -m research_agent.cli --demo
    uv run python -m research_agent.cli --input data/example.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_agent.agent import RelatedWorkAgent
from research_agent.config import configure_lm
from research_agent.data import format_cited_papers

_DEMO = {
    "title": "Efficient Retrieval-Augmented Generation for Scientific Summarization",
    "abstract": (
        "We present a retrieval-augmented approach that improves the factual "
        "grounding of scientific summaries while reducing inference cost."
    ),
    "cited_papers": [
        {
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "abstract": "Combines parametric and non-parametric memory for generation.",
        },
        {
            "title": "Dense Passage Retrieval for Open-Domain Question Answering",
            "abstract": "Learns dense representations for efficient passage retrieval.",
        },
    ],
}


def _load_input(path: str | None) -> dict:
    if path is None:
        return _DEMO
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Related Work section.")
    parser.add_argument("--input", help="Path to a JSON file with one example.")
    parser.add_argument("--demo", action="store_true", help="Run the built-in demo example.")
    parser.add_argument("--model", help="Override the LM id (e.g. anthropic/claude-opus-4-8).")
    args = parser.parse_args()

    if not args.input and not args.demo:
        parser.error("Provide --input <file> or --demo.")

    configure_lm(model=args.model)
    agent = RelatedWorkAgent()

    record = _load_input(args.input)
    prediction = agent(
        title=record["title"],
        abstract=record["abstract"],
        cited_papers=format_cited_papers(record.get("cited_papers", [])),
    )

    print("\n=== Generated Related Work ===\n")
    print(prediction.related_work)


if __name__ == "__main__":
    main()
