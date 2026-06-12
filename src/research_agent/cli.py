"""Command-line entry point: run the agent on a single example.

Usage:
    uv run python -m research_agent.cli --demo
    uv run python -m research_agent.cli --input data/example.json

The input JSON is one dataset record (see data/README.md): a `query` with
title/abstract and a `candidate_pool` of works the agent may cite by pool_id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_agent.agent import RelatedWorkAgent
from research_agent.config import configure_lm
from research_agent.data import format_candidate_pool

_DEMO = {
    "query": {
        "title": "Self-Correcting Tool Use in LLM Agents via Execution Feedback",
        "abstract": (
            "Large language model agents frequently fail multi-step tasks because "
            "erroneous tool calls go undetected. We present a framework in which agents "
            "inspect execution feedback after every tool call and revise their plan in "
            "context, without additional training."
        ),
    },
    "candidate_pool": [
        {
            "pool_id": "P01",
            "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
            "abstract": "Interleaves reasoning traces with task-specific actions in interactive environments.",
        },
        {
            "pool_id": "P02",
            "title": "Reflexion: Language Agents with Verbal Reinforcement Learning",
            "abstract": "Agents verbally reflect on task feedback and store reflections to improve later trials.",
        },
        {
            "pool_id": "P03",
            "title": "Generative Agents: Interactive Simulacra of Human Behavior",
            "abstract": "Agents simulate believable human behavior by storing and retrieving memories.",
        },
        {
            "pool_id": "P04",
            "title": "Teaching Large Language Models to Self-Debug",
            "abstract": "Models debug their predicted programs by explaining code and using execution results.",
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
    parser.add_argument("--input", help="Path to a JSON file with one dataset record.")
    parser.add_argument("--demo", action="store_true", help="Run the built-in demo example.")
    parser.add_argument("--model", help="Override the LM id (e.g. anthropic/claude-opus-4-8).")
    args = parser.parse_args()

    if not args.input and not args.demo:
        parser.error("Provide --input <file> or --demo.")

    configure_lm(model=args.model)
    agent = RelatedWorkAgent()

    record = _load_input(args.input)
    prediction = agent(
        title=record["query"]["title"],
        abstract=record["query"]["abstract"],
        cited_papers=format_candidate_pool(record.get("candidate_pool", [])),
    )

    print("\n=== Generated Related Work ===\n")
    print(prediction.related_work)


if __name__ == "__main__":
    main()
