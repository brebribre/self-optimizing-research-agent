"""Optimize the RelatedWorkAgent against a dataset (the 'self-optimizing' part).

This script is the bridge between the agent and the dataset: it loads training
examples, runs a DSPy optimizer to compile better prompts/demonstrations, and
saves the optimized program to artifacts/ for reuse.

Run once real training data exists under data/:

    uv run python scripts/optimize.py --train data/train.jsonl --dev data/dev.jsonl

The optimizer (MIPROv2 or BootstrapFewShot) and budget are deliberately
conservative for a PoC; tune as the dataset grows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from research_agent.agent import RelatedWorkAgent
from research_agent.config import configure_lm
from research_agent.data import load_examples
from research_agent.registry import METRICS, OPTIMIZERS

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize the related-work agent.")
    parser.add_argument("--train", required=True, help="Path to training JSONL.")
    parser.add_argument("--dev", help="Path to dev/validation JSONL (optional).")
    parser.add_argument("--model", help="Override the LM id.")
    parser.add_argument(
        "--metric",
        choices=list(METRICS),
        default="f1",
        help="Which metric to optimize against (the experiment to run).",
    )
    parser.add_argument(
        "--optimizer",
        choices=list(OPTIMIZERS),
        default="mipro",
        help="Which DSPy optimizer to use.",
    )
    parser.add_argument(
        "--out",
        default=str(ARTIFACTS_DIR / "optimized_agent.json"),
        help="Where to save the compiled program.",
    )
    args = parser.parse_args()

    configure_lm(model=args.model)

    trainset = load_examples(args.train)
    devset = load_examples(args.dev) if args.dev else trainset

    print(f"Loaded {len(trainset)} train / {len(devset)} dev examples.")

    optimizer_spec = OPTIMIZERS[args.optimizer]
    print(f"Optimizing with metric '{args.metric}' via {optimizer_spec.label}.")
    optimizer = optimizer_spec.build(METRICS[args.metric].fn)
    optimized = optimizer_spec.run(optimizer, RelatedWorkAgent(), trainset, devset)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(out_path))
    print(f"Saved optimized agent to {out_path}")


if __name__ == "__main__":
    main()
