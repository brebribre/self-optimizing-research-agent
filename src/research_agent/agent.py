"""The related-work generation agent.

This is intentionally simple for the PoC: a single `dspy.ChainOfThought`
module wrapping the `GenerateRelatedWork` signature. Because it's a
`dspy.Module`, it can be handed directly to a DSPy optimizer
(e.g. MIPROv2, BootstrapFewShot) to become self-optimizing — that is the
whole point of building on DSPy rather than hand-writing prompts.
"""

from __future__ import annotations

import dspy

from research_agent.signatures import GenerateRelatedWork


class RelatedWorkAgent(dspy.Module):
    """Generate a Related Work section from a paper + candidate citations."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateRelatedWork)

    def forward(
        self,
        title: str,
        abstract: str,
        cited_papers: str,
    ) -> dspy.Prediction:
        return self.generate(
            title=title,
            abstract=abstract,
            cited_papers=cited_papers,
        )
