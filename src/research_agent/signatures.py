"""DSPy signatures describing the related-work generation task.

Signatures are the declarative I/O contracts DSPy optimizes against. Keeping
them separate from the agent module makes them easy to reuse in evaluation
and optimization scripts.
"""

from __future__ import annotations

import dspy


class GenerateRelatedWork(dspy.Signature):
    """Write the Related Work section of a paper.

    Given the target paper's title and abstract plus a set of candidate
    cited papers, synthesize a coherent Related Work section that groups the
    cited works thematically, contrasts them with the target paper, and
    situates the target paper's contribution.
    """

    title: str = dspy.InputField(desc="Title of the paper being written.")
    abstract: str = dspy.InputField(desc="Abstract of the paper being written.")
    cited_papers: str = dspy.InputField(
        desc="Candidate related papers, one per line as 'Title: <title> | Abstract: <abstract>'."
    )
    related_work: str = dspy.OutputField(
        desc="A well-organized Related Work section in prose, citing the relevant works by title."
    )
