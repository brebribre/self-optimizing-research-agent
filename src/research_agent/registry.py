"""Registries of pluggable agents and metrics.

The frontend/API let users pick an agent and a metric by id. Keeping those
choices in one place (rather than hard-coded in scripts) is what makes the
system "pluggable": registering a new `dspy.Module` here makes it selectable
everywhere — the optimizer script, the API, and the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import dspy

from research_agent.agent import RelatedWorkAgent
from research_agent.metrics import (
    citation_f1,
    citation_f1_faithfulness,
    citation_faithfulness,
)

# A DSPy metric: (example, prediction, trace) -> float | bool.
Metric = Callable[..., float]


@dataclass(frozen=True)
class InputField:
    """One input the agent expects, used to render the test form."""

    name: str
    label: str
    multiline: bool = False
    placeholder: str = ""


@dataclass(frozen=True)
class AgentSpec:
    id: str
    label: str
    description: str
    factory: Callable[[], dspy.Module]
    input_fields: list[InputField] = field(default_factory=list)


@dataclass(frozen=True)
class MetricSpec:
    id: str
    label: str
    description: str
    fn: Metric


@dataclass(frozen=True)
class OptimizerSpec:
    """A selectable DSPy optimizer.

    MIPROv2 and BootstrapFewShot share the optimizer interface (both take a
    `metric` and expose `.compile(student, trainset=...)`) but differ in their
    constructor args and whether `.compile()` accepts a validation set. Each
    spec hides those differences behind `build` + `run`.
    """

    id: str
    label: str
    description: str
    build: Callable[[Metric], object]
    run: Callable[[object, dspy.Module, list, list], dspy.Module]


# --- Agents -------------------------------------------------------------------

AGENTS: dict[str, AgentSpec] = {
    "related_work": AgentSpec(
        id="related_work",
        label="Related Work Agent",
        description="Writes a Related Work section from a title, abstract, and candidate citations.",
        factory=RelatedWorkAgent,
        input_fields=[
            InputField("title", "Title", placeholder="Title of the paper being written"),
            InputField("abstract", "Abstract", multiline=True, placeholder="Abstract of that paper"),
            InputField(
                "cited_papers",
                "Candidate pool",
                multiline=True,
                placeholder="One per line:\n[P01] Title: <title> | Abstract: <abstract>",
            ),
        ],
    ),
}


# --- Metrics ------------------------------------------------------------------

METRICS: dict[str, MetricSpec] = {
    "f1": MetricSpec(
        id="f1",
        label="Citation F1",
        description="F1 of cited works against the gold set (precision + recall of citations).",
        fn=citation_f1,
    ),
    "faithfulness": MetricSpec(
        id="faithfulness",
        label="Citation Faithfulness",
        description="Fraction of generated claims entailed by their cited sources (NLI-based).",
        fn=citation_faithfulness,
    ),
    "f1_faithfulness": MetricSpec(
        id="f1_faithfulness",
        label="F1 × Faithfulness",
        description="Product of citation F1 and faithfulness — rewards correct and faithful citing.",
        fn=citation_f1_faithfulness,
    ),
}


# --- Optimizers ---------------------------------------------------------------

OPTIMIZERS: dict[str, OptimizerSpec] = {
    "mipro": OptimizerSpec(
        id="mipro",
        label="MIPROv2",
        description="Bayesian search over instructions + few-shot demos; uses a validation set.",
        build=lambda metric: dspy.MIPROv2(metric=metric, auto="light"),
        # requires_permission_to_run=False: never block a background job on stdin.
        run=lambda opt, program, trainset, valset: opt.compile(
            program, trainset=trainset, valset=valset, requires_permission_to_run=False
        ),
    ),
    "bootstrap": OptimizerSpec(
        id="bootstrap",
        label="BootstrapFewShot",
        description="Bootstraps few-shot demonstrations from the trainset (no validation set).",
        build=lambda metric: dspy.BootstrapFewShot(
            metric=metric, max_bootstrapped_demos=4, max_labeled_demos=4
        ),
        # BootstrapFewShot.compile takes no valset.
        run=lambda opt, program, trainset, valset: opt.compile(program, trainset=trainset),
    ),
}


def metric_is_ready(spec: MetricSpec) -> bool:
    """Probe whether a metric is implemented (vs. a NotImplementedError stub).

    Calls the metric with trivial inputs; a `NotImplementedError` means it is
    still a header. Any other outcome counts as "ready" (it attempted to
    compute). This stays correct automatically as stubs get implemented.
    """
    example = dspy.Example(related_work="x").with_inputs()
    prediction = dspy.Prediction(related_work="x")
    try:
        spec.fn(example, prediction, None)
        return True
    except NotImplementedError:
        return False
    except Exception:
        return True
