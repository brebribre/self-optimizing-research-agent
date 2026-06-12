"""Central configuration for the DSPy language model.

Reads provider/model settings from the environment (loaded from a local
`.env` if present) and wires them into DSPy's global settings. Keeping this
in one place means the agent, the optimizer, and any scripts all share the
same LM configuration.
"""

from __future__ import annotations

import os

import dspy
from dotenv import load_dotenv

# Sensible defaults; override via environment / .env.
DEFAULT_LM = "anthropic/claude-sonnet-4-6"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096


def build_lm(
    model: str | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dspy.LM:
    """Construct a `dspy.LM` without touching DSPy's global settings.

    Arguments take precedence over environment variables, which take
    precedence over the built-in defaults. Use the returned LM with
    `dspy.context(lm=...)` in multi-threaded code (e.g. the server) —
    `dspy.configure` is global and may only be called from the thread that
    first configured it.
    """
    load_dotenv()

    model = model or os.getenv("RESEARCH_AGENT_LM", DEFAULT_LM)
    temperature = (
        temperature
        if temperature is not None
        else float(os.getenv("RESEARCH_AGENT_TEMPERATURE", DEFAULT_TEMPERATURE))
    )
    max_tokens = (
        max_tokens
        if max_tokens is not None
        else int(os.getenv("RESEARCH_AGENT_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    )

    if model.startswith("anthropic/") and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or export it in your shell."
        )

    return dspy.LM(model, temperature=temperature, max_tokens=max_tokens)


def configure_lm(
    model: str | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dspy.LM:
    """Configure and register the global DSPy LM (single-threaded entry points).

    Suitable for the CLI and scripts, which run on the main thread. Server
    code must use `build_lm` + `dspy.context` instead.
    """
    lm = build_lm(model, temperature=temperature, max_tokens=max_tokens)
    dspy.configure(lm=lm)
    return lm
