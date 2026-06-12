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


def configure_lm(
    model: str | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dspy.LM:
    """Configure and register the global DSPy LM.

    Arguments take precedence over environment variables, which take
    precedence over the built-in defaults. Returns the constructed `dspy.LM`
    so callers can inspect or reuse it.
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

    lm = dspy.LM(model, temperature=temperature, max_tokens=max_tokens)
    dspy.configure(lm=lm)
    return lm
