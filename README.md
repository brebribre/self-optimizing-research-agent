# self-optimizing-research-agent

A proof-of-concept **DSPy** framework for a *self-optimizing research agent* that
performs **related-work generation** — given a paper's title, abstract, and a set
of candidate citations, it writes a coherent Related Work section. Because the
agent is a DSPy program, its prompts and few-shot demonstrations can be
automatically optimized against a dataset rather than hand-tuned.

## What's here

| Path | Purpose |
|------|---------|
| `src/research_agent/agent.py` | The agent — a `dspy.Module` (ChainOfThought). |
| `src/research_agent/signatures.py` | The task I/O contract DSPy optimizes. |
| `src/research_agent/config.py` | LM configuration (defaults to Claude). |
| `src/research_agent/data.py` | Loads JSONL datasets into `dspy.Example`s. |
| `src/research_agent/metrics.py` | Evaluation metric for the optimizer. |
| `src/research_agent/cli.py` | Run the agent on one example. |
| `scripts/optimize.py` | Compile/optimize the agent against a dataset. |
| `data/` | Datasets for training/optimization (see `data/README.md`). |
| `artifacts/` | Saved optimized programs. |

## Setup

This project uses [uv](https://docs.astral.sh/uv/).

```bash
# install dependencies into a managed virtualenv
uv sync

# configure your key
cp .env.example .env   # then edit .env and add ANTHROPIC_API_KEY
```

> **Note (this machine):** the user profile (`AppData`) is on a different volume
> than the project, which breaks uv's default cross-volume file moves. The repo
> pins uv's cache to `.uvcache/` (see `uv.toml`) so `uv run`/`uv sync` work as-is.
> If a fresh `uv sync` ever fails with `os error 17`, point uv's temp dir at the
> project volume too:
> ```powershell
> $env:TEMP = "$PWD\.uvtmp"; $env:TMP = $env:TEMP; uv sync --extra dev
> ```

## Run the agent

```bash
# built-in demo example
uv run python -m research_agent.cli --demo

# your own example (JSON with title/abstract/cited_papers)
uv run python -m research_agent.cli --input path/to/example.json
```

## Optimize the agent (self-optimization)

Once you have training data under `data/` (see `data/README.md` for the schema):

```bash
uv run python scripts/optimize.py --train data/train.jsonl --dev data/dev.jsonl
```

This runs a DSPy optimizer (MIPROv2) and saves the compiled program to
`artifacts/optimized_agent.json`.

## Tests

```bash
uv run pytest        # data/loader smoke tests run without an API key
```

## Configuration

All LM settings come from the environment (or `.env`):

| Variable | Default | Notes |
|----------|---------|-------|
| `ANTHROPIC_API_KEY` | — | required for the default provider |
| `RESEARCH_AGENT_LM` | `anthropic/claude-sonnet-4-6` | any LiteLLM model id |
| `RESEARCH_AGENT_TEMPERATURE` | `0.7` | |
| `RESEARCH_AGENT_MAX_TOKENS` | `4096` | |

Swap providers by changing `RESEARCH_AGENT_LM` (e.g. `openai/gpt-4o`) and
setting the matching API key.
