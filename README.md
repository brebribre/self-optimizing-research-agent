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
| `src/research_agent/registry.py` | Pluggable agent + metric registries (what the UI lists). |
| `src/research_agent/server/` | FastAPI backend: train (background jobs) + run. |
| `frontend/` | React + Vite UI to train and test agents in the browser. |
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

# choose the metric and optimizer (defaults: --metric f1 --optimizer mipro)
uv run python scripts/optimize.py --train data/train.jsonl \
    --metric f1 --optimizer bootstrap
```

This runs a DSPy optimizer — **MIPROv2** (instruction + few-shot search, uses a
validation set) or **BootstrapFewShot** (few-shot demo bootstrapping only) — and
saves the compiled program to `artifacts/optimized_agent.json`.

## Web UI (train + test in the browser)

A React frontend talks to a FastAPI backend so you can pick an agent + metric,
launch an optimization run, watch its logs, and test the compiled program — all
in the browser.

```bash
# 1. install the backend (FastAPI/uvicorn) extra
uv sync --extra server

# 2. start the API (http://127.0.0.1:8000)
uv run research-agent-server
#   ...or with autoreload during development:
#   uv run uvicorn research_agent.server.app:app --reload
```

For development, run the Vite dev server (it proxies `/api` to the backend):

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

For a single-origin deployment, build the frontend once — the backend then
serves it at `http://127.0.0.1:8000/`:

```bash
cd frontend && npm run build
```

**What it does**

- **Agents**, **metrics**, and **optimizers** are read from
  `src/research_agent/registry.py`. Register a new `dspy.Module`, metric, or
  optimizer there and it shows up in the UI.
- **Train** starts a background job (the API does not block); metrics that are
  still `NotImplementedError` stubs are flagged, and a run against one will fail
  with that error surfaced in the job logs.
- **Test** runs the base agent or any compiled artifact from `artifacts/` on an
  example you type in.

> **Note:** the metrics (`citation_f1`, `citation_faithfulness`) are headers for
> now, so training will fail until at least one is implemented. See
> `src/research_agent/metrics.py`.

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
