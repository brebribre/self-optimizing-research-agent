"""FastAPI app exposing the agent registry, training, and inference.

Endpoints (all under /api):

    GET  /api/agents              list selectable agents + their input fields
    GET  /api/metrics             list metrics (with a `ready` flag)
    GET  /api/optimizers          list selectable DSPy optimizers
    GET  /api/datasets            list JSONL datasets under data/
    GET  /api/artifacts           list compiled programs under artifacts/
    POST /api/train               start a training job -> {job_id}
    GET  /api/train               list training jobs
    GET  /api/train/{job_id}      poll a training job's status + logs
    POST /api/run                 run an agent (optionally a trained artifact)

Run it with:

    uv run uvicorn research_agent.server.app:app --reload

The built React frontend (frontend/dist) is served at / when present.
"""

from __future__ import annotations

import dspy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from research_agent.config import build_lm
from research_agent.registry import AGENTS, METRICS, OPTIMIZERS, metric_is_ready
from research_agent.server.jobs import ARTIFACTS_DIR, DATA_DIR, REPO_ROOT, job_manager

app = FastAPI(title="Self-Optimizing Research Agent")

# Dev convenience: the Vite dev server (localhost:5173) calls this API directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/response models --------------------------------------------------


class TrainRequest(BaseModel):
    agent: str
    metric: str
    optimizer: str = "mipro"
    dataset: str
    model: str | None = None


class RunRequest(BaseModel):
    agent: str
    inputs: dict[str, str]
    artifact: str | None = None  # path (relative to repo root) of a compiled program
    model: str | None = None


# --- Catalog endpoints --------------------------------------------------------


@app.get("/api/agents")
def list_agents() -> list[dict]:
    return [
        {
            "id": spec.id,
            "label": spec.label,
            "description": spec.description,
            "input_fields": [
                {
                    "name": f.name,
                    "label": f.label,
                    "multiline": f.multiline,
                    "placeholder": f.placeholder,
                }
                for f in spec.input_fields
            ],
        }
        for spec in AGENTS.values()
    ]


@app.get("/api/metrics")
def list_metrics() -> list[dict]:
    return [
        {
            "id": spec.id,
            "label": spec.label,
            "description": spec.description,
            "ready": metric_is_ready(spec),
        }
        for spec in METRICS.values()
    ]


@app.get("/api/optimizers")
def list_optimizers() -> list[dict]:
    return [
        {"id": spec.id, "label": spec.label, "description": spec.description}
        for spec in OPTIMIZERS.values()
    ]


@app.get("/api/datasets")
def list_datasets() -> list[dict]:
    out: list[dict] = []
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        try:
            count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        except OSError:
            count = 0
        out.append({"name": path.name, "examples": count})
    return out


@app.get("/api/artifacts")
def list_artifacts() -> list[dict]:
    out: list[dict] = []
    for path in sorted(ARTIFACTS_DIR.glob("*.json")):
        out.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "name": path.name,
                "modified": path.stat().st_mtime,
            }
        )
    return out


# --- Training -----------------------------------------------------------------


@app.post("/api/train")
def start_training(req: TrainRequest) -> dict:
    try:
        job = job_manager.start(req.agent, req.metric, req.optimizer, req.dataset, req.model)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job.id}


@app.get("/api/train")
def list_jobs() -> list[dict]:
    return [job.to_dict() for job in job_manager.list()]


@app.get("/api/train/{job_id}")
def get_job(job_id: str) -> dict:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


# --- Inference ----------------------------------------------------------------


@app.post("/api/run")
def run_agent(req: RunRequest) -> dict:
    if req.agent not in AGENTS:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {req.agent}")
    spec = AGENTS[req.agent]

    try:
        # Thread-local LM context: FastAPI runs sync endpoints in a threadpool,
        # so the global dspy.configure would fail on cross-thread calls.
        lm = build_lm(model=req.model)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    agent = spec.factory()
    if req.artifact:
        artifact_path = (REPO_ROOT / req.artifact).resolve()
        if ARTIFACTS_DIR.resolve() not in artifact_path.parents:
            raise HTTPException(status_code=400, detail="Artifact must live under artifacts/")
        if not artifact_path.exists():
            raise HTTPException(status_code=404, detail="Artifact not found")
        agent.load(str(artifact_path))

    # Only pass the fields this agent declares.
    field_names = {f.name for f in spec.input_fields}
    kwargs = {k: v for k, v in req.inputs.items() if k in field_names}
    missing = field_names - kwargs.keys()
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing inputs: {sorted(missing)}")

    try:
        with dspy.context(lm=lm):
            prediction: dspy.Prediction = agent(**kwargs)
    except Exception as exc:  # noqa: BLE001 — report LM/agent errors to the UI
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    result = {"related_work": getattr(prediction, "related_work", "")}
    if hasattr(prediction, "reasoning"):
        result["reasoning"] = prediction.reasoning
    return result


# --- Serve the built frontend (if present) ------------------------------------

_DIST = REPO_ROOT / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")


def run() -> None:
    """Console-script entry point: launch uvicorn."""
    import uvicorn

    uvicorn.run("research_agent.server.app:app", host="127.0.0.1", port=8000, reload=False)
