"""In-process background training jobs.

Optimization is slow and costs LM calls, so the API cannot run it inside a
request. This module runs each training run on a daemon thread and exposes its
status + streamed logs for polling. It is deliberately simple (single process,
in-memory state) — fine for a local PoC, not for multi-worker production.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import dspy

from research_agent.config import build_lm
from research_agent.data import load_examples
from research_agent.registry import AGENTS, METRICS, OPTIMIZERS

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


@dataclass
class TrainingJob:
    id: str
    agent: str
    metric: str
    optimizer: str
    dataset: str
    model: str | None = None
    status: str = "queued"  # queued | running | succeeded | failed
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    artifact_path: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent": self.agent,
            "metric": self.metric,
            "optimizer": self.optimizer,
            "dataset": self.dataset,
            "model": self.model,
            "status": self.status,
            "logs": self.logs,
            "error": self.error,
            "artifact_path": self.artifact_path,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class _LogWriter:
    """File-like object that captures optimizer stdout into a job's log list."""

    def __init__(self, job: TrainingJob) -> None:
        self._job = job
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._job.logs.append(line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self._buf.strip():
            self._job.logs.append(self._buf.rstrip())
            self._buf = ""


class JobManager:
    """Thread-safe registry of training jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    def list(self) -> list[TrainingJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get(self, job_id: str) -> TrainingJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def start(
        self, agent: str, metric: str, optimizer: str, dataset: str, model: str | None
    ) -> TrainingJob:
        if agent not in AGENTS:
            raise ValueError(f"Unknown agent: {agent}")
        if metric not in METRICS:
            raise ValueError(f"Unknown metric: {metric}")
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        dataset_path = self._resolve_dataset(dataset)

        job = TrainingJob(
            id=uuid.uuid4().hex[:12],
            agent=agent,
            metric=metric,
            optimizer=optimizer,
            dataset=dataset,
            model=model,
        )
        with self._lock:
            self._jobs[job.id] = job

        thread = threading.Thread(target=self._run, args=(job, dataset_path), daemon=True)
        thread.start()
        return job

    @staticmethod
    def _resolve_dataset(dataset: str) -> Path:
        # Confine dataset selection to files under data/ (no path traversal).
        path = (DATA_DIR / dataset).resolve()
        if DATA_DIR.resolve() not in path.parents and path.parent != DATA_DIR.resolve():
            raise ValueError(f"Dataset must live under {DATA_DIR}")
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset}")
        return path

    def _run(self, job: TrainingJob, dataset_path: Path) -> None:
        import contextlib

        job.status = "running"
        job.started_at = time.time()
        writer = _LogWriter(job)
        try:
            job.logs.append(f"Configuring LM ({job.model or 'default'})...")
            # Thread-local LM context: dspy.configure is global and rejects
            # calls from any thread other than the one that configured first.
            lm = build_lm(model=job.model)

            agent = AGENTS[job.agent].factory()
            metric = METRICS[job.metric].fn
            optimizer_spec = OPTIMIZERS[job.optimizer]

            examples = load_examples(dataset_path)
            job.logs.append(f"Loaded {len(examples)} examples from {job.dataset}.")
            # No explicit dev split in the PoC: reuse the trainset for validation.
            trainset = devset = examples

            job.logs.append(
                f"Optimizing with metric '{job.metric}' via {optimizer_spec.label}..."
            )
            optimizer = optimizer_spec.build(metric)
            with (
                dspy.context(lm=lm),
                contextlib.redirect_stdout(writer),
                contextlib.redirect_stderr(writer),
            ):
                compiled = optimizer_spec.run(optimizer, agent, trainset, devset)
            writer.flush()

            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = ARTIFACTS_DIR / f"{job.id}.json"
            compiled.save(str(out_path))
            job.artifact_path = str(out_path.relative_to(REPO_ROOT))
            job.logs.append(f"Saved compiled program to {job.artifact_path}.")
            job.status = "succeeded"
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
            writer.flush()
            job.error = f"{type(exc).__name__}: {exc}"
            job.logs.append(job.error)
            job.logs.extend(traceback.format_exc().splitlines())
            job.status = "failed"
        finally:
            job.finished_at = time.time()


# Module-level singleton used by the API.
job_manager = JobManager()
