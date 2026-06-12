import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";

// Lets the user choose agent + metric + dataset and launch an optimization job,
// then polls the job and streams its logs until it finishes.
export default function TrainPanel({ agents, metrics, optimizers, datasets, onTrained }) {
  const [agent, setAgent] = useState("");
  const [metric, setMetric] = useState("");
  const [optimizer, setOptimizer] = useState("");
  const [dataset, setDataset] = useState("");
  const [model, setModel] = useState("");
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  // Default the selects to the first available option once catalogs load.
  useEffect(() => {
    if (agents.length && !agent) setAgent(agents[0].id);
  }, [agents]);
  useEffect(() => {
    if (metrics.length && !metric) setMetric(metrics[0].id);
  }, [metrics]);
  useEffect(() => {
    if (optimizers.length && !optimizer) setOptimizer(optimizers[0].id);
  }, [optimizers]);
  useEffect(() => {
    if (datasets.length && !dataset) setDataset(datasets[0].name);
  }, [datasets]);

  useEffect(() => () => clearInterval(pollRef.current), []);

  const selectedMetric = metrics.find((m) => m.id === metric);
  const selectedOptimizer = optimizers.find((o) => o.id === optimizer);
  const running = job && (job.status === "queued" || job.status === "running");

  const poll = (id) => {
    pollRef.current = setInterval(async () => {
      try {
        const j = await api.job(id);
        setJob(j);
        if (j.status === "succeeded" || j.status === "failed") {
          clearInterval(pollRef.current);
          if (j.status === "succeeded") onTrained?.();
        }
      } catch (e) {
        setError(e.message);
        clearInterval(pollRef.current);
      }
    }, 1000);
  };

  const start = async () => {
    setError(null);
    setJob(null);
    try {
      const { job_id } = await api.startTraining({
        agent,
        metric,
        optimizer,
        dataset,
        model: model.trim() || null,
      });
      setJob({ id: job_id, status: "queued", logs: [] });
      poll(job_id);
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <section className="panel">
      <h2>1 · Train</h2>

      <label className="field">
        <span>Agent</span>
        <select value={agent} onChange={(e) => setAgent(e.target.value)}>
          {agents.map((a) => (
            <option key={a.id} value={a.id}>
              {a.label}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>Metric</span>
        <select value={metric} onChange={(e) => setMetric(e.target.value)}>
          {metrics.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
              {m.ready ? "" : " (not implemented)"}
            </option>
          ))}
        </select>
      </label>
      {selectedMetric && (
        <p className="hint">
          {selectedMetric.description}
          {!selectedMetric.ready && (
            <span className="warn">
              {" "}
              ⚠ This metric is still a stub — training will fail until it is
              implemented.
            </span>
          )}
        </p>
      )}

      <label className="field">
        <span>Optimizer</span>
        <select value={optimizer} onChange={(e) => setOptimizer(e.target.value)}>
          {optimizers.map((o) => (
            <option key={o.id} value={o.id}>
              {o.label}
            </option>
          ))}
        </select>
      </label>
      {selectedOptimizer && <p className="hint">{selectedOptimizer.description}</p>}

      <label className="field">
        <span>Dataset</span>
        <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
          {datasets.map((d) => (
            <option key={d.name} value={d.name}>
              {d.name} ({d.examples} ex)
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>Model (optional)</span>
        <input
          type="text"
          value={model}
          placeholder="e.g. anthropic/claude-opus-4-8 — blank = default"
          onChange={(e) => setModel(e.target.value)}
        />
      </label>

      <button
        className="primary"
        onClick={start}
        disabled={running || !agent || !metric || !optimizer || !dataset}
      >
        {running ? "Training…" : "Start training"}
      </button>

      {error && <div className="banner error">{error}</div>}

      {job && (
        <div className="job">
          <div className={`status status-${job.status}`}>
            Status: <strong>{job.status}</strong>
            {job.artifact_path && (
              <span className="artifact"> → {job.artifact_path}</span>
            )}
          </div>
          {job.logs?.length > 0 && (
            <pre className="logs">{job.logs.join("\n")}</pre>
          )}
        </div>
      )}
    </section>
  );
}
