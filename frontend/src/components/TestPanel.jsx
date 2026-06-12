import { useEffect, useMemo, useState } from "react";
import { api } from "../api.js";

// Runs an agent on a single user-provided example. The user can run the base
// (untrained) agent or load one of the compiled artifacts produced by training.
export default function TestPanel({ agents, artifacts }) {
  const [agent, setAgent] = useState("");
  const [artifact, setArtifact] = useState(""); // "" = base/untrained
  const [model, setModel] = useState("");
  const [inputs, setInputs] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (agents.length && !agent) setAgent(agents[0].id);
  }, [agents]);

  const spec = useMemo(() => agents.find((a) => a.id === agent), [agents, agent]);

  // Reset the input fields whenever the selected agent changes.
  useEffect(() => {
    if (!spec) return;
    const blank = {};
    spec.input_fields.forEach((f) => {
      blank[f.name] = "";
    });
    setInputs(blank);
    setResult(null);
  }, [spec]);

  const run = async () => {
    setError(null);
    setResult(null);
    setBusy(true);
    try {
      const res = await api.run({
        agent,
        inputs,
        artifact: artifact || null,
        model: model.trim() || null,
      });
      setResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel">
      <h2>2 · Test</h2>

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
        <span>Program</span>
        <select value={artifact} onChange={(e) => setArtifact(e.target.value)}>
          <option value="">Base (untrained)</option>
          {artifacts.map((a) => (
            <option key={a.path} value={a.path}>
              {a.name}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span>Model (optional)</span>
        <input
          type="text"
          value={model}
          placeholder="blank = default"
          onChange={(e) => setModel(e.target.value)}
        />
      </label>

      {spec?.input_fields.map((f) => (
        <label className="field" key={f.name}>
          <span>{f.label}</span>
          {f.multiline ? (
            <textarea
              rows={4}
              value={inputs[f.name] ?? ""}
              placeholder={f.placeholder}
              onChange={(e) => setInputs({ ...inputs, [f.name]: e.target.value })}
            />
          ) : (
            <input
              type="text"
              value={inputs[f.name] ?? ""}
              placeholder={f.placeholder}
              onChange={(e) => setInputs({ ...inputs, [f.name]: e.target.value })}
            />
          )}
        </label>
      ))}

      <button className="primary" onClick={run} disabled={busy || !agent}>
        {busy ? "Running…" : "Run agent"}
      </button>

      {error && <div className="banner error">{error}</div>}

      {result && (
        <div className="result">
          {result.reasoning && (
            <details>
              <summary>Reasoning</summary>
              <pre className="logs">{result.reasoning}</pre>
            </details>
          )}
          <h3>Related Work</h3>
          <div className="output">{result.related_work}</div>
        </div>
      )}
    </section>
  );
}
