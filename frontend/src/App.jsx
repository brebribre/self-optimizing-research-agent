import { useEffect, useState } from "react";
import { api } from "./api.js";
import TrainPanel from "./components/TrainPanel.jsx";
import TestPanel from "./components/TestPanel.jsx";

// Top-level page: loads the catalogs once, then renders the Train and Test
// panels side by side. Artifacts produced by training feed straight into Test,
// so they live in shared state and refresh when a job succeeds.
export default function App() {
  const [agents, setAgents] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [optimizers, setOptimizers] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [artifacts, setArtifacts] = useState([]);
  const [error, setError] = useState(null);

  const refreshArtifacts = () => api.artifacts().then(setArtifacts).catch(() => {});

  useEffect(() => {
    Promise.all([
      api.agents(),
      api.metrics(),
      api.optimizers(),
      api.datasets(),
      api.artifacts(),
    ])
      .then(([a, m, o, d, ar]) => {
        setAgents(a);
        setMetrics(m);
        setOptimizers(o);
        setDatasets(d);
        setArtifacts(ar);
      })
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Self-Optimizing Research Agent</h1>
        <p className="subtitle">
          Pick an agent and metric, optimize it on a dataset, then test the
          compiled program.
        </p>
      </header>

      {error && <div className="banner error">Failed to load: {error}</div>}

      <main className="grid">
        <TrainPanel
          agents={agents}
          metrics={metrics}
          optimizers={optimizers}
          datasets={datasets}
          onTrained={refreshArtifacts}
        />
        <TestPanel agents={agents} artifacts={artifacts} />
      </main>
    </div>
  );
}
