// Thin wrapper over the FastAPI backend. Uses same-origin relative URLs so it
// works both behind the Vite dev proxy and when served from frontend/dist.

async function request(path, options) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      // non-JSON error body
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  agents: () => request("/api/agents"),
  metrics: () => request("/api/metrics"),
  optimizers: () => request("/api/optimizers"),
  datasets: () => request("/api/datasets"),
  artifacts: () => request("/api/artifacts"),
  startTraining: (body) =>
    request("/api/train", { method: "POST", body: JSON.stringify(body) }),
  job: (id) => request(`/api/train/${id}`),
  run: (body) => request("/api/run", { method: "POST", body: JSON.stringify(body) }),
};
