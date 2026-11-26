"use client";

import { useEffect, useState } from "react";

type State =
  | { kind: "checking" }
  | { kind: "waking" }
  | { kind: "live"; version: string }
  | { kind: "down"; message: string };

/** Pings the API once on load. A free instance may be asleep; say so instead of looking broken. */
export function ApiStatus() {
  const [state, setState] = useState<State>({ kind: "checking" });

  useEffect(() => {
    let done = false;
    const slow = setTimeout(() => {
      if (!done) setState({ kind: "waking" });
    }, 3000);

    fetch("/api/health", { cache: "no-store" })
      .then(async (res) => {
        const body = await res.json().catch(() => null);
        if (res.ok && body?.status === "ok") {
          setState({ kind: "live", version: String(body.model_version) });
        } else {
          setState({ kind: "down", message: body?.detail ?? `HTTP ${res.status}` });
        }
      })
      .catch(() => setState({ kind: "down", message: "Could not reach the dashboard server" }))
      .finally(() => {
        done = true;
        clearTimeout(slow);
      });

    return () => clearTimeout(slow);
  }, []);

  const view = {
    checking: { color: "var(--baseline)", text: "Checking API…" },
    waking: { color: "var(--risk-medium)", text: "API waking up (up to 2 min)…" },
    live: { color: "var(--risk-low)", text: "API live" },
    down: { color: "var(--risk-high)", text: "API unavailable" },
  }[state.kind];

  return (
    <span
      className="inline-flex items-center gap-2 rounded-full border border-line px-3 py-1 text-xs text-ink-2"
      role="status"
      title={state.kind === "down" ? state.message : undefined}
    >
      <span
        aria-hidden="true"
        className={`size-2 rounded-full ${state.kind === "waking" ? "animate-pulse" : ""}`}
        style={{ background: view.color }}
      />
      {view.text}
      {state.kind === "live" && <span className="tabular">· model v{state.version}</span>}
    </span>
  );
}
