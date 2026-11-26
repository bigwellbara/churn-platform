"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { customerSchema, features, normalise, type Customer, type FeatureName } from "@/lib/customer";
import { FIELD_INFO, PRESETS } from "@/lib/fields";
import { applyLever, leversFor } from "@/lib/levers";
import type { Contribution, ExplanationResponse } from "@/lib/explain";
import { describeError, type BatchResponse, type Prediction } from "@/lib/predictions";
import { CustomerForm } from "./CustomerForm";
import { ExplanationChart } from "./ExplanationChart";
import { LeverChart, type LeverResult } from "./LeverChart";
import { RiskResult } from "./RiskResult";

interface Result {
  prediction: Prediction;
  threshold: number;
  levers: LeverResult[];
  /** Null when the API cannot explain (older API version, or a non-tree model). */
  contributions: Contribution[] | null;
}

type Status = "idle" | "loading" | "slow" | "error";

const DEBOUNCE_MS = 500;
const SLOW_MS = 3000;

export function Scorer() {
  const [customer, setCustomer] = useState<Customer>(PRESETS[0]!.customer);
  const [presetId, setPresetId] = useState<string | null>(PRESETS[0]!.id);
  const [autoTotal, setAutoTotal] = useState(true);
  const [result, setResult] = useState<Result | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const inFlight = useRef<AbortController | null>(null);

  const score = useCallback(async (c: Customer) => {
    const parsed = customerSchema.safeParse(c);
    if (!parsed.success) {
      const issue = parsed.error.issues[0]!;
      const field = issue.path[0] as FeatureName | undefined;
      setError(`${field ? FIELD_INFO[field]?.label ?? field : "Input"}: ${issue.message}`);
      setStatus("error");
      return;
    }

    // Only the newest request matters; cancel the one before it.
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    setStatus("loading");
    const slow = setTimeout(() => setStatus((s) => (s === "loading" ? "slow" : s)), SLOW_MS);

    // One request scores the customer and every retention offer together; a second, in
    // parallel, explains the customer's own score.
    const levers = leversFor(c);
    const customers = [c, ...levers.map((l) => applyLever(c, l))].map(features);
    const post = (path: string, payload: unknown) =>
      fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
    const explained = post("/api/predict/explain", features(c))
      .then(async (res) => (res.ok ? ((await res.json()) as ExplanationResponse) : null))
      .catch(() => null);
    try {
      const res = await post("/api/predict/batch", { customers });
      const body = await res.json().catch(() => null);
      if (!res.ok) throw new Error(describeError(body, res.status));
      const batch = body as BatchResponse;
      const [current, ...rest] = batch.predictions;
      // The explanation is a bonus: if it fails, the score still shows.
      const explanation = await explained;
      if (controller.signal.aborted) return;
      setResult({
        prediction: current!,
        threshold: batch.threshold,
        levers: levers.map((l, i) => ({
          id: l.id,
          label: l.label,
          probability: rest[i]!.churn_probability,
        })),
        contributions: explanation?.contributions ?? null,
      });
      setError(null);
      setStatus("idle");
    } catch (err) {
      if (controller.signal.aborted) return;
      setError(err instanceof Error ? err.message : "Something went wrong");
      setStatus("error");
    } finally {
      clearTimeout(slow);
    }
  }, []);

  // Re-score shortly after every edit, so the form works as a live what-if tool.
  useEffect(() => {
    const t = setTimeout(() => void score(customer), DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [customer, score]);

  useEffect(() => () => inFlight.current?.abort(), []);

  function update(field: FeatureName, value: string | number) {
    setPresetId(null);
    if (field === "TotalCharges") setAutoTotal(false);
    setCustomer((prev) => {
      const next = normalise({ ...prev, [field]: value } as Customer);
      const recalc = autoTotal && field !== "TotalCharges" && (field === "tenure" || field === "MonthlyCharges");
      if (recalc && Number.isFinite(next.tenure) && Number.isFinite(next.MonthlyCharges)) {
        next.TotalCharges = Math.round(next.tenure * next.MonthlyCharges * 100) / 100;
      }
      return next;
    });
  }

  function choosePreset(id: string) {
    const preset = PRESETS.find((p) => p.id === id)!;
    setPresetId(id);
    setAutoTotal(true);
    setCustomer(preset.customer);
  }

  const busy = status === "loading" || status === "slow";

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
      <section className="rounded-xl border border-line bg-surface p-5" aria-labelledby="customer-heading">
        <h2 id="customer-heading" className="font-semibold">
          Customer
        </h2>
        <p className="mt-1 text-sm text-ink-2">Start from an example, then change anything.</p>
        <div className="mt-3 flex flex-wrap gap-2" role="group" aria-label="Example customers">
          {PRESETS.map((p) => (
            <button
              key={p.id}
              type="button"
              title={p.description}
              aria-pressed={presetId === p.id}
              onClick={() => choosePreset(p.id)}
              className={`rounded-full border px-3 py-1 text-sm ${
                presetId === p.id
                  ? "border-accent bg-accent-soft text-ink"
                  : "border-line text-ink-2 hover:bg-surface-2 hover:text-ink"
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
        <div className="mt-6">
          <CustomerForm value={customer} onChange={update} autoTotal={autoTotal} />
        </div>
      </section>

      <div className="space-y-6 lg:sticky lg:top-6 lg:self-start">
        <section className="rounded-xl border border-line bg-surface p-5" aria-labelledby="risk-heading" aria-busy={busy}>
          <div className="flex items-center justify-between">
            <h2 id="risk-heading" className="font-semibold">
              Churn risk
            </h2>
            <span className="text-xs text-ink-2" aria-live="polite">
              {status === "loading" && "Scoring…"}
              {status === "slow" && "The API is waking up; the first answer can take up to 2 minutes…"}
            </span>
          </div>
          <div className={`mt-4 transition-opacity ${busy && result ? "opacity-60" : ""}`}>
            {result ? (
              <RiskResult prediction={result.prediction} threshold={result.threshold} />
            ) : (
              <div className="h-36 animate-pulse rounded-lg bg-surface-2" aria-hidden="true" />
            )}
          </div>
          {status === "error" && error && (
            <p role="alert" className="mt-4 rounded-md border border-line bg-surface-2 px-3 py-2 text-sm text-risk-high-ink">
              {error}
            </p>
          )}
        </section>

        <section className="rounded-xl border border-line bg-surface p-5" aria-labelledby="why-heading">
          <h2 id="why-heading" className="font-semibold">
            Why this score?
          </h2>
          <p className="mt-1 text-sm text-ink-2">
            The answers that moved this customer&apos;s score most, according to the model.
          </p>
          <div className={`mt-5 transition-opacity ${busy && result ? "opacity-60" : ""}`}>
            {result?.contributions ? (
              <ExplanationChart contributions={result.contributions} />
            ) : result ? (
              <p className="text-sm text-ink-2">Explanations are not available from this API version.</p>
            ) : (
              <div className="h-40 animate-pulse rounded-lg bg-surface-2" aria-hidden="true" />
            )}
          </div>
          <p className="mt-5 border-t border-line pt-3 text-xs text-ink-2">
            Exact SHAP values from the XGBoost model: each bar is how far that answer pushed the
            score from the model&apos;s starting point, and together they add up to the
            prediction. They explain the model, not the customer&apos;s real reasons for leaving.
          </p>
        </section>

        <section className="rounded-xl border border-line bg-surface p-5" aria-labelledby="levers-heading">
          <h2 id="levers-heading" className="font-semibold">
            What would lower the risk?
          </h2>
          <p className="mt-1 text-sm text-ink-2">
            The same customer scored again with one retention offer applied, monthly bill
            unchanged.
          </p>
          <div className={`mt-5 transition-opacity ${busy && result ? "opacity-60" : ""}`}>
            {result ? (
              <LeverChart
                current={result.prediction.churn_probability}
                threshold={result.threshold}
                levers={result.levers}
              />
            ) : (
              <div className="h-40 animate-pulse rounded-lg bg-surface-2" aria-hidden="true" />
            )}
          </div>
          <p className="mt-5 border-t border-line pt-3 text-xs text-ink-2">
            These are patterns the model learned from past customers, not guaranteed effects:
            customers on long contracts churn less, but a contract alone may not keep someone who
            wants to leave. Use them to decide which offers are worth testing.
          </p>
        </section>
      </div>
    </div>
  );
}
