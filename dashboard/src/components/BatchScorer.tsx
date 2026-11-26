"use client";

import { useEffect, useRef, useState } from "react";
import { features, FEATURE_NAMES } from "@/lib/customer";
import { parseCustomers, toCsv, type ParsedRow, type ParseResult } from "@/lib/csv";
import {
  describeError,
  percent,
  summarise,
  type BatchResponse,
  type Prediction,
  type RiskLevel,
} from "@/lib/predictions";
import { RiskBadge } from "./RiskBadge";

const CHUNK = 500;
const MAX_ROWS = 10_000;
const TOP_N = 25;

interface ScoredRow extends ParsedRow {
  prediction: Prediction;
}

type Phase =
  | { kind: "empty" }
  | { kind: "scoring"; done: number; total: number }
  | { kind: "done" }
  | { kind: "error"; message: string };

export function BatchScorer() {
  const [fileName, setFileName] = useState<string | null>(null);
  const [parsed, setParsed] = useState<ParseResult | null>(null);
  const [rows, setRows] = useState<ScoredRow[]>([]);
  const [threshold, setThreshold] = useState(0);
  const [phase, setPhase] = useState<Phase>({ kind: "empty" });
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    setFileName(file.name);
    setRows([]);
    const result = parseCustomers(await file.text());
    setParsed(result);
    if (result.missingColumns.length > 0 || result.rows.length === 0) {
      setPhase({ kind: "empty" });
      return;
    }
    const toScore = result.rows.slice(0, MAX_ROWS);
    const scored: ScoredRow[] = [];
    setPhase({ kind: "scoring", done: 0, total: toScore.length });

    try {
      for (let i = 0; i < toScore.length; i += CHUNK) {
        const chunk = toScore.slice(i, i + CHUNK);
        const res = await fetch("/api/predict/batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ customers: chunk.map((r) => features(r.customer)) }),
        });
        const body = await res.json().catch(() => null);
        if (!res.ok) throw new Error(describeError(body, res.status));
        const batch = body as BatchResponse;
        setThreshold(batch.threshold);
        chunk.forEach((r, j) => scored.push({ ...r, prediction: batch.predictions[j]! }));
        setPhase({ kind: "scoring", done: scored.length, total: toScore.length });
      }
      setRows(scored);
      setPhase({ kind: "done" });
    } catch (err) {
      setRows(scored);
      setPhase({ kind: "error", message: err instanceof Error ? err.message : "Scoring failed" });
    }
  }

  function loadSample(): Promise<File> {
    return fetch("/sample-customers.csv")
      .then((res) => res.text())
      .then((text) => new File([text], "sample-customers.csv", { type: "text/csv" }));
  }

  async function runSample() {
    await handleFile(await loadSample());
  }

  // A shareable link that runs the demo straight away: /batch?sample=1
  useEffect(() => {
    if (!new URLSearchParams(window.location.search).has("sample")) return;
    void loadSample().then((file) => handleFile(file));
  }, []);

  function download() {
    const hasOutcome = parsed?.hasOutcome ?? false;
    const header = [
      "line",
      "customer_id",
      "churn_probability",
      "risk_level",
      "flagged",
      ...(hasOutcome ? ["actual_churn"] : []),
      ...FEATURE_NAMES,
    ];
    const body = [...rows]
      .sort((a, b) => b.prediction.churn_probability - a.prediction.churn_probability)
      .map((r) => [
        r.line,
        r.customer.customer_id ?? "",
        r.prediction.churn_probability.toFixed(4),
        r.prediction.risk_level,
        r.prediction.churn_prediction ? "Yes" : "No",
        ...(hasOutcome ? [r.actual === undefined ? "" : r.actual ? "Yes" : "No"] : []),
        ...FEATURE_NAMES.map((f) => r.customer[f]),
      ]);
    const blob = new Blob([toCsv(header, body)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${(fileName ?? "customers").replace(/\.csv$/i, "")}-scored.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const summary =
    rows.length > 0
      ? summarise(
          rows.map((r) => ({
            probability: r.prediction.churn_probability,
            flagged: r.prediction.churn_prediction,
            risk: r.prediction.risk_level,
            actual: r.actual,
          })),
        )
      : null;
  const top = [...rows]
    .sort((a, b) => b.prediction.churn_probability - a.prediction.churn_probability)
    .slice(0, TOP_N);

  return (
    <div className="space-y-6">
      <section
        className={`rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
          dragging ? "border-accent bg-accent-soft" : "border-line bg-surface"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const file = e.dataTransfer.files[0];
          if (file) void handleFile(file);
        }}
      >
        <p className="font-medium">Drop a CSV file here</p>
        <p className="mt-1 text-sm text-ink-2">
          Same columns as the Telco dataset. The original Kaggle file works as-is, and if it has a
          Churn column you will see how the model did against what really happened.
        </p>
        <div className="mt-4 flex flex-wrap justify-center gap-3">
          <button
            type="button"
            onClick={() => input.current?.click()}
            className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-ink hover:opacity-90"
          >
            Choose file
          </button>
          <button
            type="button"
            onClick={() => void runSample()}
            className="rounded-md border border-line px-4 py-2 text-sm text-ink hover:bg-surface-2"
          >
            Try it with sample data
          </button>
        </div>
        <p className="mt-3 text-xs text-ink-2">
          The sample is 40 synthetic customers;{" "}
          <a href="/sample-customers.csv" download className="underline hover:text-ink">
            download it
          </a>{" "}
          to see the expected format.
        </p>
        <input
          ref={input}
          type="file"
          accept=".csv,text/csv"
          className="sr-only"
          aria-label="CSV file of customers"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void handleFile(file);
            e.target.value = "";
          }}
        />
        {fileName && <p className="mt-4 text-sm text-ink-2">File: {fileName}</p>}
      </section>

      {parsed && parsed.missingColumns.length > 0 && (
        <Notice>
          This file is missing {parsed.missingColumns.length} required column
          {parsed.missingColumns.length > 1 ? "s" : ""}: {parsed.missingColumns.join(", ")}.
        </Notice>
      )}
      {parsed && parsed.errors.length > 0 && (
        <Notice>
          {parsed.errors.length} row{parsed.errors.length > 1 ? "s were" : " was"} skipped because of
          invalid values. First: line {parsed.errors[0]!.line}, {parsed.errors[0]!.message}.
        </Notice>
      )}
      {parsed && parsed.rows.length > MAX_ROWS && (
        <Notice>Only the first {MAX_ROWS.toLocaleString("en-GB")} customers are scored.</Notice>
      )}

      {phase.kind === "scoring" && (
        <section className="rounded-xl border border-line bg-surface p-5" aria-live="polite">
          <div className="flex justify-between text-sm">
            <span>Scoring customers…</span>
            <span className="tabular text-ink-2">
              {phase.done.toLocaleString("en-GB")} / {phase.total.toLocaleString("en-GB")}
            </span>
          </div>
          <div className="mt-2 h-2 rounded-full bg-track">
            <div
              className="h-2 rounded-full bg-accent transition-[width]"
              style={{ width: `${(phase.done / phase.total) * 100}%` }}
            />
          </div>
          {phase.done === 0 && (
            <p className="mt-2 text-xs text-ink-2">
              If the API was asleep, the first batch can take up to 2 minutes.
            </p>
          )}
        </section>
      )}
      {phase.kind === "error" && <Notice role="alert">Scoring stopped: {phase.message}</Notice>}

      {summary && (
        <>
          <section className="grid grid-cols-2 gap-3 lg:grid-cols-4" aria-label="Summary">
            <Stat label="Customers scored" value={summary.count.toLocaleString("en-GB")} />
            <Stat
              label="Flagged for retention"
              value={summary.flagged.toLocaleString("en-GB")}
              note={`${percent(summary.flagged / summary.count)} of the list, at ${percent(threshold)} or above`}
            />
            <Stat
              label="Expected to leave"
              value={`≈ ${Math.round(summary.expectedChurners).toLocaleString("en-GB")}`}
              note="Sum of the probabilities"
            />
            {summary.outcome ? (
              <Stat
                label="Leavers caught"
                value={percent(summary.outcome.recall)}
                note={`${summary.outcome.tp} of ${summary.outcome.tp + summary.outcome.fn} who left were flagged; ${percent(summary.outcome.precision)} of flags were right`}
              />
            ) : (
              <Stat
                label="Average risk"
                value={percent(summary.expectedChurners / summary.count)}
                note="Mean churn probability"
              />
            )}
          </section>

          <section className="rounded-xl border border-line bg-surface p-5">
            <h2 className="font-semibold">Risk split</h2>
            <RiskSplit byRisk={summary.byRisk} total={summary.count} />
          </section>

          <section className="rounded-xl border border-line bg-surface p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="font-semibold">Highest risk first</h2>
                <p className="text-sm text-ink-2">
                  Top {Math.min(TOP_N, rows.length)} of {rows.length.toLocaleString("en-GB")}. The download
                  has every customer.
                </p>
              </div>
              <button
                type="button"
                onClick={download}
                className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-accent-ink hover:opacity-90"
              >
                Download all results (CSV)
              </button>
            </div>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-[40rem] text-left text-sm">
                <thead className="text-xs uppercase tracking-wider text-ink-2">
                  <tr className="border-b border-line">
                    <th className="py-2 pr-3 font-medium">Customer</th>
                    <th className="py-2 pr-3 text-right font-medium">Churn probability</th>
                    <th className="py-2 pr-3 font-medium">Risk</th>
                    <th className="py-2 pr-3 text-right font-medium">Tenure</th>
                    <th className="py-2 pr-3 font-medium">Contract</th>
                    <th className="py-2 pr-3 text-right font-medium">Monthly</th>
                    {parsed?.hasOutcome && <th className="py-2 font-medium">Actually left</th>}
                  </tr>
                </thead>
                <tbody className="tabular">
                  {top.map((r) => (
                    <tr key={r.line} className="border-b border-line last:border-0">
                      <td className="py-2 pr-3">{r.customer.customer_id ?? `Line ${r.line}`}</td>
                      <td className="py-2 pr-3 text-right font-medium">
                        {percent(r.prediction.churn_probability, 1)}
                      </td>
                      <td className="py-2 pr-3">
                        <RiskBadge level={r.prediction.risk_level} />
                      </td>
                      <td className="py-2 pr-3 text-right">{r.customer.tenure} mo</td>
                      <td className="py-2 pr-3">{r.customer.Contract}</td>
                      <td className="py-2 pr-3 text-right">{r.customer.MonthlyCharges.toFixed(2)}</td>
                      {parsed?.hasOutcome && (
                        <td className="py-2">{r.actual === undefined ? "–" : r.actual ? "Yes" : "No"}</td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </div>
  );
}

function Notice({ children, role }: { children: React.ReactNode; role?: "alert" }) {
  return (
    <p role={role} className="rounded-md border border-line bg-surface px-4 py-3 text-sm">
      {children}
    </p>
  );
}

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="rounded-xl border border-line bg-surface p-4">
      <div className="text-sm text-ink-2">{label}</div>
      <div className="mt-1 text-3xl font-semibold tracking-tight">{value}</div>
      {note && <div className="mt-1 text-xs text-ink-2">{note}</div>}
    </div>
  );
}

const SPLIT: { level: RiskLevel; color: string; label: string }[] = [
  { level: "high", color: "var(--risk-high)", label: "High" },
  { level: "medium", color: "var(--risk-medium)", label: "Medium" },
  { level: "low", color: "var(--risk-low)", label: "Low" },
];

/** One whole split into three parts: a segmented bar with a labelled legend carrying the figures. */
function RiskSplit({ byRisk, total }: { byRisk: Record<RiskLevel, number>; total: number }) {
  return (
    <div className="mt-4">
      <div className="flex h-3 gap-0.5 overflow-hidden rounded-full" aria-hidden="true">
        {SPLIT.filter((s) => byRisk[s.level] > 0).map((s) => (
          <div
            key={s.level}
            title={`${s.label} risk: ${byRisk[s.level]} (${percent(byRisk[s.level] / total)})`}
            style={{ width: `${(byRisk[s.level] / total) * 100}%`, background: s.color }}
          />
        ))}
      </div>
      <ul className="mt-3 flex flex-wrap gap-x-6 gap-y-2 text-sm">
        {SPLIT.map((s) => (
          <li key={s.level} className="flex items-center gap-2">
            <RiskBadge level={s.level} />
            <span className="tabular font-medium">{byRisk[s.level].toLocaleString("en-GB")}</span>
            <span className="tabular text-ink-2">({percent(byRisk[s.level] / total)})</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
