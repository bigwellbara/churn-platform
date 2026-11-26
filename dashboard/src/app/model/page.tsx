import type { Metadata } from "next";
import report from "@/data/model-report.json";
import { percent } from "@/lib/predictions";
import { MODEL_LABELS, type ModelReport } from "@/lib/report";

export const metadata: Metadata = { title: "The model" };

const REPO = "https://github.com/bigwellbara/churn-platform/blob/main";
const r = report as ModelReport;

function exportedOn(iso: string | null): string {
  if (!iso) return "";
  return new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
}

export default function ModelPage() {
  const t = r.test;
  const total = t.tp + t.fp + t.fn + t.tn;
  const leavers = t.tp + t.fn;
  const baseRate = leavers / total;
  const champion = r.candidates.find((c) => c.name === r.modelType);

  return (
    <div className="space-y-8">
      <div className="max-w-3xl">
        <h1 className="text-2xl font-semibold tracking-tight">The model</h1>
        <p className="mt-2 text-ink-2">
          {MODEL_LABELS[r.modelType] ?? r.modelType}, version {r.modelVersion}
          {r.exportedAt ? `, exported ${exportedOn(r.exportedAt)}` : ""}. Chosen from{" "}
          {r.candidates.length} candidates. Every number below is measured on a held-out test set
          of {total.toLocaleString("en-GB")} customers the model never saw during training or selection.
        </p>
      </div>

      <section aria-label="Test-set metrics" className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Tile
          label="ROC-AUC"
          value={t.rocAuc.toFixed(2)}
          note={`Shown one leaver and one stayer, it ranks the leaver higher ${percent(t.rocAuc)} of the time. Guessing: 0.50.`}
        />
        <Tile
          label="PR-AUC"
          value={t.prAuc.toFixed(2)}
          note={`Precision and recall across every threshold. Guessing scores ${baseRate.toFixed(2)}, the share who leave.`}
        />
        <Tile
          label="Recall"
          value={percent(t.recall)}
          note="Of customers who left, the share it flagged."
        />
        <Tile
          label="Precision"
          value={percent(t.precision)}
          note="Of customers it flagged, the share who really left."
        />
      </section>

      <section className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-line bg-surface p-5">
          <h2 className="font-semibold">What happened to the test customers</h2>
          <p className="mt-1 text-sm text-ink-2">
            Each customer lands in one box: did they leave, and did the model flag them?
          </p>
          <ConfusionMatrix tp={t.tp} fp={t.fp} fn={t.fn} tn={t.tn} />
          <p className="mt-4 text-sm">
            Of {leavers} customers who left, the model flagged <strong>{t.tp}</strong> in advance
            and missed {t.fn}. It also flagged {t.fp} who stayed: the cost of a retention offer to
            someone who did not need one.
          </p>
        </div>

        <div className="rounded-xl border border-line bg-surface p-5">
          <h2 className="font-semibold">Why flag at {percent(r.threshold)}, not 50%?</h2>
          <div className="mt-3 space-y-3 text-sm text-ink-2">
            <p>
              The model outputs a probability; the business needs a yes or no. The cut-off was
              tuned on a separate validation set to maximise <strong className="text-ink">F1</strong>,
              the balance between catching leavers (recall) and not raising false alarms
              (precision).
            </p>
            <p>
              The best balance came out at {percent(r.threshold)}. It is stored with the model, so
              the API, this dashboard and the evaluation all use exactly the same value.
            </p>
            <p>
              If the costs were known, say a missed leaver loses R3,000 of revenue and an offer
              costs R100, the threshold should be chosen from those costs instead. Cheap offers
              justify a lower threshold and more false alarms.
            </p>
          </div>
        </div>
      </section>

      <section className="rounded-xl border border-line bg-surface p-5">
        <h2 className="font-semibold">How it was chosen</h2>
        <p className="mt-1 max-w-3xl text-sm text-ink-2">
          Five models trained on the same data and compared on the validation set by PR-AUC, which
          suits a target where only about a quarter of customers leave. The baseline shows the
          score you get without learning anything.
        </p>
        <ModelComparison report={r} />
        {champion && (
          <p className="mt-4 max-w-3xl text-sm">
            The top four are close; the tuned XGBoost won on PR-AUC and, just as important, has
            the smallest <strong>overfit gap</strong> of the tree models ({champion.overfitGap.toFixed(3)}{" "}
            against {r.candidates.find((c) => c.name === "xgboost")?.overfitGap.toFixed(3)} untuned):
            tuning stopped it memorising the training data, so its scores on new customers are
            more trustworthy.
          </p>
        )}
      </section>

      <p className="text-sm text-ink-2">
        More detail:{" "}
        <a className="underline hover:text-ink" href={`${REPO}/docs/MODEL_CARD.md`}>
          model card
        </a>{" "}
        (intended use, limits, fairness checks) and{" "}
        <a className="underline hover:text-ink" href={`${REPO}/docs/TRAINING.md`}>
          training write-up
        </a>
        .
      </p>
    </div>
  );
}

function Tile({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="rounded-xl border border-line bg-surface p-4">
      <div className="text-sm text-ink-2">{label}</div>
      <div className="mt-1 text-3xl font-semibold tracking-tight">{value}</div>
      <div className="mt-1 text-xs text-ink-2">{note}</div>
    </div>
  );
}

function step(count: number, max: number): 1 | 2 | 3 | 4 {
  const share = count / max;
  if (share > 0.66) return 4;
  if (share > 0.33) return 3;
  if (share > 0.15) return 2;
  return 1;
}

/** 2x2 heatmap: one blue hue, darker = more customers; every cell is labelled in words. */
function ConfusionMatrix({ tp, fp, fn, tn }: { tp: number; fp: number; fn: number; tn: number }) {
  const max = Math.max(tp, fp, fn, tn);
  const cell = (count: number, title: string) => {
    const s = step(count, max);
    return (
      <div
        className="flex min-h-24 flex-col justify-between rounded-lg p-3"
        style={{ background: `var(--seq-${s})`, color: `var(--seq-ink-${s})` }}
        title={`${title}: ${count}`}
      >
        <span className="text-xs font-medium">{title}</span>
        <span className="tabular text-2xl font-semibold">{count.toLocaleString("en-GB")}</span>
      </div>
    );
  };
  return (
    <div className="mt-4 grid grid-cols-[auto_1fr_1fr] gap-1 text-sm" role="table" aria-label="Confusion matrix">
      <div role="row" className="contents">
        <span role="columnheader" />
        <span role="columnheader" className="pb-1 text-center text-xs text-ink-2">
          Flagged
        </span>
        <span role="columnheader" className="pb-1 text-center text-xs text-ink-2">
          Not flagged
        </span>
      </div>
      <div role="row" className="contents">
        <span role="rowheader" className="flex items-center pr-2 text-xs text-ink-2">
          Left
        </span>
        <div role="cell">{cell(tp, "Caught")}</div>
        <div role="cell">{cell(fn, "Missed")}</div>
      </div>
      <div role="row" className="contents">
        <span role="rowheader" className="flex items-center pr-2 text-xs text-ink-2">
          Stayed
        </span>
        <div role="cell">{cell(fp, "False alarm")}</div>
        <div role="cell">{cell(tn, "Correctly left alone")}</div>
      </div>
    </div>
  );
}

/** Table with an inline bar for the selection metric: the winner in the accent hue, the rest grey. */
function ModelComparison({ report }: { report: ModelReport }) {
  return (
    <div className="mt-4 overflow-x-auto">
      <table className="w-full min-w-[44rem] text-left text-sm">
        <thead className="text-xs uppercase tracking-wider text-ink-2">
          <tr className="border-b border-line">
            <th className="py-2 pr-3 font-medium">Model</th>
            <th className="w-[34%] py-2 pr-3 font-medium">Validation PR-AUC</th>
            <th className="py-2 pr-3 text-right font-medium">ROC-AUC</th>
            <th className="py-2 pr-3 text-right font-medium">Recall</th>
            <th className="py-2 text-right font-medium" title="Training score minus cross-validated score: how much it memorised">
              Overfit gap
            </th>
          </tr>
        </thead>
        <tbody className="tabular">
          {report.candidates.map((c) => {
            const winner = c.name === report.modelType;
            return (
              <tr key={c.name} className={`border-b border-line last:border-0 ${winner ? "font-medium" : ""}`}>
                <td className="py-2.5 pr-3">
                  {MODEL_LABELS[c.name] ?? c.name}
                  {winner && (
                    <span className="ml-2 rounded-full bg-accent-soft px-2 py-0.5 text-xs text-ink">
                      deployed
                    </span>
                  )}
                </td>
                <td className="py-2.5 pr-3">
                  <div className="flex items-center gap-3">
                    <div className="h-2.5 flex-1 rounded-full bg-track" aria-hidden="true">
                      <div
                        className={`h-2.5 rounded-full ${winner ? "bg-accent" : "bg-muted-bar"}`}
                        style={{ width: `${c.valPrAuc * 100}%` }}
                      />
                    </div>
                    <span className="w-10 text-right">{c.valPrAuc.toFixed(3)}</span>
                  </div>
                </td>
                <td className="py-2.5 pr-3 text-right">{c.valRocAuc.toFixed(3)}</td>
                <td className="py-2.5 pr-3 text-right">{percent(c.valRecall)}</td>
                <td className="py-2.5 text-right">{c.overfitGap.toFixed(3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-ink-2">
        Bars run from 0 to 1. Overfit gap: training ROC-AUC minus cross-validated ROC-AUC; bigger
        means the model memorised more of its training data.
      </p>
    </div>
  );
}
