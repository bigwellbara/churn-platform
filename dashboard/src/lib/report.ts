// The Model page's data: a snapshot of the training run that produced the deployed model.
// Generated from reports/metrics.json and models/champion/model_meta.json by
// `npm run sync-report`; report.test.ts fails if the snapshot drifts from those files.
// No imports, so Node can run the sync script against this file directly.

export interface Candidate {
  name: string;
  valPrAuc: number;
  valRocAuc: number;
  valRecall: number;
  valPrecision: number;
  cvRocAucMean: number;
  cvRocAucStd: number;
  overfitGap: number;
}

export interface ModelReport {
  modelType: string;
  modelVersion: string;
  threshold: number;
  exportedAt: string | null;
  selectionMetric: string;
  test: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    rocAuc: number;
    prAuc: number;
    tp: number;
    fp: number;
    fn: number;
    tn: number;
  };
  candidates: Candidate[];
}

type Num = Record<string, number>;

const round = (x: number) => Math.round(x * 10_000) / 10_000;

export function buildReport(
  metrics: { selection_metric: string; candidates: Record<string, Num> },
  meta: {
    model_type: string;
    model_version: string;
    decision_threshold: number;
    exported_at?: string | null;
    test_metrics: Num;
  },
): ModelReport {
  const t = meta.test_metrics;
  const get = (o: Num, k: string) => {
    const v = o[k];
    if (typeof v !== "number") throw new Error(`missing metric ${k}`);
    return v;
  };
  return {
    modelType: meta.model_type,
    modelVersion: String(meta.model_version),
    threshold: round(meta.decision_threshold),
    exportedAt: meta.exported_at ?? null,
    selectionMetric: metrics.selection_metric,
    test: {
      accuracy: round(get(t, "accuracy")),
      precision: round(get(t, "precision")),
      recall: round(get(t, "recall")),
      f1: round(get(t, "f1")),
      rocAuc: round(get(t, "roc_auc")),
      prAuc: round(get(t, "pr_auc")),
      tp: get(t, "tp"),
      fp: get(t, "fp"),
      fn: get(t, "fn"),
      tn: get(t, "tn"),
    },
    candidates: Object.entries(metrics.candidates)
      .map(([name, c]) => ({
        name,
        valPrAuc: round(get(c, "val_pr_auc")),
        valRocAuc: round(get(c, "val_roc_auc")),
        valRecall: round(get(c, "val_recall")),
        valPrecision: round(get(c, "val_precision")),
        cvRocAucMean: round(get(c, "cv_roc_auc_mean")),
        cvRocAucStd: round(get(c, "cv_roc_auc_std")),
        overfitGap: round(get(c, "overfit_gap")),
      }))
      .sort((a, b) => b.valPrAuc - a.valPrAuc),
  };
}

export const MODEL_LABELS: Record<string, string> = {
  dummy_baseline: "Baseline (always guess the base rate)",
  logistic_regression: "Logistic regression",
  random_forest: "Random forest",
  xgboost: "XGBoost (defaults)",
  xgboost_tuned: "XGBoost (tuned)",
};
