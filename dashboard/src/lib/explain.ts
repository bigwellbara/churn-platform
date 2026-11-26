import type { Prediction } from "./predictions";
import { FIELD_INFO } from "./fields";
import type { FeatureName } from "./customer";

// Shapes returned by POST /predict/explain (src/churn/api/schemas.py).

export interface Contribution {
  feature: string;
  value: string | number;
  /** Push on the score in log-odds: positive raises churn risk, negative lowers it. */
  impact: number;
}

export interface ExplanationResponse extends Prediction {
  model_version: string;
  threshold: number;
  base_value: number;
  contributions: Contribution[];
}

export interface Factor {
  key: string;
  label: string;
  impact: number;
  /** Number of features rolled into the "other" row; absent for a single feature. */
  count?: number;
}

// The two features the model computes from the answers (src/churn/training/features.py).
const ENGINEERED: Record<string, { label: string; format: (v: string | number) => string }> = {
  avg_monthly_spend: {
    label: "Average monthly spend so far",
    format: (v) => Number(v).toFixed(2),
  },
  num_services: {
    label: "Services in use",
    format: (v) => String(v),
  },
};

/** "Contract: Month-to-month", "Tenure: 1 month", "Senior citizen: No". */
export function describeFactor(feature: string, value: string | number): string {
  const engineered = ENGINEERED[feature];
  if (engineered) return `${engineered.label}: ${engineered.format(value)}`;

  const info = FIELD_INFO[feature as FeatureName];
  if (!info) return `${feature}: ${value}`;
  if (feature === "tenure") return `Tenure: ${value} month${Number(value) === 1 ? "" : "s"}`;
  if (feature === "MonthlyCharges" || feature === "TotalCharges") {
    return `${info.label}: ${Number(value).toFixed(2)}`;
  }
  const shown = info.display ? info.display(value) : String(value);
  return `${info.label}: ${shown}`;
}

/**
 * The `top` largest factors, plus one row summing the rest so the chart still adds up to
 * the whole score. Input is assumed sorted largest first (the API sorts it).
 */
export function topFactors(contributions: Contribution[], top = 6): Factor[] {
  const sorted = [...contributions].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
  const shown: Factor[] = sorted.slice(0, top).map((c) => ({
    key: c.feature,
    label: describeFactor(c.feature, c.value),
    impact: c.impact,
  }));
  const rest = sorted.slice(top);
  if (rest.length > 0) {
    shown.push({
      key: "other",
      label: `${rest.length} other factors combined`,
      impact: rest.reduce((sum, c) => sum + c.impact, 0),
      count: rest.length,
    });
  }
  return shown;
}

/** Plain-words strength, so nobody has to know what log-odds are. */
export function strength(impact: number): string {
  const size = Math.abs(impact);
  const direction = impact >= 0 ? "raises" : "lowers";
  if (size >= 0.5) return `strongly ${direction}`;
  if (size >= 0.15) return direction;
  return `slightly ${direction}`;
}
