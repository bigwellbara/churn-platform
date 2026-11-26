// Shapes returned by the API (src/churn/api/schemas.py).

export type RiskLevel = "low" | "medium" | "high";

export interface Prediction {
  customer_id: string | null;
  churn_probability: number;
  churn_prediction: boolean;
  risk_level: RiskLevel;
}

export interface BatchResponse {
  model_version: string;
  threshold: number;
  count: number;
  predictions: Prediction[];
}

export interface ApiError {
  error: string;
  detail: string | { field: string; message: string }[];
  request_id?: string | null;
}

export function describeError(body: unknown, status: number): string {
  const e = body as Partial<ApiError> | null;
  if (e && Array.isArray(e.detail) && e.detail.length > 0) {
    return e.detail.map((d) => `${d.field}: ${d.message}`).join("; ");
  }
  if (e && typeof e.detail === "string") return e.detail;
  return `The API answered with HTTP ${status}`;
}

export interface Scored {
  probability: number;
  flagged: boolean;
  risk: RiskLevel;
  actual?: boolean;
}

export interface Summary {
  count: number;
  byRisk: Record<RiskLevel, number>;
  flagged: number;
  /** Sum of probabilities: how many of these customers the model expects to leave. */
  expectedChurners: number;
  /** Only when every row has a known outcome. */
  outcome?: { tp: number; fp: number; fn: number; tn: number; recall: number; precision: number };
}

export function summarise(rows: Scored[]): Summary {
  const byRisk: Record<RiskLevel, number> = { low: 0, medium: 0, high: 0 };
  let flagged = 0;
  let expected = 0;
  for (const r of rows) {
    byRisk[r.risk]++;
    if (r.flagged) flagged++;
    expected += r.probability;
  }
  const summary: Summary = { count: rows.length, byRisk, flagged, expectedChurners: expected };

  if (rows.length > 0 && rows.every((r) => r.actual !== undefined)) {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    let tn = 0;
    for (const r of rows) {
      if (r.flagged && r.actual) tp++;
      else if (r.flagged) fp++;
      else if (r.actual) fn++;
      else tn++;
    }
    summary.outcome = {
      tp,
      fp,
      fn,
      tn,
      recall: tp + fn ? tp / (tp + fn) : 0,
      precision: tp + fp ? tp / (tp + fp) : 0,
    };
  }
  return summary;
}

export function percent(p: number, digits = 0): string {
  return `${(p * 100).toFixed(digits)}%`;
}
