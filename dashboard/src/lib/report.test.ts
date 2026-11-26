import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import snapshot from "../data/model-report.json";
import { buildReport, MODEL_LABELS } from "./report";

const repo = new URL("../../../", import.meta.url);
const read = (path: string) => JSON.parse(readFileSync(new URL(path, repo), "utf-8"));

describe("model report snapshot", () => {
  it("matches the training report and the shipped model (run `npm run sync-report` if not)", () => {
    const fresh = buildReport(read("reports/metrics.json"), read("models/champion/model_meta.json"));
    expect(snapshot).toEqual(fresh);
  });

  it("ranks candidates by the selection metric and labels every one", () => {
    const prAucs = snapshot.candidates.map((c) => c.valPrAuc);
    expect(prAucs).toEqual([...prAucs].sort((a, b) => b - a));
    for (const c of snapshot.candidates) expect(MODEL_LABELS[c.name]).toBeDefined();
  });

  it("has a confusion matrix that adds up to the 1,409-customer test set", () => {
    const { tp, fp, fn, tn } = snapshot.test;
    expect(tp + fp + fn + tn).toBe(1409);
  });
});

it("buildReport fails loudly on a missing metric", () => {
  expect(() =>
    buildReport(
      { selection_metric: "val_pr_auc", candidates: {} },
      { model_type: "x", model_version: "1", decision_threshold: 0.5, test_metrics: {} },
    ),
  ).toThrow(/missing metric/);
});
