import { describe, expect, it } from "vitest";
import { describeFactor, strength, topFactors, type Contribution } from "./explain";

const CONTRIBUTIONS: Contribution[] = [
  { feature: "tenure", value: 1, impact: 0.81 },
  { feature: "Contract", value: "Month-to-month", impact: 0.56 },
  { feature: "TotalCharges", value: 80, impact: 0.33 },
  { feature: "OnlineSecurity", value: "No", impact: 0.2 },
  { feature: "MultipleLines", value: "No", impact: -0.12 },
  { feature: "avg_monthly_spend", value: 80, impact: 0.05 },
  { feature: "num_services", value: 2, impact: -0.03 },
  { feature: "gender", value: "Female", impact: 0.01 },
];

describe("topFactors", () => {
  it("keeps the largest factors and sums the rest into one row", () => {
    const rows = topFactors(CONTRIBUTIONS, 5);
    expect(rows.map((r) => r.key)).toEqual([
      "tenure",
      "Contract",
      "TotalCharges",
      "OnlineSecurity",
      "MultipleLines",
      "other",
    ]);
    const other = rows.at(-1)!;
    expect(other.count).toBe(3);
    expect(other.impact).toBeCloseTo(0.05 - 0.03 + 0.01);
  });

  it("the rows still add up to the total push", () => {
    const total = CONTRIBUTIONS.reduce((s, c) => s + c.impact, 0);
    const shown = topFactors(CONTRIBUTIONS, 3).reduce((s, r) => s + r.impact, 0);
    expect(shown).toBeCloseTo(total);
  });

  it("ranks by size, not sign, even if the input is unsorted", () => {
    const rows = topFactors([...CONTRIBUTIONS].reverse(), 2);
    expect(rows.map((r) => r.key)).toEqual(["tenure", "Contract", "other"]);
  });

  it("has no 'other' row when everything fits", () => {
    expect(topFactors(CONTRIBUTIONS.slice(0, 3), 6).map((r) => r.key)).not.toContain("other");
  });
});

describe("describeFactor", () => {
  it("uses form labels and readable values", () => {
    expect(describeFactor("Contract", "Month-to-month")).toBe("Contract: Month-to-month");
    expect(describeFactor("tenure", 1)).toBe("Tenure: 1 month");
    expect(describeFactor("tenure", 24)).toBe("Tenure: 24 months");
    expect(describeFactor("SeniorCitizen", 0)).toBe("Senior citizen: No");
    expect(describeFactor("MonthlyCharges", 80)).toBe("Monthly charges: 80.00");
  });

  it("names the engineered features", () => {
    expect(describeFactor("avg_monthly_spend", 79.5)).toBe("Average monthly spend so far: 79.50");
    expect(describeFactor("num_services", 2)).toBe("Services in use: 2");
  });

  it("falls back to the raw name for anything unknown", () => {
    expect(describeFactor("mystery", "x")).toBe("mystery: x");
  });
});

it("describes strength in words", () => {
  expect(strength(0.81)).toBe("strongly raises");
  expect(strength(-0.2)).toBe("lowers");
  expect(strength(0.05)).toBe("slightly raises");
});
