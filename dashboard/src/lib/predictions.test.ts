import { describe, expect, it } from "vitest";
import { describeError, percent, summarise } from "./predictions";

describe("summarise", () => {
  it("counts risk levels, flags and expected churners", () => {
    const s = summarise([
      { probability: 0.9, flagged: true, risk: "high" },
      { probability: 0.4, flagged: false, risk: "medium" },
      { probability: 0.1, flagged: false, risk: "low" },
    ]);
    expect(s.count).toBe(3);
    expect(s.byRisk).toEqual({ low: 1, medium: 1, high: 1 });
    expect(s.flagged).toBe(1);
    expect(s.expectedChurners).toBeCloseTo(1.4);
    expect(s.outcome).toBeUndefined();
  });

  it("scores against known outcomes when every row has one", () => {
    const s = summarise([
      { probability: 0.9, flagged: true, risk: "high", actual: true },
      { probability: 0.8, flagged: true, risk: "high", actual: false },
      { probability: 0.2, flagged: false, risk: "low", actual: true },
      { probability: 0.1, flagged: false, risk: "low", actual: false },
    ]);
    expect(s.outcome).toEqual({ tp: 1, fp: 1, fn: 1, tn: 1, recall: 0.5, precision: 0.5 });
  });

  it("is safe on empty input", () => {
    expect(summarise([]).count).toBe(0);
    expect(summarise([]).outcome).toBeUndefined();
  });
});

describe("describeError", () => {
  it("joins field errors from a 422", () => {
    const body = { error: "validation_error", detail: [{ field: "tenure", message: "too small" }] };
    expect(describeError(body, 422)).toBe("tenure: too small");
  });

  it("falls back to the detail string, then the status", () => {
    expect(describeError({ detail: "Waking up" }, 504)).toBe("Waking up");
    expect(describeError(null, 502)).toBe("The API answered with HTTP 502");
  });
});

it("formats percentages", () => {
  expect(percent(0.8683)).toBe("87%");
  expect(percent(0.8683, 1)).toBe("86.8%");
});
