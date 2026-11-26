import { describe, expect, it } from "vitest";
import { customerSchema, type Customer } from "./customer";
import { PRESETS } from "./fields";
import { applyLever, leversFor } from "./levers";

const [risky, mid, loyal] = PRESETS.map((p) => p.customer) as [Customer, Customer, Customer];

describe("leversFor", () => {
  it("offers contracts, autopay and support add-ons to a new month-to-month customer", () => {
    const ids = leversFor(risky).map((l) => l.id);
    expect(ids).toEqual(["one-year", "two-year", "autopay", "security", "support", "all"]);
  });

  it("only offers what the customer does not already have", () => {
    // One-year contract, card autopay and online security already in place.
    expect(leversFor(mid).map((l) => l.id)).toEqual(["two-year", "support", "all"]);
  });

  it("offers no add-ons to a customer without internet", () => {
    const c: Customer = {
      ...risky,
      InternetService: "No",
      OnlineSecurity: "No internet service",
      OnlineBackup: "No internet service",
      DeviceProtection: "No internet service",
      TechSupport: "No internet service",
      StreamingTV: "No internet service",
      StreamingMovies: "No internet service",
    };
    const ids = leversFor(c).map((l) => l.id);
    expect(ids).not.toContain("security");
    expect(ids).not.toContain("support");
  });

  it("the combined offer uses the two-year contract, not the one-year", () => {
    const all = leversFor(risky).find((l) => l.id === "all")!;
    expect(all.change.Contract).toBe("Two year");
    expect(all.change.PaymentMethod).toBe("Credit card (automatic)");
    expect(all.change.TechSupport).toBe("Yes");
  });

  it("produces valid customers and never touches tenure or charges", () => {
    for (const c of [risky, mid, loyal]) {
      for (const lever of leversFor(c)) {
        const changed = applyLever(c, lever);
        expect(customerSchema.safeParse(changed).success).toBe(true);
        expect(changed.tenure).toBe(c.tenure);
        expect(changed.MonthlyCharges).toBe(c.MonthlyCharges);
      }
    }
  });

  it("has nothing to offer a customer who already has everything", () => {
    expect(leversFor({ ...loyal, TechSupport: "Yes", OnlineSecurity: "Yes" })).toEqual([]);
  });
});
