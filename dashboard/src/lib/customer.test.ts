import { describe, expect, it } from "vitest";
import { customerSchema, FEATURE_NAMES, features, normalise, optionsFor } from "./customer";
import { PRESETS } from "./fields";

// The API's own example (src/churn/api/schemas.py EXAMPLE_CUSTOMER).
const EXAMPLE = {
  customer_id: "7590-VHVEG",
  gender: "Female",
  SeniorCitizen: 0,
  Partner: "Yes",
  Dependents: "No",
  tenure: 1,
  PhoneService: "No",
  MultipleLines: "No phone service",
  InternetService: "DSL",
  OnlineSecurity: "No",
  OnlineBackup: "Yes",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "No",
  StreamingMovies: "No",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 29.85,
  TotalCharges: 29.85,
};

describe("customerSchema", () => {
  it("accepts the API's example customer", () => {
    expect(customerSchema.safeParse(EXAMPLE).success).toBe(true);
  });

  it("accepts every preset", () => {
    for (const p of PRESETS) expect(customerSchema.safeParse(p.customer).success).toBe(true);
  });

  it("rejects multiple lines without phone service, like the API", () => {
    const r = customerSchema.safeParse({ ...EXAMPLE, MultipleLines: "Yes" });
    expect(r.success).toBe(false);
    expect(r.error?.issues[0]?.path).toEqual(["MultipleLines"]);
  });

  it("rejects internet add-ons without internet service", () => {
    expect(customerSchema.safeParse({ ...EXAMPLE, InternetService: "No" }).success).toBe(false);
  });

  it("rejects unknown fields and out-of-range values", () => {
    expect(customerSchema.safeParse({ ...EXAMPLE, Churn: "Yes" }).success).toBe(false);
    expect(customerSchema.safeParse({ ...EXAMPLE, tenure: -5 }).success).toBe(false);
    expect(customerSchema.safeParse({ ...EXAMPLE, tenure: 1.5 }).success).toBe(false);
  });
});

describe("normalise", () => {
  it("forces dependent answers when a service is turned off", () => {
    const c = normalise({ ...PRESETS[0]!.customer, PhoneService: "No", InternetService: "No" });
    expect(c.MultipleLines).toBe("No phone service");
    expect(c.OnlineSecurity).toBe("No internet service");
    expect(c.StreamingMovies).toBe("No internet service");
    expect(customerSchema.safeParse(c).success).toBe(true);
  });

  it("resets dependent answers to No when a service is turned back on", () => {
    const off = normalise({ ...PRESETS[0]!.customer, PhoneService: "No", InternetService: "No" });
    const on = normalise({ ...off, PhoneService: "Yes", InternetService: "DSL" });
    expect(on.MultipleLines).toBe("No");
    expect(on.TechSupport).toBe("No");
    expect(customerSchema.safeParse(on).success).toBe(true);
  });
});

describe("field metadata", () => {
  it("lists the 19 model features", () => {
    expect(FEATURE_NAMES).toHaveLength(19);
    expect(FEATURE_NAMES).not.toContain("customer_id");
  });

  it("reads options from the schema", () => {
    expect(optionsFor("Contract")).toEqual(["Month-to-month", "One year", "Two year"]);
    expect(optionsFor("SeniorCitizen")).toEqual([0, 1]);
    expect(optionsFor("tenure")).toEqual([]);
  });

  it("features() drops the id", () => {
    expect(features(EXAMPLE as never)).not.toHaveProperty("customer_id");
  });
});
