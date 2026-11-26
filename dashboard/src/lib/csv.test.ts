import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parseCsv, parseCustomers, toCsv } from "./csv";

const HEADER =
  "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines," +
  "InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV," +
  "StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn";
// Rows in the exact format of the Kaggle Telco file.
const ROW_1 =
  "7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No," +
  "Month-to-month,Yes,Electronic check,29.85,29.85,No";
const ROW_NEW =
  "4472-LVYGI,Female,0,Yes,Yes,0,No,No phone service,DSL,Yes,No,Yes,Yes,Yes,No," +
  "Two year,Yes,Bank transfer (automatic),52.55, ,No";
const ROW_LEFT =
  "3668-QPYBK,Male,0,No,No,2,Yes,No,DSL,Yes,Yes,No,No,No,No," +
  "Month-to-month,Yes,Mailed check,53.85,108.15,Yes";

describe("parseCsv", () => {
  it("handles quotes, escaped quotes, embedded commas and newlines", () => {
    expect(parseCsv('a,"b,c","say ""hi""","line\nbreak"\n')).toEqual([
      ["a", "b,c", 'say "hi"', "line\nbreak"],
    ]);
  });

  it("handles CRLF, a byte-order mark and blank lines", () => {
    expect(parseCsv("﻿x,y\r\n1,2\r\n\r\n3,4")).toEqual([
      ["x", "y"],
      ["1", "2"],
      ["3", "4"],
    ]);
  });

  it("keeps empty fields", () => {
    expect(parseCsv("a,,c\n")).toEqual([["a", "", "c"]]);
  });
});

describe("toCsv", () => {
  it("escapes fields that need it and round-trips through parseCsv", () => {
    const text = toCsv(["id", "note"], [["1", 'He said "no", then left']]);
    expect(parseCsv(text)).toEqual([
      ["id", "note"],
      ["1", 'He said "no", then left'],
    ]);
  });
});

describe("parseCustomers", () => {
  it("reads the Kaggle Telco format, including the outcome column", () => {
    const r = parseCustomers([HEADER, ROW_1, ROW_LEFT].join("\n"));
    expect(r.errors).toEqual([]);
    expect(r.missingColumns).toEqual([]);
    expect(r.hasOutcome).toBe(true);
    expect(r.rows).toHaveLength(2);
    expect(r.rows[0]!.customer.customer_id).toBe("7590-VHVEG");
    expect(r.rows[0]!.customer.MonthlyCharges).toBe(29.85);
    expect(r.rows[0]!.actual).toBe(false);
    expect(r.rows[1]!.actual).toBe(true);
    expect(r.rows[1]!.line).toBe(3);
  });

  it("treats a blank TotalCharges as 0, as in the raw data for brand-new customers", () => {
    const r = parseCustomers([HEADER, ROW_NEW].join("\n"));
    expect(r.errors).toEqual([]);
    expect(r.rows[0]!.customer.TotalCharges).toBe(0);
  });

  it("reports invalid rows by line number and keeps the valid ones", () => {
    const bad = ROW_1.replace("Month-to-month", "Weekly");
    const r = parseCustomers([HEADER, ROW_1, bad].join("\n"));
    expect(r.rows).toHaveLength(1);
    expect(r.errors).toEqual([{ line: 3, message: expect.stringContaining("Contract") }]);
  });

  it("reports a non-numeric tenure", () => {
    const r = parseCustomers([HEADER, ROW_1.replace(",1,No,", ",abc,No,")].join("\n"));
    expect(r.errors[0]!.message).toContain("tenure");
  });

  it("lists missing columns and parses nothing", () => {
    const r = parseCustomers("customerID,gender\nx,Male\n");
    expect(r.missingColumns).toContain("tenure");
    expect(r.rows).toEqual([]);
  });

  it("works without id or outcome columns", () => {
    const header = HEADER.replace("customerID,", "").replace(",Churn", "");
    const row = ROW_1.replace("7590-VHVEG,", "").replace(/,No$/, "");
    const r = parseCustomers([header, row].join("\n"));
    expect(r.errors).toEqual([]);
    expect(r.hasOutcome).toBe(false);
    expect(r.rows[0]!.customer.customer_id).toBeUndefined();
  });

  it("handles an empty file", () => {
    expect(parseCustomers("").rows).toEqual([]);
  });

  it("parses the bundled sample file without errors", () => {
    const text = readFileSync(new URL("../../public/sample-customers.csv", import.meta.url), "utf-8");
    const r = parseCustomers(text);
    expect(r.missingColumns).toEqual([]);
    expect(r.errors).toEqual([]);
    expect(r.rows).toHaveLength(40);
  });
});
