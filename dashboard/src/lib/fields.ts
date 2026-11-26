import type { Customer, FeatureName } from "./customer";

export interface FieldInfo {
  label: string;
  help?: string;
  /** Display text for coded values (SeniorCitizen is 0/1 in the data). */
  display?: (value: string | number) => string;
}

export const FIELD_INFO: Record<FeatureName, FieldInfo> = {
  tenure: { label: "Tenure (months)", help: "How long they have been a customer" },
  Contract: { label: "Contract" },
  PaymentMethod: { label: "Payment method" },
  PaperlessBilling: { label: "Paperless billing" },
  MonthlyCharges: { label: "Monthly charges" },
  TotalCharges: { label: "Total charges", help: "Billed so far, about tenure × monthly" },
  gender: { label: "Gender" },
  SeniorCitizen: { label: "Senior citizen", display: (v) => (v === 1 ? "Yes" : "No") },
  Partner: { label: "Partner" },
  Dependents: { label: "Dependents" },
  PhoneService: { label: "Phone service" },
  MultipleLines: { label: "Multiple lines" },
  InternetService: { label: "Internet service" },
  OnlineSecurity: { label: "Online security" },
  OnlineBackup: { label: "Online backup" },
  DeviceProtection: { label: "Device protection" },
  TechSupport: { label: "Tech support" },
  StreamingTV: { label: "Streaming TV" },
  StreamingMovies: { label: "Streaming movies" },
};

export const FIELD_GROUPS: { title: string; fields: FeatureName[] }[] = [
  {
    title: "Account",
    fields: [
      "tenure",
      "Contract",
      "PaymentMethod",
      "PaperlessBilling",
      "MonthlyCharges",
      "TotalCharges",
    ],
  },
  {
    title: "Services",
    fields: [
      "PhoneService",
      "MultipleLines",
      "InternetService",
      "OnlineSecurity",
      "OnlineBackup",
      "DeviceProtection",
      "TechSupport",
      "StreamingTV",
      "StreamingMovies",
    ],
  },
  { title: "Household", fields: ["gender", "SeniorCitizen", "Partner", "Dependents"] },
];

// Same profiles the model gate uses (src/churn/gate.py): identical services and price,
// different commitment, so the difference in score is the contract and tenure alone.
const base: Omit<Customer, "tenure" | "Contract" | "TotalCharges"> = {
  gender: "Female",
  SeniorCitizen: 0,
  Partner: "No",
  Dependents: "No",
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "No",
  StreamingMovies: "No",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 80,
};

export const PRESETS: { id: string; label: string; description: string; customer: Customer }[] = [
  {
    id: "new",
    label: "New, no contract",
    description: "One month in, month-to-month, fibre, electronic check",
    customer: { ...base, tenure: 1, Contract: "Month-to-month", TotalCharges: 80 },
  },
  {
    id: "mid",
    label: "Two years, one-year contract",
    description: "Settled customer with DSL and a few add-ons",
    customer: {
      ...base,
      tenure: 24,
      Contract: "One year",
      InternetService: "DSL",
      OnlineSecurity: "Yes",
      OnlineBackup: "Yes",
      PaymentMethod: "Credit card (automatic)",
      MonthlyCharges: 65,
      TotalCharges: 1560,
    },
  },
  {
    id: "loyal",
    label: "Loyal, two-year contract",
    description: "Five years in, same services and price as the first",
    customer: {
      ...base,
      tenure: 60,
      Contract: "Two year",
      PaymentMethod: "Bank transfer (automatic)",
      TotalCharges: 4800,
    },
  },
];
