import { z } from "zod";

// Mirrors CustomerFeatures in src/churn/api/schemas.py. The API is the source of truth
// and validates every request again; this copy gives instant feedback in the form and
// turns CSV text into typed values.

const yesNo = z.enum(["Yes", "No"]);
const internetAddon = z.enum(["Yes", "No", "No internet service"]);

export const INTERNET_ADDONS = [
  "OnlineSecurity",
  "OnlineBackup",
  "DeviceProtection",
  "TechSupport",
  "StreamingTV",
  "StreamingMovies",
] as const;

const baseCustomer = z.object({
  customer_id: z.string().max(64).optional(),
  gender: z.enum(["Male", "Female"]),
  SeniorCitizen: z.union([z.literal(0), z.literal(1)]),
  Partner: yesNo,
  Dependents: yesNo,
  tenure: z.number().int().min(0).max(120),
  PhoneService: yesNo,
  MultipleLines: z.enum(["Yes", "No", "No phone service"]),
  InternetService: z.enum(["DSL", "Fiber optic", "No"]),
  OnlineSecurity: internetAddon,
  OnlineBackup: internetAddon,
  DeviceProtection: internetAddon,
  TechSupport: internetAddon,
  StreamingTV: internetAddon,
  StreamingMovies: internetAddon,
  Contract: z.enum(["Month-to-month", "One year", "Two year"]),
  PaperlessBilling: yesNo,
  PaymentMethod: z.enum([
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
  ]),
  MonthlyCharges: z.number().min(0).max(1_000),
  TotalCharges: z.number().min(0).max(100_000),
});

export const customerSchema = baseCustomer.strict().superRefine((c, ctx) => {
  if ((c.PhoneService === "No") !== (c.MultipleLines === "No phone service")) {
    ctx.addIssue({
      code: "custom",
      path: ["MultipleLines"],
      message: "Must be 'No phone service' exactly when PhoneService is 'No'",
    });
  }
  const noInternet = c.InternetService === "No";
  for (const field of INTERNET_ADDONS) {
    if ((c[field] === "No internet service") !== noInternet) {
      ctx.addIssue({
        code: "custom",
        path: [field],
        message: "Must be 'No internet service' exactly when InternetService is 'No'",
      });
    }
  }
});

export type Customer = z.infer<typeof baseCustomer>;
export type FeatureName = Exclude<keyof Customer, "customer_id">;

/** The 19 model inputs, in the order the training data uses. */
export const FEATURE_NAMES = Object.keys(baseCustomer.shape).filter(
  (k) => k !== "customer_id",
) as FeatureName[];

/** Allowed values of each categorical field, read from the schema itself. */
export function optionsFor(field: FeatureName): readonly (string | number)[] {
  if (field === "SeniorCitizen") return [0, 1];
  const schema = baseCustomer.shape[field];
  return schema instanceof z.ZodEnum ? (schema.options as string[]) : [];
}

/**
 * Make dependent fields agree with the fields they depend on, the way the form expects:
 * turning phone or internet service off forces the dependent answers; turning it back on
 * resets them to a plain "No".
 */
export function normalise(c: Customer): Customer {
  const next = { ...c };
  if (next.PhoneService === "No") next.MultipleLines = "No phone service";
  else if (next.MultipleLines === "No phone service") next.MultipleLines = "No";

  for (const field of INTERNET_ADDONS) {
    if (next.InternetService === "No") next[field] = "No internet service";
    else if (next[field] === "No internet service") next[field] = "No";
  }
  return next;
}

/** Model inputs only: drops customer_id, like the API's CustomerFeatures.features(). */
export function features(c: Customer): Omit<Customer, "customer_id"> {
  const { customer_id: _ignored, ...rest } = c;
  void _ignored;
  return rest;
}
