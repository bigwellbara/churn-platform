// Generates public/sample-customers.csv: 40 synthetic customers in the Telco format, for
// trying the "Score a list" page. Synthetic on purpose: the real dataset is not committed
// to this repository (see data/README.md). Seeded, so the file is reproducible:
//   node scripts/make-sample.ts
import { writeFileSync } from "node:fs";

let seed = 20261126;
function rand(): number {
  // mulberry32
  seed = (seed + 0x6d2b79f5) | 0;
  let t = seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
const pick = <T,>(xs: readonly T[]): T => xs[Math.floor(rand() * xs.length)]!;

const header = [
  "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
  "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
  "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
  "PaymentMethod", "MonthlyCharges", "TotalCharges",
];

const lines = [header.join(",")];
for (let i = 1; i <= 40; i++) {
  const contract = pick(["Month-to-month", "Month-to-month", "One year", "Two year"] as const);
  const tenure =
    contract === "Month-to-month" ? Math.floor(rand() * 30) : 12 + Math.floor(rand() * 60);
  const internet = pick(["DSL", "Fiber optic", "Fiber optic", "No"] as const);
  const phone = rand() < 0.9 ? "Yes" : "No";
  const addon = () => (internet === "No" ? "No internet service" : rand() < 0.4 ? "Yes" : "No");
  const addons = Array.from({ length: 6 }, addon);
  const base = internet === "Fiber optic" ? 70 : internet === "DSL" ? 45 : 20;
  const monthly =
    Math.round((base + addons.filter((a) => a === "Yes").length * 5 + rand() * 10) * 100) / 100;
  const total = Math.round(monthly * Math.max(tenure, 0) * 100) / 100;

  lines.push(
    [
      `SAMPLE-${String(i).padStart(3, "0")}`,
      pick(["Female", "Male"]),
      rand() < 0.16 ? 1 : 0,
      pick(["Yes", "No"]),
      rand() < 0.3 ? "Yes" : "No",
      tenure,
      phone,
      phone === "No" ? "No phone service" : pick(["Yes", "No"]),
      internet,
      ...addons,
      contract,
      pick(["Yes", "No"]),
      pick(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
      monthly.toFixed(2),
      tenure === 0 ? " " : total.toFixed(2),
    ].join(","),
  );
}

const out = new URL("../public/sample-customers.csv", import.meta.url);
writeFileSync(out, lines.join("\r\n") + "\r\n", "utf-8");
console.log(`Wrote ${lines.length - 1} customers to ${out.pathname}`);
