// Regenerate src/data/model-report.json after retraining or exporting a new champion:
//   npm run sync-report
import { readFileSync, writeFileSync } from "node:fs";
import { buildReport } from "../src/lib/report.ts";

const root = new URL("../../", import.meta.url);
const read = (path: string) => JSON.parse(readFileSync(new URL(path, root), "utf-8"));

const report = buildReport(read("reports/metrics.json"), read("models/champion/model_meta.json"));
const out = new URL("dashboard/src/data/model-report.json", root);
writeFileSync(out, JSON.stringify(report, null, 2) + "\n", "utf-8");
console.log(`Wrote ${out.pathname}: ${report.modelType} v${report.modelVersion}`);
