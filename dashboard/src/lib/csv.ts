import { customerSchema, FEATURE_NAMES, type Customer } from "./customer";

/** RFC 4180 CSV: quoted fields, "" escapes, commas and newlines inside quotes, CRLF, BOM. */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;
  const src = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;

  for (let i = 0; i < src.length; i++) {
    const ch = src[i];
    if (inQuotes) {
      if (ch === '"') {
        if (src[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && src[i + 1] === "\n") i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field !== "" || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  // Drop blank lines (a trailing newline, or empty lines in the middle).
  return rows.filter((r) => !(r.length === 1 && r[0]!.trim() === ""));
}

function escapeField(value: string | number | boolean): string {
  const s = String(value);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function toCsv(header: string[], rows: (string | number | boolean)[][]): string {
  return [header, ...rows].map((r) => r.map(escapeField).join(",")).join("\r\n") + "\r\n";
}

export interface ParsedRow {
  /** 1-based line number in the file, counting the header as line 1. */
  line: number;
  customer: Customer;
  /** Known outcome, when the file has a Churn column (the training data does). */
  actual?: boolean;
}

export interface RowError {
  line: number;
  message: string;
}

export interface ParseResult {
  rows: ParsedRow[];
  errors: RowError[];
  missingColumns: string[];
  hasOutcome: boolean;
}

const NUMERIC = new Set(["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]);

/**
 * Turn CSV text in the Telco format into validated customers. Accepts the raw Kaggle file
 * as-is: its `customerID` column becomes customer_id, a `Churn` column is read as the
 * known outcome, and a blank TotalCharges (new customers, tenure 0) becomes 0.
 */
export function parseCustomers(text: string): ParseResult {
  const [header, ...data] = parseCsv(text);
  if (!header) return { rows: [], errors: [], missingColumns: [...FEATURE_NAMES], hasOutcome: false };

  const index = new Map(header.map((h, i) => [h.trim(), i]));
  const missingColumns = FEATURE_NAMES.filter((f) => !index.has(f));
  const idColumn = index.get("customerID") ?? index.get("customer_id");
  const outcomeColumn = index.get("Churn");
  const result: ParseResult = {
    rows: [],
    errors: [],
    missingColumns,
    hasOutcome: outcomeColumn !== undefined,
  };
  if (missingColumns.length > 0) return result;

  data.forEach((cells, i) => {
    const line = i + 2;
    const raw: Record<string, unknown> = {};
    for (const name of FEATURE_NAMES) {
      const value = (cells[index.get(name)!] ?? "").trim();
      if (NUMERIC.has(name)) {
        if (value === "" && name === "TotalCharges") {
          raw[name] = 0;
        } else {
          raw[name] = value === "" ? Number.NaN : Number(value);
        }
      } else {
        raw[name] = value;
      }
    }
    if (idColumn !== undefined && cells[idColumn]?.trim()) {
      raw.customer_id = cells[idColumn]!.trim();
    }

    const parsed = customerSchema.safeParse(raw);
    if (!parsed.success) {
      const issue = parsed.error.issues[0]!;
      const where = issue.path.length ? `${issue.path.join(".")}: ` : "";
      result.errors.push({ line, message: `${where}${issue.message}` });
      return;
    }

    const row: ParsedRow = { line, customer: parsed.data };
    if (outcomeColumn !== undefined) {
      const outcome = (cells[outcomeColumn] ?? "").trim().toLowerCase();
      if (["yes", "1", "true"].includes(outcome)) row.actual = true;
      else if (["no", "0", "false"].includes(outcome)) row.actual = false;
    }
    result.rows.push(row);
  });
  return result;
}
