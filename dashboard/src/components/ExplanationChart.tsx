import { strength, topFactors, type Contribution } from "@/lib/explain";

/**
 * Diverging bars around a zero line: right (red) raises churn risk, left (blue) lowers it.
 * Bar length is the factor's push on the score, on one shared scale, so rows compare
 * directly. Every row is labelled in words; colour never carries the meaning alone.
 */
export function ExplanationChart({ contributions }: { contributions: Contribution[] }) {
  const rows = topFactors(contributions, 6);
  const scale = Math.max(...rows.map((r) => Math.abs(r.impact)), 0.05);

  return (
    <div>
      <div className="mb-2 hidden text-xs text-ink-2 sm:grid sm:grid-cols-[minmax(0,1fr)_12rem_7.5rem] sm:gap-x-3">
        <span />
        <span className="flex justify-between">
          <span>← lowers risk</span>
          <span>raises risk →</span>
        </span>
        <span />
      </div>
      <ul className="space-y-2.5">
        {rows.map((row) => {
          const up = row.impact >= 0;
          const width = (Math.abs(row.impact) / scale) * 50;
          const words = strength(row.impact);
          return (
            <li
              key={row.key}
              title={`${row.label}: ${words} risk (${row.impact >= 0 ? "+" : "−"}${Math.abs(row.impact).toFixed(2)} log-odds)`}
              className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-1 sm:grid-cols-[minmax(0,1fr)_12rem_7.5rem] sm:items-center"
            >
              <span className={`text-sm ${row.key === "other" ? "text-ink-2" : "text-ink"}`}>{row.label}</span>
              <span className="order-3 col-span-2 sm:order-none sm:col-span-1" aria-hidden="true">
                <span className="relative block h-2.5">
                  <span className="absolute inset-y-0 left-1/2 w-px bg-baseline" />
                  <span
                    className={`absolute inset-y-0 ${up ? "rounded-r-full bg-up" : "rounded-l-full bg-down"}`}
                    style={up ? { left: "50%", width: `${width}%` } : { right: "50%", width: `${width}%` }}
                  />
                </span>
              </span>
              <span className={`text-right text-xs sm:text-sm ${up ? "text-risk-high-ink" : "text-ink-2"}`}>
                {words}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
