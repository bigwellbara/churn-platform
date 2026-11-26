import { percent } from "@/lib/predictions";

export interface LeverResult {
  id: string;
  label: string;
  probability: number;
}

/**
 * One row per retention offer: a bar for the churn probability with that offer, a tick
 * at today's probability and a dashed line at the flag threshold, so each row reads
 * "from here to there" without a separate axis.
 */
export function LeverChart({
  current,
  threshold,
  levers,
}: {
  current: number;
  threshold: number;
  levers: LeverResult[];
}) {
  if (levers.length === 0) {
    return (
      <p className="text-sm text-ink-2">
        Nothing to offer: this customer already has the longest contract, automatic payment and
        the support add-ons.
      </p>
    );
  }
  const rows = [...levers].sort((a, b) => a.probability - b.probability);

  return (
    <div>
      <ul className="space-y-3">
        {rows.map((row) => {
          const delta = row.probability - current;
          const points = Math.round(delta * 100);
          const lower = delta < 0;
          const summary = `${row.label}: ${percent(current)} to ${percent(row.probability)} (${
            points <= 0 ? "" : "+"
          }${points} points)`;
          return (
            <li key={row.id} title={summary} className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 sm:grid-cols-[10rem_1fr_6.75rem] sm:items-center">
              <span className="text-sm text-ink">{row.label}</span>
              <span className="order-3 col-span-2 sm:order-none sm:col-span-1">
                <span className="relative block h-2.5 rounded-full bg-track" aria-hidden="true">
                  <span
                    className={`absolute inset-y-0 left-0 rounded-full ${lower ? "bg-down" : "bg-up"}`}
                    style={{ width: `${row.probability * 100}%` }}
                  />
                  <span
                    className="absolute -top-1 -bottom-1 w-0.5 rounded bg-ink"
                    style={{ left: `calc(${current * 100}% - 1px)` }}
                  />
                  <span
                    className="absolute -top-1 -bottom-1 border-l-2 border-dashed border-ink-2"
                    style={{ left: `calc(${threshold * 100}% - 1px)` }}
                  />
                </span>
                <span className="sr-only">{summary}</span>
              </span>
              <span className="tabular text-right text-sm">
                <span className="font-medium">{percent(row.probability)}</span>{" "}
                <span className={lower ? "text-ink-2" : "text-risk-high-ink"}>
                  ({points <= 0 ? "−" : "+"}
                  {Math.abs(points)} pts)
                </span>
              </span>
            </li>
          );
        })}
      </ul>
      <div className="mt-4 flex flex-wrap gap-x-5 gap-y-1 text-xs text-ink-2">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-0.5 bg-ink" aria-hidden="true" /> today: {percent(current)}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 border-l-2 border-dashed border-ink-2" aria-hidden="true" /> flag threshold: {percent(threshold)}
        </span>
      </div>
    </div>
  );
}
