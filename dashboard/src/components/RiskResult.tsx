import { percent, type Prediction } from "@/lib/predictions";
import { RiskBadge } from "./RiskBadge";

const EXPLAIN = {
  high: "Above the flag threshold: the model would put this customer on the retention list.",
  medium: "Below the flag threshold but at least half-way to it: worth keeping an eye on.",
  low: "Well below the flag threshold.",
};

export function RiskResult({ prediction, threshold }: { prediction: Prediction; threshold: number }) {
  const p = prediction.churn_probability;
  return (
    <div>
      <div className="flex flex-wrap items-end gap-x-4 gap-y-2">
        <div>
          <div className="text-6xl font-semibold leading-none tracking-tight">{percent(p)}</div>
          <div className="mt-2 text-sm text-ink-2">chance this customer leaves</div>
        </div>
        <div className="mb-1 ml-auto">
          <RiskBadge level={prediction.risk_level} size="lg" />
        </div>
      </div>

      {/* Meter: the probability against the 0-100% scale, with the flag threshold marked. */}
      <div className="mt-6" role="img" aria-label={`${percent(p)} churn probability; flag threshold ${percent(threshold)}`}>
        <div className="relative h-3 rounded-full bg-track">
          <div
            className="absolute inset-y-0 left-0 rounded-full bg-accent transition-[width] duration-300"
            style={{ width: `${p * 100}%` }}
          />
          <div
            className="absolute -top-1.5 -bottom-1.5 w-0.5 rounded bg-ink"
            style={{ left: `calc(${threshold * 100}% - 1px)` }}
          />
        </div>
        <div className="relative mt-1.5 h-4 text-xs text-ink-2">
          <span className="absolute left-0">0%</span>
          <span
            className="absolute -translate-x-1/2 whitespace-nowrap font-medium text-ink"
            style={{ left: `${threshold * 100}%` }}
          >
            flag at {percent(threshold)}
          </span>
          <span className="absolute right-0">100%</span>
        </div>
      </div>

      <p className="mt-4 text-sm text-ink-2">{EXPLAIN[prediction.risk_level]}</p>
    </div>
  );
}
