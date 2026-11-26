import type { RiskLevel } from "@/lib/predictions";

const RISK = {
  low: { label: "Low risk", ink: "text-risk-low-ink", dot: "var(--risk-low)" },
  medium: { label: "Medium risk", ink: "text-risk-medium-ink", dot: "var(--risk-medium)" },
  high: { label: "High risk", ink: "text-risk-high-ink", dot: "var(--risk-high)" },
} as const;

function Icon({ level }: { level: RiskLevel }) {
  const color = RISK[level].dot;
  if (level === "low") {
    return (
      <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
        <circle cx="8" cy="8" r="7" fill={color} />
        <path d="M4.8 8.2l2.1 2.1 4.3-4.6" fill="none" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  if (level === "medium") {
    return (
      <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
        <circle cx="8" cy="8" r="7" fill={color} />
        <path d="M8 4.3v4.4" stroke="#0b0b0b" strokeWidth="1.8" strokeLinecap="round" />
        <circle cx="8" cy="11.3" r="1" fill="#0b0b0b" />
      </svg>
    );
  }
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
      <path d="M8 1.5l7 12.5H1z" fill={color} />
      <path d="M8 6v3.6" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" />
      <circle cx="8" cy="11.9" r="0.95" fill="#fff" />
    </svg>
  );
}

export function RiskBadge({ level, size = "sm" }: { level: RiskLevel; size?: "sm" | "lg" }) {
  const r = RISK[level];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border border-line bg-surface font-medium ${r.ink} ${
        size === "lg" ? "px-3 py-1 text-sm" : "px-2 py-0.5 text-xs"
      }`}
    >
      <Icon level={level} />
      {r.label}
    </span>
  );
}
