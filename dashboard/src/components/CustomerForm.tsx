"use client";

import { INTERNET_ADDONS, optionsFor, type Customer, type FeatureName } from "@/lib/customer";
import { FIELD_GROUPS, FIELD_INFO } from "@/lib/fields";

const NUMBER_LIMITS: Partial<Record<FeatureName, { min: number; max: number; step: number }>> = {
  tenure: { min: 0, max: 120, step: 1 },
  MonthlyCharges: { min: 0, max: 1000, step: 0.05 },
  TotalCharges: { min: 0, max: 100000, step: 0.05 },
};

interface Props {
  value: Customer;
  onChange: (field: FeatureName, value: string | number) => void;
  autoTotal: boolean;
}

function isLocked(field: FeatureName, c: Customer): boolean {
  if (field === "MultipleLines") return c.PhoneService === "No";
  if ((INTERNET_ADDONS as readonly string[]).includes(field)) return c.InternetService === "No";
  return false;
}

export function CustomerForm({ value, onChange, autoTotal }: Props) {
  return (
    <div className="space-y-6">
      {FIELD_GROUPS.map((group) => (
        <fieldset key={group.title}>
          <legend className="mb-3 text-xs font-semibold uppercase tracking-wider text-ink-2">
            {group.title}
          </legend>
          <div className="grid grid-cols-1 gap-x-4 gap-y-3 sm:grid-cols-2">
            {group.fields.map((field) => {
              const info = FIELD_INFO[field];
              const id = `field-${field}`;
              const limits = NUMBER_LIMITS[field];
              const locked = isLocked(field, value);
              const inputClass =
                "w-full rounded-md border border-line bg-surface px-2.5 py-1.5 text-sm text-ink disabled:bg-surface-2 disabled:text-ink-2";

              return (
                <div key={field}>
                  <label htmlFor={id} className="mb-1 block text-sm text-ink">
                    {info.label}
                  </label>
                  {limits ? (
                    <input
                      id={id}
                      type="number"
                      inputMode="decimal"
                      className={`${inputClass} tabular`}
                      min={limits.min}
                      max={limits.max}
                      step={limits.step}
                      value={Number.isNaN(value[field] as number) ? "" : (value[field] as number)}
                      onChange={(e) =>
                        onChange(field, e.target.value === "" ? Number.NaN : Number(e.target.value))
                      }
                    />
                  ) : (
                    <select
                      id={id}
                      className={inputClass}
                      value={String(value[field])}
                      disabled={locked}
                      onChange={(e) =>
                        onChange(field, field === "SeniorCitizen" ? Number(e.target.value) : e.target.value)
                      }
                    >
                      {optionsFor(field)
                        .filter((o) => locked || !String(o).startsWith("No ") || String(o) === "No")
                        .map((o) => (
                          <option key={String(o)} value={String(o)}>
                            {info.display ? info.display(o) : String(o)}
                          </option>
                        ))}
                    </select>
                  )}
                  {field === "TotalCharges" && (
                    <p className="mt-1 text-xs text-ink-2">
                      {autoTotal ? "Calculated as tenure × monthly. Edit to override." : info.help}
                    </p>
                  )}
                  {field !== "TotalCharges" && info.help && (
                    <p className="mt-1 text-xs text-ink-2">{info.help}</p>
                  )}
                </div>
              );
            })}
          </div>
        </fieldset>
      ))}
    </div>
  );
}
