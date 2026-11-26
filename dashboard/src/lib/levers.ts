import type { Customer } from "./customer";

/**
 * A retention offer the business could make, expressed as the change it would make to the
 * customer's record. Scoring the changed record shows how the model's risk estimate moves.
 *
 * Only actionable fields are used (contract, payment, support add-ons), never tenure or
 * demographics, which a retention team cannot change. Charges are kept as they are: the
 * offer is assumed to come at no extra cost, which is how retention offers usually work.
 */
export interface Lever {
  id: string;
  label: string;
  change: Partial<Customer>;
}

const AUTOMATIC = ["Bank transfer (automatic)", "Credit card (automatic)"];

export function leversFor(c: Customer): Lever[] {
  const levers: Lever[] = [];

  if (c.Contract === "Month-to-month") {
    levers.push({ id: "one-year", label: "One-year contract", change: { Contract: "One year" } });
  }
  if (c.Contract !== "Two year") {
    levers.push({ id: "two-year", label: "Two-year contract", change: { Contract: "Two year" } });
  }
  if (!AUTOMATIC.includes(c.PaymentMethod)) {
    levers.push({
      id: "autopay",
      label: "Switch to automatic card payment",
      change: { PaymentMethod: "Credit card (automatic)" },
    });
  }
  if (c.InternetService !== "No") {
    if (c.OnlineSecurity === "No") {
      levers.push({
        id: "security",
        label: "Add online security",
        change: { OnlineSecurity: "Yes" },
      });
    }
    if (c.TechSupport === "No") {
      levers.push({ id: "support", label: "Add tech support", change: { TechSupport: "Yes" } });
    }
  }

  // Everything at once, using the strongest contract option, when there is more than one.
  if (levers.length > 1) {
    const combined = Object.assign(
      {},
      ...levers.filter((l) => l.id !== "one-year").map((l) => l.change),
    ) as Partial<Customer>;
    levers.push({ id: "all", label: "All of the above together", change: combined });
  }
  return levers;
}

export function applyLever(c: Customer, lever: Lever): Customer {
  return { ...c, ...lever.change };
}
