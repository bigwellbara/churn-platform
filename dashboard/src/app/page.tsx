import { Scorer } from "@/components/Scorer";

export default function ScorePage() {
  return (
    <>
      <div className="mb-6 max-w-3xl">
        <h1 className="text-2xl font-semibold tracking-tight">Score a customer</h1>
        <p className="mt-2 text-ink-2">
          How likely is this telecom customer to cancel, and which retention offer would help most?
          Every change re-scores the customer with the live model.
        </p>
      </div>
      <Scorer />
    </>
  );
}
