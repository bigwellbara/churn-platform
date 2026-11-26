import type { Metadata } from "next";
import { BatchScorer } from "@/components/BatchScorer";

export const metadata: Metadata = { title: "Score a list" };

export default function BatchPage() {
  return (
    <>
      <div className="mb-6 max-w-3xl">
        <h1 className="text-2xl font-semibold tracking-tight">Score a customer list</h1>
        <p className="mt-2 text-ink-2">
          Upload a CSV of customers and get them back ranked by churn risk: the list a retention
          team would call first. The file is read in your browser and only the customer fields
          are sent to the model.
        </p>
      </div>
      <BatchScorer />
    </>
  );
}
