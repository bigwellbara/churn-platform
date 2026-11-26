import type { Metadata } from "next";
import Link from "next/link";
import { ApiStatus } from "@/components/ApiStatus";
import { Nav } from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Churn Platform", template: "%s · Churn Platform" },
  description:
    "Score telecom customers for churn risk, rank a customer list, and see how the model was chosen.",
};

const REPO = "https://github.com/bigwellbara/churn-platform";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <header className="border-b border-line bg-surface">
          <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
            <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
              <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="12" cy="12" r="10" fill="none" stroke="var(--accent)" strokeWidth="2" />
                <path d="M7 14l3-3 3 2 4-5" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Churn Platform
            </Link>
            <Nav />
            <div className="ml-auto">
              <ApiStatus />
            </div>
          </div>
        </header>

        <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>

        <footer className="mx-auto max-w-6xl px-4 pb-10 text-sm text-ink-2">
          <div className="flex flex-wrap gap-x-5 gap-y-1 border-t border-line pt-5">
            <span>Predictions from an XGBoost model served by a FastAPI service.</span>
            <a className="underline hover:text-ink" href={REPO}>
              Source code
            </a>
            <a className="underline hover:text-ink" href={`${REPO}/blob/main/docs/MODEL_CARD.md`}>
              Model card
            </a>
          </div>
        </footer>
      </body>
    </html>
  );
}
