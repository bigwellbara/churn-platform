# Churn dashboard

A Next.js front end for the churn prediction API, made for the people who would use the
model: a retention team, not engineers.

| Page | What it does |
|---|---|
| **Score a customer** (`/`) | Edit a customer and watch the churn probability update. **Why this score?** shows the answers that pushed it up or down most (exact SHAP values from `POST /predict/explain`). The model also scores each retention offer (longer contract, autopay, support add-ons) for the same customer, so you can see which offer lowers the risk most. |
| **Score a list** (`/batch`) | Upload a CSV in the Telco format and get it back ranked by risk, with a summary and a download. The original Kaggle file works as-is; if it has a `Churn` column, the page shows how many real leavers the model caught. `/batch?sample=1` runs it on built-in sample data. |
| **The model** (`/model`) | Test-set metrics in plain words, the confusion matrix, why the threshold is 61%, and how the model was chosen from five candidates. |

## How it talks to the API

```
browser ──▶ Next.js /api/predict, /api/predict/batch, /api/predict/explain, /api/health ──▶ FastAPI (Render)
```

The browser never calls the prediction API directly. The dashboard's own route handlers
forward each request ([`src/lib/upstream.ts`](src/lib/upstream.ts)), which means:

- the API needs no CORS configuration;
- if the API has a key (`API_KEY`), it stays on the server (`CHURN_API_KEY`), never in
  page code;
- timeouts and outages become clear messages. A free Render instance can take up to two
  minutes to wake; the page says so instead of hanging.

The API stays the single source of truth for validation. The dashboard checks input with a
zod copy of the schema ([`src/lib/customer.ts`](src/lib/customer.ts)) for instant
feedback, and the API validates again.

The Model page is static: it is built from
[`src/data/model-report.json`](src/data/model-report.json), a snapshot of
`reports/metrics.json` and `models/champion/model_meta.json`. A unit test fails if the
snapshot drifts from those files; `npm run sync-report` regenerates it after retraining.

## Run it

```bash
cd dashboard
cp .env.example .env.local      # points at the live API by default
npm ci
npm run dev                     # http://localhost:3000
```

To use a local API instead: `docker compose up` in the repository root, then set
`CHURN_API_URL=http://localhost:8000` in `.env.local`.

| Command | |
|---|---|
| `npm test` | unit tests (vitest): CSV parsing, schema rules, retention offers, the proxy, report sync |
| `npm run lint` / `npm run typecheck` | ESLint, TypeScript |
| `npm run build` | production build |
| `npm run sync-report` | regenerate the Model page data after retraining |

## Deploy (Vercel, free)

1. On [vercel.com](https://vercel.com), **Add New → Project**, import this repository.
2. Set **Root Directory** to `dashboard`. The framework is detected as Next.js.
3. Add the environment variable `CHURN_API_URL` =
   `https://churn-api-latest-ofya.onrender.com` (and `CHURN_API_KEY` if the API uses one).
4. Deploy. Every push to `main` redeploys; pull requests get preview URLs.

## Supply chain

`.npmrc` sets `min-release-age=14`: npm only installs package versions published at least
14 days ago, the same policy as the Python side (`exclude-newer` in `pyproject.toml`) and
Dependabot's cooldown.
