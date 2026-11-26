# Churn Prediction Platform

![CI](https://github.com/bigwellbara/churn-platform/actions/workflows/ci.yml/badge.svg)
![Deploy](https://github.com/bigwellbara/churn-platform/actions/workflows/deploy.yml/badge.svg)
![Uptime](https://github.com/bigwellbara/churn-platform/actions/workflows/uptime.yml/badge.svg)

An end-to-end, production-style machine learning system that predicts which telecom
customers are likely to cancel (churn). It covers the full path from notebook to monitored,
deployed service: training pipeline, experiment tracking, a model registry, a FastAPI service,
Docker, monitoring with drift detection, CI/CD, cloud deployment and a Next.js dashboard.

> Status: **Week 4 complete**: model, prediction API, Docker, monitoring, drift detection,
> load testing, and CI/CD with automatic rollback. See the [roadmap](#roadmap).

## Problem

Acquiring a new customer costs far more than keeping an existing one. If the business knows
*which* customers are at risk, it can target retention offers at them instead of everyone.
The model outputs a churn probability per customer and a yes/no flag based on a tuned
decision threshold.

Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(7,043 customers, 26.5% churn). It is not committed; see [data/README.md](data/README.md).

## Architecture

```
          ┌────────────┐    log runs     ┌──────────────┐
 CSV ───▶ │  Training  │ ──────────────▶ │    MLflow    │
          │  pipeline  │  register best  │  tracking +  │
          └────────────┘ ──────────────▶ │   registry   │
                                         └──────┬───────┘
                          churn-export (pinned version → models/champion/)
                                                ▼
                                    Docker image = code + model bundle
                                                ▼
 Next.js dashboard ──HTTP──▶  FastAPI service ──▶ Postgres (prediction log)
                                     │  /metrics
                                     ▼
                         Prometheus ──▶ Grafana  (+ drift job)
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.11 automatically.

```bash
uv sync                       # create .venv and install dependencies
# put the dataset at data/raw/telco_churn.csv
uv run churn-train            # train, compare, register the best model
uv run churn-train --tune     # plus a time-boxed XGBoost hyperparameter search
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse experiments at :5000
uv run pytest                 # tests (use synthetic data, no download needed)
uv run churn-export           # copy the @champion model into models/champion/ for serving
```

### Run the API

With Docker (API + Postgres, one command):

```bash
docker compose up -d --build
# open http://localhost:8000/docs for interactive API docs
```

Without Docker: `uv run uvicorn churn.api.main:create_app --factory --reload`

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Score one customer |
| `POST /predict/batch` | Score up to 1,000 customers in one call |
| `GET /model` | Serving model version, threshold, test metrics |
| `GET /health` | Liveness, model version, database status |
| `GET /metrics` | Prometheus metrics |

```bash
curl -s localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
  "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85}'
```

Configuration is by environment variable: `MODEL_DIR`, `DATABASE_URL`, `API_KEY`
(require an `X-API-Key` header), `LOG_LEVEL`, `MAX_BATCH_SIZE`, `INFERENCE_PROCESSES`,
`MICRO_BATCHING`, `LOG_FLUSH_INTERVAL`.

### Monitoring

`docker compose up -d` also starts the drift monitor, Prometheus (http://localhost:9090,
alerts at `/alerts`) and Grafana (http://localhost:3000, dashboard provisioned from
`infra/grafana/`). Try the drift demo:

```bash
uv run churn-simulate --mode normal  --n 600   # real held-out customers: all PSI < 0.10
uv run churn-simulate --mode shifted --n 600   # price rise + marketing push: alert fires
docker compose exec drift churn-drift --once   # PSI per feature
```

Details: [docs/MONITORING.md](docs/MONITORING.md). Performance work:
[docs/LOAD_TEST.md](docs/LOAD_TEST.md).

### CI/CD and deployment

Every merge to `main` that passes CI is built once, published to GitHub Container
Registry, and deployed to **Render** (free tier). A **Google Cloud Run** pipeline is
ready too, and runs once a GCP project is configured.

- **CI** runs lint, tests and a **model quality gate** (`churn-gate`: minimum test metrics
  plus behavioural checks on known customers). It also builds the production image and
  smoke tests it at a 512 MB memory limit.
- **Deploy** releases the image **by digest** and waits until the new commit is live.
  It then smoke tests real predictions and **rolls back automatically** to the previous
  image if any check fails. On Cloud Run, the new revision gets **0% traffic** until it
  passes on a private URL.
- **Uptime** (optional) runs the smoke test against production every 10 minutes.
- **No stored cloud keys:** images are pushed with the job's own short-lived token.
  Cloud Run uses OIDC Workload Identity Federation, limited to this repo's `main` branch.

```bash
uv run churn-gate                                 # would this model be allowed to ship?
python src/churn/smoke.py https://<service-url>   # check any running deployment
```

Details, including the reasoning behind each setting and the costs:
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Results

Tuned XGBoost, held-out test set (1,409 customers), decision threshold 0.606:

| ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| **0.847** | **0.663** (random: 0.265) | **72.5%** | 57.9% | 0.644 |

The model catches 271 of 374 churners. The biggest drivers are short tenure,
month-to-month contracts and fibre internet. Full analysis is in the
[EDA notebook](notebooks/01_eda.ipynb), and limitations and the fairness check are in the
[model card](docs/MODEL_CARD.md).

**Serving performance** (50 req/s, 0 errors): profiling-driven optimisation cut `/predict`
latency from p50 1,400 ms / p95 1,700 ms to **p50 130 ms / p95 280 ms** via dynamic
micro-batching, a write-behind prediction log, and running the model in worker processes
to avoid GIL contention. Every step was measured, and the rejected hypotheses are listed
too: [docs/LOAD_TEST.md](docs/LOAD_TEST.md).

**Drift detection:** a simulated price rise and marketing push was flagged within a minute
(max PSI 1.23 on the changed features, every untouched feature stable):
[docs/MONITORING.md](docs/MONITORING.md).

## Training in one paragraph

Four models are compared: a no-skill **dummy baseline**, **logistic regression**,
**random forest** and **XGBoost**. Each is wrapped in a single scikit-learn Pipeline with
the same feature engineering and preprocessing, then evaluated with 5-fold cross-validation
and on a held-out validation set. Class imbalance is handled with class weights. The decision
threshold is tuned on validation for F1. The winner is chosen by validation **PR-AUC**, scored
once on the untouched test set, and registered in MLflow as `churn-model@champion`.
Full explanation: **[docs/TRAINING.md](docs/TRAINING.md)**.

## Repository layout

```
src/churn/
  config.py          column names, paths, settings (single source of truth)
  bundle.py          model bundle format shared by training and serving
  training/          data cleaning, features, models, evaluation, train + export CLIs
  api/               FastAPI app, schemas, micro-batching, worker-process inference,
                     write-behind Postgres log, Prometheus metrics, JSON logs
  monitoring/        PSI drift detection, drift monitor service, traffic simulator
  gate.py            model quality gate run in CI (churn-gate)
  smoke.py           deployment smoke test (stdlib only)
models/champion/     exported bundle: model.skops, metadata, drift reference profile
infra/               Prometheus config + alert rules, Grafana provisioning + dashboard,
                     gcp/bootstrap.sh (one-time cloud setup)
.github/workflows/   ci.yml (tests, model gate, image smoke test), deploy.yml (GHCR, Render,
                     Cloud Run), uptime.yml (scheduled production smoke test)
loadtest/            Locust load test, benchmarks, sampling profiler
tests/               pytest suite (synthetic data)
notebooks/           01_eda.ipynb: data exploration, model insights, fairness check
docs/                TRAINING, MODEL_CARD, MONITORING, LOAD_TEST, DEPLOYMENT
reports/             metrics.json from the latest training run
```

## Roadmap

- [x] Week 1: project setup, training pipeline, MLflow tracking and registry, tests, CI
- [x] Week 1: EDA notebook on the real dataset, first real training run, model card
- [x] Week 2: FastAPI service, Postgres logging, Docker and docker-compose
- [x] Week 3: Prometheus, Grafana, drift detection, load testing
- [ ] Grafana dashboard screenshots (image download blocked by network issues on build day)
- [x] Week 4: model quality gate, image smoke tests, CD to Render with auto-rollback,
      Cloud Run pipeline (keyless), uptime checks
- [ ] Cloud prediction log + drift monitor (managed Postgres; see DEPLOYMENT.md)
- [ ] Week 5: Next.js dashboard, docs, demo video
