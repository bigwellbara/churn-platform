# Churn Prediction Platform

![CI](https://github.com/bigwellbara/churn-platform/actions/workflows/ci.yml/badge.svg)

An end-to-end, production-style machine learning system that predicts which telecom
customers are likely to cancel (churn). It covers the full path from notebook to monitored,
deployed service: training pipeline, experiment tracking, a model registry, a FastAPI service,
Docker, monitoring with drift detection, CI/CD, cloud deployment and a Next.js dashboard.

> Status: **Week 2 complete**: model, prediction API, Postgres logging, Docker. See the
> [roadmap](#roadmap).

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
(require an `X-API-Key` header), `LOG_LEVEL`, `MAX_BATCH_SIZE`.

## Results

Tuned XGBoost, held-out test set (1,409 customers), decision threshold 0.606:

| ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| **0.847** | **0.663** (random: 0.265) | **72.5%** | 57.9% | 0.644 |

The model catches 271 of 374 churners. The biggest drivers are short tenure,
month-to-month contracts and fibre internet. Full analysis is in the
[EDA notebook](notebooks/01_eda.ipynb), and limitations and the fairness check are in the
[model card](docs/MODEL_CARD.md).

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
  api/               FastAPI app, schemas, model service, Postgres log, metrics, JSON logs
  monitoring/        drift detection                          (week 3)
models/champion/     exported model bundle served by the API (model.skops + metadata)
tests/               pytest suite (synthetic data)
notebooks/           01_eda.ipynb: data exploration, model insights, fairness check
docs/                TRAINING.md (how training works), MODEL_CARD.md
reports/             metrics.json from the latest training run
```

## Roadmap

- [x] Week 1: project setup, training pipeline, MLflow tracking and registry, tests, CI
- [x] Week 1: EDA notebook on the real dataset, first real training run, model card
- [x] Week 2: FastAPI service, Postgres logging, Docker and docker-compose
- [ ] Week 3: Prometheus, Grafana, drift detection, load testing
- [ ] Week 4: CD pipeline and cloud deployment
- [ ] Week 5: Next.js dashboard, docs, demo video
