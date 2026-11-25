# Monitoring and drift detection

```
 API :8000/metrics ──┐
                     ├──▶ Prometheus :9090 ──▶ Grafana :3000
 drift :8001/metrics ┘    (scrapes every 5 s,    (dashboard provisioned
                           evaluates alerts)      from infra/grafana/)
```

Everything is configuration in git: `infra/prometheus/` (scrape targets and alert rules)
and `infra/grafana/` (data source and dashboard JSON). `docker compose up -d` gives a
working, pre-built dashboard with no clicking.

## What is measured

| Metric | Type | Why |
|---|---|---|
| `churn_http_requests_total{method,path,status}` | counter | traffic and error rate |
| `churn_http_request_duration_seconds{method,path}` | histogram | latency percentiles (p50/p95/p99) |
| `churn_model_inference_seconds` | histogram | time in the model per call: separates "model slow" from "server slow" |
| `churn_event_loop_lag_seconds` | histogram | detects anything blocking the async server |
| `churn_micro_batch_size` | histogram | how many requests share each model call |
| `churn_predictions_total{outcome}` | counter | churn vs stay predictions over time |
| `churn_prediction_probability` | histogram | distribution of model scores |
| `churn_prediction_log_queue_depth`, `..._dropped_total`, `churn_db_write_errors_total` | gauge/counters | health of the prediction log |
| `churn_model_info{version,model_type}` | gauge | which model is serving |
| `churn_feature_psi{feature}`, `churn_drift_max_psi`, `churn_drift_features_significant` | gauges | input drift (from the drift monitor) |

Labels are deliberately low-cardinality: route templates (`/predict`), never raw URLs or
customer IDs, because every distinct label combination is a separate time series.

## Drift detection

**Why:** a model's accuracy degrades silently when incoming customers stop resembling
the training data. Real outcomes (did they churn?) arrive months later, so we monitor the
*inputs* and the model's *score distribution* instead.

**How (PSI, Population Stability Index):**

1. At export time `churn-export` writes `reference_profile.json` into the model bundle:
   decile bins for numeric features and category shares for categorical ones, from the
   training split, plus the model's score distribution on the validation split.
2. The `drift` container (`churn-drift --serve`, the same image as the API) reads the last
   500 logged predictions **for the serving model version** every 15 s, buckets them the
   same way, and computes
   `PSI = Σ (actual% − expected%) · ln(actual% / expected%)` per feature.
3. Thresholds: **< 0.10 stable, 0.10–0.25 moderate, > 0.25 significant drift**.
   Alert `DataDriftDetected` fires when any input feature stays above 0.25 for 1 minute.

**Sample size matters.** PSI has a noise floor of roughly
`(bins − 1) × (1/N_reference + 1/N_window)`. With 4,225 training rows and a 500-row
window that is about 0.02, comfortably below the 0.10 threshold. The monitor refuses to
report on fewer than 100 rows, where noise alone can exceed 0.25. (A unit test using a
tiny resampled dataset produced a false 0.29 alarm before this was accounted for.)

## Drift demo (2026-11-25, local stack)

```bash
uv run churn-simulate --mode normal  --n 600    # real held-out customers
uv run churn-simulate --mode shifted --n 600    # marketing push + 25% price rise
docker compose exec drift churn-drift --once    # PSI table
```

| | Normal traffic | Shifted traffic |
|---|---|---|
| Max input PSI | **0.046** (tenure), all stable | **1.231** (MonthlyCharges) |
| Significant features | none | MonthlyCharges 1.23 · tenure 0.73 · TotalCharges 0.54 · Contract 0.30 · model score 0.41 |
| Untouched features (gender, SeniorCitizen, ...) | stable | still stable (≤ 0.009) |
| Share flagged as churn | 32.2% (test set: 33.2%) | **55.5%** |
| `DataDriftDetected` alert | inactive | **firing** after the 1-minute `for:` window |

The monitor pointed at exactly the features the scenario changed and left the other 14
stable. That precision is what makes an alert actionable.

## Alert rules (`infra/prometheus/alerts.yml`)

| Alert | Condition | Severity |
|---|---|---|
| DataDriftDetected | max input PSI > 0.25 for 1 m | warning |
| PredictionLoggingFailing | any failed log write in 5 m | warning |
| HighLatencyP95 | `/predict` p95 > 100 ms for 5 m | warning |
| HighErrorRate | 5xx > 5% of requests for 2 m | critical |
| ServiceDown | API or drift monitor not scrapeable for 1 m | critical |

Validate with: `docker run --rm --entrypoint promtool -v ./infra/prometheus:/etc/prometheus prom/prometheus:v3.15.1 check config /etc/prometheus/prometheus.yml`

## Runbook: DataDriftDetected

1. Open the Grafana panel **Drift: PSI per feature** to see which features moved.
2. Check whether it is a data problem (a broken upstream field, a new category from a
   product launch) or a real population change (pricing, marketing campaign).
3. Check the **model score PSI** and the flagged share: is the model now flagging far more
   or fewer customers?
4. Real change: retrain on recent data (`churn-train`), compare against the current
   champion, export, deploy. Data bug: fix upstream; the model is fine.
