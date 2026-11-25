"""Prometheus metrics, exposed at GET /metrics and scraped by Prometheus (Week 3).

Label values are kept low-cardinality (route templates, not raw URLs; no customer IDs):
every unique label combination becomes a separate time series.
"""

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("churn_http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram(
    "churn_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTIONS = Counter(
    "churn_predictions_total", "Predictions served, by predicted class", ["outcome"]
)
PROBABILITY = Histogram(
    "churn_prediction_probability",
    "Distribution of predicted churn probabilities",
    buckets=tuple(round(i / 10, 1) for i in range(1, 11)),
)
BATCH_SIZE = Histogram(
    "churn_batch_size", "Customers per batch request", buckets=(1, 10, 50, 100, 250, 500, 1000)
)
DB_ERRORS = Counter("churn_db_write_errors_total", "Failed prediction-log writes")
MODEL_INFO = Gauge(
    "churn_model_info", "Currently loaded model (value is always 1)", ["version", "model_type"]
)
