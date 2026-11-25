"""Drift monitor: compares recently logged predictions with the training reference.

    churn-drift --once              print a PSI report and exit
    churn-drift --serve             run forever, exposing Prometheus metrics on --port

Runs as its own process/container so the statistics never slow down the API.
Reads DATABASE_URL and MODEL_DIR from the environment, like the API.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from prometheus_client import Gauge, start_http_server
from sqlalchemy import select
from sqlalchemy.orm import Session

from churn.api.db import PredictionLog, PredictionStore
from churn.api.log_config import configure_logging
from churn.api.settings import Settings
from churn.bundle import load_meta, load_reference
from churn.config import RAW_FEATURES
from churn.monitoring.drift import PREDICTION_KEY, SIGNIFICANT, drift_report, severity

logger = logging.getLogger("churn.drift")

FEATURE_PSI = Gauge(
    "churn_feature_psi", "PSI of each input (and the score) vs training", ["feature"]
)
MAX_PSI = Gauge("churn_drift_max_psi", "Largest input-feature PSI in the latest window")
SIGNIFICANT_FEATURES = Gauge(
    "churn_drift_features_significant", "Input features with PSI above the significant threshold"
)
WINDOW_ROWS = Gauge("churn_drift_window_rows", "Predictions in the evaluated window")
LAST_RUN = Gauge("churn_drift_last_run_timestamp_seconds", "Time of the last completed check")


def load_window(
    store: PredictionStore, model_version: str, window: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """The most recent `window` predictions made by this model version."""
    query = (
        select(PredictionLog.features, PredictionLog.churn_probability)
        .where(PredictionLog.model_version == model_version)
        .order_by(PredictionLog.id.desc())
        .limit(window)
    )
    with Session(store.engine) as session:
        rows = session.execute(query).all()
    features = pd.DataFrame([r.features for r in rows], columns=RAW_FEATURES)
    scores = np.asarray([r.churn_probability for r in rows], dtype=float)
    return features, scores


def check(
    store: PredictionStore,
    reference: dict[str, Any],
    model_version: str,
    window: int = 500,
    min_rows: int = 100,
) -> dict[str, float] | None:
    """One drift check. Updates the Prometheus gauges; returns PSI per feature, or None
    when there are too few predictions for a meaningful comparison."""
    features, scores = load_window(store, model_version, window)
    WINDOW_ROWS.set(len(features))
    if len(features) < min_rows:
        logger.info(
            "Not enough predictions for a drift check",
            extra={"fields": {"rows": len(features), "min_rows": min_rows}},
        )
        return None

    report = drift_report(reference, features, scores)
    inputs = {k: v for k, v in report.items() if k != PREDICTION_KEY}
    for feature, value in report.items():
        FEATURE_PSI.labels(feature).set(value)
    MAX_PSI.set(max(inputs.values()))
    SIGNIFICANT_FEATURES.set(sum(v >= SIGNIFICANT for v in inputs.values()))
    LAST_RUN.set(time.time())

    drifted = {k: round(v, 3) for k, v in report.items() if v >= SIGNIFICANT}
    logger.log(
        logging.WARNING if drifted else logging.INFO,
        "Drift check complete",
        extra={
            "fields": {
                "rows": len(features),
                "max_psi": round(max(inputs.values()), 4),
                "significant": drifted,
            }
        },
    )
    return report


def format_report(report: dict[str, float]) -> str:
    lines = [f"{'feature':<18}{'PSI':>8}  status", "-" * 38]
    for feature, value in sorted(report.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"{feature:<18}{value:>8.3f}  {severity(value)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PSI drift monitor for logged predictions.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Print one report and exit (default)")
    mode.add_argument("--serve", action="store_true", help="Run forever, expose /metrics")
    parser.add_argument("--window", type=int, default=500, help="Recent predictions to compare")
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--interval", type=float, default=30.0, help="Seconds between checks")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)

    settings = Settings()
    configure_logging(settings.log_level)
    if settings.database_url is None:
        raise SystemExit("DATABASE_URL must be set")
    reference = load_reference(settings.model_dir)
    version = str(load_meta(settings.model_dir)["model_version"])
    store = PredictionStore(settings.database_url.get_secret_value())

    if not args.serve:
        report = check(store, reference, version, args.window, args.min_rows)
        print(format_report(report) if report else "Not enough predictions yet.")
        return

    start_http_server(args.port)
    logger.info(
        "Drift monitor started",
        extra={"fields": {"model_version": version, "port": args.port, "interval": args.interval}},
    )
    while True:
        try:
            check(store, reference, version, args.window, args.min_rows)
        except Exception:
            logger.exception("Drift check failed")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
