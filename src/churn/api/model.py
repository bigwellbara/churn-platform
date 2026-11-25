"""Loads the model bundle once and turns validated customers into predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from churn.api.schemas import CustomerFeatures, Prediction
from churn.bundle import load_bundle
from churn.config import RAW_FEATURES


def risk_level(probability: float, threshold: float) -> str:
    """high: flagged as churner. medium: at least half-way to the threshold. low: the rest."""
    if probability >= threshold:
        return "high"
    if probability >= threshold / 2:
        return "medium"
    return "low"


def limit_threads(pipeline: Pipeline, threads: int) -> None:
    """Set n_jobs on every pipeline step that has it.

    Training used n_jobs=-1 (all cores), which is right for fitting thousands of rows but
    wrong for serving: each single-row prediction would wake every core's thread, and
    under concurrent load those threads fight over CPUs. Measured on 16 cores, n_jobs=1
    cut per-prediction time from ~32 ms to ~11 ms (see docs/LOAD_TEST.md).
    """
    for _, step in pipeline.steps:
        if "n_jobs" in step.get_params(deep=False):
            step.set_params(n_jobs=threads)


class ModelService:
    def __init__(self, pipeline: Pipeline, meta: dict[str, Any], threads: int | None = 1):
        if threads is not None:
            limit_threads(pipeline, threads)
        self.pipeline = pipeline
        self.meta = meta
        self.version = str(meta["model_version"])
        self.threshold = float(meta["decision_threshold"])
        self.model_type = meta.get("model_type", "unknown")

    @classmethod
    def from_dir(cls, model_dir: Path, threads: int | None = 1) -> ModelService:
        pipeline, meta = load_bundle(model_dir)
        return cls(pipeline, meta, threads=threads)

    def score(self, features: list[dict[str, Any]]) -> list[float]:
        """Raw model inputs in, churn probabilities out. Picklable in and out, so it can
        run in a separate worker process (see churn.api.inference)."""
        # One DataFrame, one predict_proba call for the whole batch: vectorised scoring
        # is far faster than looping over rows. Column order matches training.
        frame = pd.DataFrame(features, columns=RAW_FEATURES)
        return self.pipeline.predict_proba(frame)[:, 1].tolist()

    def to_predictions(
        self, customers: list[CustomerFeatures], probabilities: list[float]
    ) -> list[Prediction]:
        return [
            Prediction(
                customer_id=customer.customer_id,
                churn_probability=round(float(p), 6),
                churn_prediction=bool(p >= self.threshold),
                risk_level=risk_level(float(p), self.threshold),
            )
            for customer, p in zip(customers, probabilities, strict=True)
        ]

    def predict(self, customers: list[CustomerFeatures]) -> list[Prediction]:
        """In-process convenience: score and apply the threshold."""
        return self.to_predictions(customers, self.score([c.features() for c in customers]))
