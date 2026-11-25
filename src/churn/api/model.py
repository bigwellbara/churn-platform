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


class ModelService:
    def __init__(self, pipeline: Pipeline, meta: dict[str, Any]):
        self.pipeline = pipeline
        self.meta = meta
        self.version = str(meta["model_version"])
        self.threshold = float(meta["decision_threshold"])
        self.model_type = meta.get("model_type", "unknown")

    @classmethod
    def from_dir(cls, model_dir: Path) -> ModelService:
        pipeline, meta = load_bundle(model_dir)
        return cls(pipeline, meta)

    def predict(self, customers: list[CustomerFeatures]) -> list[Prediction]:
        # One DataFrame, one predict_proba call for the whole batch: vectorised scoring
        # is far faster than looping over rows. Column order matches training.
        frame = pd.DataFrame([c.features() for c in customers], columns=RAW_FEATURES)
        probabilities = self.pipeline.predict_proba(frame)[:, 1]
        return [
            Prediction(
                customer_id=customer.customer_id,
                churn_probability=round(float(p), 6),
                churn_prediction=bool(p >= self.threshold),
                risk_level=risk_level(float(p), self.threshold),
            )
            for customer, p in zip(customers, probabilities, strict=True)
        ]
