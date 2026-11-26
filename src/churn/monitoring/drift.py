"""Population Stability Index (PSI) drift detection.

A *reference profile* records how each feature was distributed in the training data:
quantile bins for numeric features, category shares for categorical ones, plus the
distribution of the model's own scores. Recent live inputs are bucketed the same way
and compared with:

    PSI = sum over buckets of (actual% - expected%) * ln(actual% / expected%)

Rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant drift.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from churn.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

N_BINS = 10
EPSILON = 1e-4  # stands in for empty buckets, so ln(0) never happens
MODERATE, SIGNIFICANT = 0.10, 0.25
PREDICTION_KEY = "prediction"


# ---------------------------------------------------------------- reference profile


def _numeric_profile(values: pd.Series, n_bins: int = N_BINS) -> dict[str, list[float]]:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    # Inner quantile edges; np.unique merges duplicates for spiky features like tenure.
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return {"edges": edges.tolist(), "proportions": _bucket_numeric(values, edges).tolist()}


def _categorical_profile(values: pd.Series) -> dict[str, list]:
    shares = values.astype(str).value_counts(normalize=True)
    return {"categories": shares.index.tolist(), "proportions": shares.to_numpy().tolist()}


def build_reference(features: pd.DataFrame, scores: np.ndarray | None = None) -> dict[str, Any]:
    """Profile of 'normal' data. Built from the training split at export time."""
    return {
        "n_rows": int(len(features)),
        "numeric": {c: _numeric_profile(features[c]) for c in NUMERIC_FEATURES},
        "categorical": {c: _categorical_profile(features[c]) for c in CATEGORICAL_FEATURES},
        PREDICTION_KEY: _numeric_profile(pd.Series(scores)) if scores is not None else None,
    }


# ---------------------------------------------------------------- bucketing and PSI


def _bucket_numeric(values, edges) -> np.ndarray:
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    counts = np.bincount(np.digitize(values, np.asarray(edges)), minlength=len(edges) + 1)
    return counts / max(counts.sum(), 1)


def _bucket_categorical(values: pd.Series, categories: list[str]) -> np.ndarray:
    """Shares per known category, plus a final 'unseen category' bucket."""
    values = values.astype(str)
    shares = [float((values == c).mean()) for c in categories]
    return np.asarray([*shares, float((~values.isin(categories)).mean())])


def psi(expected, actual) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPSILON, None)
    a = np.clip(np.asarray(actual, dtype=float), EPSILON, None)
    return float(np.sum((a - e) * np.log(a / e)))


def severity(value: float) -> str:
    if value >= SIGNIFICANT:
        return "significant"
    if value >= MODERATE:
        return "moderate"
    return "stable"


def drift_report(
    reference: dict[str, Any], current: pd.DataFrame, scores: np.ndarray | None = None
) -> dict[str, float]:
    """PSI for every feature (and the model score) of `current` versus the reference."""
    report = {}
    for col, prof in reference["numeric"].items():
        report[col] = psi(prof["proportions"], _bucket_numeric(current[col], prof["edges"]))
    for col, prof in reference["categorical"].items():
        expected = [*prof["proportions"], 0.0]  # the reference has no unseen categories
        report[col] = psi(expected, _bucket_categorical(current[col], prof["categories"]))
    pred_prof = reference.get(PREDICTION_KEY)
    if pred_prof is not None and scores is not None:
        report[PREDICTION_KEY] = psi(
            pred_prof["proportions"], _bucket_numeric(scores, pred_prof["edges"])
        )
    return report
