"""Per-customer explanations: which answers pushed this customer's churn score up or down.

Uses XGBoost's built-in SHAP values (TreeSHAP, `pred_contribs=True`): for one prediction,
each model column gets a contribution in log-odds, and

    base_value + sum(contributions) == log(p / (1 - p))

exactly, where p is the predicted probability. So the explanation is not an
approximation fitted after the fact; it is a decomposition of the score the model gave.

The model sees 48 columns (one-hot categories, scaled numbers, two engineered features).
People think in the 19 answers they gave, so contributions are summed back to the field
they came from: Contract_Month-to-month + Contract_One year + Contract_Two year become
"Contract". Summing is valid because SHAP values are additive. The two engineered
features (avg_monthly_spend, num_services) are reported under their own names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from churn.config import CATEGORICAL_FEATURES, ENGINEERED_FEATURES, RAW_FEATURES


class ExplanationsUnavailable(Exception):
    """The deployed model type has no exact per-prediction explanation (not a tree booster)."""


def group_columns(columns: list[str]) -> list[str]:
    """Map each transformed column name to the input field (or engineered feature) it
    came from. Fails loudly on anything unexpected, so a pipeline change cannot silently
    mislabel an explanation."""
    numeric = {f"num__{name}": name for name in RAW_FEATURES + ENGINEERED_FEATURES}
    # Longest prefix first, so a field whose name starts with another's still maps right.
    categorical = sorted(CATEGORICAL_FEATURES, key=len, reverse=True)
    groups = []
    for column in columns:
        if column in numeric:
            groups.append(numeric[column])
            continue
        match = next((f for f in categorical if column.startswith(f"cat__{f}_")), None)
        if match is None:
            raise ValueError(f"Cannot map model column {column!r} back to an input field")
        groups.append(match)
    return groups


@dataclass(frozen=True)
class Contribution:
    feature: str
    value: Any
    impact: float


@dataclass(frozen=True)
class Explanation:
    base_value: float
    contributions: list[Contribution]


class Explainer:
    def __init__(self, pipeline: Pipeline):
        model = pipeline.steps[-1][1]
        if not hasattr(model, "get_booster"):
            raise ExplanationsUnavailable(
                f"{type(model).__name__} has no exact per-prediction explanation"
            )
        self._engineer = pipeline.steps[0][1]
        self._preprocess = Pipeline(pipeline.steps[1:-1])
        self._booster = model.get_booster()
        self._groups = group_columns(list(self._preprocess.get_feature_names_out()))
        self._fields = list(dict.fromkeys(self._groups))

    def explain(self, features: dict[str, Any]) -> Explanation:
        engineered = self._engineer.transform(pd.DataFrame([features], columns=RAW_FEATURES))
        matrix = self._preprocess.transform(engineered)
        dmatrix = xgb.DMatrix(matrix, feature_names=self._booster.feature_names)
        row = self._booster.predict(dmatrix, pred_contribs=True)[0]
        *per_column, bias = row.tolist()

        totals = dict.fromkeys(self._fields, 0.0)
        for field, impact in zip(self._groups, per_column, strict=True):
            totals[field] += impact

        values = engineered.iloc[0]
        contributions = [
            Contribution(feature=field, value=_plain(values[field]), impact=round(impact, 4))
            for field, impact in totals.items()
        ]
        contributions.sort(key=lambda c: abs(c.impact), reverse=True)
        return Explanation(base_value=round(bias, 4), contributions=contributions)


def _plain(value: Any) -> Any:
    """numpy scalars to plain Python, with engineered numbers rounded for display."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, 2)
    return value
