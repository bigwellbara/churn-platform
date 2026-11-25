"""Candidate models compared during training.

Each one goes from simple to more powerful. The dummy baseline sets the floor every real
model has to beat. See docs/TRAINING.md for why each model is here.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.config import RANDOM_STATE
from churn.training.features import FeatureEngineer, build_preprocessor


def build_candidates(pos_weight: float) -> dict[str, BaseEstimator]:
    """Return unfitted models keyed by name.

    pos_weight = (#non-churners / #churners). Every model gets some form of class
    weighting so it does not just predict "stays" for everyone.
    """
    return {
        "dummy_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": build_xgboost(pos_weight),
    }


def build_xgboost(pos_weight: float, **overrides) -> XGBClassifier:
    params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
        "scale_pos_weight": pos_weight,
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    params.update(overrides)
    return XGBClassifier(**params)


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """Feature engineering, then preprocessing, then the model, as one deployable object."""
    return Pipeline(
        [
            ("features", FeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )


# Search space for optional hyperparameter tuning (train.py --tune). Kept small on
# purpose: the plan caps tuning time, because engineering matters more than +0.5% AUC.
XGB_SEARCH_SPACE = {
    "model__n_estimators": [200, 300, 400, 600],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "model__max_depth": [3, 4, 5, 6],
    "model__min_child_weight": [1, 3, 5],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
}
