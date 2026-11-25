"""Feature engineering and preprocessing, packaged as scikit-learn transformers.

Everything here lives inside the model Pipeline, so the API applies exactly the same
transformations at prediction time as were used in training (no training/serving skew).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERIC_FEATURES,
    SERVICE_COLUMNS,
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain features. Stateless: learns nothing from the data.

    - avg_monthly_spend: TotalCharges / tenure. Compared with MonthlyCharges it shows
      whether a customer's bill has recently gone up (a common churn trigger).
    - num_services: how many add-on services the customer has. More services means
      more switching cost, so these customers tend to churn less.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in NUMERIC_FEATURES:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X["avg_monthly_spend"] = X["TotalCharges"] / X["tenure"].clip(lower=1)
        X["num_services"] = (X[SERVICE_COLUMNS] == "Yes").sum(axis=1) + (
            X["InternetService"] != "No"
        ).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray([*self.feature_names_in_, *ENGINEERED_FEATURES], dtype=object)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore": an unseen category at prediction time becomes an
            # all-zero vector instead of crashing the API.
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES + ENGINEERED_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
