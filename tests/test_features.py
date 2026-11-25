import numpy as np
import pandas as pd

from churn.config import ENGINEERED_FEATURES, TARGET
from churn.training.data import clean
from churn.training.features import FeatureEngineer, build_preprocessor
from churn.training.models import build_pipeline, build_xgboost


def test_feature_engineer_adds_columns(raw_df):
    X = clean(raw_df).drop(columns=[TARGET])
    out = FeatureEngineer().fit_transform(X)
    for col in ENGINEERED_FEATURES:
        assert col in out.columns
    assert (out["num_services"] >= 0).all()
    assert (out["num_services"] <= 9).all()


def test_num_services_counts_correctly(raw_df):
    X = clean(raw_df).drop(columns=[TARGET]).head(1).copy()
    services = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    X[services] = "No"
    X["InternetService"] = "DSL"
    X["PhoneService"] = "Yes"
    X["TechSupport"] = "Yes"
    out = FeatureEngineer().fit_transform(X)
    assert out["num_services"].iloc[0] == 3  # phone + tech support + internet


def test_avg_monthly_spend_handles_zero_tenure(raw_df):
    X = clean(raw_df).drop(columns=[TARGET])
    out = FeatureEngineer().fit_transform(X)
    assert np.isfinite(out["avg_monthly_spend"]).all()


def test_preprocessor_output_has_no_nans(raw_df):
    X = clean(raw_df).drop(columns=[TARGET])
    X.loc[X.index[:5], "MonthlyCharges"] = np.nan
    X.loc[X.index[:5], "Contract"] = np.nan
    Xt = build_preprocessor().fit_transform(FeatureEngineer().fit_transform(X))
    assert not np.isnan(Xt).any()


def test_pipeline_tolerates_unseen_category(raw_df):
    df = clean(raw_df)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    pipe = build_pipeline(build_xgboost(pos_weight=2.0, n_estimators=20)).fit(X, y)
    new = X.head(2).copy()
    new["PaymentMethod"] = "Crypto"  # never seen in training
    prob = pipe.predict_proba(new)[:, 1]
    assert ((prob >= 0) & (prob <= 1)).all()


def test_pipeline_accepts_string_total_charges(raw_df):
    df = clean(raw_df)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    pipe = build_pipeline(build_xgboost(pos_weight=2.0, n_estimators=20)).fit(X, y)
    row = X.head(1).copy()
    row["TotalCharges"] = row["TotalCharges"].astype(str)
    assert pipe.predict_proba(pd.DataFrame(row))[:, 1].shape == (1,)
