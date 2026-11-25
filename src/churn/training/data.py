"""Loading, cleaning and splitting the raw Telco churn data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import ID_COLUMN, RANDOM_STATE, RAW_FEATURES, TARGET, TEST_SIZE, VAL_SIZE


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for where to put it."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types and encode the target. Returns only model features plus the target."""
    missing = [c for c in [*RAW_FEATURES, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    df = df.copy()

    # Strip stray whitespace from text columns. pandas 3 stores text as the "str" dtype;
    # "object" still catches any mixed-type columns.
    for col in df.select_dtypes(include=["str", "object"]).columns:
        df[col] = df[col].str.strip()

    # TotalCharges loads as text because a few rows hold a blank string. Those rows are
    # all brand-new customers (tenure == 0) who have not been billed yet, so 0 is the
    # truthful value rather than a guess. Any other non-numeric value stays NaN and is
    # handled by the imputer inside the model pipeline.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    new_customers = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[new_customers, "TotalCharges"] = 0.0

    target = df[TARGET].map({"Yes": 1, "No": 0})
    if target.isna().any():
        bad = df.loc[target.isna(), TARGET].unique().tolist()
        raise ValueError(f"Unexpected target values: {bad}")
    df[TARGET] = target.astype(int)

    # customerID is an identifier, not a signal; keeping it would let models memorise rows.
    df = df.drop(columns=[ID_COLUMN], errors="ignore")
    return df[[*RAW_FEATURES, TARGET]]


def split_data(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> Splits:
    """Stratified 60/20/20 train/validation/test split.

    Stratifying keeps the churn rate identical across splits, which matters on imbalanced
    data. The test set is touched exactly once, after the winning model is chosen.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=random_state
    )
    val_fraction_of_rest = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=val_fraction_of_rest,
        stratify=y_rest,
        random_state=random_state,
    )
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test)
