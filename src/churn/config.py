"""Single source of truth for paths, column names and training settings.

Training, the API schema and the drift check all import from here, so a column rename
only has to happen in one place.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "telco_churn.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLFLOW_ARTIFACT_ROOT = PROJECT_ROOT / "mlartifacts"

TARGET = "Churn"
ID_COLUMN = "customerID"

# Continuous columns: imputed with the median and standardised.
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Columns created by churn.training.features.FeatureEngineer.
ENGINEERED_FEATURES = ["avg_monthly_spend", "num_services"]

# Discrete columns: one-hot encoded.
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Add-on services counted by the num_services feature.
SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # fraction of the full dataset

EXPERIMENT_NAME = "churn-prediction"
REGISTERED_MODEL_NAME = "churn-model"
CHAMPION_ALIAS = "champion"

# Metric used to pick the winning model. PR-AUC (average precision) focuses on the
# minority (churn) class, which matters more than overall accuracy on imbalanced data.
SELECTION_METRIC = "val_pr_auc"


def default_tracking_uri() -> str:
    """MLflow tracking URI: env var if set, otherwise a local SQLite file."""
    return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
