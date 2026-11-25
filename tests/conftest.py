"""Shared fixtures. Tests use a synthetic dataset shaped like the real Telco CSV, so they
run in CI without downloading anything."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_synthetic_telco(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    tenure = rng.integers(0, 73, n)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])

    def internet_addon() -> np.ndarray:
        return np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n))

    monthly = rng.uniform(18, 119, n).round(2)
    total = (monthly * tenure).round(2).astype(str).astype(object)
    total[tenure == 0] = " "  # mimic the real file's blank TotalCharges for new customers

    # Churn depends on contract, tenure and fibre, roughly like the real data.
    logit = (
        -1.2
        + 1.6 * (contract == "Month-to-month")
        - 0.04 * tenure
        + 0.9 * (internet == "Fiber optic")
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    return pd.DataFrame(
        {
            "customerID": [f"{i:04d}-SYNTH" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": np.where(
                phone == "No", "No phone service", rng.choice(["Yes", "No"], n)
            ),
            "InternetService": internet,
            "OnlineSecurity": internet_addon(),
            "OnlineBackup": internet_addon(),
            "DeviceProtection": internet_addon(),
            "TechSupport": internet_addon(),
            "StreamingTV": internet_addon(),
            "StreamingMovies": internet_addon(),
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": np.where(churn, "Yes", "No"),
        }
    )


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_synthetic_telco()


@pytest.fixture
def csv_path(tmp_path, raw_df):
    path = tmp_path / "telco.csv"
    raw_df.to_csv(path, index=False)
    return path
