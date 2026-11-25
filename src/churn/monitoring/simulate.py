"""Send realistic traffic to the API, either normal or deliberately shifted.

    uv run churn-simulate --mode normal  --n 1000
    uv run churn-simulate --mode shifted --n 1000

Normal traffic samples real customers from the held-out test split: never seen by the
model, but drawn from the same population, so drift should stay low.

Shifted traffic applies a business scenario on top:
  * a marketing push brings in a wave of new (0-6 month) month-to-month customers,
    with DSL sign-ups moved onto fibre;
  * a 25% price rise on monthly charges for everyone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from churn.config import RAW_DATA_PATH
from churn.training.data import clean, load_raw, split_data


def sample_customers(n: int, data_path: Path = RAW_DATA_PATH, seed: int = 0) -> pd.DataFrame:
    test = split_data(clean(load_raw(data_path))).X_test
    return test.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)


def apply_shift(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    new = rng.random(len(df)) < 0.5
    df.loc[new, "tenure"] = rng.integers(0, 7, int(new.sum()))
    df.loc[new, "Contract"] = "Month-to-month"
    df.loc[new & (df["InternetService"] == "DSL"), "InternetService"] = "Fiber optic"
    df["MonthlyCharges"] = (df["MonthlyCharges"] * 1.25).round(2).clip(upper=1_000)
    df.loc[new, "TotalCharges"] = (df.loc[new, "MonthlyCharges"] * df.loc[new, "tenure"]).round(2)
    return df


def send(df: pd.DataFrame, client: httpx.Client, batch_size: int = 100) -> dict[str, float]:
    """POST the customers in batches; return a summary of what the model said."""
    probabilities: list[float] = []
    for start in range(0, len(df), batch_size):
        records = df.iloc[start : start + batch_size].to_dict(orient="records")
        customers = [{"customer_id": f"sim-{start + i}", **r} for i, r in enumerate(records)]
        response = client.post("/predict/batch", json={"customers": customers})
        response.raise_for_status()
        probabilities += [p["churn_probability"] for p in response.json()["predictions"]]
    threshold = response.json()["threshold"]
    probs = np.asarray(probabilities)
    return {
        "sent": len(probs),
        "mean_churn_probability": round(float(probs.mean()), 3),
        "flagged_share": round(float((probs >= threshold).mean()), 3),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Send normal or drifted traffic to the API.")
    parser.add_argument("--mode", choices=["normal", "shifted"], default="normal")
    parser.add_argument("--n", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    df = sample_customers(args.n, args.data, args.seed)
    if args.mode == "shifted":
        df = apply_shift(df, args.seed)
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    with httpx.Client(base_url=args.url, headers=headers, timeout=30) as client:
        summary = send(df, client, args.batch_size)
    print({"mode": args.mode, **summary})


if __name__ == "__main__":
    main()
