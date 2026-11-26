"""Model quality gate: may this model bundle be shipped?

CI runs `churn-gate` on every pull request, so a change that swaps in a worse model, a
broken export or a model that behaves strangely fails the build *before* it can be
deployed. Unit tests check the code; this checks the artefact.

Checks:
  1. The bundle loads with only its declared trusted types (skops allowlist).
  2. Held-out test metrics, recorded at training time, clear minimum floors.
  3. Behavioural checks on hand-written customers (the "CheckList" idea from NLP testing,
     Ribeiro et al. 2020): probabilities are valid and deterministic, and the model moves
     in the direction the business expects, e.g. a loyal two-year customer must look
     less risky than a brand-new month-to-month one.
  4. The drift-detection reference profile is present.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from churn.bundle import load_bundle, load_reference
from churn.config import PROJECT_ROOT, RAW_FEATURES

# Floors, not targets. Current champion: ROC-AUC 0.847, PR-AUC 0.663, recall 0.725
# (docs/MODEL_CARD.md). Base rate is 0.265, so PR-AUC 0.60 is still ~2.3x random.
DEFAULT_FLOORS = {"roc_auc": 0.80, "pr_auc": 0.60, "recall": 0.65}

_BASE = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.0,
}
# Brand-new customer, no commitment: the profile the EDA found churns most.
RISKY = {**_BASE, "tenure": 1, "Contract": "Month-to-month", "TotalCharges": 80.0}
# Same services and price, but five years in and on a two-year contract.
LOYAL = {
    **_BASE,
    "tenure": 60,
    "Contract": "Two year",
    "PaymentMethod": "Bank transfer (automatic)",
    "TotalCharges": 4800.0,
}


@dataclass
class GateResult:
    passed: list[str]
    failed: list[str]

    @property
    def ok(self) -> bool:
        return not self.failed


def run_gate(model_dir: Path, floors: dict[str, float] | None = None) -> GateResult:
    floors = DEFAULT_FLOORS if floors is None else floors
    passed: list[str] = []
    failed: list[str] = []

    def record(ok: bool, message: str) -> None:
        (passed if ok else failed).append(message)

    try:
        pipeline, meta = load_bundle(model_dir)
    except Exception as err:  # nothing else can be checked without a model
        return GateResult(passed, [f"bundle does not load: {err}"])
    record(True, f"bundle loads (model v{meta['model_version']}, {meta.get('model_type')})")

    metrics: dict[str, Any] = meta.get("test_metrics", {})
    for name, floor in floors.items():
        value = metrics.get(name)
        if value is None:
            record(False, f"test {name} missing from model metadata")
        else:
            record(value >= floor, f"test {name} {value:.3f} (floor {floor:.2f})")

    frame = pd.DataFrame([RISKY, LOYAL], columns=RAW_FEATURES)
    first = pipeline.predict_proba(frame)[:, 1]
    second = pipeline.predict_proba(frame)[:, 1]
    record(bool(((first >= 0) & (first <= 1)).all()), f"probabilities in [0, 1]: {first.round(3)}")
    record(bool((first == second).all()), "predictions are deterministic")
    risky, loyal = float(first[0]), float(first[1])
    record(
        loyal < risky,
        f"loyal two-year customer scores below new month-to-month ({loyal:.3f} < {risky:.3f})",
    )

    try:
        reference = load_reference(model_dir)
        has_all = {"numeric", "categorical", "prediction"} <= reference.keys()
        record(has_all, f"drift reference profile present ({reference.get('n_rows')} rows)")
    except FileNotFoundError as err:
        record(False, str(err))

    return GateResult(passed, failed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a model bundle is fit to ship.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "champion")
    for name, floor in DEFAULT_FLOORS.items():
        parser.add_argument(f"--min-{name.replace('_', '-')}", type=float, default=floor)
    args = parser.parse_args(argv)
    floors = {name: getattr(args, f"min_{name}") for name in DEFAULT_FLOORS}

    result = run_gate(args.model_dir, floors)
    for line in result.passed:
        print(f"  PASS  {line}")
    for line in result.failed:
        print(f"  FAIL  {line}")
    print("Model gate passed." if result.ok else "Model gate FAILED.")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
