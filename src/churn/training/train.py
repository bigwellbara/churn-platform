"""End-to-end training: compare models, log to MLflow, register the best one.

Usage:
    uv run churn-train                    # train all candidates, register the winner
    uv run churn-train --tune             # also run a time-boxed XGBoost search
    uv run churn-train --data path.csv --no-register
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from churn.bundle import trusted_types_for
from churn.config import (
    CHAMPION_ALIAS,
    EXPERIMENT_NAME,
    MLFLOW_ARTIFACT_ROOT,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
    SELECTION_METRIC,
    default_tracking_uri,
)
from churn.training.data import Splits, clean, load_raw, split_data
from churn.training.evaluate import compute_metrics, find_best_threshold, save_confusion_matrix
from churn.training.models import (
    XGB_SEARCH_SPACE,
    build_candidates,
    build_pipeline,
    build_xgboost,
)

logger = logging.getLogger("churn.train")

# A train ROC-AUC this much higher than validation ROC-AUC is flagged as overfitting.
OVERFIT_GAP_WARNING = 0.05


@dataclass
class CandidateResult:
    name: str
    run_id: str
    pipeline: Pipeline
    threshold: float
    metrics: dict[str, float]


def _setup_mlflow(tracking_uri: str, artifact_root: Path) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        artifact_root.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_root.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def _loggable_params(pipeline: Pipeline) -> dict[str, object]:
    params = pipeline.named_steps["model"].get_params()
    return {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}


def tune_xgboost(splits: Splits, pos_weight: float, n_iter: int, cv: StratifiedKFold) -> dict:
    """Randomised search over XGB_SEARCH_SPACE, scored by cross-validated PR-AUC."""
    search = RandomizedSearchCV(
        build_pipeline(build_xgboost(pos_weight)),
        param_distributions=XGB_SEARCH_SPACE,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,  # XGBoost already uses all cores
    )
    search.fit(splits.X_train, splits.y_train)
    best = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    logger.info("Tuning best CV PR-AUC=%.4f params=%s", search.best_score_, best)
    return best


def evaluate_candidate(
    name: str,
    pipeline: Pipeline,
    splits: Splits,
    cv: StratifiedKFold,
    figures_dir: Path,
) -> CandidateResult:
    with mlflow.start_run(run_name=name) as run:
        # 1. Cross-validation on the training set: how stable is the model across folds?
        cv_auc = cross_val_score(
            clone(pipeline), splits.X_train, splits.y_train, cv=cv, scoring="roc_auc"
        )

        # 2. Fit once on the full training set.
        pipeline.fit(splits.X_train, splits.y_train)
        train_prob = pipeline.predict_proba(splits.X_train)[:, 1]
        val_prob = pipeline.predict_proba(splits.X_val)[:, 1]

        # 3. Choose the decision threshold on validation (never on test).
        threshold = find_best_threshold(splits.y_val, val_prob)
        train_m = compute_metrics(splits.y_train, train_prob, threshold)
        val_m = compute_metrics(splits.y_val, val_prob, threshold)

        # 4. Overfitting check: big train/validation gap = memorising, not learning.
        gap = train_m["roc_auc"] - val_m["roc_auc"]
        if gap > OVERFIT_GAP_WARNING:
            logger.warning("%s may be overfitting: train-val ROC-AUC gap=%.3f", name, gap)

        metrics = {
            "cv_roc_auc_mean": float(np.mean(cv_auc)),
            "cv_roc_auc_std": float(np.std(cv_auc)),
            "train_roc_auc": train_m["roc_auc"],
            "overfit_gap": gap,
            "threshold": threshold,
            **{f"val_{k}": v for k, v in val_m.items()},
        }

        mlflow.set_tag("model_type", name)
        mlflow.log_params(_loggable_params(pipeline))
        mlflow.log_metrics(metrics)
        fig = save_confusion_matrix(
            splits.y_val, val_prob, threshold, figures_dir / f"cm_val_{name}.png", name
        )
        mlflow.log_artifact(str(fig), artifact_path="figures")

        logger.info(
            "%-20s cv_auc=%.4f±%.4f val_auc=%.4f val_pr_auc=%.4f val_f1=%.4f gap=%.3f",
            name,
            metrics["cv_roc_auc_mean"],
            metrics["cv_roc_auc_std"],
            val_m["roc_auc"],
            val_m["pr_auc"],
            val_m["f1"],
            gap,
        )
        return CandidateResult(name, run.info.run_id, pipeline, threshold, metrics)


def register_champion(best: CandidateResult, splits: Splits) -> str:
    """Log the winning pipeline as an MLflow model, register it and alias it 'champion'."""
    with mlflow.start_run(run_id=best.run_id):
        input_example = splits.X_val.head(5)
        signature = infer_signature(input_example, best.pipeline.predict_proba(input_example)[:, 1])
        info = mlflow.sklearn.log_model(
            best.pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=trusted_types_for(best.pipeline),
        )

    version = str(info.registered_model_version)
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version)
    # The API needs the tuned threshold, so it travels with the model version.
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, version, "decision_threshold", f"{best.threshold:.6f}"
    )
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "model_type", best.name)
    logger.info("Registered %s v%s as @%s", REGISTERED_MODEL_NAME, version, CHAMPION_ALIAS)
    return version


def run_training(
    data_path: Path | None = None,
    tracking_uri: str | None = None,
    artifact_root: Path | None = None,
    reports_dir: Path | None = None,
    tune: bool = False,
    tune_iterations: int = 25,
    register: bool = True,
    cv_folds: int = 5,
) -> dict:
    data_path = data_path or RAW_DATA_PATH
    reports_dir = reports_dir or REPORTS_DIR
    _setup_mlflow(tracking_uri or default_tracking_uri(), artifact_root or MLFLOW_ARTIFACT_ROOT)
    figures_dir = reports_dir / "figures"

    df = clean(load_raw(data_path))
    splits = split_data(df)
    pos_weight = float((splits.y_train == 0).sum() / (splits.y_train == 1).sum())
    logger.info(
        "Data: %d rows, churn rate %.1f%%; train/val/test = %d/%d/%d",
        len(df),
        100 * df["Churn"].mean(),
        len(splits.X_train),
        len(splits.X_val),
        len(splits.X_test),
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    candidates = build_candidates(pos_weight)
    if tune:
        best_params = tune_xgboost(splits, pos_weight, tune_iterations, cv)
        candidates["xgboost_tuned"] = build_xgboost(pos_weight, **best_params)

    results = [
        evaluate_candidate(name, build_pipeline(model), splits, cv, figures_dir)
        for name, model in candidates.items()
    ]

    best = max(results, key=lambda r: r.metrics[SELECTION_METRIC])
    logger.info("Best model by %s: %s", SELECTION_METRIC, best.name)

    # The test set is used exactly once, on the chosen model, for an unbiased estimate.
    test_prob = best.pipeline.predict_proba(splits.X_test)[:, 1]
    test_metrics = compute_metrics(splits.y_test, test_prob, best.threshold)
    with mlflow.start_run(run_id=best.run_id):
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.set_tag("champion_candidate", "true")
        fig = save_confusion_matrix(
            splits.y_test, test_prob, best.threshold, figures_dir / "cm_test_best.png", best.name
        )
        mlflow.log_artifact(str(fig), artifact_path="figures")

    version = register_champion(best, splits) if register else None

    summary = {
        "selection_metric": SELECTION_METRIC,
        "best_model": best.name,
        "registered_version": version,
        "decision_threshold": best.threshold,
        "test_metrics": test_metrics,
        "candidates": {r.name: r.metrics for r in results},
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and register the churn model.")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH, help="Path to raw CSV")
    parser.add_argument("--tune", action="store_true", help="Run XGBoost hyperparameter search")
    parser.add_argument("--tune-iterations", type=int, default=25)
    parser.add_argument("--no-register", action="store_true", help="Skip model registry")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run_training(
        data_path=args.data,
        tune=args.tune,
        tune_iterations=args.tune_iterations,
        register=not args.no_register,
    )
    print(json.dumps({k: summary[k] for k in ("best_model", "test_metrics")}, indent=2))


if __name__ == "__main__":
    main()
