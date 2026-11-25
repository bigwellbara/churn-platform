"""End-to-end: train on synthetic data, log to a temporary MLflow store, register."""

import json

import mlflow
import mlflow.sklearn
import pytest
from mlflow.tracking import MlflowClient

import churn.training.train as train_module
from churn.bundle import load_bundle
from churn.config import CHAMPION_ALIAS, REGISTERED_MODEL_NAME
from churn.training.data import clean
from churn.training.export import export_model
from churn.training.train import main, run_training


@pytest.fixture
def mlflow_env(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    yield {"tracking_uri": uri, "artifact_root": tmp_path / "artifacts"}
    mlflow.set_tracking_uri(None)


def test_full_training_registers_champion(csv_path, raw_df, tmp_path, mlflow_env):
    summary = run_training(
        data_path=csv_path,
        reports_dir=tmp_path / "reports",
        cv_folds=3,
        tune=True,
        tune_iterations=2,
        **mlflow_env,
    )
    mlflow_uri = {"tracking_uri": mlflow_env["tracking_uri"]}

    cands = summary["candidates"]
    assert set(cands) >= {"dummy_baseline", "logistic_regression", "random_forest", "xgboost"}
    assert "xgboost_tuned" in cands
    # A real model must beat the no-skill baseline.
    assert cands[summary["best_model"]]["val_pr_auc"] > cands["dummy_baseline"]["val_pr_auc"]
    assert summary["best_model"] != "dummy_baseline"
    assert 0.0 < summary["decision_threshold"] < 1.0

    saved = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert saved["best_model"] == summary["best_model"]

    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    assert str(mv.version) == summary["registered_version"]
    assert float(mv.tags["decision_threshold"]) == pytest.approx(
        summary["decision_threshold"], abs=1e-5
    )

    # The registered artifact is a working pipeline that accepts raw feature columns.
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    X = clean(raw_df.head(10)).drop(columns=["Churn"])
    probs = model.predict_proba(X)[:, 1]
    assert ((probs >= 0) & (probs <= 1)).all()

    # Export the pinned version into a serving bundle and check it round-trips.
    meta = export_model(tmp_path / "bundle", version=summary["registered_version"], **mlflow_uri)
    assert meta["model_version"] == summary["registered_version"]
    assert meta["exported_from"] == f"version {summary['registered_version']}"
    assert set(meta["test_metrics"]) >= {"roc_auc", "pr_auc", "recall"}
    pipeline, loaded = load_bundle(tmp_path / "bundle")
    assert loaded["decision_threshold"] == pytest.approx(summary["decision_threshold"], abs=1e-5)
    assert pipeline.predict_proba(X)[:, 1] == pytest.approx(probs)


def test_cli_no_register(csv_path, tmp_path, mlflow_env, monkeypatch, capsys):
    # Redirect every default output path so the test never touches the real repo folders.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_env["tracking_uri"])
    monkeypatch.setattr(train_module, "MLFLOW_ARTIFACT_ROOT", mlflow_env["artifact_root"])
    monkeypatch.setattr(train_module, "REPORTS_DIR", tmp_path / "cli_reports")
    main(["--data", str(csv_path), "--no-register"])
    assert "best_model" in capsys.readouterr().out
    assert (tmp_path / "cli_reports" / "metrics.json").exists()
