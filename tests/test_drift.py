"""PSI maths, the reference profile, the drift monitor and the traffic simulator."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from churn.api.main import create_app
from churn.api.schemas import CustomerFeatures
from churn.api.settings import Settings
from churn.bundle import load_meta, load_reference
from churn.config import TARGET
from churn.monitoring import monitor
from churn.monitoring.drift import build_reference, drift_report, psi, severity
from churn.monitoring.simulate import apply_shift, sample_customers, send
from churn.training.data import clean

# ------------------------------------------------------------------ PSI maths


def test_psi_is_zero_for_identical_distributions():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0.0)


def test_psi_grows_with_shift():
    small = psi([0.5, 0.5], [0.45, 0.55])
    large = psi([0.5, 0.5], [0.1, 0.9])
    assert 0 < small < 0.1 < 0.25 < large


def test_psi_handles_empty_buckets():
    assert np.isfinite(psi([0.5, 0.5, 0.0], [0.4, 0.4, 0.2]))


@pytest.mark.parametrize(
    ("value", "label"), [(0.05, "stable"), (0.1, "moderate"), (0.3, "significant")]
)
def test_severity_bands(value, label):
    assert severity(value) == label


# ------------------------------------------------------------------ reference + report


@pytest.fixture
def features(raw_df):
    return clean(raw_df).drop(columns=[TARGET])


def test_same_data_shows_no_drift(features):
    report = drift_report(build_reference(features), features)
    assert max(report.values()) < 0.01


def test_shifted_data_is_detected(features):
    report = drift_report(build_reference(features), apply_shift(features, seed=1))
    assert report["tenure"] > 0.25
    assert report["Contract"] > 0.25
    assert report["MonthlyCharges"] > 0.25
    assert report["gender"] < 0.1  # untouched by the scenario


def test_unseen_category_counts_as_drift(features):
    current = features.copy()
    current["PaymentMethod"] = "Crypto"
    assert drift_report(build_reference(features), current)["PaymentMethod"] > 0.25


def test_shifted_rows_are_still_valid_api_input(features):
    for row in apply_shift(features).to_dict(orient="records"):
        CustomerFeatures(**row)


# ------------------------------------------------------------------ end to end


@pytest.fixture
def api(model_bundle_dir, tmp_path):
    db_url = f"sqlite:///{(tmp_path / 'preds.db').as_posix()}"
    settings = Settings(model_dir=model_bundle_dir, database_url=db_url, log_flush_interval=60)
    with TestClient(create_app(settings)) as c:
        yield c


def send_logged(df, api, batch_size=100):
    """Send traffic, then flush the write-behind buffer so the monitor can see it."""
    summary = send(df, api, batch_size)
    api.app.state.log_writer.flush()
    return summary


def test_monitor_end_to_end(api, large_csv_path, model_bundle_dir):
    """Traffic goes through the real API into the DB; the monitor reads it back."""
    store = api.app.state.store
    reference = load_reference(model_bundle_dir)
    version = load_meta(model_bundle_dir)["model_version"]

    # Too little data: no verdict yet.
    send_logged(sample_customers(20, large_csv_path, seed=1), api, batch_size=20)
    assert monitor.check(store, reference, version, window=500, min_rows=100) is None

    # Normal traffic: no input feature reaches even "moderate" drift.
    send_logged(sample_customers(500, large_csv_path, seed=2), api, batch_size=100)
    normal = monitor.check(store, reference, version, window=500, min_rows=100)
    assert max(v for k, v in normal.items() if k != "prediction") < 0.10

    # Shifted traffic: the monitor and its Prometheus gauges flag it.
    send_logged(apply_shift(sample_customers(500, large_csv_path, seed=3)), api, batch_size=100)
    shifted = monitor.check(store, reference, version, window=500, min_rows=100)
    assert shifted["tenure"] > 0.25
    assert REGISTRY.get_sample_value("churn_drift_max_psi") > 0.25
    assert REGISTRY.get_sample_value("churn_drift_features_significant") >= 2
    # Synthetic data is already 55% month-to-month, so the Contract shift is "moderate" here.
    assert REGISTRY.get_sample_value("churn_feature_psi", {"feature": "Contract"}) > 0.10
    assert REGISTRY.get_sample_value("churn_feature_psi", {"feature": "gender"}) < 0.10

    table = monitor.format_report(shifted)
    assert "tenure" in table and "significant" in table


def test_monitor_ignores_other_model_versions(api, csv_path, model_bundle_dir):
    send_logged(sample_customers(150, csv_path), api)
    reference = load_reference(model_bundle_dir)
    assert monitor.check(api.app.state.store, reference, "999", min_rows=10) is None


def test_monitor_cli_once(api, csv_path, model_bundle_dir, tmp_path, monkeypatch, capsys):
    send_logged(sample_customers(150, csv_path), api)
    monkeypatch.setenv("MODEL_DIR", str(model_bundle_dir))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'preds.db').as_posix()}")
    monitor.main(["--once", "--min-rows", "100"])
    assert "PSI" in capsys.readouterr().out


def test_simulate_summary(api, csv_path):
    summary = send(sample_customers(50, csv_path), api, batch_size=20)
    assert summary["sent"] == 50
    assert 0 <= summary["flagged_share"] <= 1


def test_sample_customers_comes_from_test_split(csv_path):
    df = sample_customers(30, csv_path)
    assert len(df) == 30 and TARGET not in df.columns
    assert isinstance(df, pd.DataFrame)
