"""API tests: happy paths, validation, auth, metrics, logging and failure modes."""

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from sqlalchemy import select
from sqlalchemy.orm import Session

from churn.api.db import PredictionLog
from churn.api.main import create_app
from churn.api.settings import Settings


@pytest.fixture
def make_client(model_bundle_dir, tmp_path):
    opened = []

    def _make(**overrides):
        options = {
            "model_dir": model_bundle_dir,
            "database_url": f"sqlite:///{(tmp_path / 'predictions.db').as_posix()}",
            "log_flush_interval": 60,  # tests flush explicitly, for determinism
            **overrides,
        }
        client = TestClient(create_app(Settings(**options)), raise_server_exceptions=False)
        client.__enter__()  # runs the lifespan: loads the model, connects the DB
        opened.append(client)
        return client

    yield _make
    for client in opened:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()


# ------------------------------------------------------------------ health / info


def test_health_reports_model_and_database(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "7", "database": "connected"}


def test_health_without_database(make_client):
    assert make_client(database_url=None).get("/health").json()["database"] == "disabled"


def test_unreachable_database_degrades_gracefully(make_client, customer):
    c = make_client(database_url="postgresql+psycopg://u:p@127.0.0.1:1/none?connect_timeout=1")
    assert c.get("/health").json()["database"] == "unavailable"
    assert c.post("/predict", json=customer).status_code == 200  # predictions still served


def test_model_info(client):
    body = client.get("/model").json()
    assert body["model_version"] == "7"
    assert body["decision_threshold"] == 0.5
    assert body["test_metrics"]["roc_auc"] == 0.8
    assert "tenure" in body["features"]


def test_openapi_docs_list_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/predict", "/predict/batch", "/health", "/model"} <= set(paths)


# ------------------------------------------------------------------ predictions


def test_predict_single(client, customer):
    r = client.post("/predict", json=customer)
    assert r.status_code == 200
    body = r.json()
    assert body["customer_id"] == customer["customer_id"]
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] == (body["churn_probability"] >= body["threshold"])
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["model_version"] == "7"
    assert r.headers["X-Request-ID"]


def test_request_id_is_propagated(client, customer):
    r = client.post("/predict", json=customer, headers={"X-Request-ID": "trace-123"})
    assert r.headers["X-Request-ID"] == "trace-123"


def test_predict_batch_preserves_order_and_matches_single(client, customer):
    loyal = {**customer, "customer_id": "loyal", "tenure": 70, "Contract": "Two year"}
    batch = [customer, loyal, {**customer, "customer_id": None}]
    r = client.post("/predict/batch", json={"customers": batch})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert [p["customer_id"] for p in body["predictions"]] == [
        customer["customer_id"],
        "loyal",
        None,
    ]
    single = client.post("/predict", json=customer).json()["churn_probability"]
    assert body["predictions"][0]["churn_probability"] == pytest.approx(single)


def test_batch_limits(make_client, customer):
    c = make_client(max_batch_size=2)
    assert c.post("/predict/batch", json={"customers": []}).status_code == 422
    r = c.post("/predict/batch", json={"customers": [customer] * 3})
    assert r.status_code == 413
    assert "limit of 2" in r.json()["detail"]


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"tenure": -1}, "tenure"),
        ({"tenure": 3.5}, "tenure"),
        ({"Contract": "Weekly"}, "Contract"),
        ({"MonthlyCharges": "abc"}, "MonthlyCharges"),
        ({"SeniorCitizen": 2}, "SeniorCitizen"),
        ({"tenur": 5}, "tenur"),  # typo'd field name is rejected, not ignored
    ],
)
def test_invalid_input_is_rejected_with_field_name(client, customer, change, field):
    r = client.post("/predict", json={**customer, **change})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_error"
    assert field in [d["field"] for d in body["detail"]]
    assert body["request_id"]


def test_missing_field_is_rejected(client, customer):
    del customer["gender"]
    r = client.post("/predict", json=customer)
    assert r.status_code == 422
    assert r.json()["detail"][0]["field"] == "gender"


def test_inconsistent_services_are_rejected(client, customer):
    r = client.post("/predict", json={**customer, "InternetService": "No"})
    assert r.status_code == 422
    assert "No internet service" in r.json()["detail"][0]["message"]


def test_batch_error_points_at_the_bad_row(client, customer):
    r = client.post("/predict/batch", json={"customers": [customer, {**customer, "tenure": -5}]})
    assert r.status_code == 422
    assert r.json()["detail"][0]["field"] == "customers.1.tenure"


# ------------------------------------------------------------------ auth


def test_api_key_required_when_configured(make_client, customer):
    c = make_client(api_key="s3cret")
    assert c.post("/predict", json=customer).status_code == 401
    assert c.post("/predict", json=customer, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/predict", json=customer, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert c.get("/health").status_code == 200  # health stays open for probes


# ------------------------------------------------------------------ logging & metrics


def test_predictions_are_logged_to_database(client, customer):
    client.post("/predict", json=customer, headers={"X-Request-ID": "req-1"})
    client.post("/predict/batch", json={"customers": [customer, customer]})
    store = client.app.state.store
    assert store.count() == 0  # buffered: nothing written on the request path
    assert client.app.state.log_writer.flush() == 3
    assert store.count() == 3
    with Session(store.engine) as s:
        row = s.scalars(select(PredictionLog).where(PredictionLog.request_id == "req-1")).one()
    assert row.features["tenure"] == customer["tenure"]
    assert "customer_id" not in row.features
    assert row.model_version == "7"


def test_database_failure_does_not_break_predictions(client, customer, monkeypatch):
    def broken_save(rows):
        raise RuntimeError("database is down")

    monkeypatch.setattr(client.app.state.store, "save", broken_save)
    before = REGISTRY.get_sample_value("churn_db_write_errors_total") or 0
    dropped = REGISTRY.get_sample_value("churn_prediction_log_dropped_total") or 0
    assert client.post("/predict", json=customer).status_code == 200
    client.app.state.log_writer.flush()
    assert REGISTRY.get_sample_value("churn_db_write_errors_total") == before + 1
    assert REGISTRY.get_sample_value("churn_prediction_log_dropped_total") == dropped + 1


def test_buffered_rows_are_flushed_on_shutdown(make_client, customer):
    client = make_client()
    store = client.app.state.store
    client.post("/predict/batch", json={"customers": [customer] * 5})
    client.__exit__(None, None, None)  # graceful shutdown
    assert store.count() == 5


def test_log_queue_is_bounded(make_client, customer):
    from churn.api.log_writer import PredictionLogWriter

    client = make_client()
    tiny = PredictionLogWriter(client.app.state.store, max_queue=2)  # never started
    client.app.state.log_writer = tiny
    dropped = REGISTRY.get_sample_value("churn_prediction_log_dropped_total") or 0
    client.post("/predict/batch", json={"customers": [customer] * 5})
    assert REGISTRY.get_sample_value("churn_prediction_log_dropped_total") == dropped + 3
    assert tiny.flush() == 2


def test_metrics_endpoint(client, customer):
    client.post("/predict", json=customer)
    text = client.get("/metrics").text
    assert 'churn_http_requests_total{method="POST",path="/predict",status="200"}' in text
    assert "churn_predictions_total" in text
    assert "churn_prediction_probability_bucket" in text
    assert 'churn_model_info{model_type="xgboost_test",version="7"} 1.0' in text


# ------------------------------------------------------------------ failure modes


def test_unexpected_error_returns_json_500(client, customer, monkeypatch):
    def explode(customers):
        raise RuntimeError("boom")

    monkeypatch.setattr(client.app.state.model, "score", explode)
    r = client.post("/predict", json=customer, headers={"X-Request-ID": "req-500"})
    assert r.status_code == 500
    assert r.json() == {
        "error": "internal_error",
        "detail": "Internal server error",
        "request_id": "req-500",
    }


def test_startup_fails_fast_without_model(tmp_path):
    app = create_app(Settings(model_dir=tmp_path / "missing", database_url=None))
    with pytest.raises(FileNotFoundError, match="churn-export"), TestClient(app):
        pass
