"""Per-customer explanations (POST /predict/explain, churn.api.explain)."""

import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from churn.api.explain import Explainer, ExplanationsUnavailable, group_columns
from churn.api.main import create_app
from churn.api.model import ModelService
from churn.api.settings import Settings
from churn.config import ENGINEERED_FEATURES, RAW_FEATURES
from churn.gate import LOYAL, RISKY

CHAMPION = Path(__file__).resolve().parents[1] / "models" / "champion"
ALL_FEATURES = set(RAW_FEATURES) | set(ENGINEERED_FEATURES)


def logit(p: float) -> float:
    return math.log(p / (1 - p))


@pytest.fixture
def client(model_bundle_dir):
    settings = Settings(model_dir=model_bundle_dir, database_url=None)
    with TestClient(create_app(settings), raise_server_exceptions=False) as c:
        yield c


# ------------------------------------------------------------------ column grouping


def test_group_columns_maps_one_hot_and_numeric_columns_to_their_field():
    columns = [
        "num__tenure",
        "num__avg_monthly_spend",
        "cat__Contract_Month-to-month",
        "cat__StreamingTV_No internet service",
        "cat__PaymentMethod_Bank transfer (automatic)",
    ]
    assert group_columns(columns) == [
        "tenure",
        "avg_monthly_spend",
        "Contract",
        "StreamingTV",
        "PaymentMethod",
    ]


def test_group_columns_refuses_unknown_columns():
    with pytest.raises(ValueError, match="Cannot map"):
        group_columns(["num__tenure", "cat__Mystery_Yes"])


# ------------------------------------------------------------------ explainer


def test_explanation_adds_up_to_the_predicted_probability(model_bundle_dir, customer):
    service = ModelService.from_dir(model_bundle_dir)
    features = {k: v for k, v in customer.items() if k != "customer_id"}
    explanation = service.explainer.explain(features)
    p = service.score([features])[0]

    total = explanation.base_value + sum(c.impact for c in explanation.contributions)
    assert total == pytest.approx(logit(p), abs=2e-3)  # rounding to 4 dp, 21 terms


def test_every_feature_appears_once_largest_first(model_bundle_dir, customer):
    service = ModelService.from_dir(model_bundle_dir)
    features = {k: v for k, v in customer.items() if k != "customer_id"}
    contributions = service.explainer.explain(features).contributions

    assert {c.feature for c in contributions} == ALL_FEATURES
    assert len(contributions) == len(ALL_FEATURES)
    impacts = [abs(c.impact) for c in contributions]
    assert impacts == sorted(impacts, reverse=True)


def test_shipped_champion_blames_the_contract_in_the_right_direction():
    """Behavioural check on the real model: a month-to-month contract pushes risk up for
    the risky profile, a two-year contract pushes it down for the loyal one."""
    service = ModelService.from_dir(CHAMPION)
    risky = {c.feature: c for c in service.explainer.explain(RISKY).contributions}
    loyal = {c.feature: c for c in service.explainer.explain(LOYAL).contributions}

    assert risky["Contract"].impact > 0
    assert loyal["Contract"].impact < 0
    assert risky["tenure"].impact > 0 > loyal["tenure"].impact
    assert risky["Contract"].value == "Month-to-month"
    # Engineered features carry their computed value: 80 total over 1 month.
    assert risky["avg_monthly_spend"].value == 80.0


def test_non_tree_models_have_no_explainer():
    from sklearn.linear_model import LogisticRegression

    from churn.training.models import build_pipeline

    with pytest.raises(ExplanationsUnavailable):
        Explainer(build_pipeline(LogisticRegression()))


# ------------------------------------------------------------------ endpoint


def test_explain_endpoint_matches_predict(client, customer):
    explained = client.post("/predict/explain", json=customer)
    assert explained.status_code == 200
    body = explained.json()
    single = client.post("/predict", json=customer).json()

    assert body["churn_probability"] == single["churn_probability"]
    assert body["risk_level"] == single["risk_level"]
    assert body["customer_id"] == customer["customer_id"]
    assert body["model_version"] == "7"
    total = body["base_value"] + sum(c["impact"] for c in body["contributions"])
    assert total == pytest.approx(logit(body["churn_probability"]), abs=2e-3)

    by_feature = {c["feature"]: c for c in body["contributions"]}
    assert by_feature["tenure"]["value"] == customer["tenure"]
    assert by_feature["Contract"]["value"] == customer["Contract"]


def test_explain_validates_input(client, customer):
    response = client.post("/predict/explain", json={**customer, "tenure": -5})
    assert response.status_code == 422
    assert response.json()["detail"][0]["field"] == "tenure"


def test_explain_requires_the_api_key_when_configured(model_bundle_dir, customer):
    settings = Settings(model_dir=model_bundle_dir, database_url=None, api_key="s3cret")
    with TestClient(create_app(settings)) as c:
        assert c.post("/predict/explain", json=customer).status_code == 401
        ok = c.post("/predict/explain", json=customer, headers={"X-API-Key": "s3cret"})
        assert ok.status_code == 200


def test_explain_returns_501_when_the_model_cannot_be_explained(client, customer):
    client.app.state.model.explainer = None
    response = client.post("/predict/explain", json=customer)
    assert response.status_code == 501
    assert "not available" in response.json()["detail"]


def test_explain_is_in_the_openapi_docs(client):
    assert "/predict/explain" in client.get("/openapi.json").json()["paths"]
