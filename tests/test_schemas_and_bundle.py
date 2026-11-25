import json

import pytest
from pydantic import ValidationError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from churn.api.model import ModelService, risk_level
from churn.api.schemas import EXAMPLE_CUSTOMER, CustomerFeatures
from churn.bundle import META_FILE, load_bundle, trusted_types_for, write_bundle
from churn.config import TARGET
from churn.training.data import clean

# ------------------------------------------------------------------ schemas


def test_schema_accepts_every_row_of_training_shaped_data(raw_df):
    """The allowed values in the schema must cover everything the model was trained on."""
    rows = clean(raw_df).drop(columns=[TARGET]).to_dict(orient="records")
    for row in rows:
        CustomerFeatures(**row)


def test_example_customer_is_valid():
    assert CustomerFeatures(**EXAMPLE_CUSTOMER).features()["tenure"] == 1


def test_phone_consistency_rule():
    with pytest.raises(ValidationError, match="No phone service"):
        CustomerFeatures(**{**EXAMPLE_CUSTOMER, "PhoneService": "Yes"})


@pytest.mark.parametrize(
    ("p", "level"), [(0.9, "high"), (0.6, "high"), (0.59, "medium"), (0.3, "medium"), (0.29, "low")]
)
def test_risk_level_bands(p, level):
    assert risk_level(p, threshold=0.6) == level


# ------------------------------------------------------------------ bundle


def test_bundle_round_trip(model_bundle_dir):
    pipeline, meta = load_bundle(model_bundle_dir)
    assert meta["model_version"] == "7"
    assert any(t.startswith("churn.") for t in meta["trusted_types"])
    service = ModelService(pipeline, meta)
    [pred] = service.predict([CustomerFeatures(**EXAMPLE_CUSTOMER)])
    assert 0 <= pred.churn_probability <= 1


def test_load_bundle_missing_dir_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="churn-export"):
        load_bundle(tmp_path)


def test_load_bundle_rejects_incomplete_metadata(model_bundle_dir, tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    for f in model_bundle_dir.iterdir():
        (broken / f.name).write_bytes(f.read_bytes())
    meta = json.loads((broken / META_FILE).read_text())
    del meta["decision_threshold"]
    (broken / META_FILE).write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="decision_threshold"):
        load_bundle(broken)


class Suspicious(BaseEstimator, TransformerMixin):
    """A type from outside our trusted modules, standing in for a tampered model."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def test_untrusted_types_are_refused(tmp_path):
    pipeline = Pipeline([("step", Suspicious())])
    with pytest.raises(RuntimeError, match="Suspicious"):
        trusted_types_for(pipeline)
    with pytest.raises(RuntimeError):
        write_bundle(pipeline, {"model_version": "1", "decision_threshold": 0.5}, tmp_path)
