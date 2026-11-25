import pandas as pd
import pytest

from churn.config import ID_COLUMN, RAW_FEATURES, TARGET
from churn.training.data import clean, load_raw, split_data


def test_load_raw_missing_file_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_raw(tmp_path / "nope.csv")


def test_clean_fixes_blank_total_charges_for_new_customers(raw_df):
    out = clean(raw_df)
    new = out["tenure"] == 0
    assert new.any()
    assert (out.loc[new, "TotalCharges"] == 0).all()
    assert pd.api.types.is_float_dtype(out["TotalCharges"])


def test_clean_encodes_target_and_drops_id(raw_df):
    out = clean(raw_df)
    assert set(out[TARGET].unique()) <= {0, 1}
    assert ID_COLUMN not in out.columns
    assert list(out.columns) == [*RAW_FEATURES, TARGET]


def test_clean_strips_whitespace(raw_df):
    raw_df.loc[0, "Contract"] = "  Two year "
    assert clean(raw_df).loc[0, "Contract"] == "Two year"


def test_clean_rejects_unknown_target(raw_df):
    raw_df.loc[0, TARGET] = "Maybe"
    with pytest.raises(ValueError, match="Unexpected target"):
        clean(raw_df)


def test_clean_rejects_missing_columns(raw_df):
    with pytest.raises(ValueError, match="missing expected columns"):
        clean(raw_df.drop(columns=["Contract"]))


def test_split_is_stratified_and_disjoint(raw_df):
    df = clean(raw_df)
    s = split_data(df)
    total = len(s.X_train) + len(s.X_val) + len(s.X_test)
    assert total == len(df)
    assert abs(len(s.X_test) / total - 0.2) < 0.01
    assert abs(len(s.X_val) / total - 0.2) < 0.01
    # churn rate is preserved in every split
    rate = df[TARGET].mean()
    for y in (s.y_train, s.y_val, s.y_test):
        assert abs(y.mean() - rate) < 0.03
    # no row appears in two splits
    assert not set(s.X_train.index) & set(s.X_test.index)
    assert not set(s.X_val.index) & set(s.X_test.index)
