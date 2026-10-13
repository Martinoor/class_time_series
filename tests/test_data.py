"""Tests for submission building -- the step where a silent error costs a day."""

import numpy as np
import pandas as pd
import pytest

from data import make_submission
from metrics import TARGETS


# The real sample file lists NO2 before CO.
SAMPLE_ORDER = ["valeur_NO2", "valeur_CO", "valeur_O3", "valeur_PM10", "valeur_PM25"]


def make_sample(n=504):
    ids = pd.date_range("2024-09-03 23:00", periods=n, freq="h").strftime("%Y-%m-%d %H")
    sub = pd.DataFrame({"id": ids})
    for col in SAMPLE_ORDER:
        sub[col] = 0.0
    return sub


def test_submission_preserves_id_order_and_columns(tmp_path):
    sample = make_sample()
    preds = pd.DataFrame({c: np.arange(len(sample), dtype=float) for c in TARGETS})
    out = make_submission(preds, sample, tmp_path / "s.csv")
    assert out["id"].equals(sample["id"])
    assert list(out.columns) == list(sample.columns)
    assert len(out) == len(sample)


def test_submission_is_written_to_disk(tmp_path):
    sample = make_sample()
    preds = pd.DataFrame({c: np.ones(len(sample)) for c in TARGETS})
    path = tmp_path / "nested" / "s.csv"
    make_submission(preds, sample, path)
    assert path.exists()
    assert len(pd.read_csv(path)) == len(sample)


def test_submission_aligns_on_id_when_predictions_are_shuffled(tmp_path):
    sample = make_sample()
    preds = pd.DataFrame({"id": sample["id"], **{c: np.arange(504, dtype=float) for c in TARGETS}})
    shuffled = preds.sample(frac=1.0, random_state=0)
    out = make_submission(shuffled, sample, tmp_path / "s.csv")
    # Row i must still carry prediction i despite the shuffle.
    assert out.loc[0, "valeur_NO2"] == 0.0
    assert out.loc[503, "valeur_NO2"] == 503.0


def test_negative_predictions_are_clipped_to_zero(tmp_path):
    sample = make_sample()
    preds = pd.DataFrame({c: np.full(len(sample), -5.0) for c in TARGETS})
    out = make_submission(preds, sample, tmp_path / "s.csv")
    assert (out[TARGETS] >= 0).all().all()


def test_missing_prediction_rows_raise(tmp_path):
    sample = make_sample()
    preds = pd.DataFrame({"id": sample["id"].iloc[:100], **{c: np.ones(100) for c in TARGETS}})
    with pytest.raises(ValueError, match="no prediction"):
        make_submission(preds, sample, tmp_path / "s.csv")


def test_nan_predictions_raise(tmp_path):
    sample = make_sample()
    preds = pd.DataFrame({c: np.ones(len(sample)) for c in TARGETS})
    preds.loc[5, "valeur_O3"] = np.nan
    with pytest.raises(ValueError, match="no prediction"):
        make_submission(preds, sample, tmp_path / "s.csv")


def test_wrong_row_count_without_ids_raises(tmp_path):
    sample = make_sample()
    preds = pd.DataFrame({c: np.ones(10) for c in TARGETS})
    with pytest.raises(ValueError, match="rows but the submission needs"):
        make_submission(preds, sample, tmp_path / "s.csv")


def test_duplicate_ids_in_predictions_raise(tmp_path):
    sample = make_sample()
    preds = pd.DataFrame({"id": sample["id"], **{c: np.ones(504) for c in TARGETS}})
    preds = pd.concat([preds, preds.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate ids"):
        make_submission(preds, sample, tmp_path / "s.csv")


def test_submission_follows_sample_column_order_not_targets_order(tmp_path):
    sample = make_sample()
    # Predictions arrive in TARGETS order (CO first)...
    preds = pd.DataFrame({c: np.ones(len(sample)) for c in TARGETS})
    out = make_submission(preds, sample, tmp_path / "s.csv")
    written = pd.read_csv(tmp_path / "s.csv")
    # ...but the file must match the sample (NO2 first).
    assert list(written.columns) == list(sample.columns)
    assert list(out.columns) == ["id"] + SAMPLE_ORDER


def test_hour_only_ids_parse_to_the_right_hour(tmp_path):
    from data import ID_FORMAT

    ts = pd.to_datetime(pd.Series(["2024-09-03 23", "2024-09-04 00"]), format=ID_FORMAT)
    assert ts.iloc[0] == pd.Timestamp("2024-09-03 23:00")
    assert ts.iloc[1] - ts.iloc[0] == pd.Timedelta(hours=1)


def test_load_raw_on_the_real_files():
    from pathlib import Path

    from data import RAW, load_raw

    if not (Path(RAW) / "train.csv").exists():
        pytest.skip("Kaggle data not downloaded")
    d = load_raw()
    assert len(d["test"]) == 504
    assert d["train"]["datetime"].is_monotonic_increasing
    assert d["test"]["datetime"].min() - d["train"]["datetime"].max() == pd.Timedelta(hours=1)


def test_submission_ignores_the_parsed_datetime_column(tmp_path):
    """`load_raw` adds a `datetime` column to the sample; it must not become a target."""
    sample = make_sample()
    sample["datetime"] = pd.to_datetime(sample["id"], format="%Y-%m-%d %H")
    preds = pd.DataFrame({c: np.ones(len(sample)) for c in TARGETS})
    make_submission(preds, sample, tmp_path / "s.csv")
    written = pd.read_csv(tmp_path / "s.csv")
    assert list(written.columns) == ["id"] + SAMPLE_ORDER
