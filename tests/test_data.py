"""Tests for submission building -- the step where a silent error costs a day."""

import numpy as np
import pandas as pd
import pytest

from data import make_submission
from metrics import TARGETS


def make_sample(n=504):
    ids = pd.date_range("2025-09-24 00:00", periods=n, freq="h").astype(str)
    sub = pd.DataFrame({"id": ids})
    for col in TARGETS:
        sub[col] = 0.0
    return sub


def test_submission_preserves_id_order_and_columns(tmp_path):
    sample = make_sample()
    preds = pd.DataFrame({c: np.arange(len(sample), dtype=float) for c in TARGETS})
    out = make_submission(preds, sample, tmp_path / "s.csv")
    assert out["id"].equals(sample["id"])
    assert list(out.columns) == ["id"] + TARGETS
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
