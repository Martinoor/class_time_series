"""Tests for rolling-origin backtesting."""

import numpy as np
import pandas as pd
import pytest

from validation import (
    HORIZON_3W,
    assert_no_overlap,
    describe_folds,
    rolling_origin_folds,
)

N = 24 * 365 * 3  # three years of hourly data


def test_horizon_is_three_weeks():
    assert HORIZON_3W == 504


def test_every_fold_validates_exactly_one_horizon():
    folds = rolling_origin_folds(N, horizon=504, n_folds=4)
    assert len(folds) == 4
    assert all(len(f.valid_idx) == 504 for f in folds)


def test_training_is_always_strictly_before_validation():
    """The property that makes this valid time-series CV."""
    folds = rolling_origin_folds(N, horizon=504, n_folds=4)
    for f in folds:
        assert f.train_idx.max() < f.valid_idx.min()
    assert_no_overlap(folds)


def test_folds_do_not_overlap_each_other_in_validation():
    folds = rolling_origin_folds(N, horizon=504, n_folds=4)
    seen = np.concatenate([f.valid_idx for f in folds])
    assert len(seen) == len(np.unique(seen)), "validation blocks overlap"


def test_expanding_training_sets_grow():
    folds = rolling_origin_folds(N, horizon=504, n_folds=4, expanding=True)
    sizes = [len(f.train_idx) for f in folds]
    assert sizes == sorted(sizes)
    assert all(f.train_idx[0] == 0 for f in folds)


def test_sliding_window_training_sets_are_constant():
    folds = rolling_origin_folds(
        N, horizon=504, n_folds=4, min_train=24 * 365, expanding=False
    )
    sizes = {len(f.train_idx) for f in folds}
    assert len(sizes) == 1


def test_last_fold_ends_on_the_final_row():
    folds = rolling_origin_folds(N, horizon=504, n_folds=4)
    assert folds[-1].valid_idx[-1] == N - 1


def test_gap_is_respected():
    folds = rolling_origin_folds(N, horizon=504, n_folds=2, gap=48)
    for f in folds:
        assert f.valid_idx[0] - f.train_idx[-1] == 49  # 48h gap + 1


def test_too_few_rows_raises_rather_than_shrinking_cv():
    with pytest.raises(ValueError, match="too few"):
        rolling_origin_folds(1000, horizon=504, n_folds=4)


def test_bad_horizon_raises():
    with pytest.raises(ValueError, match="horizon must be"):
        rolling_origin_folds(N, horizon=0)


def test_assert_no_overlap_catches_a_bad_fold():
    from validation import Fold

    bad = [Fold(k=0, train_idx=np.arange(0, 100), valid_idx=np.arange(50, 150))]
    with pytest.raises(AssertionError):
        assert_no_overlap(bad)


def test_describe_folds_reports_real_timestamps():
    df = pd.DataFrame({"datetime": pd.date_range("2021-01-01", periods=N, freq="h")})
    folds = rolling_origin_folds(N, horizon=504, n_folds=3)
    table = describe_folds(df, folds)
    assert len(table) == 3
    assert (table["valid_hours"] == 504).all()
    # Each fold's validation window must start after the previous one's.
    assert table["valid_start"].is_monotonic_increasing
    span = table.loc[0, "valid_end"] - table.loc[0, "valid_start"]
    assert span == pd.Timedelta(hours=503)
