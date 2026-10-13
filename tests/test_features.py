"""Tests for the feature builder.

The central property under test: no feature at time t may contain information
from time >= t for the pollutant series. The trick used throughout is a
"poison pill" series -- the value at index i is i, so a leaked future value is
immediately identifiable by its magnitude.
"""

import numpy as np
import pandas as pd
import pytest

from features import (
    DAY,
    TARGETS,
    WEEK,
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_features,
    reindex_hourly,
)


def make_frame(n=24 * 60, start="2023-01-01 00:00"):
    """Hourly frame where every target equals its own row index."""
    ts = pd.date_range(start, periods=n, freq="h")
    df = pd.DataFrame({"datetime": ts})
    for col in TARGETS:
        df[col] = np.arange(n, dtype=float)
    return df


# --- lags -------------------------------------------------------------------


def test_lag_holds_the_past_value_exactly():
    df = make_frame(200)
    out = add_lag_features(df, cols=["valeur_NO2"], lags=[24], min_lag=1)
    # Row 100 must see row 76, i.e. value 76.
    assert out.loc[100, "valeur_NO2_lag24"] == 76.0
    assert out.loc[24, "valeur_NO2_lag24"] == 0.0


def test_lag_leading_rows_are_nan_not_backfilled():
    df = make_frame(200)
    out = add_lag_features(df, cols=["valeur_NO2"], lags=[24], min_lag=1)
    # The first 24 rows have no 24h history; they must be NaN, never filled
    # from later rows.
    assert out["valeur_NO2_lag24"].iloc[:24].isna().all()


def test_no_lag_feature_ever_exceeds_its_own_row_index():
    """The core no-leakage assertion for lags.

    With value == index, any lag feature at row t must be < t. A value >= t
    would mean a future observation leaked in.
    """
    df = make_frame(24 * 40)
    out = add_lag_features(
        df, cols=TARGETS, lags=[DAY, 2 * DAY, WEEK, 2 * WEEK], min_lag=1
    )
    lag_cols = [c for c in out.columns if "_lag" in c]
    assert lag_cols, "no lag columns were created"
    idx = np.arange(len(out), dtype=float)
    for col in lag_cols:
        vals = out[col].to_numpy()
        seen = ~np.isnan(vals)
        assert (vals[seen] < idx[seen]).all(), f"{col} contains a value from the present or future"


def test_lag_shorter_than_min_lag_is_rejected():
    df = make_frame(200)
    with pytest.raises(ValueError, match="shorter than min_lag"):
        add_lag_features(df, cols=["valeur_NO2"], lags=[1, 24], min_lag=24)


def test_block_horizon_lags_are_all_available():
    """For a 504h block forecast, every pollutant lag must be >= 504."""
    df = make_frame(24 * 400)
    out = build_features(df, horizon=504)
    lag_cols = [c for c in out.columns if "_lag" in c]
    for col in lag_cols:
        k = int(col.split("_lag")[1])
        assert k >= 504, f"{col} would not be observable 504 hours ahead"


# --- rolling windows --------------------------------------------------------


def test_rolling_window_excludes_the_current_row():
    df = make_frame(300)
    out = add_rolling_features(
        df, cols=["valeur_NO2"], windows=[24], shift=1, stats=("max",)
    )
    # Window for row 100 is [76..99] after a 1h shift, so the max is 99 < 100.
    assert out.loc[100, "valeur_NO2_rollmax24_s1"] == 99.0


def test_rolling_mean_matches_hand_computation():
    df = make_frame(300)
    out = add_rolling_features(
        df, cols=["valeur_NO2"], windows=[24], shift=1, stats=("mean",)
    )
    expected = np.mean(np.arange(76, 100))  # rows 76..99
    assert out.loc[100, "valeur_NO2_rollmean24_s1"] == pytest.approx(expected)


def test_no_rolling_feature_ever_reaches_the_current_row():
    """The core no-leakage assertion for rolling stats."""
    df = make_frame(24 * 40)
    out = add_rolling_features(
        df, cols=TARGETS, windows=[DAY, WEEK], shift=DAY, stats=("mean", "std", "min", "max")
    )
    idx = np.arange(len(out), dtype=float)
    max_cols = [c for c in out.columns if "rollmax" in c]
    assert max_cols, "no rolling-max columns were created"
    for col in max_cols:
        vals = out[col].to_numpy()
        seen = ~np.isnan(vals)
        # The max of a causal window is the tightest bound available: it must
        # still be strictly below the current row's own value (== its index).
        assert (vals[seen] < idx[seen]).all(), f"{col} saw the present or the future"


def test_rolling_shift_zero_is_rejected():
    df = make_frame(200)
    with pytest.raises(ValueError, match="must be >= 1"):
        add_rolling_features(df, cols=["valeur_NO2"], windows=[24], shift=0)


def test_rolling_respects_block_horizon_shift():
    df = make_frame(24 * 400)
    out = build_features(df, horizon=504)
    roll_cols = [c for c in out.columns if "_roll" in c]
    assert roll_cols
    for col in roll_cols:
        shift = int(col.split("_s")[-1])
        assert shift >= 504, f"{col} uses a {shift}h shift, too recent for a 504h forecast"


# --- calendar ---------------------------------------------------------------


def test_calendar_features_match_the_timestamp():
    df = pd.DataFrame({"datetime": pd.to_datetime(["2024-07-14 15:00"])})
    out = add_calendar_features(df)
    assert out.loc[0, "hour"] == 15
    assert out.loc[0, "dayofweek"] == 6  # a Sunday
    assert out.loc[0, "month"] == 7
    assert out.loc[0, "is_weekend"] == 1
    assert out.loc[0, "is_holiday"] == 1  # Bastille Day


def test_cyclical_hour_encoding_wraps_around():
    df = pd.DataFrame({"datetime": pd.to_datetime(["2024-01-01 23:00", "2024-01-02 00:00"])})
    out = add_calendar_features(df)
    d = np.hypot(
        out.loc[0, "hour_sin"] - out.loc[1, "hour_sin"],
        out.loc[0, "hour_cos"] - out.loc[1, "hour_cos"],
    )
    # Hours 23 and 0 must be adjacent on the circle, not 23 units apart.
    assert d < 0.3


def test_calendar_features_do_not_depend_on_targets():
    """Calendar features must be computable for rows with no observed target."""
    df = make_frame(50)
    df.loc[40:, TARGETS] = np.nan
    out = add_calendar_features(df)
    cal = ["hour", "dayofweek", "month", "is_weekend", "is_holiday", "hour_sin"]
    assert out.loc[40:, cal].notna().all().all()


# --- hourly reindexing ------------------------------------------------------


def test_reindex_hourly_inserts_missing_hours_as_nan():
    ts = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 04:00"]
    )
    df = pd.DataFrame({"datetime": ts, "valeur_NO2": [1.0, 2.0, 3.0]})
    out = reindex_hourly(df)
    assert len(out) == 5
    assert out["valeur_NO2"].isna().sum() == 2
    assert out["datetime"].diff().dropna().eq(pd.Timedelta(hours=1)).all()


def test_reindex_then_lag_keeps_lags_in_clock_hours():
    """A gap must not silently shorten a lag.

    Without reindexing, shift(2) over a frame with a missing hour spans 3 real
    hours. After reindexing, lag-2 means exactly 2 hours.
    """
    ts = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00"]
    )
    df = pd.DataFrame({"datetime": ts, "valeur_NO2": [10.0, 20.0, 30.0]})
    out = add_lag_features(reindex_hourly(df), cols=["valeur_NO2"], lags=[2], min_lag=1)
    row3 = out[out["datetime"] == pd.Timestamp("2024-01-01 03:00")].iloc[0]
    # 2 hours before 03:00 is 01:00, whose value is 20.
    assert row3["valeur_NO2_lag2"] == 20.0


def test_reindex_drops_duplicate_timestamps():
    ts = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 01:00"])
    df = pd.DataFrame({"datetime": ts, "valeur_NO2": [1.0, 99.0, 2.0]})
    out = reindex_hourly(df)
    assert len(out) == 2
    assert out.loc[0, "valeur_NO2"] == 99.0  # keep="last"


def test_build_features_is_row_order_independent():
    """Shuffled input must give the same features, because build sorts first."""
    df = make_frame(24 * 400)
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    a = build_features(df, horizon=504)
    b = build_features(shuffled, horizon=504)
    pd.testing.assert_frame_equal(a, b)
