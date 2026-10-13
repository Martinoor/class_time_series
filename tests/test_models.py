"""Tests for the baselines."""

import numpy as np
import pandas as pd
import pytest

from metrics import TARGETS, mae
from models import LastValue, MovingAverage, SeasonalMean, SeasonalNaive


def make_frame(n=24 * 60, start="2023-01-01 00:00", seed=0):
    """Hourly frame with a real daily + weekly cycle plus noise."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="h")
    h = ts.hour.to_numpy()
    d = ts.dayofweek.to_numpy()
    signal = 50 + 20 * np.sin(2 * np.pi * h / 24) + 10 * (d < 5)
    df = pd.DataFrame({"datetime": ts})
    for col in TARGETS:
        df[col] = signal + rng.normal(0, 2, n)
    return df


def split(df, horizon=504):
    return df.iloc[:-horizon].copy(), df.iloc[-horizon:].copy()


def test_all_baselines_return_the_right_shape_and_columns():
    train, valid = split(make_frame())
    for model in [
        LastValue(),
        SeasonalNaive(24),
        SeasonalNaive(168),
        SeasonalMean(),
        MovingAverage(168),
    ]:
        pred = model.fit(train).predict(valid)
        assert pred.shape == (len(valid), len(TARGETS)), model.name
        assert list(pred.columns) == TARGETS, model.name
        assert pred.notna().all().all(), f"{model.name} produced NaNs"


def test_predictions_are_indexed_like_the_request():
    train, valid = split(make_frame())
    pred = SeasonalNaive(168).fit(train).predict(valid)
    assert pred.index.equals(valid.index)


def test_last_value_is_constant_and_equals_the_last_observation():
    train, valid = split(make_frame())
    pred = LastValue().fit(train).predict(valid)
    for col in TARGETS:
        assert pred[col].nunique() == 1
        assert pred[col].iloc[0] == pytest.approx(train[col].iloc[-1])


def test_last_value_skips_trailing_nans():
    train, valid = split(make_frame())
    train.loc[train.index[-3:], TARGETS] = np.nan
    pred = LastValue().fit(train).predict(valid)
    assert pred.notna().all().all()
    for col in TARGETS:
        assert pred[col].iloc[0] == pytest.approx(train[col].dropna().iloc[-1])


def test_seasonal_naive_copies_the_value_one_season_ago():
    train, valid = split(make_frame(), horizon=168)
    pred = SeasonalNaive(168).fit(train).predict(valid)
    expected = train[TARGETS].iloc[-168:].reset_index(drop=True)
    np.testing.assert_allclose(pred.to_numpy(), expected.to_numpy())


def test_seasonal_naive_tiles_when_horizon_exceeds_the_season():
    train, valid = split(make_frame(), horizon=504)
    pred = SeasonalNaive(168).fit(train).predict(valid)
    # 504 = 3 x 168, so the block must repeat three times.
    a = pred.iloc[:168].to_numpy()
    np.testing.assert_allclose(a, pred.iloc[168:336].to_numpy())
    np.testing.assert_allclose(a, pred.iloc[336:504].to_numpy())


def test_seasonal_naive_needs_a_full_season():
    short = make_frame(100)
    with pytest.raises(ValueError, match="need at least"):
        SeasonalNaive(168).fit(short)


def test_weekly_seasonal_naive_beats_last_value_on_seasonal_data():
    """The baselines must be ordered as the report will claim they are."""
    train, valid = split(make_frame())
    naive = LastValue().fit(train).predict(valid)
    seasonal = SeasonalNaive(168).fit(train).predict(valid)
    truth = valid[TARGETS]
    assert mae(truth.to_numpy(), seasonal.to_numpy()) < mae(
        truth.to_numpy(), naive.to_numpy()
    )


def test_seasonal_mean_beats_seasonal_naive_on_noisy_data():
    """Averaging the season should beat copying one noisy week."""
    train, valid = split(make_frame(seed=1))
    truth = valid[TARGETS].to_numpy()
    sn = SeasonalNaive(168).fit(train).predict(valid).to_numpy()
    sm = SeasonalMean().fit(train).predict(valid).to_numpy()
    assert mae(truth, sm) < mae(truth, sn)


def test_seasonal_mean_handles_unseen_hour_dow_combinations():
    """A short training set misses some (hour, dow) cells; fill must cover them."""
    train = make_frame(48)  # two days only
    valid = make_frame(504, start="2023-01-03 00:00")
    pred = SeasonalMean(lookback_days=None).fit(train).predict(valid)
    assert pred.notna().all().all()


def test_moving_average_equals_the_window_mean():
    train, valid = split(make_frame())
    pred = MovingAverage(168).fit(train).predict(valid)
    for col in TARGETS:
        assert pred[col].iloc[0] == pytest.approx(train[col].iloc[-168:].mean())


def test_predict_before_fit_raises():
    _, valid = split(make_frame())
    for model in [LastValue(), SeasonalNaive(168), SeasonalMean(), MovingAverage()]:
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(valid)


def test_baselines_never_see_validation_data():
    """Fitting on train only must give identical predictions if validation changes."""
    df = make_frame()
    train, valid = split(df)
    pred_a = SeasonalMean().fit(train).predict(valid)
    tampered = valid.copy()
    tampered[TARGETS] = -999.0
    pred_b = SeasonalMean().fit(train).predict(tampered)
    pd.testing.assert_frame_equal(pred_a, pred_b)


def test_seasonal_naive_fills_gaps_from_the_same_hour_in_earlier_weeks():
    """A hole in the last week is filled from t-336, not by repeating a stale value."""
    train, valid = split(make_frame(), horizon=168)
    gappy = train.copy()
    # Knock out the final 69 hours of NO2, as at the real forecast origin.
    gappy.loc[gappy.index[-69:], "valeur_NO2"] = np.nan
    pred = SeasonalNaive(168).fit(gappy).predict(valid)
    two_weeks_back = train["valeur_NO2"].to_numpy()[-336:-168]
    # The last 69 positions of the copied week must come from two weeks back.
    np.testing.assert_allclose(pred["valeur_NO2"].to_numpy()[-69:], two_weeks_back[-69:])
    # And they must not be a flat forward fill.
    assert pred["valeur_NO2"].iloc[-69:].nunique() > 1


def test_seasonal_naive_does_not_touch_observed_values():
    train, valid = split(make_frame(), horizon=168)
    gappy = train.copy()
    gappy.loc[gappy.index[-10:], "valeur_NO2"] = np.nan
    pred = SeasonalNaive(168).fit(gappy).predict(valid)
    np.testing.assert_allclose(
        pred["valeur_NO2"].to_numpy()[:-10], train["valeur_NO2"].to_numpy()[-168:-10]
    )


def test_seasonal_naive_falls_back_to_last_value_when_history_is_empty():
    train, valid = split(make_frame(24 * 30), horizon=24)
    gappy = train.copy()
    gappy.loc[gappy.index[-24 * 20 :], "valeur_CO"] = np.nan  # longer than max_back weeks
    pred = SeasonalNaive(168, max_back=1).fit(gappy).predict(valid)
    assert pred.notna().all().all()
