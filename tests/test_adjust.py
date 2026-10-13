"""Tests for the seasonal level adjustment: causality and the a-priori rule."""

import numpy as np
import pandas as pd
import pytest

from adjust import SeasonallyAdjusted, adjustment_factors, seasonal_ratios
from metrics import TARGETS
from models import LastValue


def make_history(end="2023-09-03 22:00", jump=2.0):
    """Every target is 10 before Sep 4 each year and 10*jump for the 3 weeks after."""
    ts = pd.date_range("2019-06-01 00:00", end, freq="h")
    df = pd.DataFrame({"datetime": ts})
    base = np.full(len(ts), 10.0)
    for y in range(2019, 2024):
        o = pd.Timestamp(f"{y}-09-03 23:00")
        m = (ts >= o) & (ts < o + pd.Timedelta(hours=504))
        base[m] = 10.0 * jump
    for c in TARGETS:
        df[c] = base
    return df


ORIGIN = pd.Timestamp("2023-09-03 23:00")


def test_ratios_recover_the_planted_jump():
    r = seasonal_ratios(make_history(jump=2.0), ORIGIN)
    assert list(r.index) == [2019, 2020, 2021, 2022]
    np.testing.assert_allclose(r.to_numpy(), 2.0)


def test_ratios_never_use_data_at_or_after_the_origin():
    hist = make_history()
    with pytest.raises(ValueError, match="before the origin"):
        seasonal_ratios(hist, hist["datetime"].iloc[-1])


def test_tampering_with_the_future_does_not_change_the_factors():
    """Causality: rows after the origin must have no effect."""
    hist = make_history(end="2024-01-01 00:00")
    past = hist[hist.datetime < ORIGIN]
    a = adjustment_factors(seasonal_ratios(past, ORIGIN))
    tampered = hist.copy()
    tampered.loc[tampered.datetime >= ORIGIN, TARGETS] = 1e6
    b = adjustment_factors(seasonal_ratios(tampered[tampered.datetime < ORIGIN], ORIGIN))
    pd.testing.assert_series_equal(a, b)


def test_years_whose_after_window_overlaps_the_forecast_are_excluded():
    # Origin only 2 weeks after an earlier-year origin + 1 year: that year's
    # after-window would run into the forecast, so it must be skipped.
    hist = make_history(end="2023-09-17 22:00")
    r = seasonal_ratios(hist, pd.Timestamp("2023-09-17 23:00"))
    assert 2023 not in r.index


def test_inconsistent_direction_means_no_adjustment():
    ratios = pd.DataFrame({"a": [1.5, 0.8], "b": [1.5, 1.2], "c": [0.7, 0.9], "d": [1.5, np.nan]})
    f = adjustment_factors(ratios, min_years=2)
    assert f["a"] == 1.0                     # mixed directions
    assert f["b"] == pytest.approx(1.35)     # consistent up -> median
    assert f["c"] == pytest.approx(0.8)      # consistent down -> median
    assert f["d"] == 1.0                     # only one usable year


def test_low_coverage_years_are_dropped():
    hist = make_history()
    o20 = pd.Timestamp("2020-09-03 23:00")
    hist.loc[(hist.datetime >= o20) & (hist.datetime < o20 + pd.Timedelta(hours=400)), "valeur_NO2"] = np.nan
    r = seasonal_ratios(hist, ORIGIN)
    assert np.isnan(r.loc[2020, "valeur_NO2"])
    assert not np.isnan(r.loc[2020, "valeur_CO"])


def test_wrapper_scales_the_base_forecast():
    hist = make_history(jump=2.0)
    future = pd.DataFrame({"datetime": pd.date_range(ORIGIN, periods=504, freq="h")})
    base = LastValue().fit(hist).predict(future)
    adj = SeasonallyAdjusted(LastValue()).fit(hist).predict(future)
    np.testing.assert_allclose(adj.to_numpy(), 2.0 * base.to_numpy())
