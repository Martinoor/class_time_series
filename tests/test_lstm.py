"""Leakage tests for the LSTM data pipeline (no training here: fast)."""

import numpy as np
import pandas as pd

from lstm import build_inputs, make_samples
from metrics import TARGETS


def frame(n=2000):
    ts = pd.date_range("2023-01-01", periods=n, freq="h")
    df = pd.DataFrame({"datetime": ts})
    for c in TARGETS:
        df[c] = np.arange(n, dtype=float)
    return df


def test_sample_inputs_strictly_precede_targets():
    df = frame()
    x, z = build_inputs(df, np.zeros(5), np.ones(5), TARGETS)
    xs, ys, origins = make_samples(x, z, lookback=48, horizon=24, stride=7, last_origin=1500)
    # value == row index, so the last input value must be origin-1 and the first target origin
    for i, o in enumerate(origins):
        assert xs[i, -1, 0] == o - 1
        assert ys[i, 0, 0] == o


def test_no_target_window_reaches_past_last_origin():
    df = frame()
    x, z = build_inputs(df, np.zeros(5), np.ones(5), TARGETS)
    _, ys, origins = make_samples(x, z, lookback=48, horizon=24, stride=1, last_origin=1500)
    assert (origins + 24 <= 1500).all()
    assert ys.max() < 1500


def test_missing_inputs_are_zeroed_and_flagged_not_interpolated():
    df = frame(100)
    df.loc[10:12, "valeur_NO2"] = np.nan
    x, z = build_inputs(df, np.zeros(5), np.ones(5), TARGETS)
    no2 = TARGETS.index("valeur_NO2")
    assert (x[10:13, no2] == 0).all()                 # zero, not an interpolated 10..12
    assert (x[10:13, len(TARGETS) + no2] == 1).all()  # flagged missing
    assert np.isnan(z[10:13, no2]).all()              # targets stay NaN for the masked loss
