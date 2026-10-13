"""Tests for validation windows and scoring."""

import numpy as np
import pandas as pd
import pytest

from evaluation import make_windows, mae_by_horizon_block, score, split_at
from metrics import TARGETS


def make_train(start="2022-01-01 00:00", end="2024-09-03 22:00"):
    ts = pd.date_range(start, end, freq="h")
    df = pd.DataFrame({"datetime": ts})
    for i, col in enumerate(TARGETS):
        df[col] = np.arange(len(ts), dtype=float) + i
    return df


def test_windows_never_let_history_reach_the_window():
    w = make_windows(make_train(), n_rolling=3)
    for name, (hist, win) in w.items():
        assert hist["datetime"].max() < win["datetime"].min(), name
        assert len(win) == 504, name


def test_primary_window_is_the_last_504_hours():
    tr = make_train()
    hist, win = make_windows(tr)["primary"]
    assert win["datetime"].iloc[0] == pd.Timestamp("2024-08-13 23:00")
    assert win["datetime"].iloc[-1] == tr["datetime"].iloc[-1]
    assert len(hist) + len(win) == len(tr)


def test_sep2023_window_matches_the_test_calendar_weeks():
    _, win = make_windows(make_train())["sep2023"]
    assert win["datetime"].iloc[0] == pd.Timestamp("2023-09-03 23:00")
    assert win["datetime"].iloc[-1] == pd.Timestamp("2023-09-24 22:00")


def test_split_at_rejects_a_short_window():
    with pytest.raises(ValueError, match="expected 504"):
        split_at(make_train(), pd.Timestamp("2024-09-01 00:00"))


def test_score_average_is_the_mean_of_per_pollutant_maes():
    truth = pd.DataFrame({c: [0.0, 0.0] for c in TARGETS})
    pred = pd.DataFrame({c: [float(i), float(i)] for i, c in enumerate(TARGETS)})
    s = score(truth, pred)
    assert s["avg_mae"] == pytest.approx(np.mean(range(len(TARGETS))))


def test_score_skips_missing_truth_per_pollutant():
    truth = pd.DataFrame({c: [1.0, 1.0] for c in TARGETS})
    truth.loc[1, "valeur_NO2"] = np.nan
    pred = pd.DataFrame({c: [2.0, 100.0] for c in TARGETS})
    s = score(truth, pred)
    assert s["mae_valeur_NO2"] == pytest.approx(1.0)
    assert s["n_valeur_NO2"] == 1
    assert s["n_valeur_CO"] == 2


def test_horizon_blocks_cover_the_whole_window():
    truth = pd.DataFrame({c: np.zeros(504) for c in TARGETS})
    pred = pd.DataFrame({c: np.r_[np.ones(168), 2 * np.ones(168), 3 * np.ones(168)] for c in TARGETS})
    b = mae_by_horizon_block(truth, pred)
    assert list(b["block"]) == ["h1-168", "h169-336", "h337-504"]
    assert list(b["NO2"]) == [1.0, 2.0, 3.0]
