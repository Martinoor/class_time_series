"""Validation windows and scoring shared by every model.

Windows (see the report, section 2.1):

* ``primary``  -- the last 504 h of train (2024-08-13 23 -> 2024-09-03 22).
  The PDF asks for one validation period shared by all comparisons, and the
  final model is chosen on this window.
* ``sep2023``  -- the same calendar weeks one year earlier
  (2023-09-03 23 -> 2023-09-24 22). The primary window is the August
  holidays while the test is early September, so this checks whether
  conclusions survive the seasonal shift.
* ``fold_k``   -- rolling-origin folds ending at the primary window, used only
  to report the spread of scores (mean +/- std) for cheap models.

In every window the model sees only rows strictly before the window start.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from metrics import TARGETS, mae, smape, wape
from validation import HORIZON_3W, rolling_origin_folds

TIME_COL = "datetime"
RESULTS = Path(__file__).resolve().parents[1] / "results"

SEP2023_START = pd.Timestamp("2023-09-03 23:00")


def split_at(df: pd.DataFrame, start: pd.Timestamp, horizon: int = HORIZON_3W):
    """(history, window) with history = every row strictly before `start`."""
    ts = df[TIME_COL]
    history = df[ts < start].copy()
    window = df[(ts >= start) & (ts < start + pd.Timedelta(hours=horizon))].copy()
    if len(window) != horizon:
        raise ValueError(f"window at {start} has {len(window)} rows, expected {horizon}")
    return history, window


def make_windows(train: pd.DataFrame, n_rolling: int = 0) -> dict:
    """Named (history, window) pairs. `train` must be sorted with a full hourly index."""
    train = train.sort_values(TIME_COL).reset_index(drop=True)
    windows = {}
    primary_start = train[TIME_COL].iloc[-HORIZON_3W]
    windows["primary"] = split_at(train, primary_start)
    windows["sep2023"] = split_at(train, SEP2023_START)
    if n_rolling:
        folds = rolling_origin_folds(len(train), horizon=HORIZON_3W, n_folds=n_rolling)
        for f in folds:
            start = train[TIME_COL].iloc[f.valid_idx[0]]
            windows[f"fold_{f.k}"] = split_at(train, start)
    return windows


def score(truth: pd.DataFrame, pred: pd.DataFrame, targets=None) -> dict:
    """MAE per pollutant and their mean, plus WAPE/SMAPE extras.

    `avg_mae` is the mean of the five per-pollutant MAEs (the PDF's "average
    MAE"). Truth rows that are NaN are skipped per pollutant. `n_<target>`
    records how many hours were actually scored, since that varies with
    sensor outages.
    """
    targets = list(targets) if targets is not None else list(TARGETS)
    out = {}
    for col in targets:
        y, p = truth[col].to_numpy(), pred[col].to_numpy()
        out[f"mae_{col}"] = mae(y, p)
        out[f"wape_{col}"] = wape(y, p)
        out[f"smape_{col}"] = smape(y, p)
        out[f"n_{col}"] = int((~np.isnan(y)).sum())
    for metric in ("mae", "wape", "smape"):
        out[f"avg_{metric}"] = _mean_defined([out[f"{metric}_{c}"] for c in targets])
    return out


def _mean_defined(values) -> float:
    """Mean over the defined (non-NaN) values; NaN if none, without a warning."""
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if vals.size else np.nan


def mae_table(rows: list[dict], targets=None) -> pd.DataFrame:
    """Compact model x window table of per-pollutant MAE and average MAE."""
    targets = list(targets) if targets is not None else list(TARGETS)
    df = pd.DataFrame(rows)
    cols = ["model", "window"] + [f"mae_{c}" for c in targets] + ["avg_mae"]
    out = df[cols].copy()
    out.columns = ["model", "window"] + [c.replace("valeur_", "") for c in targets] + ["avg_MAE"]
    return out


def mae_by_horizon_block(truth, pred, block: int = 168, targets=None) -> pd.DataFrame:
    """MAE per pollutant for each `block`-hour slice of the horizon (week 1, 2, 3)."""
    targets = list(targets) if targets is not None else list(TARGETS)
    truth = truth.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    rows = []
    for b, lo in enumerate(range(0, len(truth), block), start=1):
        sl = slice(lo, lo + block)
        r = {"block": f"h{lo + 1}-{min(lo + block, len(truth))}"}
        for col in targets:
            r[col.replace("valeur_", "")] = mae(truth[col].to_numpy()[sl], pred[col].to_numpy()[sl])
        rows.append(r)
    return pd.DataFrame(rows)


def log_results(rows: list[dict], path: Path | str | None = None) -> pd.DataFrame:
    """Append to results/leaderboard_local.csv, replacing earlier rows for the same (model, window)."""
    path = Path(path) if path is not None else RESULTS / "leaderboard_local.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        keys = set(zip(new["model"], new["window"]))
        old = old[[(m, w) not in keys for m, w in zip(old["model"], old["window"])]]
        if len(old):
            new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def run_model(make, windows: dict, names=None, keep_preds=("primary", "sep2023"), label=None):
    """Fit a fresh model per window on that window's history, predict, score.

    `make` is a zero-argument factory, so no state leaks between windows.
    Returns (rows, preds) where preds maps window name -> prediction frame
    (only for `keep_preds`, to keep memory small).
    """
    rows, preds = [], {}
    for wname in names or list(windows):
        hist, win = windows[wname]
        model = make()
        pred = model.fit(hist).predict(win)
        name = label or model.name
        rows.append({"model": name, "window": wname, "window_start": win[TIME_COL].iloc[0], **score(win, pred)})
        if wname in keep_preds:
            preds[wname] = pred
    return rows, preds


def summarize(rows: list[dict], windows=("primary", "sep2023")) -> pd.DataFrame:
    """One line per model: avg MAE on the named windows plus rolling-fold mean/std."""
    t = mae_table(rows)
    out = t[t.window.isin(windows)].pivot(index="model", columns="window", values="avg_MAE")
    roll = t[t.window.str.startswith("fold_")].groupby("model")["avg_MAE"]
    out["rolling_mean"] = roll.mean()
    out["rolling_std"] = roll.std()
    return out
