"""Forecast error metrics.

The official Kaggle metric for this competition is MAE. The others are reported
alongside it because the course asks for them, and because MAE alone is
misleading when five pollutants live on different scales.

Every function here takes array-likes, drops pairs where the truth is NaN, and
returns a plain float. y = 0 is handled explicitly: see `smape` and `wape`.
"""

from __future__ import annotations

import numpy as np

# Pollutant columns, in the order the submission file expects them.
TARGETS = ["valeur_CO", "valeur_NO2", "valeur_O3", "valeur_PM10", "valeur_PM25"]


def _clean(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Flatten to 1-D float arrays and drop positions where y_true is NaN.

    Missing ground truth is a property of the data (sensor outages), not of the
    forecast, so those positions must not count toward the error.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    keep = ~np.isnan(y_true)
    if not keep.any():
        return np.array([]), np.array([])
    return y_true[keep], y_pred[keep]


def mae(y_true, y_pred) -> float:
    """Mean absolute error. The official competition metric."""
    yt, yp = _clean(y_true, y_pred)
    if yt.size == 0:
        return np.nan
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error. Reported for context: it punishes missed peaks."""
    yt, yp = _clean(y_true, y_pred)
    if yt.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error = sum|e| / sum|y|, as a fraction.

    Scale-free, so it makes errors comparable across pollutants, and it is
    well-defined whenever at least one y is non-zero. Individual y = 0 rows are
    harmless here because the normalisation is over the *sum*, not per row --
    this is exactly why WAPE is preferred to MAPE on this data.
    """
    yt, yp = _clean(y_true, y_pred)
    if yt.size == 0:
        return np.nan
    denom = float(np.sum(np.abs(yt)))
    if denom == 0.0:
        # Every observed value is zero: a percentage error is undefined.
        return np.nan
    return float(np.sum(np.abs(yt - yp)) / denom)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE as a fraction, in [0, 2].

    Uses the convention smape = mean( |e| / ((|y| + |yhat|) / 2) ) with the
    y = yhat = 0 case defined as 0 error rather than 0/0. That case is real
    here: a pollutant can read exactly 0 and a model can predict exactly 0,
    and calling that a 100% error would be wrong.
    """
    yt, yp = _clean(y_true, y_pred)
    if yt.size == 0:
        return np.nan
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    num = np.abs(yt - yp)
    # Where the denominator is 0, numerator is necessarily 0 too -> ratio 0.
    ratio = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)
    return float(np.mean(ratio))


def mase(y_true, y_pred, y_naive) -> float:
    """MAE scaled by the MAE of a naive forecast. < 1 means we beat the naive.

    `y_naive` must be the naive prediction for the *same* positions as y_pred
    (e.g. the seasonal-naive forecast), not the in-sample training series.
    """
    yt, yp = _clean(y_true, y_pred)
    _, yn = _clean(y_true, y_naive)
    if yt.size == 0:
        return np.nan
    denom = float(np.mean(np.abs(yt - yn)))
    if denom == 0.0:
        return np.nan
    return float(np.mean(np.abs(yt - yp)) / denom)


def evaluate(y_true, y_pred, y_naive=None) -> dict[str, float]:
    """All metrics at once, for one pollutant or one pooled array."""
    out = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
    if y_naive is not None:
        out["mase"] = mase(y_true, y_pred, y_naive)
    return out


def evaluate_frame(df_true, df_pred, targets=None, df_naive=None):
    """Per-pollutant metrics plus the pooled MAE that Kaggle scores.

    Returns a DataFrame indexed by pollutant with a final `POOLED` row. The
    pooled MAE is the average over all (timestamp x pollutant) cells, which is
    the literal reading of the competition's formula; the per-pollutant rows
    exist because that pooled number is dominated by the large-scale
    pollutants and hides what the model is actually doing on CO.
    """
    import pandas as pd

    targets = list(targets) if targets is not None else list(TARGETS)
    rows = {}
    for col in targets:
        naive = df_naive[col] if df_naive is not None else None
        rows[col] = evaluate(df_true[col], df_pred[col], naive)

    pooled_naive = df_naive[targets].to_numpy() if df_naive is not None else None
    rows["POOLED"] = evaluate(
        df_true[targets].to_numpy(), df_pred[targets].to_numpy(), pooled_naive
    )
    return pd.DataFrame(rows).T
