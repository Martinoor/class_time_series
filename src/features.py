"""Feature engineering for the Paris air-quality forecast.

Leakage discipline, which is the whole point of this module:

* **Pollutant features are strictly causal.** A lag-k feature at time t holds
  the value at t-k, k >= 1. A rolling feature at time t is computed on a window
  that ends at t-1 (never including t itself). Both are produced by shifting
  first and only then rolling, so no future value can enter a row.
* **Calendar features are exogenous and known in advance**, so hour/day/month
  and French holidays may be used contemporaneously at time t.
* **Weather is exogenous too, and the competition rules explicitly permit using
  test-period weather as if perfectly forecast**, so weather may also be used
  at time t. This is unusual and is flagged in the report.

The forecast is a 504-hour (3-week) block made in one go, with no observed
pollutant values inside the test window. So a lag-1 feature is *not available*
for most of the horizon. Use `min_lag` to restrict the lag set to horizons that
are actually computable: for a block forecast of H hours, every pollutant lag
must be >= H, or the feature must come from a recursive scheme. The
`direct` strategy below is the safe default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGETS = ["valeur_CO", "valeur_NO2", "valeur_O3", "valeur_PM10", "valeur_PM25"]

# Hourly data: these are the seasonalities that matter for urban air quality.
HOUR = 1
DAY = 24
WEEK = 24 * 7


def add_calendar_features(df: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
    """Hour / day-of-week / month / holiday features, all known ahead of time.

    Cyclical encodings are used for hour and day-of-week so that hour 23 sits
    next to hour 0 for distance-based and linear models. Tree models can use
    the raw integers, which are kept as well.
    """
    out = df.copy()
    ts = pd.to_datetime(out[time_col])

    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["day"] = ts.dt.day
    out["month"] = ts.dt.month
    out["dayofyear"] = ts.dt.dayofyear
    out["weekofyear"] = ts.dt.isocalendar().week.astype(int).to_numpy()
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    # Rush hours drive NO2 and CO in particular.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)

    out["is_holiday"] = _french_holiday_flag(ts)
    return out


def _french_holiday_flag(ts: pd.Series) -> np.ndarray:
    """1 on French public holidays, 0 otherwise.

    Uses the `holidays` package when installed; falls back to the fixed-date
    French holidays, which covers 8 of the 11 (the Easter-linked ones are
    moveable and are skipped in the fallback). The fallback is deliberately
    silent-but-documented rather than an exception, so the pipeline still runs
    in a fresh kernel without the optional dependency.
    """
    dates = pd.to_datetime(ts).dt.date
    try:
        import holidays as _holidays

        fr = _holidays.France(years=sorted({d.year for d in dates}))
        return np.fromiter((1 if d in fr else 0 for d in dates), dtype=int, count=len(dates))
    except ImportError:
        fixed = {(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)}
        return np.fromiter(
            (1 if (d.month, d.day) in fixed else 0 for d in dates), dtype=int, count=len(dates)
        )


def add_lag_features(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    lags: list[int] = (DAY, 2 * DAY, WEEK, 2 * WEEK),
    min_lag: int = 1,
) -> pd.DataFrame:
    """Append `{col}_lag{k}` for each column and each lag k.

    `min_lag` guards against building a feature the forecast cannot have: when
    predicting a block of H hours with no observations inside it, pass
    `min_lag=H` and any shorter lag is rejected loudly rather than silently
    leaking an unavailable value.
    """
    cols = list(cols) if cols is not None else list(TARGETS)
    bad = [k for k in lags if k < min_lag]
    if bad:
        raise ValueError(
            f"lags {bad} are shorter than min_lag={min_lag}; they would not be "
            f"available at prediction time for a {min_lag}-step block forecast"
        )
    out = df.copy()
    for col in cols:
        for k in lags:
            out[f"{col}_lag{k}"] = out[col].shift(k)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    windows: list[int] = (DAY, WEEK),
    shift: int = DAY,
    stats: tuple[str, ...] = ("mean", "std", "min", "max"),
) -> pd.DataFrame:
    """Append rolling statistics that end `shift` hours before the current row.

    The shift is applied **before** the rolling window, so the window for row t
    spans [t-shift-window+1, t-shift] and cannot touch t. `shift` must be >= 1;
    for a block forecast of H hours it should be >= H for the same reason as
    `min_lag` above.
    """
    cols = list(cols) if cols is not None else list(TARGETS)
    if shift < 1:
        raise ValueError(f"shift must be >= 1 to stay causal, got {shift}")
    out = df.copy()
    for col in cols:
        shifted = out[col].shift(shift)
        for w in windows:
            roll = shifted.rolling(window=w, min_periods=max(2, w // 4))
            for stat in stats:
                out[f"{col}_roll{stat}{w}_s{shift}"] = getattr(roll, stat)()
    return out


def build_features(
    df: pd.DataFrame,
    horizon: int,
    time_col: str = "datetime",
    targets: list[str] | None = None,
    lags: list[int] | None = None,
    windows: list[int] = (DAY, WEEK),
) -> pd.DataFrame:
    """Full causal feature matrix for a direct `horizon`-step block forecast.

    Every pollutant-derived feature is at least `horizon` hours old, so the
    same feature set is computable for the first and the last hour of the test
    block. Calendar features are exogenous and unrestricted.

    The frame must be sorted by time and have a complete hourly index -- call
    `reindex_hourly` first, otherwise `shift(k)` counts rows, not hours, and a
    gap silently turns a lag-24 into something else.
    """
    targets = list(targets) if targets is not None else list(TARGETS)
    if lags is None:
        # Round the horizon up to whole days, then add the natural seasonal lags.
        base = int(np.ceil(horizon / DAY) * DAY)
        lags = sorted({base, base + DAY, base + WEEK, base + 2 * WEEK})

    out = df.sort_values(time_col).reset_index(drop=True)
    out = add_calendar_features(out, time_col=time_col)
    out = add_lag_features(out, cols=targets, lags=lags, min_lag=horizon)
    out = add_rolling_features(
        out, cols=targets, windows=windows, shift=max(horizon, DAY)
    )
    return out


def reindex_hourly(
    df: pd.DataFrame, time_col: str = "datetime", freq: str = "h"
) -> pd.DataFrame:
    """Return the frame on a gap-free hourly index, missing hours as NaN rows.

    Required before any shift/rolling: those operate on row position, so a
    missing hour would otherwise shift the meaning of every lag after it.
    """
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.sort_values(time_col).drop_duplicates(subset=time_col, keep="last")
    full = pd.date_range(out[time_col].min(), out[time_col].max(), freq=freq)
    out = out.set_index(time_col).reindex(full).rename_axis(time_col).reset_index()
    return out
