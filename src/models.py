"""Baselines and models.

Everything here follows one interface: `fit(train_df)` then
`predict(index_or_frame)` returning a DataFrame of the five target columns
indexed like the request. That makes baselines and learned models
interchangeable inside the backtest loop.

The baselines are not decoration. Seasonal naive on hourly urban air quality is
genuinely strong -- the daily traffic cycle is the dominant signal -- so a model
that does not beat `SeasonalNaive(WEEK)` is not a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import DAY, WEEK  # noqa: F401  (re-exported for convenience)
from metrics import TARGETS

SEED = 42


class BaseForecaster:
    """Minimal interface shared by baselines and learned models."""

    name = "base"

    def fit(self, train: pd.DataFrame, time_col: str = "datetime"):
        raise NotImplementedError

    def predict(self, future: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
        raise NotImplementedError


class LastValue(BaseForecaster):
    """Naive: carry the last observed value forward across the whole horizon.

    The weakest sensible reference. On a 504-hour horizon it is bad by
    construction because it throws away the daily cycle, but it bounds the
    problem from below and the report needs that number.
    """

    name = "naive_last"

    def __init__(self, targets: list[str] | None = None):
        self.targets = list(targets) if targets is not None else list(TARGETS)
        self.last_: pd.Series | None = None

    def fit(self, train: pd.DataFrame, time_col: str = "datetime"):
        tr = train.sort_values(time_col)
        # Last *observed* value per target, not the last row (which may be NaN).
        self.last_ = tr[self.targets].ffill().iloc[-1]
        return self

    def predict(self, future: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
        if self.last_ is None:
            raise RuntimeError("call fit() before predict()")
        out = pd.DataFrame(
            np.tile(self.last_.to_numpy(), (len(future), 1)),
            columns=self.targets,
            index=future.index,
        )
        return out


class SeasonalNaive(BaseForecaster):
    """Seasonal naive: predict the value one season ago.

    With `season=WEEK` (168 h) this reproduces both the hour-of-day and the
    day-of-week pattern, which is why it is the baseline to beat. For horizons
    longer than the season, the forecast tiles the last full season forward.

    Missing values in the last season are filled from the same position in
    earlier seasons (t - 2*season, t - 3*season, ...), up to `max_back`
    seasons back. This keeps the seasonal shape and uses only past data. A
    plain forward fill would instead repeat one stale reading across the gap:
    NO2 is unobserved for the 69 h before the competition's forecast origin,
    so a forward fill would flatten most of a day and a half of the week being
    copied. Anything still missing after `max_back` seasons falls back to the
    series' last observed value.
    """

    def __init__(
        self, season: int = WEEK, targets: list[str] | None = None, max_back: int = 8
    ):
        if season < 1:
            raise ValueError(f"season must be >= 1, got {season}")
        self.season = season
        self.max_back = max_back
        self.targets = list(targets) if targets is not None else list(TARGETS)
        self.tail_: pd.DataFrame | None = None
        self.name = f"seasonal_naive_{season}h"

    def fit(self, train: pd.DataFrame, time_col: str = "datetime"):
        tr = train.sort_values(time_col)
        if len(tr) < self.season:
            raise ValueError(
                f"need at least {self.season} training rows for a {self.season}h season, "
                f"got {len(tr)}"
            )
        values = tr[self.targets].to_numpy(dtype=float)
        tail = values[-self.season :].copy()
        # Fill holes from the same hour-of-season in older seasons, newest first.
        for k in range(2, self.max_back + 2):
            hole = np.isnan(tail)
            if not hole.any():
                break
            start = len(values) - k * self.season
            if start < 0:
                break
            older = values[start : start + self.season]
            tail[hole] = older[hole]
        # Last resort: the most recent observed value of each series.
        hole = np.isnan(tail)
        if hole.any():
            last_obs = tr[self.targets].ffill().iloc[-1].to_numpy(dtype=float)
            tail[hole] = np.broadcast_to(last_obs, tail.shape)[hole]
        self.tail_ = pd.DataFrame(tail, columns=self.targets)
        return self

    def predict(self, future: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
        if self.tail_ is None:
            raise RuntimeError("call fit() before predict()")
        h = len(future)
        reps = int(np.ceil(h / self.season))
        tiled = pd.concat([self.tail_] * reps, ignore_index=True).iloc[:h]
        tiled.index = future.index
        return tiled


class SeasonalMean(BaseForecaster):
    """Climatological baseline: mean by (hour-of-day, day-of-week) over a lookback.

    Stronger and much smoother than seasonal naive, because it averages away
    the single-week noise that `SeasonalNaive` copies verbatim. Falls back to
    the hour-of-day mean, then the global mean, for cells never seen.
    """

    def __init__(self, lookback_days: int | None = 365, targets: list[str] | None = None):
        self.lookback_days = lookback_days
        self.targets = list(targets) if targets is not None else list(TARGETS)
        self.name = f"seasonal_mean_{lookback_days or 'all'}d"
        self.by_hour_dow_: pd.DataFrame | None = None
        self.by_hour_: pd.DataFrame | None = None
        self.global_: pd.Series | None = None

    def fit(self, train: pd.DataFrame, time_col: str = "datetime"):
        tr = train.sort_values(time_col).copy()
        tr[time_col] = pd.to_datetime(tr[time_col])
        if self.lookback_days is not None:
            cutoff = tr[time_col].max() - pd.Timedelta(days=self.lookback_days)
            tr = tr[tr[time_col] >= cutoff]
        tr["_hour"] = tr[time_col].dt.hour
        tr["_dow"] = tr[time_col].dt.dayofweek
        self.by_hour_dow_ = tr.groupby(["_hour", "_dow"])[self.targets].mean()
        self.by_hour_ = tr.groupby("_hour")[self.targets].mean()
        self.global_ = tr[self.targets].mean()
        return self

    def predict(self, future: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
        if self.by_hour_dow_ is None:
            raise RuntimeError("call fit() before predict()")
        ts = pd.to_datetime(future[time_col])
        keys = pd.MultiIndex.from_arrays([ts.dt.hour, ts.dt.dayofweek])
        out = self.by_hour_dow_.reindex(keys)
        # Fill any unseen (hour, dow) cell from the hour-of-day mean, then global.
        hour_fill = self.by_hour_.reindex(ts.dt.hour)
        out = out.to_numpy()
        hour_fill = hour_fill.to_numpy()
        gap = np.isnan(out)
        out[gap] = hour_fill[gap]
        still = np.isnan(out)
        if still.any():
            out[still] = np.broadcast_to(
                self.global_.to_numpy(), out.shape
            )[still]
        return pd.DataFrame(out, columns=self.targets, index=future.index)


class MovingAverage(BaseForecaster):
    """Flat forecast at the mean of the last `window` observed hours."""

    def __init__(self, window: int = WEEK, targets: list[str] | None = None):
        self.window = window
        self.targets = list(targets) if targets is not None else list(TARGETS)
        self.name = f"moving_average_{window}h"
        self.level_: pd.Series | None = None

    def fit(self, train: pd.DataFrame, time_col: str = "datetime"):
        tr = train.sort_values(time_col)
        self.level_ = tr[self.targets].iloc[-self.window :].mean()
        return self

    def predict(self, future: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
        if self.level_ is None:
            raise RuntimeError("call fit() before predict()")
        return pd.DataFrame(
            np.tile(self.level_.to_numpy(), (len(future), 1)),
            columns=self.targets,
            index=future.index,
        )
