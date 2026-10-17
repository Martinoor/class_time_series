"""Seasonal level adjustment on top of any forecaster.

Motivation (report, sections 1 and 3): forecasters anchored on the recent level
under-predict early September, because the weeks before the origin are the
August holidays and traffic pollutants rebound afterwards. Chronos-2 cannot
learn this in context: its 8192 h (341-day) window ends just short of the
previous September.

The rule is fixed in advance, so it cannot be tuned on a validation window:

1. For each earlier year y, place an origin at the same calendar time as the
   real origin. ratio_y = mean(504 h after) / mean(504 h before), per pollutant.
   Only years whose *after* window ends strictly before the real origin are
   used, and both windows must be >= 50 % observed.
2. Adjust a pollutant only if every usable ratio_y is on the same side of 1
   (all > 1 or all < 1) and there are at least `min_years` of them.
   Otherwise leave it untouched.
3. Adjusted forecast = base forecast x median(ratio_y).

Only training history before the origin is used, so this is causal. It uses
pollutant history (allowed) and the calendar (allowed), nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import TARGETS
from models import BaseForecaster

H = 504


def seasonal_ratios(
    history: pd.DataFrame,
    origin: pd.Timestamp,
    targets=None,
    horizon: int = H,
    min_coverage: float = 0.5,
    time_col: str = "datetime",
) -> pd.DataFrame:
    """ratio_y per earlier year (rows) and pollutant (columns); NaN where unusable."""
    targets = list(targets) if targets is not None else list(TARGETS)
    ts = history[time_col]
    if ts.max() >= origin:
        raise ValueError("history must end before the origin")
    rows = {}
    for back in range(1, 20):
        o = origin - pd.DateOffset(years=back)
        if o - pd.Timedelta(hours=horizon) < ts.min():
            break
        after_end = o + pd.Timedelta(hours=horizon)
        if after_end > origin:  # its "after" window would overlap the forecast
            continue
        before = history[(ts >= o - pd.Timedelta(hours=horizon)) & (ts < o)][targets]
        after = history[(ts >= o) & (ts < after_end)][targets]
        r = {}
        for c in targets:
            ok = (before[c].notna().mean() >= min_coverage) and (after[c].notna().mean() >= min_coverage)
            r[c] = after[c].mean() / before[c].mean() if ok and before[c].mean() > 0 else np.nan
        rows[o.year] = r
    return pd.DataFrame(rows).T.sort_index()


def adjustment_factors(ratios: pd.DataFrame, min_years: int = 2) -> pd.Series:
    """Median ratio where the direction is consistent across years, else 1."""
    factors = {}
    for c in ratios.columns:
        r = ratios[c].dropna()
        consistent = len(r) >= min_years and ((r > 1).all() or (r < 1).all())
        factors[c] = float(r.median()) if consistent else 1.0
    return pd.Series(factors)


class SeasonallyAdjusted(BaseForecaster):
    """Wrap a base forecaster and rescale its forecast by `adjustment_factors`."""

    def __init__(self, base: BaseForecaster, min_years: int = 2, targets=None):
        self.base = base
        self.min_years = min_years
        self.targets = list(targets) if targets is not None else list(TARGETS)
        self.name = f"{base.name}+seasonal_adj"
        self.history_ = None
        self.ratios_: pd.DataFrame | None = None
        self.factors_: pd.Series | None = None

    def fit(self, train: pd.DataFrame, time_col: str = "datetime"):
        self.history_ = train.sort_values(time_col)
        self.base.fit(train, time_col=time_col)
        return self

    def predict(self, future: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
        origin = pd.Timestamp(future[time_col].iloc[0])
        self.ratios_ = seasonal_ratios(self.history_, origin, self.targets, horizon=len(future))
        self.factors_ = adjustment_factors(self.ratios_, self.min_years)
        pred = self.base.predict(future, time_col=time_col)
        return pred * self.factors_[self.targets].to_numpy()
