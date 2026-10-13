"""Time-aware backtesting.

Random K-fold is invalid here: it trains on the future to predict the past. The
only validation used in this project is rolling-origin (a.k.a. expanding
window) backtesting, where each fold's validation block sits strictly after its
training block and has the **same length as the Kaggle test horizon** (504
hours), so a fold score is directly comparable to the leaderboard.

A `gap` is supported for completeness but defaults to 0: because the forecast
is a single block issued at the train/test boundary, the validation block
should start immediately after training ends, exactly like the real task.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZON_3W = 24 * 21  # 504 hours, the competition horizon


@dataclass(frozen=True)
class Fold:
    """One rolling-origin fold, as positional index arrays."""

    k: int
    train_idx: np.ndarray
    valid_idx: np.ndarray

    def __repr__(self) -> str:  # concise, for printing a fold list
        return (
            f"Fold(k={self.k}, train={len(self.train_idx)}h, "
            f"valid={len(self.valid_idx)}h)"
        )


def rolling_origin_folds(
    n: int,
    horizon: int = HORIZON_3W,
    n_folds: int = 4,
    min_train: int | None = None,
    gap: int = 0,
    expanding: bool = True,
) -> list[Fold]:
    """Build `n_folds` folds over `n` rows, newest fold last.

    The last fold's validation block ends at the final row, and each earlier
    fold is shifted back by one horizon. With `expanding=True` every fold
    trains on all history up to its cut (which mirrors the real submission,
    where the whole training set is available); with `expanding=False` the
    training window is held at a fixed length.

    Raises if the requested folds do not fit in `n` rows, rather than silently
    returning fewer -- a quietly shortened CV is how people end up comparing
    models on different folds.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if min_train is None:
        # Need at least a year to learn the annual cycle; fall back to 4 horizons.
        min_train = max(4 * horizon, 24 * 365)

    needed = min_train + gap + n_folds * horizon
    if n < needed:
        raise ValueError(
            f"{n} rows is too few for {n_folds} folds of horizon {horizon} "
            f"with min_train={min_train} and gap={gap} (need {needed}). "
            f"Reduce n_folds, horizon or min_train."
        )

    folds = []
    for i in range(n_folds):
        # Fold 0 is the oldest; the newest fold ends on the last row.
        valid_end = n - (n_folds - 1 - i) * horizon
        valid_start = valid_end - horizon
        train_end = valid_start - gap
        train_start = 0 if expanding else max(0, train_end - min_train)
        folds.append(
            Fold(
                k=i,
                train_idx=np.arange(train_start, train_end),
                valid_idx=np.arange(valid_start, valid_end),
            )
        )
    return folds


def describe_folds(df: pd.DataFrame, folds: list[Fold], time_col: str = "datetime"):
    """Human-readable fold table, for the report's validation section."""
    ts = pd.to_datetime(df[time_col]).to_numpy()
    rows = []
    for f in folds:
        rows.append(
            {
                "fold": f.k,
                "train_start": ts[f.train_idx[0]],
                "train_end": ts[f.train_idx[-1]],
                "train_hours": len(f.train_idx),
                "valid_start": ts[f.valid_idx[0]],
                "valid_end": ts[f.valid_idx[-1]],
                "valid_hours": len(f.valid_idx),
            }
        )
    return pd.DataFrame(rows)


def assert_no_overlap(folds: list[Fold]) -> None:
    """Fail loudly if any fold trains on rows it also validates on."""
    for f in folds:
        overlap = np.intersect1d(f.train_idx, f.valid_idx)
        if overlap.size:
            raise AssertionError(
                f"fold {f.k} trains and validates on {overlap.size} shared rows"
            )
        if len(f.train_idx) and f.train_idx[-1] >= f.valid_idx[0]:
            raise AssertionError(
                f"fold {f.k} trains up to row {f.train_idx[-1]} but validates from "
                f"row {f.valid_idx[0]}: training data is in the future of the cut"
            )
