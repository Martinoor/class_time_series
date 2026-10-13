"""Loading and auditing the Kaggle files.

`id` in the raw files is an hour-resolution CET timestamp string such as
"2024-09-03 23" (no minutes). It is parsed with an explicit format into a
`datetime` column; the `id` column is preserved untouched, because the
submission must carry it back verbatim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from metrics import TARGETS

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
TIME_COL = "datetime"
ID_FORMAT = "%Y-%m-%d %H"  # e.g. "2024-09-03 23"


def load_raw(raw_dir: Path | str = RAW) -> dict[str, pd.DataFrame]:
    """Read train/test/sample_submission and parse the timestamp column."""
    raw_dir = Path(raw_dir)
    missing = [
        f for f in ("train.csv", "test.csv", "sample_submission.csv")
        if not (raw_dir / f).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"missing {missing} in {raw_dir}. Download with "
            f"`kaggle competitions download -c x-hec-ts-2025-predicting-air-quality-in-paris "
            f"-p {raw_dir} && unzip -o {raw_dir}/*.zip -d {raw_dir}`, or place the files there."
        )

    out = {}
    for key in ("train", "test", "sample_submission"):
        df = pd.read_csv(raw_dir / f"{key}.csv")
        if "id" in df.columns:
            df[TIME_COL] = pd.to_datetime(df["id"], format=ID_FORMAT)
            df = df.sort_values(TIME_COL).reset_index(drop=True)
        out[key] = df
    return out


def audit(train: pd.DataFrame, test: pd.DataFrame, targets=None) -> dict:
    """Facts the report's data section needs, computed in one pass.

    Returned rather than printed so the notebook can render it and the report
    can quote it without re-deriving anything.
    """
    targets = list(targets) if targets is not None else list(TARGETS)
    present = [c for c in targets if c in train.columns]

    t0, t1 = train[TIME_COL].min(), train[TIME_COL].max()
    s0, s1 = test[TIME_COL].min(), test[TIME_COL].max()
    expected_hours = int((t1 - t0) / pd.Timedelta(hours=1)) + 1

    info = {
        "train_start": t0,
        "train_end": t1,
        "train_rows": len(train),
        "train_expected_hours": expected_hours,
        "train_missing_hours": expected_hours - train[TIME_COL].nunique(),
        "train_duplicate_timestamps": int(train[TIME_COL].duplicated().sum()),
        "test_start": s0,
        "test_end": s1,
        "test_rows": len(test),
        "test_hours_span": int((s1 - s0) / pd.Timedelta(hours=1)) + 1,
        "gap_train_to_test_hours": (s0 - t1) / pd.Timedelta(hours=1),
        "targets_present": present,
    }

    per_target = {}
    for col in present:
        s = train[col]
        per_target[col] = {
            "missing_pct": round(100 * s.isna().mean(), 2),
            "zeros": int((s == 0).sum()),
            "negatives": int((s < 0).sum()),
            "min": s.min(),
            "median": s.median(),
            "max": s.max(),
            "mean": s.mean(),
            "std": s.std(),
        }
    info["per_target"] = pd.DataFrame(per_target).T
    return info


def seasonal_shift_check(train: pd.DataFrame, test: pd.DataFrame, targets=None):
    """Compare the test window against the same calendar window in past years.

    The test block is three specific weeks, so the right reference is not the
    whole training set but the same weeks in earlier years. A large difference
    here is distribution shift the report must mention.
    """
    targets = list(targets) if targets is not None else list(TARGETS)
    present = [c for c in targets if c in train.columns]
    doy_lo = test[TIME_COL].dt.dayofyear.min()
    doy_hi = test[TIME_COL].dt.dayofyear.max()

    doy = train[TIME_COL].dt.dayofyear
    in_window = (doy >= doy_lo) & (doy <= doy_hi)
    rows = {
        "same_weeks_past_years": train.loc[in_window, present].mean(),
        "whole_train": train[present].mean(),
    }
    out = pd.DataFrame(rows)
    out["ratio"] = out["same_weeks_past_years"] / out["whole_train"].replace(0, np.nan)
    return out


def make_submission(
    predictions: pd.DataFrame,
    sample_submission: pd.DataFrame,
    path: Path | str,
    targets=None,
) -> pd.DataFrame:
    """Write a submission aligned to `sample_submission` by `id`.

    Guards the mistakes that silently cost a submission: a reordered or
    altered `id` column, missing rows, NaN or negative concentrations, and
    columns written in a different order from the sample file (which lists
    NO2 before CO, unlike `TARGETS`).
    """
    if targets is None:
        # Follow the sample file's own column order, not ours. Skip the parsed
        # timestamp that `load_raw` adds: it is ours, not part of the format.
        targets = [c for c in sample_submission.columns if c not in ("id", TIME_COL)]
    targets = list(targets)
    sub = sample_submission[["id"]].copy()

    pred = predictions.copy()
    if "id" not in pred.columns:
        if len(pred) != len(sub):
            raise ValueError(
                f"predictions have {len(pred)} rows but the submission needs {len(sub)}; "
                "pass an `id` column so rows can be aligned explicitly"
            )
        pred["id"] = sub["id"].to_numpy()

    merged = sub.merge(pred[["id"] + targets], on="id", how="left")
    if len(merged) != len(sub):
        raise ValueError("merge changed the row count: duplicate ids in the predictions")

    bad = merged[targets].isna().any(axis=1).sum()
    if bad:
        raise ValueError(f"{bad} submission rows have no prediction")

    # Concentrations cannot be negative; clipping is strictly better under MAE.
    neg = (merged[targets] < 0).to_numpy().sum()
    if neg:
        merged[targets] = merged[targets].clip(lower=0)

    if not merged["id"].equals(sample_submission["id"]):
        raise ValueError("the id column was reordered; Kaggle requires it unchanged")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged
