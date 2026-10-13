"""Fit on all of train and write a dated Kaggle submission for one model.

Usage: python3 scripts/make_submission.py chronos2_joint
"""
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_raw, make_submission
from foundation import Chronos2Forecaster
from models import LastValue, SeasonalNaive

MODELS = {
    "persistence": lambda: LastValue(),
    "seasonal_naive_168h": lambda: SeasonalNaive(168),
    "chronos2_joint": lambda: Chronos2Forecaster(mode="joint"),
    "chronos2_independent": lambda: Chronos2Forecaster(mode="independent"),
}

name = sys.argv[1] if len(sys.argv) > 1 else "chronos2_joint"
d = load_raw()
pred = MODELS[name]().fit(d["train"]).predict(d["test"])
pred["id"] = d["test"]["id"].to_numpy()
path = ROOT / "submissions" / f"{date.today().isoformat()}_{name}.csv"
sub = make_submission(pred, d["sample_submission"], path)
print(f"wrote {path.relative_to(ROOT)}: {sub.shape}, columns {list(sub.columns)}")
print(sub.describe().loc[["min", "mean", "max"]].round(3).to_string())
