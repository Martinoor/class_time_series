"""Score Chronos-2 + seasonal level adjustment (rule fixed a priori, see src/adjust.py)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from adjust import SeasonallyAdjusted
from data import load_raw
from evaluation import log_results, mae_table, make_windows, score
from foundation import Chronos2Forecaster

train = load_raw()["train"]
windows = make_windows(train, n_rolling=6)
rows, factors = [], {}
for wname, (hist, win) in windows.items():
    for mode in ("independent", "joint"):
        m = SeasonallyAdjusted(Chronos2Forecaster(mode=mode)).fit(hist)
        pred = m.predict(win)
        rows.append({"model": m.name, "window": wname, "window_start": win["datetime"].iloc[0], **score(win, pred)})
        factors[wname] = m.factors_.rename(lambda c: c[7:]).round(2)
        if wname in ("primary", "sep2023"):
            out = ROOT / "results" / "preds"
            pred.assign(datetime=win["datetime"].to_numpy()).to_csv(out / f"{m.name}__{wname}.csv", index=False)

log_results(rows)
print("adjustment factors chosen by the rule (1.0 = untouched):")
print(pd.DataFrame(factors).T.to_string())
lb = pd.read_csv(ROOT / "results" / "leaderboard_local.csv")
tab = mae_table(lb.to_dict("records"))
keep = ["persistence", "seasonal_naive_24h", "seasonal_naive_168h", "weekly_profile_4w",
        "chronos2_independent", "chronos2_joint",
        "chronos2_independent+seasonal_adj", "chronos2_joint+seasonal_adj"]
tab = tab[tab.model.isin(keep)]
for w in ("sep2023", "primary"):
    print(f"\n=== {w} ===")
    print(tab[tab.window == w].set_index("model").loc[keep].drop(columns="window").round(3).to_string())
r = tab[tab.window.str.startswith("fold_")]
print("\n=== rolling folds avg_MAE ===")
print(r.groupby("model")["avg_MAE"].agg(["mean", "std"]).loc[keep].round(3).to_string())
