"""Score the baselines on every validation window and log them."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from data import load_raw
from evaluation import log_results, mae_table, make_windows, score
from models import LastValue, SeasonalMean, SeasonalNaive

train = load_raw()["train"]
windows = make_windows(train, n_rolling=6)

models = {
    "persistence": lambda: LastValue(),
    "seasonal_naive_24h": lambda: SeasonalNaive(24),
    "seasonal_naive_168h": lambda: SeasonalNaive(168),
    # Diagnostic, not a PDF baseline: the 4-week mean (hour x weekday) profile.
    # Window fixed a priori at 4 weeks, not tuned. It tests whether seasonal
    # naive loses to persistence because it copies one noisy week.
    "weekly_profile_4w": lambda: SeasonalMean(lookback_days=28),
}

rows = []
for wname, (hist, win) in windows.items():
    for mname, make in models.items():
        pred = make().fit(hist).predict(win)
        rows.append({"model": mname, "window": wname,
                     "window_start": win["datetime"].iloc[0], **score(win, pred)})

log_results(rows)
tab = mae_table(rows)
pd.set_option("display.width", 160)
print("=== primary and sep2023 ===")
print(tab[tab.window.isin(["primary", "sep2023"])].round(3).to_string(index=False))
print("\n=== rolling folds (fold_5 == primary): avg_MAE mean +/- std ===")
roll = tab[tab.window.str.startswith("fold_")]
print(roll.groupby("model")[["CO", "NO2", "O3", "PM10", "PM25", "avg_MAE"]].agg(["mean", "std"]).round(3).to_string())
print("\nscored hours per window (primary):",
      {c.replace("n_valeur_", ""): r for c, r in pd.DataFrame(rows).query("window=='primary'").iloc[0].items() if c.startswith("n_")})
print("scored hours per window (sep2023):",
      {c.replace("n_valeur_", ""): r for c, r in pd.DataFrame(rows).query("window=='sep2023'").iloc[0].items() if c.startswith("n_")})
