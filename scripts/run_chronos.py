"""Score zero-shot Chronos-2 (joint and independent) on every validation window."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from data import load_raw
from evaluation import log_results, mae_table, make_windows, score
from foundation import Chronos2Forecaster, load_pipeline, run_info

train = load_raw()["train"]
windows = make_windows(train, n_rolling=6)

rows, preds = [], {}
for wname, (hist, win) in windows.items():
    for mode in ("independent", "joint"):
        m = Chronos2Forecaster(mode=mode).fit(hist)
        pred = m.predict(win)
        preds[(m.name, wname)] = pred
        rows.append({"model": m.name, "window": wname, "window_start": win["datetime"].iloc[0],
                     "seconds": round(m.seconds_, 1), **score(win, pred)})
        print(f"{wname:8s} {mode:11s} avg_MAE={rows[-1]['avg_mae']:.3f}  ({m.seconds_:.1f}s)", flush=True)

log_results(rows)
# Keep primary-window predictions for plots and horizon analysis.
out = ROOT / "results" / "preds"
out.mkdir(parents=True, exist_ok=True)
for (name, wname), p in preds.items():
    if wname in ("primary", "sep2023"):
        p.assign(datetime=windows[wname][1]["datetime"].to_numpy()).to_csv(out / f"{name}__{wname}.csv", index=False)
(ROOT / "results" / "chronos_run_info.json").write_text(json.dumps(run_info(load_pipeline()), indent=2))

tab = mae_table(rows)
print("\n=== primary and sep2023 ===")
print(tab[tab.window.isin(["primary", "sep2023"])].round(3).to_string(index=False))
r = tab[tab.window.str.startswith("fold_")]
print("\n=== rolling folds avg_MAE mean +/- std ===")
print(r.groupby("model")[["CO", "NO2", "O3", "PM10", "PM25", "avg_MAE"]].agg(["mean", "std"]).round(3).to_string())
