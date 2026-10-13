"""Train and score the LSTM on each validation window (one training per window)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from data import load_raw
from evaluation import log_results, make_windows, score
from lstm import LSTMForecaster

T = ["valeur_CO", "valeur_NO2", "valeur_O3", "valeur_PM10", "valeur_PM25"]
train = load_raw()["train"]
windows = make_windows(train, n_rolling=6)
rows, logs = [], {}
order = ["primary", "sep2023"] + [f"fold_{k}" for k in range(5)]  # fold_5 == primary
for wname in order:
    hist, win = windows[wname]
    m = LSTMForecaster().fit(hist)
    pred = m.predict(win)
    s = score(win, pred)
    rows.append({"model": m.name, "window": wname, "window_start": win["datetime"].iloc[0],
                 "seconds": round(m.seconds_), "best_epoch": m.best_epoch_, **s})
    logs[wname] = m.log_
    # daily range = mean over days of (max - min) within the day: does the forecast have a daily cycle?
    day = win["datetime"].dt.floor("D").to_numpy()
    rng = lambda d: d.groupby(day).agg(lambda g: g.max() - g.min()).mean()
    amp = pd.DataFrame({"observed": rng(win[T]), "lstm": rng(pred)}).T.round(2).rename(columns=lambda c: c[7:])
    print(f"{wname:8s} avg_MAE={s['avg_mae']:.3f}  best_epoch={m.best_epoch_}  {m.seconds_:.0f}s", flush=True)
    print(amp.to_string(), flush=True)
    if wname in ("primary", "sep2023"):
        pred.assign(datetime=win["datetime"].to_numpy()).to_csv(ROOT / "results" / "preds" / f"lstm__{wname}.csv", index=False)
    log_results(rows[-1:])
rows.append({**rows[0], "window": "fold_5"})  # same training as primary
log_results(rows[-1:])
(ROOT / "results" / "lstm_training_logs.json").write_text(json.dumps(logs))
print("done", flush=True)
