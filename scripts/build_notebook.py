"""Generate notebooks/main.ipynb from the cell list below.

Keeping the notebook as code makes it reviewable and reproducible. Run:
    python3 scripts/build_notebook.py
then execute it top to bottom (Colab, or `jupyter nbconvert --execute`).
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Paris air quality forecasting — X-HEC Time Series challenge

**Author:** Martino Orioli · **Course:** X-HEC MSc, Time Series (Capgemini)

**Task.** Forecast hourly concentrations of CO, NO₂, O₃, PM10 and PM2.5 in Paris for the 504 hours
(3 weeks) after the end of the training data, and compare simple baselines with a zero-shot
time-series foundation model (Chronos-2).

**What this notebook does**, in the order of the assignment:
1. Data understanding and preparation
2. Validation strategy and baselines (persistence, seasonal naive 168 h)
3. Zero-shot Chronos-2 forecasting
4. Comparative experiments: independent vs joint, foundation model vs LSTM
5. One targeted fix for the September level shift
6. Results and discussion
7. Final forecast and submission

The modelling code lives in `src/` and is unit-tested in `tests/` (metrics at y = 0, no-leakage
checks on every feature and validation split). This notebook calls it instead of re-implementing it.

**How to run.** Locally: open from the project folder and run all cells. In Google Colab: copy the
whole project folder (including `data/raw/`) to Google Drive, open this notebook from there, set
`DRIVE_PROJECT_DIR` in the next cell if your path differs, choose a GPU runtime, then run all.
""")

code(r"""
# --- Setup: environment, paths, seeds -------------------------------------------------------
import os, sys, subprocess, random, warnings
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
DRIVE_PROJECT_DIR = "/content/drive/MyDrive/class_time_serie"  # edit if you copied it elsewhere

if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "chronos-forecasting==2.3.2", "holidays>=0.40"], check=True)
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT_DIR = Path(DRIVE_PROJECT_DIR)
else:
    PROJECT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

assert (PROJECT_DIR / "src").exists(), f"src/ not found under {PROJECT_DIR}"
assert (PROJECT_DIR / "data" / "raw" / "train.csv").exists(), "put the Kaggle files in data/raw/"
sys.path.insert(0, str(PROJECT_DIR / "src"))
os.chdir(PROJECT_DIR)

import numpy as np
import pandas as pd
import torch
from IPython.display import Markdown, display

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)

from data import load_raw, audit, make_submission
from metrics import TARGETS
from evaluation import (make_windows, run_model, mae_table, summarize, score,
                        mae_by_horizon_block, log_results)
from models import LastValue, SeasonalNaive, SeasonalMean
from foundation import Chronos2Forecaster, load_pipeline, run_info
from adjust import SeasonallyAdjusted, seasonal_ratios, adjustment_factors
from lstm import LSTMForecaster
import plots

SHORT = {c: c.replace("valeur_", "") for c in TARGETS}
print("project:", PROJECT_DIR, "| Colab:", IN_COLAB, "| torch", torch.__version__,
      "| CUDA:", torch.cuda.is_available(), "| MPS:", torch.backends.mps.is_available())
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 1. Data understanding and preparation

### 1.1 Load and check the structure
""")
code(r"""
raw = load_raw()
train, test, sample = raw["train"], raw["test"], raw["sample_submission"]
print("train columns:", list(train.columns))
print("test columns: ", list(test.columns))
print("timestamp column: 'id' (string 'YYYY-MM-DD HH', CET) -> parsed into 'datetime'")
print("pollutant columns:", TARGETS)
display(train.head(3))

for name, d in [("train", train), ("test", test)]:
    t = d["datetime"]
    full = pd.date_range(t.min(), t.max(), freq="h")
    print(f"{name:5s}: {len(d):6d} rows  {t.min()} -> {t.max()} | sorted: {t.is_monotonic_increasing} "
          f"| duplicate timestamps: {t.duplicated().sum()} | missing hours in index: {len(full.difference(t))}")
print("gap between train end and test start:", test.datetime.min() - train.datetime.max())
print("test ids identical to sample_submission ids:", test["id"].equals(sample["id"]))
""")
md(r"""
The index is clean: hourly, sorted, no duplicated and no missing timestamps, and the test block
(504 h) starts exactly one hour after the last training row. Missingness is in the **values**, not
in the timestamps.

### 1.2 Data quality per pollutant
""")
code(r"""
info = audit(train, test)
q = info["per_target"].rename(index=SHORT)
display(q.round(3))
""")
code(r"""
_ = plots.series_overview(train)
_ = plots.missingness_heatmap(train)
""")
md(r"""
**Quality findings**
- **CO is missing for most of 2021 and early 2022** (a sensor outage of more than a year) and
  ~31 % overall. **PM10** is missing ~18 % of hours, in scattered blocks.
- **NO₂ has an outage right before the forecast origin** (see 1.4): its last observed value is
  69 h before the test period starts.
- **O₃ has zeros and a few small negative values** (sensor noise around 0). That is why plain
  MAPE is never used here, and why final predictions are clipped at 0.
- Episodes (winter particle peaks, CO spikes) reach several times the 99th percentile. They
  dominate squared errors but matter less under MAE.

### 1.3 Patterns: daily, weekly, yearly, and between pollutants
""")
code(r"""
_ = plots.daily_weekly_profiles(train)
_ = plots.autocorrelation(train)
_ = plots.correlation_matrix(train)
acf = pd.DataFrame({SHORT[c]: [train[c].autocorr(k) for k in (24, 168, 336)] for c in TARGETS},
                   index=["lag 24 h", "lag 168 h", "lag 336 h"]).round(2)
display(acf)
""")
md(r"""
**What the plots show**
- **Hourly / daily:** every pollutant has a strong 24 h cycle (ACF peaks at 1, 2, 3 … days).
  NO₂ and CO show the traffic signature: morning (7–9 h) and evening (18–21 h) rush-hour peaks.
  O₃ is the mirror image: it is produced by sunlight and destroyed by NO, so it peaks in the
  early afternoon and dips at the morning rush.
- **Weekly:** the weekday/weekend gap is large for NO₂, CO and PM10 (no morning rush at weekends).
  In the ACF, NO₂ and O₃ keep the same correlation at 168 h and 336 h, the sign of a stable
  weekly component. For PM10/PM2.5 the ACF decays steadily: particles follow weather episodes
  more than the calendar.
- **Yearly:** O₃ peaks in summer; NO₂, CO and particles are higher in winter (heating, inversions).
- **Between pollutants:** NO₂ and CO move together (traffic); PM10 and PM2.5 are strongly
  correlated (PM2.5 is a subset of PM10); O₃ is negatively correlated with NO₂.

### 1.4 The test period: gaps and distribution shift
""")
code(r"""
T = TARGETS
origin = test.datetime.min()
last_no2 = train.loc[train.valeur_NO2.last_valid_index(), "datetime"]
print(f"forecast origin {origin}; NO2 last observed {last_no2} "
      f"({(origin - last_no2) / pd.Timedelta(hours=1):.0f} h before the origin)")

# Level in the 3 weeks after 4 Sep vs the 3 weeks before, in earlier years
rows = {}
for y in range(2020, 2024):
    o = pd.Timestamp(f"{y}-09-03 23:00")
    b = train[(train.datetime >= o - pd.Timedelta(hours=504)) & (train.datetime < o)][T].mean()
    a = train[(train.datetime >= o) & (train.datetime < o + pd.Timedelta(hours=504))][T].mean()
    rows[y] = (a / b)
display(Markdown("**Ratio of the mean level in Sep 4–24 to the 3 weeks before, by year:**"))
display(pd.DataFrame(rows).T.rename(columns=SHORT).round(2))
recent = train[train.datetime >= origin - pd.Timedelta(hours=504)][T].mean().rename(SHORT)
display(Markdown("**Mean level in the 3 weeks before the test origin (2024):**"))
display(recent.round(2).to_frame("2024").T)
_ = plots.september_rebound(train)
""")
md(r"""
**Distribution shift.** The test period (4–24 September 2024) follows the August holidays. In
every year with data, **NO₂ rises sharply after the summer break** (×1.3 to ×1.8), CO less so;
particles go either way depending on weather; O₃ does not rise. The rise starts ~3 days before
4 September (schools re-open) — exactly the hours where NO₂ is unobserved in 2024. Any model that
anchors on the recent level will **under-forecast NO₂ in the test period**. This motivates the
second validation window below.

### 1.5 Missing values: how they are handled
- **Never interpolated across time.** Interpolating a gap uses the value *after* the gap; inside a
  validation or test window that is the future we are trying to predict, so the score would be
  optimistic and the method impossible to run for real. Even inside the training history, a
  causal approach is used so the same code is valid at the forecast origin.
- **Missing truth is skipped in scoring** (per pollutant), never filled.
- **Per model:** persistence uses the last *observed* value; seasonal naive fills a hole in the
  copied week from the same hour 2, 3, … weeks earlier; Chronos-2 accepts NaN in its context
  natively; the LSTM gets 0 (= training mean after scaling) **plus a missingness flag**, and its
  loss ignores missing targets.
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 2. Validation strategy and baselines

**Why not a random split?** Neighbouring hours are strongly correlated, so a random split puts
hours from the same day into both train and validation: the model is evaluated on points it has
effectively seen, and it is allowed to use the future to predict the past. The real task is to
forecast a **whole block of 504 future hours from the past only**, so validation must do exactly
that.

**Windows.** Every window is a 504 h block; models only see rows strictly before its start.
- `primary` — the last 504 h of train (13 Aug → 3 Sep 2024). The assignment's main comparison.
- `sep2023` — the same calendar weeks one year earlier (3 → 24 Sep 2023). It reproduces the
  test period's season, including the post-holiday rebound. **The final model is chosen on this
  window.** To be transparent about timing: this was decided *after* the September under-forecast
  of Chronos-2 had been noticed on this window, but *before* the LSTM and the seasonal fix were scored.
- `fold_0 … fold_5` — six consecutive 504 h folds ending at `primary` (`fold_5` = `primary`),
  used to show how much scores vary from one 3-week period to the next.
""")
code(r"""
windows = make_windows(train, n_rolling=6)
display(pd.DataFrame([{"window": k, "history ends": h.datetime.max(), "window start": w.datetime.min(),
                        "window end": w.datetime.max(), "hours": len(w),
                        **{f"scored {SHORT[c]}": int(w[c].notna().sum()) for c in TARGETS}}
                       for k, (h, w) in windows.items()]))
""")
code(r"""
ALL_ROWS, PREDS = [], {}

def add(rows_preds, name=None):
    rows, preds = rows_preds
    ALL_ROWS.extend(rows)
    for w, p in preds.items():
        PREDS[(name or rows[0]["model"], w)] = p

add(run_model(lambda: LastValue(), windows, label="persistence"))
add(run_model(lambda: SeasonalNaive(168), windows))
add(run_model(lambda: SeasonalNaive(24), windows))
add(run_model(lambda: SeasonalMean(lookback_days=28), windows, label="weekly_profile_4w"))
BASELINES = ["persistence", "seasonal_naive_24h", "seasonal_naive_168h", "weekly_profile_4w"]
tab = mae_table(ALL_ROWS)
display(tab[tab.window.isin(["primary", "sep2023"])].round(3))
display(summarize(ALL_ROWS).round(3))
""")
md(r"""
**Baselines.** Persistence repeats the last observed value for all 504 hours. Seasonal naive
168 h copies the last observed week. Seasonal naive 24 h and the 4-week weekly profile (mean by
hour × weekday over the last 28 days, window fixed in advance) are extra references, used only
to explain the result below.

**Why a 168 h lag?** Hourly urban pollution is driven by human activity on a weekly calendar:
weekdays have rush hours, weekends do not (section 1.3). A 168 h lag is the shortest lag that
matches *both* the hour of day and the day of the week, so a Monday forecast is copied from a
Monday, not from a Sunday.

**But in practice it does not beat persistence on average MAE.** Copying one specific week also
copies that week's noise and episodes. Under MAE, a central forecast is rewarded; averaging four
weeks of the same hour × weekday profile keeps the weekly shape and removes most of the noise, and
it beats both. So the weekly *shape* is real; copying a single week is just a noisy estimate of it.

**Why MAE per pollutant, not only the average?** The pollutants are on very different scales:
CO is in mg/m³ (MAE ≈ 0.05) while O₃ is in µg/m³ (MAE ≈ 15). The average is dominated by O₃ and
NO₂, and CO is invisible in it. A model can be much better or worse on CO, or on PM2.5, without the
average moving. The best model also differs by pollutant.
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 3. Zero-shot forecasting with Chronos-2

**Foundation model.** Chronos-2 is a transformer pretrained on a very large and diverse collection
of time series (real and synthetic). It learns general forecasting skills (seasonality, trends,
level shifts, uncertainty) once, and is then applied to new series. A model trained only on this
dataset knows nothing except these 4.7 years of Paris data.

**Zero-shot** means the pretrained weights are used as-is: no training, no fine-tuning on Paris
data. At prediction time the model only reads a context window (here up to 8,192 hours of
history, strictly before the forecast origin) and outputs quantile forecasts for the next 504 h.

**Two modes, same model:**
- *independent*: each pollutant is passed as its own univariate series;
- *joint*: the five pollutants are passed as one multivariate target, and Chronos-2's group
  attention lets each forecast attend to the others' history.
""")
code(r"""
pipe = load_pipeline()
RUN_INFO = run_info(pipe)
display(pd.Series(RUN_INFO, name="Chronos-2 run").to_frame())
""")
code(r"""
add(run_model(lambda: Chronos2Forecaster(mode="independent"), windows))
add(run_model(lambda: Chronos2Forecaster(mode="joint"), windows))
CHRONOS = ["chronos2_independent", "chronos2_joint"]
tab = mae_table(ALL_ROWS)
display(tab[tab.window.isin(["primary", "sep2023"]) & tab.model.isin(BASELINES + CHRONOS)].round(3))
display(summarize(ALL_ROWS).loc[BASELINES + CHRONOS].round(3))
""")
md(r"""
**Settings recorded above:** model `amazon/chronos-2`, `chronos-forecasting` version, device,
max context 8,192 h, prediction length 504 (emitted in one pass: the model outputs up to 1,024
steps directly, no autoregressive roll-out), quantiles 0.1/0.5/0.9, point forecast = **median**.

**Why the median when the metric is MAE?** For any predictive distribution, the value that
minimises the expected absolute error is its median (the mean minimises squared error). So the
0.5 quantile is the MAE-optimal point forecast. It is also robust: pollution distributions are
right-skewed with episodes, which pull the mean up but barely move the median.
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 4. Comparative experiments

### 4.1 Independent vs joint (Chronos-2)
The two Chronos-2 rows above use the same model, context and windows; only the grouping changes.
""")
code(r"""
t = mae_table(ALL_ROWS)
cmp = t[t.model.isin(CHRONOS) & t.window.isin(["primary", "sep2023", "fold_4"])].set_index(["window", "model"])
display(cmp.round(3))
""")
md(r"""
**Advantage of joint forecasting:** a pollutant can borrow information from the others. This
matters most when its own history is missing: in the `primary` window NO₂ has been unobserved for
~13 days before the origin, and joint forecasting improves NO₂ the most — the correlated series
(CO, O₃) carry the missing information.

**Risk:** errors and spurious relationships propagate across pollutants. If the relationship seen
in the context does not hold in the forecast period, one series can drag the others off.
`fold_4` covers the Paris 2024 Olympics (26 Jul – 11 Aug), an unusual traffic regime, and there
the joint model is worse than the independent one on every pollutant. The joint model is also more
variable across folds (rolling std above).

### 4.2 Foundation model vs a neural network trained on this data (LSTM)
A deliberately simple LSTM: it reads the last 336 h (5 pollutants, 5 missingness flags, calendar
features) and a linear head emits all 504 × 5 values at once. It is trained per window on that
window's history only, with masked L1 loss, Adam, and early stopping on the last 504 h of the
history. Calendar features include day-of-year, so — unlike Chronos-2 — it can learn the
September rebound from earlier years. Settings were fixed in advance, not tuned.
""")
code(r"""
LSTM_SEEDS = [42, 7, 123]
# Seed 42 on every window (comparable with the other models' rolling folds)...
add(run_model(lambda: LSTMForecaster(seed=42), windows, label="lstm"))
# ...and two extra seeds on the two decision windows, to see how much is seed noise.
seed_rows = []
for s in LSTM_SEEDS[1:]:
    r, _ = run_model(lambda: LSTMForecaster(seed=s), windows, names=["primary", "sep2023"], label=f"lstm_seed{s}")
    seed_rows += r
seeds = mae_table([r for r in ALL_ROWS if r["model"] == "lstm" and r["window"] in ("primary", "sep2023")] + seed_rows)
display(seeds.round(3))
display(seeds.groupby("window")["avg_MAE"].agg(["mean", "std", "min", "max"]).round(3))
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 5. One targeted fix: a seasonal level adjustment

The September under-forecast is a property of the *context*, not of Chronos-2 itself: its 8,192 h
window ends just short of the previous September, so it never sees the rebound. The fix
re-scales the forecast with what history says about this time of year. **The rule was fixed
before looking at its results:**
1. For each earlier year, place an origin at the same calendar time; ratio = mean of the 504 h
   after / mean of the 504 h before (both windows ≥ 50 % observed; only years fully before the
   real origin).
2. Adjust a pollutant only if every year's ratio points the same way (all > 1 or all < 1),
   with at least 2 years. Otherwise leave it.
3. Forecast × median ratio.

It only uses pollutant history before the origin and the calendar, so it is allowed and causal
(a unit test checks that changing future values does not change the factors).
""")
code(r"""
add(run_model(lambda: SeasonallyAdjusted(Chronos2Forecaster(mode="independent")), windows))
add(run_model(lambda: SeasonallyAdjusted(Chronos2Forecaster(mode="joint")), windows))
ADJUSTED = ["chronos2_independent+seasonal_adj", "chronos2_joint+seasonal_adj"]

factors = {}
for w in ["primary", "sep2023"] + [f"fold_{k}" for k in range(5)]:
    h, win = windows[w]
    factors[w] = adjustment_factors(seasonal_ratios(h, win.datetime.iloc[0]))
display(Markdown("**Factors chosen by the rule (1.0 = left unchanged):**"))
display(pd.DataFrame(factors).T.rename(columns=SHORT).round(2))
""")
md(r"""
On `sep2023` the rule picks only CO and NO₂ (from 2020–22 data alone) and reduces the NO₂ error.
At other origins, with only 2–3 earlier years, a consistent direction can happen by chance, and the
rule sometimes rescales a pollutant on noise (e.g. PM2.5 on `primary`), which hurts. Its gain on
`sep2023` is also optimistic, because the idea came from looking at that period.
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 6. Results and discussion

### 6.1 All models, per pollutant and on average
""")
code(r"""
MODELS = BASELINES + CHRONOS + ADJUSTED + ["lstm"]
tab = mae_table(ALL_ROWS)
for w in ["primary", "sep2023"]:
    display(Markdown(f"**Window `{w}`** (MAE; lower is better)"))
    display(tab[tab.window == w].set_index("model").loc[MODELS].drop(columns="window").round(3))
    _ = plots.mae_comparison(tab, w, MODELS)
display(Markdown("**Summary: average MAE per window and spread over the rolling folds**"))
SUMMARY = summarize(ALL_ROWS).loc[MODELS]
display(SUMMARY.round(3))
best_per = {w: tab[(tab.window == w) & tab.model.isin(MODELS)].set_index("model")[[SHORT[c] for c in TARGETS] + ["avg_MAE"]].idxmin()
            for w in ["primary", "sep2023"]}
display(Markdown("**Best model per pollutant:**"))
display(pd.DataFrame(best_per))
""")
md(r"""
### 6.2 Does the error grow over the 504-hour horizon?
MAE for week 1, week 2 and week 3 of the window.
""")
code(r"""
for w in ["primary", "sep2023"]:
    truth = windows[w][1]
    blocks = {m: mae_by_horizon_block(truth, PREDS[(m, w)]) for m in
              ["persistence", "seasonal_naive_168h", "chronos2_joint", "lstm"]}
    display(Markdown(f"**`{w}`**"))
    display(pd.concat(blocks, names=["model"]).round(3))
    _ = plots.horizon_profile(blocks, f"results_horizon_{w}", f"MAE by week of the horizon, window '{w}'")
""")
md(r"""
### 6.3 Observed vs predicted
""")
code(r"""
for w in ["primary", "sep2023"]:
    _ = plots.observed_vs_predicted(windows[w][1],
        {m: PREDS[(m, w)] for m in ["seasonal_naive_168h", "chronos2_joint", "lstm"]},
        f"results_obs_pred_{w}", f"Observed vs predicted, window '{w}'")
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 7. Final forecast and submission

**Selection rule:** the model with the lowest average MAE on `sep2023`, the window that matches the
test season (fixed before the LSTM and the seasonal fix were scored; see section 2). For the LSTM, the mean over its three seeds is
used, since a single seed is not a stable estimate. The competition's leaderboard scores are
never used for selection.
""")
code(r"""
sel = summarize(ALL_ROWS)["sep2023"].loc[MODELS].copy()
sel.loc["lstm"] = seeds[seeds.window == "sep2023"]["avg_MAE"].mean()
sel = sel.sort_values()
display(sel.round(3).to_frame("sep2023 avg MAE (LSTM: mean of 3 seeds)"))
FINAL_MODEL = sel.index[0]
print("final model:", FINAL_MODEL)
""")
code(r"""
FACTORIES = {
    "persistence": lambda: LastValue(),
    "seasonal_naive_24h": lambda: SeasonalNaive(24),
    "seasonal_naive_168h": lambda: SeasonalNaive(168),
    "weekly_profile_4w": lambda: SeasonalMean(lookback_days=28),
    "chronos2_independent": lambda: Chronos2Forecaster(mode="independent"),
    "chronos2_joint": lambda: Chronos2Forecaster(mode="joint"),
    "chronos2_independent+seasonal_adj": lambda: SeasonallyAdjusted(Chronos2Forecaster(mode="independent")),
    "chronos2_joint+seasonal_adj": lambda: SeasonallyAdjusted(Chronos2Forecaster(mode="joint")),
    "lstm": lambda: LSTMForecaster(seed=42),
}
final = FACTORIES[FINAL_MODEL]().fit(train)          # all of train = history before the test origin
test_pred = final.predict(test)
test_pred["id"] = test["id"].to_numpy()
out_path = PROJECT_DIR / "submissions" / f"final_{FINAL_MODEL.replace('+', '_')}.csv"
submission = make_submission(test_pred, sample, out_path)

# Format checks required by the assignment
written = pd.read_csv(out_path)
checks = {
    "columns identical to sample, same order": list(written.columns) == list(sample.columns[:6]),
    "ids identical to sample, same order": written["id"].equals(sample["id"]),
    "504 rows": len(written) == 504,
    "no missing predictions": int(written.isna().sum().sum()) == 0,
    "no negative concentrations": bool((written.iloc[:, 1:] >= 0).all().all()),
}
display(pd.Series(checks, name="check"))
assert all(checks.values())
print("wrote", out_path.relative_to(PROJECT_DIR))
display(written.head())
""")
code(r"""
log_results(ALL_ROWS + seed_rows)   # results/leaderboard_local.csv
print("results logged:", len(ALL_ROWS) + len(seed_rows), "rows")
""")
md(r"""
## 8. Limitations (expanded in the report)
- **Data:** long sensor outages (CO in 2021, NO₂ right before the test origin) mean every model
  starts NO₂ from stale information, and only four earlier Septembers exist to learn the
  post-holiday rebound from.
- **Evaluation:** one 504 h window is a small sample with strong autocorrelation, so differences
  of a few percent between models are within the fold-to-fold spread. The selection window was
  chosen after noticing the September shift, and the seasonal fix was designed after seeing it.
- **Model:** Chronos-2's context (8,192 h) cannot reach the previous year's same season, and
  zero-shot means it cannot learn location-specific calendar effects; no weather information is
  used, which drives particles and ozone.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"},
                  "accelerator": "GPU", "colab": {"provenance": []}}
out = ROOT / "notebooks" / "main.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
