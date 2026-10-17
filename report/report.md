# Paris air quality forecasting with baselines, a zero-shot foundation model and an LSTM

**Martino Orioli** · X-HEC MSc, Time Series · October 2026
Notebook: `notebooks/main.ipynb` · Code: `src/` · Tests: `tests/`

---

## 1. Problem and data

The task is to forecast hourly concentrations of five pollutants in Paris (CO in mg/m³; NO₂, O₃,
PM10 and PM2.5 in µg/m³) for the **504 hours (3 weeks)** after the end of the training data,
4–24 September 2024. The training set runs from 1 January 2020 to 3 September 2024 (40,991 hours).
The test block starts one hour after the last training row. Evaluation is MAE, per pollutant and
on average.

**Data quality.** The hourly index is complete (no missing or duplicated timestamps). The gaps
are in the values:

| | CO | NO₂ | O₃ | PM10 | PM2.5 |
|---|---|---|---|---|---|
| missing (% of hours) | 30.6 | 8.0 | 1.7 | 17.5 | 4.4 |
| median | 0.18 | 17.6 | 50.7 | 15.6 | 8.7 |
| 99th percentile | 0.58 | 71.4 | 123.2 | 59.2 | 41.1 |
| max | 4.31 | 131.0 | 193.1 | 128.5 | 111.1 |

CO is missing for most of 2021 and early 2022. **NO₂ is unobserved for the last 69 hours before
the forecast origin**, after a 331-hour outage in August 2024. O₃ has 84 zeros and 10 small
negative values (sensor noise around zero), so plain MAPE is undefined and is never used.

**Patterns** (notebook §1.3):
- **Daily:** a strong 24 h cycle for every pollutant. NO₂ and CO peak at the morning and evening
  rush hours. O₃ peaks in the early afternoon and dips at the morning rush, because sunlight
  produces it and NO destroys it.
- **Weekly:** weekends lose the morning rush for NO₂, CO and PM10. For NO₂ and O₃, autocorrelation
  is the same at 168 h and 336 h (0.35/0.35 and 0.53/0.52), the sign of a stable weekly component.
  For particles it decays (PM2.5: 0.15 → 0.09), so they follow weather episodes more than the calendar.
- **Yearly:** O₃ is high in summer; NO₂, CO and particles are higher in winter.
- **Between pollutants:** NO₂–CO correlate at 0.71 (both traffic), PM10–PM2.5 at 0.89, and
  O₃–NO₂ at −0.64.

**Distribution shift in the test period.** The test weeks follow the August holidays. In every
year with data, NO₂ in 4–24 September is ×1.3 to ×1.8 its level in the three preceding weeks,
as traffic returns. CO rises less (×1.0–1.4), particles go either way (×0.8–2.1), and O₃ does not
rise. The rise starts about three days before 4 September, exactly the hours where NO₂ is missing
in 2024. The three weeks before the 2024 origin also have the lowest NO₂ of any year (10.6 µg/m³).
Any model that anchors on the recent level should therefore under-forecast NO₂.

## 2. Methodology

### 2.1 Validation
Every validation window is a 504-hour block, and models see only rows strictly before its start.
- **`primary`:** the last 504 h of training data (13 Aug → 3 Sep 2024). The main comparison.
- **`sep2023`:** the same calendar weeks one year earlier (3 → 24 Sep 2023). It reproduces the
  test season and its post-holiday rebound. **The final model is chosen on this window.** This was
  decided after Chronos-2's September under-forecast had been noticed there, but before the LSTM
  and the seasonal adjustment were scored.
- **Six rolling folds** ending at `primary`, used only to measure how much scores vary between
  3-week periods.

### 2.2 Models
- **Persistence:** the last observed value, repeated for 504 h.
- **Seasonal naive 168 h:** the last observed week, tiled three times. A gap in that week is
  filled from the same hour 2, 3, … weeks earlier, never by interpolation. Two extra references
  are used only to explain the baselines: seasonal naive 24 h, and a 4-week weekly profile (mean
  by hour × weekday over the last 28 days, window fixed in advance).
- **Chronos-2, zero-shot** (`amazon/chronos-2`, 119.5 M parameters, `chronos-forecasting` 2.3.2):
  context of up to 8,192 h before the origin, 504 steps emitted in one pass, point forecast = the
  median. Two modes: *independent* (each pollutant its own series) and *joint* (the five pollutants
  as one multivariate target, linked by group attention).
- **LSTM:** reads the last 336 h (5 pollutants, 5 missingness flags, calendar features: hour,
  weekday, day of year, French holidays). A linear head emits all 504 × 5 values at once. Trained
  per window with masked L1 loss and early stopping on the last 504 h of that window's history.
  Settings were fixed in advance; three seeds are reported.
- **Seasonal level adjustment** (one targeted fix): rescale a forecast by the median ratio between
  the 504 h after and the 504 h before the same calendar origin in earlier years. It is applied only
  where every earlier year points the same way. The rule was fixed before it was scored.

### 2.3 Leakage controls
- Scalers are fitted on each window's history only.
- Missing inputs are never interpolated: they are flagged, or filled from the past only.
- Missing truth is skipped when scoring.
- No weather data is used, since the assignment forbids weather observations from the
  validation or test period.
- The only external information is the French public-holiday calendar (Python `holidays` package),
  which is known in advance.
- Unit tests check each of these, using a series whose value equals its row index, so any future
  value entering a feature would be visible by its size.

## 3. Results

Average MAE (mean of the five per-pollutant MAEs); lower is better.

| model | `primary` | `sep2023` | rolling folds, mean ± std |
|---|---|---|---|
| Persistence | 6.70 | 10.63 | 7.48 ± 1.23 |
| Seasonal naive 168 h | 7.36 | 10.88 | 7.92 ± 1.01 |
| Seasonal naive 24 h *(reference)* | 6.43 | 10.43 | 7.53 ± 1.38 |
| Weekly profile 4 wk *(reference)* | 6.66 | 9.66 | 6.45 ± 0.55 |
| Chronos-2 independent | 5.87 | 9.18 | 5.94 ± 0.41 |
| Chronos-2 joint | **5.57** | 9.20 | **5.88** ± 0.83 |
| Chronos-2 indep. + seasonal adj. | 6.10 | 9.00 | 6.01 ± 0.47 |
| **Chronos-2 joint + seasonal adj. (final)** | 5.86 | 8.86 | 5.98 ± 0.90 |
| LSTM, 3 seeds (mean ± std) | 6.53 ± 0.63 | **8.84** ± 0.05 | 6.15 ± 0.74 *(seed 42 only)* |

Per-pollutant MAE on `primary`:

| model | CO | NO₂ | O₃ | PM10 | PM2.5 |
|---|---|---|---|---|---|
| Persistence | 0.046 | 7.43 | 16.16 | 5.62 | 4.24 |
| Seasonal naive 168 h | 0.044 | 9.47 | 17.30 | 6.45 | 3.56 |
| Chronos-2 independent | 0.048 | 7.71 | 12.92 | 5.42 | 3.26 |
| Chronos-2 joint | 0.037 | **6.35** | **12.79** | **5.29** | 3.37 |
| LSTM (mean of 3 seeds) | **0.035** | 8.27 | 14.67 | 5.79 | 3.89 |

Per-pollutant MAE on `sep2023` (the test season):

| model | CO | NO₂ | O₃ | PM10 | PM2.5 |
|---|---|---|---|---|---|
| Persistence | 0.060 | 15.82 | 23.52 | 7.90 | 5.82 |
| Seasonal naive 168 h | 0.064 | 15.77 | 25.35 | 7.88 | 5.32 |
| Chronos-2 independent | 0.062 | 15.85 | **18.09** | 6.78 | 5.10 |
| Chronos-2 joint | 0.061 | 15.44 | 18.22 | 6.90 | 5.36 |
| Chronos-2 joint + seasonal adj. | 0.058 | **13.76** | 18.22 | 6.90 | 5.36 |
| LSTM (seed 42) | 0.060 | 14.89 | 18.67 | **6.01** | **4.64** |

**Error over the horizon** (Chronos-2 joint, MAE by week 1 / 2 / 3):

| window | NO₂ | O₃ | PM2.5 |
|---|---|---|---|
| `primary` | 4.9 / 7.0 / 7.6 | 12.1 / 12.6 / 13.7 | 2.8 / 2.9 / 4.5 |
| `sep2023` | 25.9 / 12.8 / 7.5 | 27.7 / 17.6 / 9.4 | 9.9 / 4.5 / 1.7 |

In August the error grows with the horizon, as expected when the context ages. In September 2023
the pattern is reversed for **every** model: week 1 contains the post-holiday NO₂ surge (7-day mean
around 40 µg/m³ against about 15 before), which no model anticipated, and the later weeks are calmer.
So what drives the error is less the distance from the origin than whether the period contains a
regime change the context does not show.

**Final choice.** On `sep2023`, the agreed decision window, the LSTM (8.84, mean of 3 seeds) and
Chronos-2 joint + seasonal adjustment (8.86) differ by 0.02. That is less than the LSTM's own
seed-to-seed standard deviation on the window (0.045), so it is not evidence either way. I treated
them as tied and broke the tie on the six rolling folds, where Chronos-2 + adjustment is better
(5.98 vs 6.15). It is also better on `primary` (5.86 vs 6.53), and it is deterministic, so the
Colab run reproduces it. This tie-break was decided *after* seeing the results, and is reported
as such.

**Test result (Kaggle).** The final submission scored MAE **5.57 public / 6.80 private**. The plain
Chronos-2 joint forecast, uploaded earlier as a first reference, scored **4.92 / 6.17**. The two
forecasts differ only in NO₂ and CO, so on the test period the seasonal adjustment made the forecast
*worse*. September 2024 evidently did not rebound by the ×1.76 that the earlier years suggested for
NO₂. This is the risk flagged above: the adjustment's validation gain rested on two or three past
years, and on the same window that motivated it. I kept the final model as chosen. Switching to the
better test score after the fact would turn the test set into a validation set. The competition is
closed, so the private score was visible, but neither score was used for any decision.

**Main findings**
1. **Zero-shot Chronos-2 beats every baseline in every window.** On `primary` it is about 13 %
   better than the best baseline, and it has the lowest fold-to-fold spread.
2. **Seasonal naive 168 h does not beat persistence on average.** It copies one specific noisy
   week. Averaging four weeks of the same hour × weekday profile beats both, so the weekly shape is
   real, and the problem is the noise in a single copied week.
3. **Joint vs independent depends on the situation.** Joint wins on `primary` mostly through NO₂
   (6.35 vs 7.71), the series whose own history was missing for 13 days. It borrows from correlated
   pollutants. But joint is worse on every pollutant during the Paris Olympics fold, and twice as
   variable across folds.
4. **Every model under-forecasts in September.** On `sep2023`, Chronos-2 under-predicts NO₂ by
   45–47 %. Its 8,192-hour context ends just before the previous September, so it never sees the
   rebound. The LSTM, which gets day-of-year and trains on all years, ties for best in that window, but
   it is the weakest trained model in August and unstable across seeds there (6.00–7.22). The seasonal adjustment recovers part of the gap on `sep2023`, but it fires on
   noise at other origins.

## 4. Answers to the discussion questions

**Why is this time-series forecasting, not standard regression?** The observations are ordered
and strongly dependent in time: today resembles yesterday and last week. The goal is to
extrapolate a whole block of future values from the past alone, with no future inputs except the
calendar. Standard regression assumes independent rows and treats the features of the point being
predicted as known. Here those features (recent pollutant values) are not available 504 hours
ahead.

**Why would a random split be misleading?** Neighbouring hours are nearly identical. A random
split puts hours of the same day in both train and validation, so the model is scored on points it
has effectively already seen, and it learns from the future to predict the past. The error comes
out far too optimistic, and the setup cannot be reproduced at forecast time.

**Which patterns are visible?** Daily, weekly and yearly cycles, with the evidence summarized in
§1: rush-hour peaks and weekend dips in the profiles, ACF peaks at 24, 168 and 336 h, and seasonal
waves in the series plot.

**How are missing values handled, and why not interpolate with future data?** Missing truth is
skipped when scoring. Missing inputs are flagged and filled only from the past, or passed as NaN
to Chronos-2, which accepts them. Interpolation inside a validation or test window would use values
from the period being forecast. That is leakage: the score would be optimistic, and the method could
not run for real. The same causal rule is used inside the training history, so the code behaves
identically at the forecast origin.

**Why is a 168 h lag useful?** Urban pollution follows a weekly human calendar. 168 h is the
shortest lag that matches both the hour of day and the day of week, so a Monday 8 h forecast is
copied from a Monday 8 h. In this data, the weekly component is clear for NO₂ and O₃. Copying a
single week is still a noisy estimate of that pattern, though (finding 2).

**Why report MAE per pollutant?** The scales differ by two orders of magnitude: CO's MAE is
about 0.04, O₃'s about 13. The average is dominated by O₃ and NO₂, and CO is invisible in it. The best
model also differs by pollutant: seasonal naive 168 h is competitive on PM2.5, while Chronos-2
joint wins NO₂ and CO.

**What is a time-series foundation model?** A large model pretrained on a broad corpus of time
series from many domains. It learns general forecasting skills once (seasonality, trend, level
shifts, uncertainty) and is applied to new series without task-specific training. A model trained
only on this dataset learns from 4.7 years of one city. That is less general knowledge, but
everything it learns is specific to Paris.

**What does zero-shot mean here?** Chronos-2's weights are never updated on Paris data. It reads
the pollutant history before the origin as context, and outputs the forecast directly.

**Why could forecasting all pollutants together help? One advantage, one risk.** Advantage:
correlated series inform each other, which matters most when one is missing. NO₂ improved most in
joint mode when its own history was 13 days stale. Risk: errors and spurious links propagate. If
the relationship seen in the context breaks, as in the Olympics period, one series drags the
others off.

**Target, past-only covariate or known-future covariate?**
- Pollutant history: the **target**. Its own history is the context.
- Historical weather: a **past-only covariate**.
- Hour of day: a **known-future covariate**.
- Future weather observations: not allowed here, since actual observations from the forecast
  period are unknown at forecast time. If they were real weather *forecasts*, they would be
  known-future covariates.

**Why can the median be a good choice for MAE?** The median of a predictive distribution
minimises the expected absolute error; the mean minimises squared error. Pollution is
right-skewed, and episodes pull the mean up but barely move the median.

**Which approach works best: per-pollutant, joint, or LSTM?** It depends on the period, and the
per-pollutant MAE shows why.
- In August (`primary`), **joint** is best (5.57, against 5.87 for independent and 6.53 for the
  LSTM). Its gain comes almost entirely from NO₂ (6.35 vs 7.71) and CO (0.037 vs 0.048), the two
  traffic pollutants. NO₂'s own history was missing for 13 days, and CO, its closest correlate
  (0.71), carried the information. Independent is slightly better on PM2.5 (3.26 vs 3.37).
- In September 2023, the **LSTM** is best on average (8.84), entirely through particles (3-seed means: PM10 6.2,
  PM2.5 4.7, against 6.8–6.9 and 5.1–5.4 for Chronos-2). Unlike Chronos-2, it learns from all
  earlier Septembers via its day-of-year input.
- But the LSTM is unstable where it has less to go on. Its August score ranges from 6.00 to 7.22
  across seeds, against a 0.05 spread in September.
- Independent Chronos-2 has the lowest fold-to-fold spread (0.41): the safest, if not the best.

**Best model overall, and for every pollutant?** Overall, Chronos-2 joint (+ seasonal adjustment
for September): best on `primary` and over the rolling folds, and tied best on `sep2023`. It is
**not** best for every pollutant. Independent Chronos-2 wins PM2.5 in August and O₃ in September,
and the LSTM wins particles in September and CO in August (by 0.002, i.e. nothing). The average
hides this because it is dominated by O₃ and NO₂.

**Where does the foundation model fail compared with seasonal naive?** On average it never loses,
but its advantage disappears in two places.
- **In September, on NO₂ and PM2.5:** 15.44 vs 15.77, and 5.36 vs 5.32. Seasonal naive copies a
  real recent week, while Chronos-2's median forecast is anchored on the low holiday level and
  smoothed. When the level jumps, neither is right, and Chronos-2's usual advantage (better shape,
  less noise) is swamped by the level error.
- **For PM10 in the first week of August** (4.64 vs 4.08): copying last week's particle pattern
  beats a smoothed median when a short episode repeats.

The observed-vs-predicted plots (notebook §6.3) show both sides:
- **A clear success:** O₃'s daily cycle in August, whose amplitude and timing Chronos-2 tracks
  (MAE 12.8, against 17.3 for seasonal naive).
- **A clear failure:** the NO₂ surge in the first week of September 2023, which every model
  misses by a wide margin.

## 5. Limitations and next steps

- **Data:** long sensor outages. NO₂ is unobserved for the 69 h before the test origin and for
  13 days before the `primary` origin, and CO is missing for most of 2021. Only four earlier
  Septembers exist to learn the post-holiday rebound from.
- **Evaluation:** a 504-hour window is a small, autocorrelated sample. Differences of a few
  percent are within the fold-to-fold spread (std 0.4–1.4). The selection window and the seasonal
  fix were both chosen after noticing the September shift, so their gains were optimistic. The test
  confirmed this: the adjustment that won on `sep2023` cost about 0.6 MAE on the test period.
- **Model:** Chronos-2's context cannot reach the previous year's same season, and zero-shot
  means it cannot learn Paris-specific calendar effects. No weather is used, and weather drives
  particles and ozone.
- **Next steps:** pass calendar covariates to Chronos-2 (supported, as known-future inputs);
  fine-tune it on Paris; use weather forecasts if the rules allowed them; ensemble Chronos-2 with
  the LSTM, whose errors fall in different months.

## References
- Ansari et al., *Chronos-2: From Univariate to Universal Forecasting* (Amazon, 2025), and the
  Chronos-2 quick-start notebook.
- Kaggle competition *X/HEC Time Series 2025/2026: Predicting the level of air pollutants in Paris*:
  description and data.
- French public holidays: the `holidays` Python package (calendar information, known in advance).
- Paris 2024 Olympic Games dates (26 July – 11 August 2024), used only to interpret one validation fold.
