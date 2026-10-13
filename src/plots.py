"""Figures for the notebook and report.

Conventions (from the data-viz guidance used for this project):
* each pollutant keeps one fixed color everywhere (color follows the entity);
* pollutants have different units, so they get their own panels -- never a
  shared or dual y-axis;
* every panel is titled with its pollutant and unit, so identity never depends
  on color alone (three of the five hues are low-contrast on white);
* thin lines, recessive grid, text in neutral ink.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import TARGETS

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# Reference categorical palette, light mode, slots 1-5 in fixed order.
POLLUTANT_COLOR = {
    "valeur_CO": "#2a78d6",    # blue
    "valeur_NO2": "#eb6834",   # orange
    "valeur_O3": "#1baf7a",    # aqua
    "valeur_PM10": "#eda100",  # yellow
    "valeur_PM25": "#e87ba4",  # magenta
}
UNIT = {"valeur_CO": "mg/m³", "valeur_NO2": "µg/m³", "valeur_O3": "µg/m³",
        "valeur_PM10": "µg/m³", "valeur_PM25": "µg/m³"}
LABEL = {"valeur_CO": "CO", "valeur_NO2": "NO₂", "valeur_O3": "O₃",
         "valeur_PM10": "PM10", "valeur_PM25": "PM2.5"}

# Model colors, also fixed: observed truth is neutral ink, not a series hue.
MODEL_COLOR = {
    "observed": "#0b0b0b",
    "chronos2_joint": "#2a78d6",
    "chronos2_joint+seasonal_adj": "#4a3aa7",
    "chronos2_independent": "#1baf7a",
    "seasonal_naive_168h": "#eb6834",
    "persistence": "#e34948",
    "lstm": "#008300",
}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Sequential blue ramp (100 -> 700) and the diverging pair with a gray midpoint.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a7a6", "#e66767", "#b8302f"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.4, "legend.frameon": False, "legend.fontsize": 8.5,
        "font.family": "sans-serif", "figure.dpi": 110, "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    return path


def _title(col):
    return f"{LABEL[col]} ({UNIT[col]})"


def series_overview(train, time_col="datetime"):
    """Daily means per pollutant, 2020 -> 2024. Gaps stay gaps (no interpolation)."""
    style()
    daily = train.set_index(time_col)[TARGETS].resample("D").mean()
    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(10, 9), sharex=True)
    for ax, col in zip(axes, TARGETS):
        ax.plot(daily.index, daily[col], color=POLLUTANT_COLOR[col], lw=0.9)
        roll = daily[col].rolling(30, min_periods=10).mean()
        ax.plot(roll.index, roll, color=INK, lw=1.4, alpha=0.75)
        ax.set_title(_title(col))
    axes[0].text(0.995, 1.02, "thin: daily mean   bold: 30-day rolling mean",
                 transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
    fig.suptitle("Pollutant levels, daily means", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, "eda_series"), fig


def missingness_heatmap(train, time_col="datetime"):
    """% missing per pollutant per month, sequential blue (darker = more missing)."""
    from matplotlib.colors import LinearSegmentedColormap

    style()
    m = train.set_index(time_col)[TARGETS].isna().resample("MS").mean() * 100
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd"] + SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(10, 2.6))
    im = ax.imshow(m.T.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=100, interpolation="nearest")
    ax.set_yticks(range(len(TARGETS)), [LABEL[c] for c in TARGETS])
    years = [i for i, d in enumerate(m.index) if d.month == 1]
    ax.set_xticks(years, [m.index[i].year for i in years])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cb.set_label("% of hours missing", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Missing values by month")
    fig.tight_layout()
    return _save(fig, "eda_missing"), fig


def daily_weekly_profiles(train, time_col="datetime"):
    """Mean by hour of day, weekdays vs weekends: evidence for daily and weekly cycles."""
    style()
    ts = train[time_col]
    df = train.assign(hour=ts.dt.hour, weekend=ts.dt.dayofweek >= 5)
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(13, 2.9))
    for ax, col in zip(axes, TARGETS):
        prof = df.groupby(["weekend", "hour"])[col].mean().unstack(0)
        c = POLLUTANT_COLOR[col]
        ax.plot(prof.index, prof[False], color=c, lw=2)
        ax.plot(prof.index, prof[True], color=c, lw=1.4, ls=(0, (3, 2)))
        ax.set_title(_title(col))
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel("hour of day")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=INK2, lw=2), Line2D([], [], color=INK2, lw=1.4, ls=(0, (3, 2)))]
    fig.legend(handles, ["weekday (Mon-Fri)", "weekend (Sat-Sun)"], loc="upper right",
               ncol=2, bbox_to_anchor=(0.995, 1.0))
    fig.suptitle("Average daily profile, weekdays vs weekends", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, "eda_profiles"), fig


def autocorrelation(train, max_lag=24 * 15):
    """ACF to 15 days; markers at 24 h, 168 h and 336 h."""
    style()
    lags = np.arange(1, max_lag + 1)
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(13, 2.7), sharey=True)
    for ax, col in zip(axes, TARGETS):
        s = train[col]
        acf = [s.autocorr(k) for k in lags]
        ax.plot(lags / 24, acf, color=POLLUTANT_COLOR[col], lw=1.1)
        for k in (24, 168, 336):
            ax.axvline(k / 24, color=INK2, lw=0.6, ls=":")
        ax.set_title(LABEL[col])
        ax.set_xlabel("lag (days)")
        ax.set_xticks([0, 1, 7, 14])
    axes[0].set_ylabel("autocorrelation")
    fig.suptitle("Autocorrelation: a daily cycle for every pollutant, a weekly one mainly for NO₂ (dotted: 1, 7, 14 days)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, "eda_acf"), fig


def correlation_matrix(train):
    """Pairwise correlation of hourly values, diverging blue <-> red with a gray midpoint."""
    from matplotlib.colors import LinearSegmentedColormap

    style()
    c = train[TARGETS].corr()
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    im = ax.imshow(c.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    names = [LABEL[t] for t in TARGETS]
    ax.set_xticks(range(5), names)
    ax.set_yticks(range(5), names)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(5):
        for j in range(5):
            v = c.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if abs(v) > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    ax.set_title("Correlation between pollutants (hourly)")
    fig.tight_layout()
    return _save(fig, "eda_correlation"), fig


def september_rebound(train, time_col="datetime"):
    """NO2 and CO around 4 Sep, one line per year: the post-holiday rebound.

    Plotted as a centred 7-day mean of daily means. The centring uses later
    days, which is fine for a descriptive chart -- it is never a model input.
    """
    style()
    years = sorted(train[time_col].dt.year.unique())[:-1]  # skip the partial last year
    shades = ["#9ec5f4", "#6da7ec", "#3987e5", "#184f95"]  # light = older
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
    for ax, col in zip(axes, ["valeur_NO2", "valeur_CO"]):
        for y, shade in zip(years, shades[-len(years):]):
            o = pd.Timestamp(f"{y}-09-04")
            w = train[(train[time_col] >= o - pd.Timedelta(days=35)) & (train[time_col] < o + pd.Timedelta(days=28))]
            d = w.set_index(time_col)[col].resample("D").mean()
            d = d.rolling(7, center=True, min_periods=4).mean()
            d = d[(d.index >= o - pd.Timedelta(days=28)) & (d.index < o + pd.Timedelta(days=21))]
            if d.notna().sum() < 10:
                continue  # this pollutant was offline that year
            x = (d.index - o).days
            ax.plot(x, d.to_numpy(), color=shade, lw=1.8)
            last = d.dropna()
            ax.annotate(str(y), (x[d.index.get_loc(last.index[-1])], last.iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=7.5, color=INK2, va="center")
        ax.axvline(0, color=INK, lw=0.8)
        ax.axvspan(0, 21, color="#f0efec", zorder=0)
        ax.text(10.5, 0.03, "3 weeks from 4 Sep (the test period's calendar)",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_title(_title(col))
        ax.set_xlabel("days from 4 September")
        ax.set_xlim(-28, 25)
    fig.suptitle("Levels rise after the summer break (7-day mean around 4 September, one line per year)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, "eda_september"), fig


MODEL_LABEL = {
    "persistence": "Persistence",
    "seasonal_naive_24h": "Seasonal naive 24 h",
    "seasonal_naive_168h": "Seasonal naive 168 h",
    "weekly_profile_4w": "Weekly profile, 4 wk (diagnostic)",
    "chronos2_independent": "Chronos-2 independent",
    "chronos2_joint": "Chronos-2 joint",
    "chronos2_independent+seasonal_adj": "Chronos-2 indep. + seasonal adj.",
    "chronos2_joint+seasonal_adj": "Chronos-2 joint + seasonal adj.",
    "lstm": "LSTM",
}


def mae_comparison(table: pd.DataFrame, window: str, models: list[str], name: str | None = None):
    """Per-pollutant MAE by model, one panel per pollutant (scales differ).

    Bars are neutral; the best model in each panel is highlighted and is the
    only one carrying a value label, so the panel answers "who wins here?"
    """
    style()
    sub = table[(table.window == window) & table.model.isin(models)].set_index("model").loc[models]
    cols = ["CO", "NO2", "O3", "PM10", "PM25", "avg_MAE"]
    titles = ["CO", "NO₂", "O₃", "PM10", "PM2.5", "Average MAE"]
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 0.42 * len(models) + 1.4), sharey=True)
    y = np.arange(len(models))
    for ax, col, title in zip(axes, cols, titles):
        v = sub[col].to_numpy()
        best = int(np.nanargmin(v))
        colors = ["#c3c2b7"] * len(v)
        colors[best] = "#2a78d6"
        ax.barh(y, v, color=colors, height=0.66)
        ax.text(v[best], best, f" {v[best]:.3g}", va="center", ha="left", fontsize=8, color=INK)
        ax.set_title(title)
        ax.set_xlim(0, np.nanmax(v) * 1.28)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis() if ax is axes[0] else None
    axes[0].set_yticks(y, [MODEL_LABEL.get(m, m) for m in models])
    fig.suptitle(f"MAE per pollutant, window '{window}' (blue = best in panel; lower is better)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, name or f"results_mae_{window}"), fig


def observed_vs_predicted(window_df: pd.DataFrame, preds: dict[str, pd.DataFrame], name: str,
                          title: str, time_col: str = "datetime"):
    """Observed (ink) against each model's forecast, one panel per pollutant."""
    from matplotlib.lines import Line2D

    style()
    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(11, 9.5), sharex=True)
    t = window_df[time_col].to_numpy()
    for ax, col in zip(axes, TARGETS):
        ax.plot(t, window_df[col].to_numpy(), color=INK, lw=0.9)
        for m, p in preds.items():
            ax.plot(t, p[col].to_numpy(), color=MODEL_COLOR.get(m, "#4a3aa7"), lw=1.3, alpha=0.95)
        ax.set_title(_title(col))
    handles = [Line2D([], [], color=INK, lw=1.2)] + [Line2D([], [], color=MODEL_COLOR.get(m, "#4a3aa7"), lw=2) for m in preds]
    fig.legend(handles, ["Observed"] + [MODEL_LABEL.get(m, m) for m in preds], loc="upper right",
               ncol=len(preds) + 1, bbox_to_anchor=(0.995, 1.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, name), fig


def horizon_profile(blocks: dict[str, pd.DataFrame], name: str, title: str):
    """MAE per pollutant for week 1 / 2 / 3 of the horizon, one line per model."""
    from matplotlib.lines import Line2D

    style()
    cols = ["CO", "NO2", "O3", "PM10", "PM25"]
    titles = ["CO", "NO₂", "O₃", "PM10", "PM2.5"]
    fig, axes = plt.subplots(1, len(cols), figsize=(13, 2.9))
    for ax, col, tt in zip(axes, cols, titles):
        for m, b in blocks.items():
            ax.plot([1, 2, 3], b[col].to_numpy(), color=MODEL_COLOR.get(m, "#4a3aa7"), lw=2, marker="o", ms=4)
        ax.set_xticks([1, 2, 3], ["wk 1", "wk 2", "wk 3"])
        ax.set_title(tt)
    axes[0].set_ylabel("MAE")
    handles = [Line2D([], [], color=MODEL_COLOR.get(m, "#4a3aa7"), lw=2, marker="o", ms=4) for m in blocks]
    fig.legend(handles, [MODEL_LABEL.get(m, m) for m in blocks], loc="upper right", ncol=len(blocks),
               bbox_to_anchor=(0.995, 1.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, name), fig
