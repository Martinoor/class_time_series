"""Zero-shot Chronos-2 forecaster.

Zero-shot means the pretrained weights are used as-is: the model is never
trained or fine-tuned on this Paris data. At prediction time it only reads
the context window (history strictly before the forecast origin) and returns
quantile forecasts for the next 504 hours in one pass (Chronos-2 can emit up
to 64 x 16 = 1024 steps directly, so there is no autoregressive roll-out).

Two modes, for the PDF's independent-vs-joint comparison:

* ``independent`` -- each pollutant is its own univariate series. The model
  sees one pollutant's history at a time.
* ``joint`` -- the five pollutants are passed as one multivariate target, so
  Chronos-2's group attention lets each forecast attend to the other four
  series' history.

The point forecast is the median (0.5 quantile), which is the MAE-optimal
summary of a predictive distribution.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from metrics import TARGETS
from models import BaseForecaster

MODEL_ID = "amazon/chronos-2"
_PIPELINES: dict = {}  # cache: loading takes ~30 s, do it once per device


def best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(device: str | None = None):
    """Load (once) and return the Chronos-2 pipeline on `device`."""
    import torch
    from chronos import Chronos2Pipeline

    device = device or best_device()
    if device not in _PIPELINES:
        pipe = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=device, dtype=torch.float32)
        _PIPELINES[device] = pipe
    return _PIPELINES[device]


def run_info(pipe) -> dict:
    """What the report must record: model, version, hardware, settings."""
    import platform

    import chronos
    import torch

    dev = next(pipe.model.parameters()).device
    info = {
        "model": MODEL_ID,
        "chronos_forecasting": getattr(chronos, "__version__", "unknown"),
        "torch": torch.__version__,
        "device": str(dev),
        "params_millions": round(sum(p.numel() for p in pipe.model.parameters()) / 1e6, 1),
        "context_length_max": pipe.model.config.chronos_config["context_length"],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    if dev.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(dev)
    return info


class Chronos2Forecaster(BaseForecaster):
    """Zero-shot Chronos-2 behind the same fit/predict interface as the baselines.

    `fit` does not train anything -- it only stores the context. That is what
    zero-shot means, and it keeps the interface identical to the baselines.
    """

    def __init__(
        self,
        mode: str = "joint",
        context_length: int | None = None,
        device: str | None = None,
        targets: list[str] | None = None,
        quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    ):
        if mode not in ("joint", "independent"):
            raise ValueError(f"mode must be 'joint' or 'independent', got {mode!r}")
        self.mode = mode
        self.context_length = context_length
        self.device = device
        self.targets = list(targets) if targets is not None else list(TARGETS)
        self.quantile_levels = list(quantile_levels)
        self.name = f"chronos2_{mode}" + (f"_ctx{context_length}" if context_length else "")
        self.history_: pd.DataFrame | None = None
        self.quantiles_: pd.DataFrame | None = None  # long format, kept for interval plots
        self.seconds_: float | None = None

    def fit(self, train: pd.DataFrame, time_col: str = "datetime"):
        tr = train.sort_values(time_col)
        if self.context_length:
            tr = tr.iloc[-self.context_length :]
        self.history_ = tr[[time_col] + self.targets].reset_index(drop=True)
        return self

    def _context(self, time_col: str) -> tuple[pd.DataFrame, str | list[str]]:
        h = self.history_
        if self.mode == "joint":
            return h.assign(item_id="paris"), self.targets
        # independent: stack the pollutants as separate univariate series
        long = h.melt(id_vars=time_col, value_vars=self.targets,
                      var_name="item_id", value_name="target")
        return long, "target"

    def predict(self, future: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
        if self.history_ is None:
            raise RuntimeError("call fit() before predict()")
        expected_start = self.history_[time_col].iloc[-1] + pd.Timedelta(hours=1)
        if pd.Timestamp(future[time_col].iloc[0]) != expected_start:
            raise ValueError(
                f"forecast must start right after the context ({expected_start}), "
                f"got {future[time_col].iloc[0]}"
            )

        pipe = load_pipeline(self.device)
        ctx, target = self._context(time_col)
        t0 = time.time()
        out = pipe.predict_df(
            ctx,
            id_column="item_id",
            timestamp_column=time_col,
            target=target,
            prediction_length=len(future),
            quantile_levels=self.quantile_levels,
            freq="h",
        )
        self.seconds_ = time.time() - t0

        # Name the series consistently across modes: joint -> target_name, independent -> item_id
        out["series"] = out["target_name"] if self.mode == "joint" else out["item_id"]
        self.quantiles_ = out
        wide = out.pivot(index=time_col, columns="series", values="predictions")[self.targets]

        # Align to the requested timestamps explicitly, never by position.
        wide = wide.reindex(pd.to_datetime(future[time_col]).to_numpy())
        if wide.isna().any().any():
            raise ValueError("Chronos-2 output does not cover every requested timestamp")
        wide.index = future.index
        wide.columns.name = None
        return wide
