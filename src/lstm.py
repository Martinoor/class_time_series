"""A simple LSTM trained on this dataset, the PDF's neural-network comparator.

Direct multi-horizon design: an LSTM encoder reads the last `lookback` hours
and a linear head emits all `horizon` x 5 values at once. Predicting the
whole block directly avoids feeding the model's own errors back in over 504
steps.

Inputs per hour: the 5 pollutants (z-scored), 5 missingness flags, and
calendar features (hour, weekday, day-of-year as sin/cos; French holiday).
Calendar values are known in advance, so they are allowed by the PDF. The
day-of-year feature gives the LSTM a way to learn the September rebound,
which Chronos-2 cannot see in its 341-day context.

Leakage controls:
* scaler statistics come from the training history only;
* missing inputs are replaced by 0 (= the training mean after scaling) and
  flagged -- never interpolated, which would use later values;
* missing targets are masked out of the loss, not filled;
* early stopping uses the last `horizon` hours of the history, and no training
  sample's target window may overlap that block.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from features import add_calendar_features
from metrics import TARGETS
from models import BaseForecaster

CAL_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_holiday"]


def build_inputs(df: pd.DataFrame, mean: np.ndarray, std: np.ndarray, targets, time_col="datetime"):
    """Per-hour feature matrix (n, 2*len(targets) + len(CAL_COLS)) and scaled targets."""
    y = df[targets].to_numpy(dtype=np.float32)
    z = (y - mean) / std
    miss = np.isnan(z)
    x_poll = np.where(miss, 0.0, z)
    cal = add_calendar_features(df[[time_col]], time_col=time_col)[CAL_COLS].to_numpy(dtype=np.float32)
    x = np.concatenate([x_poll, miss.astype(np.float32), cal], axis=1).astype(np.float32)
    return x, z.astype(np.float32)


def make_samples(x, z, lookback, horizon, stride, last_origin):
    """Origins o with input rows [o-lookback, o) and target rows [o, o+horizon).

    Only origins with o + horizon <= last_origin are produced, so a target
    window never reaches past `last_origin`.
    """
    origins = np.arange(lookback, last_origin - horizon + 1, stride)
    xs = np.stack([x[o - lookback : o] for o in origins])
    ys = np.stack([z[o : o + horizon] for o in origins])
    return xs, ys, origins


class LSTMForecaster(BaseForecaster):
    def __init__(
        self,
        lookback: int = 336,
        horizon: int = 504,
        hidden: int = 64,
        layers: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 40,
        patience: int = 5,
        stride: int = 6,
        seed: int = 42,
        device: str | None = None,
        targets=None,
        verbose: bool = False,
    ):
        self.lookback, self.horizon = lookback, horizon
        self.hidden, self.layers, self.dropout = hidden, layers, dropout
        self.lr, self.batch_size = lr, batch_size
        self.max_epochs, self.patience, self.stride = max_epochs, patience, stride
        self.seed, self.device, self.verbose = seed, device, verbose
        self.targets = list(targets) if targets is not None else list(TARGETS)
        self.name = "lstm"
        self.history_ = None
        self.log_: list[dict] = []
        self.seconds_: float | None = None

    # -- model -------------------------------------------------------------
    def _net(self, n_in):
        import torch.nn as nn

        horizon, n_out = self.horizon, len(self.targets)

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.lstm = nn.LSTM(n_in, self.hidden, num_layers=self.layers, batch_first=True,
                                 dropout=self.dropout if self.layers > 1 else 0.0)
                s.drop = nn.Dropout(self.dropout)
                s.head = nn.Linear(self.hidden, horizon * n_out)

            def forward(s, x):
                _, (h, _) = s.lstm(x)
                return s.head(s.drop(h[-1])).view(-1, horizon, n_out)

        return Net()

    @staticmethod
    def _masked_l1(pred, y):
        import torch

        mask = ~torch.isnan(y)
        return (pred[mask] - y[mask]).abs().mean()

    # -- interface ---------------------------------------------------------
    def fit(self, train: pd.DataFrame, time_col: str = "datetime"):
        import torch

        t0 = time.time()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = self.device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device_ = dev

        tr = train.sort_values(time_col).reset_index(drop=True)
        self.history_ = tr
        vals = tr[self.targets].to_numpy(dtype=np.float32)
        self.mean_ = np.nanmean(vals, axis=0)
        self.std_ = np.nanstd(vals, axis=0) + 1e-6
        x, z = build_inputs(tr, self.mean_, self.std_, self.targets, time_col)

        n = len(tr)
        es_start = n - self.horizon  # early-stopping origin: the last `horizon` rows
        xs, ys, _ = make_samples(x, z, self.lookback, self.horizon, self.stride, last_origin=es_start)
        x_es = torch.from_numpy(x[es_start - self.lookback : es_start][None])
        y_es = torch.from_numpy(z[es_start:][None])

        net = self._net(x.shape[1]).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        X, Y = torch.from_numpy(xs), torch.from_numpy(ys)
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = np.inf, None, 0
        for epoch in range(self.max_epochs):
            net.train()
            perm = torch.randperm(len(X), generator=gen)
            losses = []
            for i in range(0, len(X), self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb, yb = X[idx].to(dev), Y[idx].to(dev)
                opt.zero_grad()
                loss = self._masked_l1(net(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            net.eval()
            with torch.no_grad():
                es = self._masked_l1(net(x_es.to(dev)), y_es.to(dev)).item()
            self.log_.append({"epoch": epoch, "train_l1": float(np.mean(losses)), "es_l1": es})
            if self.verbose:
                print(f"  epoch {epoch:2d} train {np.mean(losses):.4f}  es {es:.4f}", flush=True)
            if es < best - 1e-4:
                best, bad = es, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.net_ = net.eval()
        self.best_epoch_ = int(np.argmin([r["es_l1"] for r in self.log_]))
        self.n_samples_ = len(X)
        self.seconds_ = time.time() - t0
        return self

    def predict(self, future: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
        import torch

        if self.history_ is None:
            raise RuntimeError("call fit() before predict()")
        if len(future) != self.horizon:
            raise ValueError(f"model emits {self.horizon} steps, asked for {len(future)}")
        expected = self.history_[time_col].iloc[-1] + pd.Timedelta(hours=1)
        if pd.Timestamp(future[time_col].iloc[0]) != expected:
            raise ValueError(f"forecast must start at {expected}")
        x, _ = build_inputs(self.history_, self.mean_, self.std_, self.targets, time_col)
        with torch.no_grad():
            out = self.net_(torch.from_numpy(x[-self.lookback :][None]).to(self.device_))
        z = out.cpu().numpy()[0]
        pred = z * self.std_ + self.mean_
        return pd.DataFrame(pred, columns=self.targets, index=future.index)
