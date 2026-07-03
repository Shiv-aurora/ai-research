"""Small pooled GRU forecaster of next-day log-RV.

Deliberately modest (single-layer GRU, hidden 32): the neural row of the pool
exists to show the conformal layer is forecaster-agnostic, not to win the
point-forecast race. Input per example: a 60-day window of log_rv plus the
current market-state snapshot; output: next-day log_rv.

Training subsamples windows for speed; inference is fully vectorized. Device
auto-selects MPS on Apple Silicon.

PROCESS-ISOLATION RULE (macOS): torch and lightgbm bundle separate
libomp.dylib copies; importing both in one process segfaults. Never put this
forecaster in the same walk-forward run as LGBMForecaster — produce GRU
forecasts in their own process (scripts write parquet; hedge_combine merges
prediction frames afterward). Tests live in tests/torch_isolated/.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.forecasters.base import Forecaster

STATE_COLS = ["vix_pctl", "mkt_rv_pctl", "term_spread", "credit_spread",
              "stock_rv_pctl"]
WINDOW = 60


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _GRUNet(nn.Module):
    def __init__(self, n_state: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_state, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, seq, state):
        _, h = self.gru(seq)
        return self.head(torch.cat([h[-1], state], dim=1)).squeeze(-1)


def _windows(panel: pd.DataFrame, state_cols: list[str]):
    """(seq, state, target, row_index) arrays for all rows with full windows.

    seq[i] is the WINDOW-length history of log_rv ENDING at row i (inclusive);
    target is log_rv at i+1 within ticker.
    """
    seqs, states, targets, rows = [], [], [], []
    for _, g in panel.groupby("ticker"):
        lv = g["log_rv"].values
        st = g[state_cols].fillna(0.0).values
        tgt = np.append(lv[1:], np.nan)
        for i in range(WINDOW - 1, len(g)):
            seqs.append(lv[i - WINDOW + 1: i + 1])
            states.append(st[i])
            targets.append(tgt[i])
            rows.append(g.index[i])
    return (np.asarray(seqs, dtype=np.float32),
            np.asarray(states, dtype=np.float32),
            np.asarray(targets, dtype=np.float32),
            np.asarray(rows))


class GRUForecaster(Forecaster):
    name = "gru"

    def __init__(self, epochs: int = 3, batch: int = 1024,
                 max_train_windows: int = 80_000, lr: float = 1e-3,
                 seed: int = 42) -> None:
        self.epochs, self.batch, self.max_train = epochs, batch, max_train_windows
        self.lr, self.seed = lr, seed
        self.net_: _GRUNet | None = None
        self.state_cols_: list[str] = []
        self.mu_, self.sd_ = 0.0, 1.0

    def fit(self, train: pd.DataFrame) -> "GRUForecaster":
        torch.manual_seed(self.seed)
        self.state_cols_ = [c for c in STATE_COLS if c in train.columns]
        seq, st, tgt, _ = _windows(train.reset_index(drop=True), self.state_cols_)
        ok = ~np.isnan(tgt)
        seq, st, tgt = seq[ok], st[ok], tgt[ok]
        rng = np.random.default_rng(self.seed)
        if len(seq) > self.max_train:
            idx = rng.choice(len(seq), self.max_train, replace=False)
            seq, st, tgt = seq[idx], st[idx], tgt[idx]

        self.mu_, self.sd_ = float(seq.mean()), float(seq.std() + 1e-8)
        seq = (seq - self.mu_) / self.sd_
        tgt_n = (tgt - self.mu_) / self.sd_

        dev = _device()
        net = _GRUNet(len(self.state_cols_)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(seq).unsqueeze(-1), torch.from_numpy(st),
            torch.from_numpy(tgt_n))
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch, shuffle=True)
        net.train()
        for _ in range(self.epochs):
            for xb, sb, yb in dl:
                opt.zero_grad()
                loss = loss_fn(net(xb.to(dev), sb.to(dev)), yb.to(dev))
                loss.backward()
                opt.step()
        self.net_ = net.eval()
        return self

    @torch.no_grad()
    def predict(self, frame: pd.DataFrame) -> pd.Series:
        f = frame.reset_index()
        seq, st, _, rows = _windows(f, self.state_cols_)
        preds = pd.Series(np.nan, index=frame.index, name=self.name)
        if len(seq) == 0:
            return preds
        seq = (seq - self.mu_) / self.sd_
        dev = _device()
        outs = []
        for i in range(0, len(seq), 8192):
            xb = torch.from_numpy(seq[i:i + 8192]).unsqueeze(-1).to(dev)
            sb = torch.from_numpy(st[i:i + 8192]).to(dev)
            outs.append(self.net_(xb, sb).cpu().numpy())
        vals = np.concatenate(outs) * self.sd_ + self.mu_
        preds.loc[f.loc[rows, "index"].values] = vals
        return preds
