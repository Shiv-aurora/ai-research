"""POGO: parameter-free group-conditional online conformal (Bharti et al.
2026, arXiv:2606.00419), ported to the panel volatility setting.

POGO parameterizes the issued threshold as tau_t = <theta_t, c(X_t)> over
group-membership scores c and learns theta by coin-betting: each group j
keeps a wealth process W_j whose bets are sized by a Universal Portfolio
mixture (Jeffreys prior) over two-outcome "stock returns" built from the
pinball subgradient — no learning rate anywhere. Guarantee: per-group
miscoverage -> alpha at O(sqrt(log/T_j)) for every group.

Port choices (documented in the paper):
  * per-side: the original tracks one radius for a nonnegative score
    |y - m|; volatility errors are skewed, so we run two independent POGO
    instances on the signed standardized score s and -s at level
    alpha/2 each, exactly as every other method in our stack.
  * groups: the paper's interface takes memberships as given; we supply
    the same hard trailing VIX-percentile bins the headline method uses
    (market-level, identical across stocks on a day).
  * panel: POGO is a single-stream algorithm. The pooled variant treats
    each stock-day as one betting round, processing day t's cross-section
    sequentially after all of day t's intervals were issued with the
    morning theta (strictly causal, same convention as the pooled RC
    update). The per-stock variant runs one POGO per ticker.

Universal Portfolio integral is discretized on an interior midpoint grid
with Jeffreys weights; the grid is fine enough that results are unchanged
at double the resolution.
"""

import numpy as np
import pandas as pd

from src.conformal.scores import standardized_scores

N_GRID = 101


class _PogoSide:
    """One-sided POGO over K groups at miscoverage level a_side."""

    def __init__(self, K: int, a_side: float):
        self.K = K
        self.a = a_side
        # interior midpoint grid for the Universal Portfolio integral
        edges = np.linspace(0.0, 1.0, N_GRID + 1)
        self.lam_grid = (edges[:-1] + edges[1:]) / 2.0
        dens = 1.0 / np.sqrt(self.lam_grid * (1.0 - self.lam_grid))
        self.mu = dens / dens.sum()
        self.logw = np.zeros((K, N_GRID))   # log cumulative products
        self.W = np.full(K, 1.0 / K)        # wealth per group
        self.theta = np.zeros(K)

    def refresh_theta(self) -> None:
        """Compute lambda_t from history and set theta_t (pre-outcome)."""
        w = np.exp(self.logw - self.logw.max(axis=1, keepdims=True))
        num = (w * self.mu * self.lam_grid).sum(axis=1)
        den = (w * self.mu).sum(axis=1)
        lam = num / den
        beta = (lam - self.a) / (self.a * (1.0 - self.a))
        self.theta = self.W * beta
        self._beta = beta

    def tau(self, c: np.ndarray) -> float:
        return float(self.theta @ c)

    def update(self, s: float, c: np.ndarray) -> None:
        """One betting round with realized score s and memberships c."""
        covered = s <= self.theta @ c
        g = ((1.0 if covered else 0.0) - (1.0 - self.a)) * c   # (K,)
        self.W = self.W * (1.0 - self._beta * g)
        w1 = 1.0 - g / self.a
        w2 = 1.0 + g / (1.0 - self.a)
        ret = (self.lam_grid[None, :] * w1[:, None]
               + (1.0 - self.lam_grid[None, :]) * w2[:, None])
        self.logw += np.log(np.maximum(ret, 1e-300))
        self.refresh_theta()


def run_pogo_panel(
    preds: pd.DataFrame,
    membership: pd.DataFrame,     # date-indexed, K columns (hard or soft)
    forecast_col: str,
    alpha: float = 0.10,
    warmup_days: int = 100,
    scale_window: int = 250,
) -> pd.DataFrame:
    """Pooled POGO over a walk-forward predictions frame. Output schema
    matches run_panel_mondrian (q_lo/q_hi in standardized units)."""
    K = membership.shape[1]
    a_side = alpha / 2.0

    parts = [standardized_scores(g.sort_values("date"), forecast_col,
                                 window=scale_window)
             for _, g in preds.groupby("ticker")]
    df = pd.concat(parts).dropna(subset=["s_std"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    lo = _PogoSide(K, a_side)
    hi = _PogoSide(K, a_side)
    lo.refresh_theta()
    hi.refresh_theta()

    dates = df["date"].unique()
    warmup_end = dates[min(warmup_days, len(dates) - 1)]

    out_qlo = np.zeros(len(df))
    out_qhi = np.zeros(len(df))
    out_cov = np.zeros((len(df), 3), dtype=bool)
    warm_flag = np.zeros(len(df), dtype=bool)

    for d, block in df.groupby("date", sort=True):
        idx = block.index.values
        pi = membership.reindex([d]).values
        if np.isnan(pi).any():
            pi = np.full((1, K), 1.0 / K)
        c = pi[0] / pi[0].sum()

        ql, qh = lo.tau(c), hi.tau(c)
        s = block["s_std"].values
        cov_lo = (-s) <= ql
        cov_hi = s <= qh
        out_qlo[idx] = ql
        out_qhi[idx] = qh
        out_cov[idx, 0] = cov_lo & cov_hi
        out_cov[idx, 1] = cov_lo
        out_cov[idx, 2] = cov_hi

        if d <= warmup_end:
            warm_flag[idx] = True
        # sequential betting rounds through the day's cross-section (the
        # issued tau above used the pre-day theta; updating during warmup
        # is what lets wealth accumulate before evaluation starts)
        for si in s:
            lo.update(-si, c)
            hi.update(si, c)

    df["q_lo"] = out_qlo
    df["q_hi"] = out_qhi
    df["covered"] = out_cov[:, 0]
    df["covered_lo"] = out_cov[:, 1]
    df["covered_hi"] = out_cov[:, 2]
    df["warmup"] = warm_flag
    df["width"] = (df["q_lo"] + df["q_hi"]) * df["sigma_hat"]
    return df
