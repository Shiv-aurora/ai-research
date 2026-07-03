"""Transparent quantile-bin regime estimator (the interpretable baseline).

Composite stress index = mean of the available market-state percentiles
(vix_pctl, mkt_rv_pctl). Regime k on day t is determined by which trailing-
quantile bin the index falls into — thresholds computed from the index's own
trailing window, so membership at t uses only information up to t.

Outputs hard one-hot memberships (this estimator's whole point is
transparency); the soft-Mondrian calibrator consumes them as degenerate
probabilities.
"""

import numpy as np
import pandas as pd

STRESS_COLS = ["vix_pctl", "mkt_rv_pctl"]


def stress_index(market: pd.DataFrame) -> pd.Series:
    cols = [c for c in STRESS_COLS if c in market.columns]
    return market[cols].mean(axis=1)


def quantile_bin_memberships(
    market: pd.DataFrame,
    n_regimes: int = 3,
    window: int = 750,
) -> pd.DataFrame:
    """market: one row per DATE with state columns. Returns (date x K) one-hot
    membership frame aligned to market.index."""
    idx = stress_index(market)
    edges = np.linspace(0, 1, n_regimes + 1)[1:-1]  # interior quantile levels

    member = np.zeros((len(idx), n_regimes))
    vals = idx.values
    for t in range(len(idx)):
        lo = max(0, t - window)
        hist = vals[lo: t + 1]
        thresholds = np.quantile(hist, edges) if len(hist) >= 30 else None
        if thresholds is None or np.isnan(vals[t]):
            member[t, :] = 1.0 / n_regimes  # uninformative during warmup
        else:
            k = int(np.searchsorted(thresholds, vals[t], side="right"))
            member[t, k] = 1.0
    out = pd.DataFrame(member, index=market.index,
                       columns=[f"regime_{k}" for k in range(n_regimes)])
    return out
