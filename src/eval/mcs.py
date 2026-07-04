"""Model Confidence Set (Hansen, Lunde & Nason 2011), Tmax variant.

Input: a T x M frame of per-period losses (for interval methods, the daily
cross-sectional mean interval score). The MCS is the subset of methods that
cannot be rejected as equal-predictive-ability at level alpha; each method
gets an MCS p-value (the level at which it would first be eliminated).

Bootstrap: moving-block over the time dimension (losses are daily series —
dependence handled by the block), studentized Tmax elimination rule.
"""

import numpy as np
import pandas as pd


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                   alpha: float) -> np.ndarray:
    """Winkler/interval score (lower is better): width plus 2/alpha times
    the distance by which y escapes the interval."""
    return ((hi - lo)
            + (2.0 / alpha) * np.maximum(lo - y, 0.0)
            + (2.0 / alpha) * np.maximum(y - hi, 0.0))


def _block_indices(T: int, block: int, n_boot: int,
                   rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :])
    return idx.reshape(n_boot, -1)[:, :T]


def mcs(losses: pd.DataFrame, alpha: float = 0.10, n_boot: int = 1000,
        block: int = 20, seed: int = 0) -> pd.DataFrame:
    """Returns a frame indexed by method with columns mcs_pvalue, mean_loss,
    in_mcs (at level alpha). Methods eliminated early have low p-values."""
    L = losses.dropna()
    T, M = L.shape
    rng = np.random.default_rng(seed)
    idx = _block_indices(T, block, n_boot, rng)
    arr = L.values
    names = list(L.columns)

    include = list(range(M))
    pvals: dict[str, float] = {}
    p_running = 0.0

    while len(include) > 1:
        sub = arr[:, include]                      # (T, m)
        dbar = sub.mean(axis=0) - sub.mean()       # relative mean loss
        # bootstrap distribution of dbar
        boot = sub[idx]                            # (n_boot, T, m)
        bmean = boot.mean(axis=1)
        bd = bmean - bmean.mean(axis=1, keepdims=True)
        se = np.sqrt(((bd - dbar) ** 2).mean(axis=0))
        se = np.maximum(se, 1e-12)
        tstat = dbar / se
        tmax = tstat.max()
        tmax_boot = ((bd - dbar) / se).max(axis=1)
        p = float((tmax_boot >= tmax).mean())
        p_running = max(p_running, p)
        worst = int(np.argmax(tstat))
        if p < alpha:
            pvals[names[include[worst]]] = p_running
            include.pop(worst)
        else:
            break

    # survivors: running max of test p-values (>= alpha because the final
    # test failed to reject); a sole survivor gets 1.0 by convention
    if len(include) == 1:
        pvals[names[include[0]]] = 1.0
    else:
        for i in include:
            pvals[names[i]] = p_running

    out = pd.DataFrame({
        "mean_loss": L.mean(),
        "mcs_pvalue": pd.Series(pvals),
    })
    out["in_mcs"] = out["mcs_pvalue"] >= alpha
    return out.sort_values("mean_loss")
