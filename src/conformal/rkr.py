"""Group-conditional online conformal via no-regret FTRL (Ramalingam,
Kiyani & Roth, ICML 2025, arXiv:2502.10947), per stock.

Their Algorithm 1 parameterizes the threshold linearly in group
memberships, tau_hat_t = <theta_t, g_t>, and runs FTRL on the pinball
loss; with a quadratic regularizer this is online gradient descent:

  miss  (s_t > tau_hat):  theta += eta * q * g_t
  cover (s_t <= tau_hat): theta -= eta * (1 - q) * g_t

with q = 1 - a_side the target level. Group-conditional coverage then
deviates by at most ||grad R(theta)||_inf / T_i for group i.

Port choices: per stock on the signed raw-residual stream (the same
scale and per-side convention as the ACI/DtACI baselines); groups are
OVERLAPPING — a constant marginal group plus the four hard VIX bins —
which is the instantiation that distinguishes this method from plain
Mondrian ACI (with one-hot groups alone, OGD-FTRL reduces to per-group
ACI up to step reparameterization). eta defaults to the ACI baseline's
0.05 on this scale; pass others for rate sensitivity.
"""

import numpy as np
import pandas as pd


def run_rkr(scores: np.ndarray, groups: np.ndarray, alpha: float = 0.10,
            warmup: int = 100, eta: float = 0.05) -> pd.DataFrame:
    """Two-sided RKR-FTRL(OGD) over a stream of signed scores.

    groups: (n, k) membership matrix in [0,1] per round (may overlap).
    Output schema matches run_aci for drop-in evaluation.
    """
    n, k = groups.shape
    a_side = alpha / 2.0
    q = 1.0 - a_side

    # initialize the marginal coordinate at the warmup quantile so both
    # sides start from a sane level (group coords start at zero)
    th_hi = np.zeros(k)
    th_lo = np.zeros(k)
    th_hi[0] = float(np.quantile(scores[:warmup], q))
    th_lo[0] = float(np.quantile(-scores[:warmup], q))

    out = np.zeros((n, 5))
    for t in range(n):
        g = groups[t]
        s = scores[t]
        q_hi = float(th_hi @ g)
        q_lo = float(th_lo @ g)
        cov_lo, cov_hi = (-s) <= q_lo, s <= q_hi
        out[t] = (q_lo, q_hi, cov_lo and cov_hi, cov_lo, cov_hi)
        if t >= warmup:
            th_hi += eta * (q * g if not cov_hi else -(1.0 - q) * g)
            th_lo += eta * (q * g if not cov_lo else -(1.0 - q) * g)

    df = pd.DataFrame(out, columns=["q_lo", "q_hi", "covered",
                                    "covered_lo", "covered_hi"])
    df["warmup"] = np.arange(n) < warmup
    return df


def marginal_plus_bins(vix_pctl: np.ndarray,
                       cuts=(0.5, 0.8, 0.95)) -> np.ndarray:
    """Overlapping groups: [marginal, calm, normal, elevated, stress]."""
    n = len(vix_pctl)
    g = np.zeros((n, len(cuts) + 2))
    g[:, 0] = 1.0
    kbin = np.searchsorted(cuts, vix_pctl, side="right")
    ok = ~np.isnan(vix_pctl)
    g[np.arange(n)[ok], 1 + kbin[ok]] = 1.0
    return g
