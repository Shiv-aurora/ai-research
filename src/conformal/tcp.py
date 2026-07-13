"""Temporal Conformal Prediction with Robbins-Monro correction (TCP-RM).

Port of Aich, Aich & Jain (2025, arXiv:2507.05470) to our per-stock
residual-score setting so it is comparable row-for-row with the other
baselines. Their method: a rolling split-conformal layer (calibration slice
of the last w_cal = 60 observations, finite-sample-corrected order
statistic ceil((m+1)(1-alpha))) around a quantile forecaster, plus an
online Robbins-Monro offset

    C^RM_{t+1} = C^RM_t + gamma_t (err_t - alpha),
    gamma_t = gamma_0 / (1 + lam * t)^beta,   beta in (1/2, 1]

added to the conformal threshold (effective threshold C_t + C^RM_t). We
keep the calibration window, the order statistic, the RM recursion, and
their default gamma_0 = 0.01 verbatim; the port replaces their quantile
forecaster's out-of-bound score with our signed forecast residuals, with
each side tracked at alpha/2 (the same two-sided convention every other
per-stock baseline in the comparison uses). The decaying step is the
method's signature (and its liability on a 16-year nonstationary stream);
we do not tune it.
"""

import numpy as np
import pandas as pd

W_CAL = 60           # their calibration slice (of a 252-day rolling year)
GAMMA_0 = 0.01       # their default RM step
LAM = 0.01           # decay constants satisfying their beta in (1/2, 1]
BETA = 0.75


def run_tcp_rm(
    scores: np.ndarray,
    alpha: float = 0.10,
    warmup: int = 100,
) -> pd.DataFrame:
    """Online two-sided TCP-RM over one stock's signed score stream.

    scores: s_t = y_t - m_t in time order. At each t the interval
    [m_t - q_lo_t, m_t + q_hi_t] is formed BEFORE seeing y_t. Each side's
    threshold is the finite-sample rolling quantile of the last W_CAL
    side-scores plus that side's Robbins-Monro offset. Warmup rows (first
    `warmup`, matching the other baselines) are flagged and excluded from
    evaluation.
    """
    n = len(scores)
    a_side = alpha / 2.0
    c_lo, c_hi = 0.0, 0.0          # RM offsets, start at zero (their init)
    out = np.zeros((n, 5))
    steps = 0                       # RM time index t (post-warmup updates)
    for t in range(n):
        lo_win = -scores[max(0, t - W_CAL):t]
        hi_win = scores[max(0, t - W_CAL):t]
        m = len(hi_win)
        if m >= 5:
            k = min(m - 1, int(np.ceil((m + 1) * (1 - a_side))) - 1)
            q_lo = float(np.sort(lo_win)[k]) + c_lo
            q_hi = float(np.sort(hi_win)[k]) + c_hi
        else:                       # first days: no calibration slice yet
            q_lo = q_hi = np.inf
        s = scores[t]
        cov_lo = (-s) <= q_lo
        cov_hi = s <= q_hi
        out[t] = (q_lo, q_hi, cov_lo and cov_hi, cov_lo, cov_hi)
        if t >= warmup:
            gamma = GAMMA_0 / (1.0 + LAM * steps) ** BETA
            c_lo += gamma * ((0.0 if cov_lo else 1.0) - a_side)
            c_hi += gamma * ((0.0 if cov_hi else 1.0) - a_side)
            steps += 1

    df = pd.DataFrame(out, columns=["q_lo", "q_hi", "covered", "covered_lo",
                                    "covered_hi"])
    df["warmup"] = np.arange(n) < warmup
    return df
