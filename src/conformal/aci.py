"""Adaptive Conformal Inference (Gibbs & Candès 2021), online, per stock or pooled.

Vanilla ACI tracks a miscoverage level alpha_t online:
    alpha_{t+1} = alpha_t + gamma * (alpha - err_t),  err_t = 1{y_t not covered}
and forms intervals from the trailing empirical quantile of nonconformity
scores at level 1 - alpha_t.

We implement the (equivalent up to parameterization) quantile-tracking form
used by later work (Angelopoulos et al.): track the score threshold q_t
directly by online pinball-loss gradient descent:
    q_{t+1} = q_t + eta * (err_t - alpha_side)
(widen on a miss, decay gently while covered; note the sign is OPPOSITE to
the alpha_t-tracking parameterization of Gibbs & Candes, where lower alpha_t
means wider intervals). This is the K=1 special case of the soft-Mondrian
method (P4) — that reduction is enforced by a regression test.

Scores here are SIGNED residuals s_t = y_t - m_t handled as a two-sided pair
(lower/upper thresholds tracked separately), matching the skewed error
distribution of log-RV forecasts.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ACIState:
    q_lo: float   # threshold for lower tail (on -s scale)
    q_hi: float   # threshold for upper tail (on +s scale)


def run_aci(
    scores: np.ndarray,
    alpha: float = 0.10,
    eta: float = 0.05,
    warmup: int = 100,
) -> pd.DataFrame:
    """Online two-sided ACI over a single stream of signed scores.

    scores: s_t = y_t - m_t in time order. At each t we form the interval
    [m_t - q_lo_t, m_t + q_hi_t] BEFORE seeing y_t, then update. Each side is
    tracked at level alpha/2. Warmup thresholds come from the first `warmup`
    scores' empirical quantiles (those rows are marked warmup=True and
    excluded from evaluation).

    Returns per-t: q_lo, q_hi, covered, covered_lo, covered_hi, warmup flag.
    """
    n = len(scores)
    a_side = alpha / 2.0
    q_lo = float(np.quantile(-scores[:warmup], 1 - a_side))
    q_hi = float(np.quantile(scores[:warmup], 1 - a_side))

    out = np.zeros((n, 5))
    for t in range(n):
        s = scores[t]
        cov_lo = (-s) <= q_lo
        cov_hi = s <= q_hi
        covered = cov_lo and cov_hi
        out[t] = (q_lo, q_hi, covered, cov_lo, cov_hi)
        if t >= warmup:
            # per-side pinball updates: widen on miss, decay gently on cover
            q_lo += eta * ((0.0 if cov_lo else 1.0) - a_side)
            q_hi += eta * ((0.0 if cov_hi else 1.0) - a_side)
        # during warmup thresholds stay at the empirical initialization

    df = pd.DataFrame(out, columns=["q_lo", "q_hi", "covered", "covered_lo",
                                    "covered_hi"])
    df["warmup"] = np.arange(n) < warmup
    return df


def run_aci_panel(
    preds: pd.DataFrame,
    forecast_col: str,
    alpha: float = 0.10,
    eta: float = 0.05,
    warmup: int = 100,
) -> pd.DataFrame:
    """Run per-stock ACI streams over a walk-forward predictions frame.

    preds: columns ticker, date, target, <forecast_col>; time-sorted within
    ticker. Returns preds + interval columns and coverage indicators, with
    warmup rows flagged.
    """
    frames = []
    for ticker, g in preds.sort_values("date").groupby("ticker"):
        g = g.dropna(subset=["target", forecast_col]).copy()
        if len(g) <= warmup + 50:
            continue
        scores = (g["target"] - g[forecast_col]).values
        aci = run_aci(scores, alpha=alpha, eta=eta, warmup=warmup)
        aci.index = g.index
        frames.append(pd.concat([g, aci], axis=1))
    return pd.concat(frames).sort_values(["ticker", "date"])
