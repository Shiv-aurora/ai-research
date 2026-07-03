"""Scale-free online gradient descent conformal tracker (SF-OGD).

Bhatnagar et al. (2023)-style: pinball-gradient steps with a scale-free
step size eta_t = eta0 / sqrt(sum of squared past gradients), giving anytime
regret guarantees without tuning eta to the horizon. Two-sided via
independent per-side trackers; same output schema as run_aci.
"""

import numpy as np
import pandas as pd


def run_sfogd(scores: np.ndarray, alpha: float = 0.10, eta0: float = 1.0,
              warmup: int = 100) -> pd.DataFrame:
    n = len(scores)
    a_side = alpha / 2.0
    q_lo = float(np.quantile(-scores[:warmup], 1 - a_side))
    q_hi = float(np.quantile(scores[:warmup], 1 - a_side))
    g2_lo = g2_hi = 1e-8

    out = np.zeros((n, 5))
    for t in range(n):
        s = scores[t]
        cov_lo, cov_hi = (-s) <= q_lo, s <= q_hi
        out[t] = (q_lo, q_hi, cov_lo and cov_hi, cov_lo, cov_hi)
        if t >= warmup:
            g_lo = (0.0 if cov_lo else 1.0) - a_side
            g_hi = (0.0 if cov_hi else 1.0) - a_side
            g2_lo += g_lo * g_lo
            g2_hi += g_hi * g_hi
            q_lo += eta0 / np.sqrt(g2_lo) * g_lo
            q_hi += eta0 / np.sqrt(g2_hi) * g_hi

    df = pd.DataFrame(out, columns=["q_lo", "q_hi", "covered", "covered_lo",
                                    "covered_hi"])
    df["warmup"] = np.arange(n) < warmup
    return df
