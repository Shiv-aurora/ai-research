"""DtACI-style adaptive conformal: expert aggregation over ACI learning rates.

Follows Gibbs & Candes (2022) "Conformal inference for online prediction with
arbitrary distribution shifts": run K threshold-tracking ACI experts with
different learning rates eta_k; combine with exponential weights on pinball
loss of each expert's threshold, with uniform mixing (sigma) so experts can
recover. We use the probability-weighted AVERAGE threshold rather than
randomized expert selection — the standard deterministic variant; the
difference is immaterial for coverage in practice and it removes evaluation
randomness. Two-sided via independent lower/upper trackers, as in aci.py.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_ETAS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2)


@dataclass
class _SideTracker:
    """One-sided DtACI tracker at miscoverage level a_side."""
    a_side: float
    etas: tuple = DEFAULT_ETAS
    meta_eta: float = 5.0
    sigma: float = 0.01
    q: np.ndarray = field(default=None)
    logw: np.ndarray = field(default=None)

    def init(self, q0: float) -> None:
        self.q = np.full(len(self.etas), q0, dtype=float)
        self.logw = np.zeros(len(self.etas))

    def threshold(self) -> float:
        w = np.exp(self.logw - self.logw.max())
        w /= w.sum()
        return float(w @ self.q)

    def update(self, s: float) -> None:
        """s: realized one-sided score (coverage means s <= threshold)."""
        # pinball loss of each expert's threshold at level 1 - a_side
        tau = 1.0 - self.a_side
        diff = s - self.q
        pinball = np.where(diff > 0, tau * diff, (tau - 1.0) * diff)
        self.logw -= self.meta_eta * pinball
        self.logw -= self.logw.max()
        # uniform mixing so dormant experts can revive
        w = np.exp(self.logw)
        w = (1 - self.sigma) * w / w.sum() + self.sigma / len(w)
        self.logw = np.log(w)
        # per-expert threshold updates (widen on own-miss)
        err = (s > self.q).astype(float)
        self.q += np.asarray(self.etas) * (err - self.a_side)


def run_dtaci(scores: np.ndarray, alpha: float = 0.10,
              warmup: int = 100, etas: tuple = DEFAULT_ETAS) -> pd.DataFrame:
    """Two-sided DtACI over a stream of signed scores. Same output schema as
    run_aci for drop-in evaluation."""
    n = len(scores)
    a_side = alpha / 2.0
    lo = _SideTracker(a_side, etas=etas)
    hi = _SideTracker(a_side, etas=etas)
    lo.init(float(np.quantile(-scores[:warmup], 1 - a_side)))
    hi.init(float(np.quantile(scores[:warmup], 1 - a_side)))

    out = np.zeros((n, 5))
    for t in range(n):
        s = scores[t]
        q_lo, q_hi = lo.threshold(), hi.threshold()
        cov_lo, cov_hi = (-s) <= q_lo, s <= q_hi
        out[t] = (q_lo, q_hi, cov_lo and cov_hi, cov_lo, cov_hi)
        if t >= warmup:
            lo.update(-s)
            hi.update(s)

    df = pd.DataFrame(out, columns=["q_lo", "q_hi", "covered", "covered_lo",
                                    "covered_hi"])
    df["warmup"] = np.arange(n) < warmup
    return df
