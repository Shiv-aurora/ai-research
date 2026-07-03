"""Soft-Mondrian regime-conditional online conformal calibration — the method.

Maintains per-regime threshold trackers and combines them at prediction time
with the filtered regime probabilities pi_t (membership uncertainty is a
first-class input, not an afterthought):

    issued threshold:  q_hat_t = sum_k pi_t(k) * q_t(k)
    update, each k:    q_{t+1}(k) = q_t(k) + eta_k * pi_t(k) * (err_t - a_side)

where err_t is the miss indicator of the ISSUED interval. With hard
memberships (pi one-hot) this is exactly per-regime (Mondrian) ACI; with
K = 1 it is exactly vanilla ACI — both reductions are enforced by regression
tests. Two-sided intervals track lower/upper separately (skewed errors);
the same machinery at level alpha_u on the upper side gives the one-sided
risk head.

Per-regime learning rates: stress regimes are short-lived and need fast
adaptation; calm regimes reward small steps. `eta_by_regime` sets this
explicitly; the DtACI-style per-regime learning-rate aggregation is layered
on later (P4 completion) and reduces to this with a singleton eta grid.
"""

import numpy as np
import pandas as pd


class SoftMondrianCalibrator:
    def __init__(
        self,
        n_regimes: int,
        alpha: float = 0.10,
        eta_by_regime: list[float] | float = 0.05,
        one_sided: bool = False,
    ) -> None:
        self.K = n_regimes
        self.alpha = alpha
        self.a_side = alpha if one_sided else alpha / 2.0
        self.one_sided = one_sided
        if np.isscalar(eta_by_regime):
            eta_by_regime = [float(eta_by_regime)] * n_regimes
        assert len(eta_by_regime) == n_regimes
        self.eta = np.asarray(eta_by_regime, dtype=float)
        self.q_lo: np.ndarray | None = None   # (K,)
        self.q_hi: np.ndarray | None = None

    def init_thresholds(self, warmup_scores: np.ndarray) -> None:
        """Initialize all regimes from pooled warmup quantiles."""
        q_hi0 = float(np.quantile(warmup_scores, 1 - self.a_side))
        self.q_hi = np.full(self.K, q_hi0)
        if not self.one_sided:
            q_lo0 = float(np.quantile(-warmup_scores, 1 - self.a_side))
            self.q_lo = np.full(self.K, q_lo0)

    def issued(self, pi: np.ndarray) -> tuple[float | None, float]:
        q_hi = float(pi @ self.q_hi)
        q_lo = None if self.one_sided else float(pi @ self.q_lo)
        return q_lo, q_hi

    def update(self, s: float, pi: np.ndarray) -> dict:
        """Observe score s under membership pi; returns issued thresholds and
        coverage indicators (evaluated BEFORE the update, as issued)."""
        q_lo, q_hi = self.issued(pi)
        cov_hi = s <= q_hi
        cov_lo = True if self.one_sided else (-s) <= q_lo
        step = self.eta * pi
        self.q_hi += step * ((0.0 if cov_hi else 1.0) - self.a_side)
        if not self.one_sided:
            self.q_lo += step * ((0.0 if cov_lo else 1.0) - self.a_side)
        return {"q_lo": np.nan if q_lo is None else q_lo, "q_hi": q_hi,
                "covered": cov_lo and cov_hi,
                "covered_lo": cov_lo, "covered_hi": cov_hi}


def run_soft_mondrian(
    scores: np.ndarray,
    regime_probs: np.ndarray,
    alpha: float = 0.10,
    eta_by_regime: list[float] | float = 0.05,
    one_sided: bool = False,
    warmup: int = 100,
) -> pd.DataFrame:
    """Stream interface over aligned (scores, regime_probs) arrays.

    regime_probs: (n, K) rows summing to 1 (filtered probabilities, or one-hot
    for hard regimes). Output schema matches run_aci for drop-in evaluation.
    """
    n, K = regime_probs.shape
    assert len(scores) == n
    cal = SoftMondrianCalibrator(K, alpha=alpha, eta_by_regime=eta_by_regime,
                                 one_sided=one_sided)
    cal.init_thresholds(scores[:warmup])

    rows = []
    for t in range(n):
        pi = regime_probs[t]
        if t < warmup:
            q_lo, q_hi = cal.issued(pi)
            s = scores[t]
            cov_hi = s <= q_hi
            cov_lo = True if one_sided else (-s) <= q_lo
            rows.append({"q_lo": np.nan if q_lo is None else q_lo, "q_hi": q_hi,
                         "covered": cov_lo and cov_hi,
                         "covered_lo": cov_lo, "covered_hi": cov_hi})
        else:
            rows.append(cal.update(scores[t], pi))

    df = pd.DataFrame(rows)
    df["warmup"] = np.arange(n) < warmup
    return df
