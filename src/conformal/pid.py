"""Conformal PID control (Angelopoulos, Candes & Tibshirani 2023), per
stock, as a direct baseline.

The paper's corrector is explicitly motivated by conformal PID's
quantile-tracking + integrator structure, so the method itself must
appear in the comparison. Implementation follows the paper's canonical
components on a single score stream:

  P (quantile tracking):  q_{t+1} = q_t + eta * (err_t - a)
  I (integrator):         r_t = saturation( sum_{s<=t} (err_s - a) )
                          with the tan saturation function
                          r_t = K_I * tan( x * log(T)/(C * T) ) applied
                          to the running error sum x (their default
                          nonlinear saturation; C=20)
  D (scorecaster):        optional forecast of the score's quantile; we
                          use the trailing-500-day empirical (1-a)
                          quantile, their simplest instantiation.

Issued threshold: q_t + r_t (+ scorecaster delta when enabled). Per
side at alpha/2 on the signed standardized-score stream, matching every
other per-stock baseline in the stack. eta defaults to 0.05 on the
raw-residual scale (the ACI baseline's rate); e15-style rate sweeps are
exposed via the eta argument so "PID + faster rate" is testable.
"""

import numpy as np
import pandas as pd

C_SAT = 20.0


class _PidSide:
    def __init__(self, a_side: float, eta: float, T: int, k_i: float):
        self.a = a_side
        self.eta = eta
        self.q = 0.0
        self.err_sum = 0.0
        self.k_i = k_i
        self.scale = np.log(T) / (C_SAT * T)

    def threshold(self) -> float:
        r = self.k_i * np.tan(np.clip(self.err_sum * self.scale,
                                      -1.5, 1.5))
        return self.q + r

    def update(self, s: float) -> None:
        err = 1.0 if s > self.threshold() else 0.0
        self.q += self.eta * (err - self.a)
        self.err_sum += err - self.a


def run_conformal_pid(scores: np.ndarray, alpha: float = 0.10,
                      warmup: int = 100, eta: float = 0.05,
                      k_i: float = 1.0,
                      scorecaster: bool = True) -> pd.DataFrame:
    """Two-sided conformal PID over a stream of signed scores. Output
    schema matches run_aci for drop-in evaluation."""
    n = len(scores)
    a_side = alpha / 2.0
    lo = _PidSide(a_side, eta, n, k_i)
    hi = _PidSide(a_side, eta, n, k_i)
    lo.q = float(np.quantile(-scores[:warmup], 1 - a_side))
    hi.q = float(np.quantile(scores[:warmup], 1 - a_side))

    s_ser = pd.Series(scores)
    if scorecaster:
        # trailing empirical quantile of each side's score, shifted so
        # only past observations enter (their simplest scorecaster);
        # expressed as a delta against its own warmup level so the
        # tracking component keeps ownership of the base level
        q_hi_sc = s_ser.rolling(500, min_periods=warmup).quantile(
            1 - a_side).shift(1)
        q_lo_sc = (-s_ser).rolling(500, min_periods=warmup).quantile(
            1 - a_side).shift(1)
        d_hi = (q_hi_sc - hi.q).fillna(0.0).values
        d_lo = (q_lo_sc - lo.q).fillna(0.0).values
    else:
        d_hi = np.zeros(n)
        d_lo = np.zeros(n)

    out = np.zeros((n, 5))
    for t in range(n):
        s = scores[t]
        q_lo_t = lo.threshold() + d_lo[t]
        q_hi_t = hi.threshold() + d_hi[t]
        cov_lo, cov_hi = (-s) <= q_lo_t, s <= q_hi_t
        out[t] = (q_lo_t, q_hi_t, cov_lo and cov_hi, cov_lo, cov_hi)
        if t >= warmup:
            lo.update(-s - d_lo[t])
            hi.update(s - d_hi[t])

    df = pd.DataFrame(out, columns=["q_lo", "q_hi", "covered",
                                    "covered_lo", "covered_hi"])
    df["warmup"] = np.arange(n) < warmup
    return df
