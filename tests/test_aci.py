"""ACI tests: marginal validity, adaptation, and the E1 mechanism in vitro —
marginal validity coexisting with conditional miscoverage across states."""

import numpy as np
import pandas as pd

from src.conformal.aci import run_aci
from src.eval.coverage import coverage_by_state, marginal_coverage


def test_marginal_coverage_iid():
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, 6000)
    res = run_aci(scores, alpha=0.10, eta=0.05)
    cov = res.loc[~res.warmup, "covered"].mean()
    assert abs(cov - 0.90) < 0.02


def test_adapts_to_variance_shift():
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(0, 1, 4000), rng.normal(0, 3, 4000)])
    res = run_aci(scores, alpha=0.10, eta=0.05)
    # still (approximately) marginally valid overall despite the shift
    cov = res.loc[~res.warmup, "covered"].mean()
    assert abs(cov - 0.90) < 0.03
    # and thresholds grew after the shift
    assert res.q_hi.iloc[-500:].mean() > 2 * res.q_hi.iloc[3000:3500].mean()


def test_marginal_validity_masks_state_conditional_miscoverage():
    """The paper's motivating mechanism, on synthetic data: scores whose scale
    depends on a persistent state. Vanilla ACI is marginally valid but
    under-covers in the high-vol state and over-covers in the calm state."""
    rng = np.random.default_rng(2)
    n = 20000
    # persistent two-state regime (long calm spells, shorter stress spells)
    state = np.zeros(n)
    s = 0
    for t in range(n):
        p_switch = 0.005 if s == 0 else 0.02
        if rng.random() < p_switch:
            s = 1 - s
        state[t] = s
    scores = rng.normal(0, 1 + 3 * state)

    res = run_aci(scores, alpha=0.10, eta=0.02)
    res["state_pctl"] = state  # 0 = calm, 1 = stress
    res["width"] = res.q_lo + res.q_hi

    assert abs(marginal_coverage(res) - 0.90) < 0.03  # marginally fine

    by = coverage_by_state(res, "state_pctl", bins=[-0.1, 0.5, 1.1],
                           labels=["calm", "stress"])
    assert by.loc["calm", "coverage"] > 0.92      # over-covers calm
    assert by.loc["stress", "coverage"] < 0.85    # under-covers stress
