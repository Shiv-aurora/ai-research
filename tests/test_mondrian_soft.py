"""Soft-Mondrian calibrator tests: exact ACI reduction, hand-computed traces,
and the headline in-vitro result — regime-conditional coverage where vanilla
ACI conditionally fails."""

import numpy as np
import pandas as pd

from src.conformal.aci import run_aci
from src.conformal.mondrian_soft import run_soft_mondrian
from src.eval.coverage import coverage_by_state, marginal_coverage


def _two_state_stream(n=20000, seed=2):
    """Persistent calm/stress regime with state-dependent score scale —
    the same generator as the ACI conditional-failure test."""
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    s = 0
    for t in range(n):
        p_switch = 0.005 if s == 0 else 0.02
        if rng.random() < p_switch:
            s = 1 - s
        state[t] = s
    scores = rng.normal(0, 1 + 3 * state)
    return scores, state


def test_k1_reduces_exactly_to_aci():
    rng = np.random.default_rng(7)
    scores = rng.normal(0, 1, 3000)
    pi = np.ones((3000, 1))
    ours = run_soft_mondrian(scores, pi, alpha=0.10, eta_by_regime=0.05)
    ref = run_aci(scores, alpha=0.10, eta=0.05)
    np.testing.assert_allclose(ours["q_hi"].values, ref["q_hi"].values, atol=1e-12)
    np.testing.assert_allclose(ours["q_lo"].values, ref["q_lo"].values, atol=1e-12)
    assert (ours["covered"].values == ref["covered"].values).all()


def test_hand_computed_update_trace():
    """Two regimes, soft membership, one covered step and one miss."""
    scores = np.array([0.0] * 10 + [0.5, 9.0])
    pi = np.tile([0.75, 0.25], (12, 1))
    res = run_soft_mondrian(scores, pi, alpha=0.20, eta_by_regime=[0.1, 0.4],
                            warmup=10)
    # warmup quantile of scores[:10]=0 at level 0.9 -> q_hi0 = q_lo0 = 0
    # t=10: issued q_hi = 0. s=0.5 -> miss (err=1).
    assert res.q_hi.iloc[10] == 0.0 and not res.covered_hi.iloc[10]
    # updates: q_hi[0] += 0.1*0.75*(1-0.1) = 0.0675 ; q_hi[1] += 0.4*0.25*0.9 = 0.09
    # t=11 issued: 0.75*0.0675 + 0.25*0.09 = 0.073125
    assert abs(res.q_hi.iloc[11] - 0.073125) < 1e-12


def test_fixes_conditional_coverage_where_aci_fails():
    scores, state = _two_state_stream()
    # oracle hard memberships (regime estimation is layered on separately)
    pi = np.zeros((len(scores), 2))
    pi[np.arange(len(scores)), state] = 1.0
    res = run_soft_mondrian(scores, pi, alpha=0.10,
                            eta_by_regime=[0.02, 0.08])
    res["state_pctl"] = state.astype(float)
    res["width"] = res.q_lo + res.q_hi

    assert abs(marginal_coverage(res) - 0.90) < 0.02
    by = coverage_by_state(res, "state_pctl", bins=[-0.1, 0.5, 1.1],
                           labels=["calm", "stress"])
    # vanilla ACI on this stream: calm > 0.92, stress < 0.85 (see test_aci).
    # regime-conditional tracking must repair BOTH directions:
    assert abs(by.loc["calm", "coverage"] - 0.90) < 0.02
    assert abs(by.loc["stress", "coverage"] - 0.90) < 0.03


def test_soft_membership_guarantee_and_graceful_degradation():
    """What the method actually guarantees under membership uncertainty
    (Prop 2): pi-WEIGHTED per-regime coverage ~ 1-alpha, for any blur.
    Conditional-on-TRUTH coverage degrades with membership error (honesty
    remark), but with realistically sharp filtered probabilities (90/10)
    the repair over vanilla ACI must survive.

    Empirical design note (verified): updating each regime against its OWN
    threshold instead of the issued one breaks marginal validity (92.6%)
    and worsens stress coverage (72% vs 84%) under blur — the issued-interval
    update rule is load-bearing."""
    scores, state = _two_state_stream(seed=11)
    pi_true = np.zeros((len(scores), 2))
    pi_true[np.arange(len(scores)), state] = 1.0

    for blur in (0.2, 0.05):
        pi = pi_true * (1 - 2 * blur) + blur
        res = run_soft_mondrian(scores, pi, alpha=0.10,
                                eta_by_regime=[0.02, 0.08])
        d = res[~res.warmup]
        # Prop-2 object: pi-weighted per-regime coverage
        for k in range(2):
            w = pi[len(scores) - len(d):, k]
            cov_w = (d["covered"].values * w).sum() / w.sum()
            # tolerance reflects the finite-sample bound O(eta) + O(1/sqrt(T_k)):
            # the stress regime gets ~4.8k effective visits here
            assert abs(cov_w - 0.90) < 0.03, (blur, k, cov_w)

    # realistic sharpness: conditional-on-truth repair must beat vanilla ACI
    pi_sharp = pi_true * 0.9 + 0.05
    res = run_soft_mondrian(scores, pi_sharp, alpha=0.10,
                            eta_by_regime=[0.02, 0.08])
    res["state_pctl"] = state.astype(float)
    by = coverage_by_state(res, "state_pctl", bins=[-0.1, 0.5, 1.1],
                           labels=["calm", "stress"])
    assert by.loc["stress", "coverage"] > 0.855   # vanilla ACI: ~0.83
    assert by.loc["calm", "coverage"] < 0.92      # vanilla ACI: >0.92


def test_one_sided_head():
    rng = np.random.default_rng(5)
    scores = rng.normal(0, 1, 10000)
    pi = np.ones((10000, 1))
    res = run_soft_mondrian(scores, pi, alpha=0.05, one_sided=True)
    cov = res.loc[~res.warmup, "covered"].mean()
    assert abs(cov - 0.95) < 0.015
    assert res["q_lo"].isna().all()
