"""RKR FTRL-OGD baseline sanity: marginal + group-conditional validity
with overlapping groups, and causality."""

import numpy as np

from src.conformal.rkr import marginal_plus_bins, run_rkr


def test_rkr_group_conditional_validity():
    rng = np.random.default_rng(0)
    n = 20000
    # persistent group-1 blocks with 3x score scale
    g1 = (np.arange(n) // 200) % 5 == 0
    s = rng.standard_normal(n) * np.where(g1, 3.0, 1.0)
    groups = np.zeros((n, 3))
    groups[:, 0] = 1.0
    groups[np.arange(n), 1 + g1.astype(int)] = 1.0
    res = run_rkr(s, groups, alpha=0.10, warmup=500, eta=0.05)
    d = res[~res.warmup]
    m = d["covered"].values
    assert abs(m.mean() - 0.90) < 0.02
    assert abs(m[g1[500:]].mean() - 0.90) < 0.05
    assert abs(m[~g1[500:]].mean() - 0.90) < 0.03


def test_rkr_causality():
    rng = np.random.default_rng(1)
    n = 900
    s = rng.standard_normal(n)
    groups = marginal_plus_bins(rng.uniform(0, 1, n))
    a = run_rkr(s, groups, warmup=300)
    s2 = s.copy()
    s2[-1] = 50.0
    b = run_rkr(s2, groups, warmup=300)
    assert np.allclose(a["q_hi"].values, b["q_hi"].values)
