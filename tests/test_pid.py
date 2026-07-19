"""Conformal PID baseline sanity: marginal validity, shift response,
causality."""

import numpy as np

from src.conformal.pid import run_conformal_pid


def test_pid_iid_marginal_validity():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(6000)
    res = run_conformal_pid(s, alpha=0.10, warmup=300)
    d = res[~res.warmup]
    assert abs(d["covered"].mean() - 0.90) < 0.02


def test_pid_adapts_to_level_shift():
    rng = np.random.default_rng(1)
    s = rng.standard_normal(4000)
    s[2000:] += 3.0
    res = run_conformal_pid(s, alpha=0.10, warmup=300)
    late = res.iloc[3000:]
    assert late["covered"].mean() > 0.85       # integrator catches up
    assert (res["q_hi"].iloc[3500] - res["q_hi"].iloc[1500]) > 1.5


def test_pid_causality():
    rng = np.random.default_rng(2)
    s = rng.standard_normal(900)
    a = run_conformal_pid(s, alpha=0.10, warmup=300)
    s2 = s.copy()
    s2[-1] = 50.0
    b = run_conformal_pid(s2, alpha=0.10, warmup=300)
    assert np.allclose(a["q_hi"].values, b["q_hi"].values)
