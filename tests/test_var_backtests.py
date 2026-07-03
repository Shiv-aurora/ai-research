"""VaR backtest tests on streams with known properties."""

import numpy as np

from src.eval.var_backtests import (christoffersen_independence, dq_test,
                                    kupiec_pof)


def test_kupiec_accepts_correct_rate():
    rng = np.random.default_rng(0)
    e = rng.random(4000) < 0.05
    r = kupiec_pof(e, 0.05)
    assert r["p"] > 0.05


def test_kupiec_rejects_double_rate():
    rng = np.random.default_rng(1)
    e = rng.random(4000) < 0.10
    r = kupiec_pof(e, 0.05)
    assert r["p"] < 0.01


def test_christoffersen_rejects_clustered_hits():
    # persistent exceedances: once exceeding, 60% chance to exceed again
    rng = np.random.default_rng(2)
    e = np.zeros(4000, dtype=bool)
    for t in range(1, 4000):
        p = 0.6 if e[t - 1] else 0.03
        e[t] = rng.random() < p
    r = christoffersen_independence(e)
    assert r["p"] < 0.01
    # and iid hits pass
    e_iid = rng.random(4000) < 0.05
    assert christoffersen_independence(e_iid)["p"] > 0.05


def test_dq_rejects_dynamics_accepts_iid():
    rng = np.random.default_rng(3)
    e_iid = rng.random(4000) < 0.05
    assert dq_test(e_iid, 0.05)["p"] > 0.05
    e = np.zeros(4000, dtype=bool)
    for t in range(1, 4000):
        p = 0.5 if e[t - 1] else 0.03
        e[t] = rng.random() < p
    assert dq_test(e, 0.05)["p"] < 0.01
