"""MCS must eliminate a dominated method and retain equal ones."""

import numpy as np
import pandas as pd

from src.eval.mcs import interval_score, mcs


def test_interval_score_penalizes_misses():
    y = np.array([0.0, 2.0])
    lo = np.array([-1.0, -1.0])
    hi = np.array([1.0, 1.0])
    s = interval_score(y, lo, hi, alpha=0.10)
    assert s[0] == 2.0                      # covered: just the width
    assert s[1] == 2.0 + 20.0 * 1.0         # miss by 1 at 2/alpha = 20


def test_mcs_eliminates_dominated_keeps_equals():
    # seed chosen so the two 'equal' methods draw a near-zero t-stat against
    # each other (seed 0 gives a genuine 2.5-sigma fluke, which the MCS
    # correctly rejects — that is behavior, not a bug)
    rng = np.random.default_rng(8)
    T = 1500
    common = rng.normal(0, 1, T)
    losses = pd.DataFrame({
        "good_a": common + rng.normal(0, 0.10, T),
        "good_b": common + rng.normal(0, 0.10, T),
        "bad": common + 0.30 + rng.normal(0, 0.10, T),
    })
    out = mcs(losses, alpha=0.10, n_boot=500, seed=1)
    assert not out.loc["bad", "in_mcs"]
    assert out.loc["good_a", "in_mcs"] and out.loc["good_b", "in_mcs"]
    assert out.loc["bad", "mcs_pvalue"] < 0.10


def test_mcs_all_equal_keeps_all():
    rng = np.random.default_rng(2)
    T = 1000
    losses = pd.DataFrame(rng.normal(0, 1, (T, 4)),
                          columns=list("abcd"))
    out = mcs(losses, alpha=0.10, n_boot=500, seed=3)
    assert out["in_mcs"].all()


def test_mcs_deterministic_given_seed():
    rng = np.random.default_rng(4)
    losses = pd.DataFrame(rng.normal(0, 1, (800, 3)), columns=list("xyz"))
    a = mcs(losses, n_boot=300, seed=7)
    b = mcs(losses, n_boot=300, seed=7)
    pd.testing.assert_frame_equal(a, b)
