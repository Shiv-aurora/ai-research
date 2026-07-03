"""Regime-layer tests: state recovery on synthetic data and strict causality."""

import numpy as np
import pandas as pd

from src.regimes.online_hmm import online_hmm_memberships
from src.regimes.quantile_bins import quantile_bin_memberships


def make_market(n=2500, seed=4):
    """Two-state synthetic market: stress coordinate jumps between levels."""
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    s = 0
    for t in range(n):
        if rng.random() < (0.01 if s == 0 else 0.03):
            s = 1 - s
        state[t] = s
    vix_pctl = np.clip(0.3 + 0.5 * state + rng.normal(0, 0.08, n), 0, 1)
    mkt_rv_pctl = np.clip(0.35 + 0.45 * state + rng.normal(0, 0.08, n), 0, 1)
    dates = pd.bdate_range("2012-01-02", periods=n)
    market = pd.DataFrame({"vix_pctl": vix_pctl, "mkt_rv_pctl": mkt_rv_pctl},
                          index=dates)
    return market, state


def test_hmm_recovers_states():
    market, state = make_market()
    probs = online_hmm_memberships(market, ["vix_pctl", "mkt_rv_pctl"],
                                   n_regimes=2, min_train=500)
    post = probs.iloc[600:]
    hard = post.values.argmax(axis=1)
    acc = (hard == state[600:]).mean()
    assert acc > 0.9
    # canonical ordering: regime 1 (stress) probability higher on stress days
    assert post.loc[state[600:] == 1, "regime_1"].mean() > 0.8


def test_hmm_is_causal():
    market, _ = make_market(n=1500, seed=6)
    p1 = online_hmm_memberships(market, ["vix_pctl", "mkt_rv_pctl"],
                                n_regimes=2, min_train=500)
    tampered = market.copy()
    tampered.iloc[-200:, :] = 0.99  # rewrite the future
    p2 = online_hmm_memberships(tampered, ["vix_pctl", "mkt_rv_pctl"],
                                n_regimes=2, min_train=500)
    pd.testing.assert_frame_equal(p1.iloc[:1290], p2.iloc[:1290])


def test_quantile_bins_hard_and_causal():
    market, state = make_market(n=1200, seed=8)
    m = quantile_bin_memberships(market, n_regimes=3, window=500)
    post = m.iloc[100:]
    # hard memberships: rows are one-hot
    assert ((post.max(axis=1) == 1.0) & (post.sum(axis=1) == 1.0)).all()
    # causality
    tampered = market.copy()
    tampered.iloc[-100:, :] = 0.0
    m2 = quantile_bin_memberships(tampered, n_regimes=3, window=500)
    pd.testing.assert_frame_equal(m.iloc[:1100], m2.iloc[:1100])
    # stress days land in the top bin far more often
    top = m.values.argmax(axis=1) == 2
    assert top[state == 1].mean() > 3 * top[state == 0].mean()
