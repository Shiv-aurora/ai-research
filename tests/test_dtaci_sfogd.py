"""DtACI and SF-OGD baseline tests: marginal validity and shift adaptation."""

import numpy as np

from src.conformal.dtaci import run_dtaci
from src.conformal.sfogd import run_sfogd


def _iid_scores(n=8000, seed=0):
    return np.random.default_rng(seed).normal(0, 1, n)


def _shift_scores(n=8000, seed=1):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0, 1, n // 2), rng.normal(0, 3, n // 2)])


def test_dtaci_marginal_iid():
    res = run_dtaci(_iid_scores(), alpha=0.10)
    cov = res.loc[~res.warmup, "covered"].mean()
    assert abs(cov - 0.90) < 0.02


def test_dtaci_adapts_to_shift():
    """DtACI guarantees regret-driven adaptation, not instant conditional
    recovery — assert the threshold converges toward the new truth (4.93 for
    N(0,3) at the 95th pctl) and coverage improves after the transition."""
    res = run_dtaci(_shift_scores(), alpha=0.10)
    assert res.q_hi.iloc[-200:].mean() > 4.0
    cov_transition = res.covered.iloc[4000:5000].mean()
    cov_late = res.covered.iloc[6500:].mean()
    assert cov_late > cov_transition + 0.02


def test_sfogd_marginal_iid():
    res = run_sfogd(_iid_scores(seed=2), alpha=0.10)
    cov = res.loc[~res.warmup, "covered"].mean()
    assert abs(cov - 0.90) < 0.02


def test_sfogd_adapts_to_shift():
    res = run_sfogd(_shift_scores(seed=3), alpha=0.10)
    # thresholds must grow after the variance triples
    assert res.q_hi.iloc[-500:].mean() > 1.8 * res.q_hi.iloc[3500:3900].mean()
