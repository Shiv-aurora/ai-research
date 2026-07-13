"""POGO port sanity: marginal validity, group-conditional adaptation,
and causality."""

import numpy as np
import pandas as pd
import pytest

from src.conformal.pogo import run_pogo_panel


def _frame(scores: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    # forecast 0, target = score; window MAD of iid N(0,1) ~ stable, so
    # s_std tracks the raw score up to a slowly-varying positive scale
    return pd.DataFrame({"ticker": "AAA", "date": dates,
                         "target": scores, "m": 0.0})


def test_pogo_iid_marginal_validity():
    rng = np.random.default_rng(0)
    n = 6000
    dates = pd.bdate_range("2005-01-03", periods=n)
    df = _frame(rng.standard_normal(n), dates)
    member = pd.DataFrame(1.0, index=dates, columns=["g0"])
    res = run_pogo_panel(df, member, "m", alpha=0.10, warmup_days=300)
    d = res[~res.warmup]
    assert abs(d["covered"].mean() - 0.90) < 0.02


def test_pogo_group_conditional_adaptation():
    rng = np.random.default_rng(1)
    n = 12000
    dates = pd.bdate_range("2005-01-03", periods=n)
    # group 1 days carry 3x-scale scores in persistent blocks
    g1 = (np.arange(n) // 250) % 4 == 0
    raw = rng.standard_normal(n) * np.where(g1, 3.0, 1.0)
    df = _frame(raw, dates)
    member = pd.DataFrame({"g0": (~g1).astype(float),
                           "g1": g1.astype(float)}, index=dates)
    res = run_pogo_panel(df, member, "m", alpha=0.10, warmup_days=500)
    d = res[~res.warmup].copy()
    d["g1"] = g1[np.searchsorted(dates, d["date"].values)]
    cov0 = d.loc[~d.g1, "covered"].mean()
    cov1 = d.loc[d.g1, "covered"].mean()
    assert abs(cov0 - 0.90) < 0.03
    assert abs(cov1 - 0.90) < 0.06     # rare group: looser but must adapt
    # a marginal tracker at the pooled quantile would sit far below 0.9
    # on g1; POGO must not
    assert cov1 > 0.80


def test_pogo_causality():
    rng = np.random.default_rng(2)
    n = 900
    dates = pd.bdate_range("2005-01-03", periods=n)
    raw = rng.standard_normal(n)
    member = pd.DataFrame(1.0, index=dates, columns=["g0"])
    a = run_pogo_panel(_frame(raw, dates), member, "m", warmup_days=300)
    raw2 = raw.copy()
    raw2[-1] = 50.0                       # perturb only the last outcome
    b = run_pogo_panel(_frame(raw2, dates), member, "m", warmup_days=300)
    assert np.allclose(a["q_hi"].values, b["q_hi"].values)
