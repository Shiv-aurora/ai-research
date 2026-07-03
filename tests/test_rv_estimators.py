"""Unit tests for RV/BPV estimators on synthetic tick data with known truth."""

import numpy as np
import pandas as pd
import pytest

from src.data.rv_estimators import DayMeasures, clean_trades, realized_measures


def make_gbm_day(sigma_daily: float, n_seconds: int = 23400, seed: int = 7,
                 jump_at: int | None = None, jump_size: float = 0.0) -> pd.Series:
    """One 09:30-16:00 day of 1-second prices from a driftless GBM with the
    given daily volatility of log returns; optional single jump."""
    rng = np.random.default_rng(seed)
    dt_sigma = sigma_daily / np.sqrt(n_seconds)
    r = rng.normal(0.0, dt_sigma, n_seconds)
    if jump_at is not None:
        r[jump_at] += jump_size
    log_p = np.log(100.0) + np.cumsum(r)
    idx = pd.date_range("2024-06-03 09:30:00", periods=n_seconds, freq="s",
                        tz="US/Eastern")
    return pd.Series(np.exp(log_p), index=idx)


def test_rv_recovers_known_variance():
    sigma = 0.02  # 2% daily vol
    prices = make_gbm_day(sigma)
    m = realized_measures(prices)
    # 5-min RV is a noisy estimate of integrated variance; 3 std errors of the
    # asymptotic RV error (sqrt(2/n)*IV, n=78 returns) is ~48% relative.
    assert m.rv == pytest.approx(sigma**2, rel=0.5)
    assert m.n_returns >= 70


def test_bpv_close_to_rv_without_jumps():
    prices = make_gbm_day(0.02, seed=11)
    m = realized_measures(prices)
    assert m.bpv == pytest.approx(m.rv, rel=0.35)


def test_jump_inflates_rv_but_not_bpv():
    sigma = 0.01
    no_jump = realized_measures(make_gbm_day(sigma, seed=3))
    jump = realized_measures(
        make_gbm_day(sigma, seed=3, jump_at=11700, jump_size=0.03),
        clean=False,  # a genuine price jump must not be filtered as an outlier here
    )
    assert jump.rv > no_jump.rv * 5  # 3% jump dominates 1% diffusive day
    # BPV is jump-robust: inflated far less than RV
    assert (jump.bpv - no_jump.bpv) < 0.3 * (jump.rv - no_jump.rv)


def test_subsampling_reduces_variance_of_estimator():
    sigma = 0.02
    est_sub, est_plain = [], []
    for seed in range(20):
        prices = make_gbm_day(sigma, seed=seed)
        est_sub.append(realized_measures(prices, subsample=True).rv)
        est_plain.append(realized_measures(prices, subsample=False).rv)
    assert np.var(est_sub) < np.var(est_plain)


def test_cleaning_removes_bad_ticks():
    prices = make_gbm_day(0.02, seed=5)
    dirty = prices.copy()
    # An isolated bad tick between grid points is invisible to 5-min sampling,
    # so contaminate a short burst straddling the 12:00:00 grid point
    # (index 9000 = 09:30 + 9000s) plus a zero price elsewhere.
    dirty.iloc[8995:9006] = dirty.iloc[8995:9006] * 1.2
    dirty.iloc[6000] = 0.0
    cleaned = clean_trades(dirty)
    assert len(cleaned) <= len(dirty) - 12
    m_dirty = realized_measures(dirty, clean=False)
    m_clean = realized_measures(dirty, clean=True)
    m_true = realized_measures(prices, clean=False)
    # the sampled burst badly inflates dirty RV; cleaning must undo it
    assert m_dirty.rv > 5 * m_true.rv
    assert abs(m_clean.rv - m_true.rv) < abs(m_dirty.rv - m_true.rv) * 0.05


def test_out_of_session_ticks_dropped():
    prices = make_gbm_day(0.02, seed=9)
    pre = pd.Series(
        [99.0],
        index=pd.DatetimeIndex([pd.Timestamp("2024-06-03 08:00:00", tz="US/Eastern")]),
    )
    with_premarket = pd.concat([pre, prices])
    assert len(clean_trades(with_premarket)) == len(clean_trades(prices))


def test_degenerate_inputs():
    empty = pd.Series(
        [], dtype=float,
        index=pd.DatetimeIndex([], tz="US/Eastern"),
    )
    m = realized_measures(empty)
    assert isinstance(m, DayMeasures) and np.isnan(m.rv)
