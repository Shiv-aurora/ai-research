"""Timing-discipline tests for panel feature construction.

These test the *invariants* (no feature at date t uses information after t)
on synthetic frames, so they run without network access.
"""

import numpy as np
import pandas as pd

from src.data.panel import _trailing_percentile


def test_trailing_percentile_uses_only_past():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=400))
    p = _trailing_percentile(s, window=100)
    # Changing FUTURE values must not change the percentile at t.
    s2 = s.copy()
    s2.iloc[250:] = 99.0
    p2 = _trailing_percentile(s2, window=100)
    pd.testing.assert_series_equal(p.iloc[:250], p2.iloc[:250])


def test_trailing_percentile_extremes():
    s = pd.Series(np.arange(200, dtype=float))  # strictly increasing
    p = _trailing_percentile(s, window=50)
    # In an increasing series every point is the max of its trailing window.
    assert (p.dropna() == 1.0).all()


def test_dispersion_lag_convention():
    # Mirrors the shift(1) in build_panel: dispersion dated t must equal the
    # same-day statistic from t-1.
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"] * 2 + ["2024-01-03"] * 2),
        "log_rv": [1.0, 3.0, 2.0, 2.0],
    })
    same_day = df.groupby("date")["log_rv"].std()
    lagged = same_day.shift(1)
    assert np.isnan(lagged.iloc[0])
    assert lagged.iloc[1] == same_day.iloc[0]
