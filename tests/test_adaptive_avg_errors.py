"""Adaptive average_errors path: on a constant-n panel, a per-day step on
the mean error with grid n*g and corrector n*eta_corr must reproduce the
summed path with grid g and corrector eta_corr exactly."""

import numpy as np
import pandas as pd

from src.conformal.panel_hierarchical import run_panel_mondrian

N_STOCKS = 10


def _panel():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2005-01-03", periods=1200)
    rows = []
    for i in range(N_STOCKS):
        rows.append(pd.DataFrame({
            "ticker": f"S{i}", "date": dates,
            "target": rng.standard_normal(len(dates)), "m": 0.0}))
    return pd.concat(rows, ignore_index=True)


def test_averaged_equals_summed_when_rescaled():
    preds = _panel()
    dates = pd.bdate_range("2005-01-03", periods=1200)
    member = pd.DataFrame(1.0, index=dates, columns=["g0"])
    grid = (0.002, 0.008, 0.032)
    a = run_panel_mondrian(preds, member, "m", alpha=0.10, adaptive=True,
                           eta_grid=grid, eta_corr=0.002, warmup_days=100)
    b = run_panel_mondrian(preds, member, "m", alpha=0.10, adaptive=True,
                           eta_grid=tuple(g * N_STOCKS for g in grid),
                           eta_corr=0.002 * N_STOCKS,
                           average_errors=True, warmup_days=100)
    assert np.allclose(a["q_hi"].values, b["q_hi"].values)
    assert np.allclose(a["q_lo"].values, b["q_lo"].values)
