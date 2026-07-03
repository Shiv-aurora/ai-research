"""Panel calibrator tests: pooling accelerates rare-regime convergence."""

import numpy as np
import pandas as pd

from src.conformal.panel_hierarchical import run_panel_mondrian
from src.conformal.scores import trailing_scale


def make_panel_stream(n_days=3000, n_stocks=20, seed=0):
    """Panel of forecasts whose residual scale doubles in a shared rare
    stress regime (~8% of days) and differs by stock."""
    rng = np.random.default_rng(seed)
    state = np.zeros(n_days, dtype=int)
    s = 0
    for t in range(n_days):
        if rng.random() < (0.01 if s == 0 else 0.12):
            s = 1 - s
        state[t] = s
    dates = pd.bdate_range("2013-01-01", periods=n_days)
    rows = []
    for i in range(n_stocks):
        base = 0.5 + 0.1 * i / n_stocks
        resid = rng.normal(0, base * (1 + state))  # scale doubles in stress
        rows.append(pd.DataFrame({
            "ticker": f"S{i:02d}", "date": dates,
            "target": resid, "m": np.zeros(n_days),
        }))
    df = pd.concat(rows, ignore_index=True)
    member = pd.DataFrame(np.eye(2)[state], index=dates,
                          columns=["regime_0", "regime_1"])
    return df, member, state


def test_trailing_scale_is_causal():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 1, 400))
    s1 = trailing_scale(r)
    r2 = r.copy()
    r2.iloc[300:] = 50.0
    s2 = trailing_scale(r2)
    pd.testing.assert_series_equal(s1.iloc[:301], s2.iloc[:301])


def test_pooled_panel_repairs_rare_regime():
    df, member, state = make_panel_stream()
    res = run_panel_mondrian(df, member, "m", alpha=0.10,
                             eta_by_regime=[0.005, 0.05], warmup_days=150)
    res["state"] = res["date"].map(pd.Series(state, index=member.index))
    d = res[~res.warmup]
    marg = d["covered"].mean()
    cov_stress = d.loc[d.state == 1, "covered"].mean()
    cov_calm = d.loc[d.state == 0, "covered"].mean()
    assert abs(marg - 0.90) < 0.02
    # rare regime gets n_stocks x visits -> tight conditional coverage
    assert abs(cov_stress - 0.90) < 0.025
    assert abs(cov_calm - 0.90) < 0.015


def test_offsets_absorb_stock_heterogeneity():
    """A stock whose scale estimate is systematically wrong (misscaled
    forecasts) should still get near-target coverage via its offset."""
    df, member, _ = make_panel_stream(seed=3)
    # corrupt one stock's forecasts with a constant bias
    df.loc[df.ticker == "S00", "m"] += 0.4
    res = run_panel_mondrian(df, member, "m", alpha=0.10,
                             eta_by_regime=[0.005, 0.05],
                             eta_offset=0.01, warmup_days=150)
    d = res[~res.warmup]
    cov_biased = d.loc[d.ticker == "S00", "covered"].mean()
    assert cov_biased > 0.86
