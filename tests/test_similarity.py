"""KNN-state conformal: causal, near-nominal on regime-switching panels."""

import numpy as np
import pandas as pd

from src.conformal.similarity import run_knn_state_conformal
from tests.test_panel_hierarchical import make_panel_stream


def _market_state(member: pd.DataFrame, state: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"stress_ind": state.astype(float)},
                        index=member.index)


def test_knn_state_conformal_coverage():
    df, member, state = make_panel_stream()
    ms = _market_state(member, state)
    res = run_knn_state_conformal(df, ms, "m", alpha=0.10, k=200,
                                  warmup_days=150)
    res["state"] = res["date"].map(pd.Series(state, index=member.index))
    d = res[~res.warmup]
    # Even with a perfectly informative state, similarity selection has no
    # coverage feedback: it under-covers the rare regime early (too few
    # similar days exist) and cannot track within-episode drift. Marginal
    # stays near-nominal; the rare-regime gap vs our tracked method is the
    # point of this baseline.
    assert abs(d["covered"].mean() - 0.90) < 0.02
    assert abs(d.loc[d.state == 0, "covered"].mean() - 0.90) < 0.02
    knn_stress = d.loc[d.state == 1, "covered"].mean()
    assert knn_stress > 0.75                      # sane, not collapsed

    from src.conformal.panel_hierarchical import run_panel_mondrian
    rc = run_panel_mondrian(df, member, "m", alpha=0.10, adaptive=True,
                            warmup_days=150)
    rc["state"] = rc["date"].map(pd.Series(state, index=member.index))
    rc_stress = rc.loc[(~rc.warmup) & (rc.state == 1), "covered"].mean()
    assert rc_stress > knn_stress + 0.03          # tracking beats similarity


def test_knn_state_conformal_is_causal():
    df, member, state = make_panel_stream(n_days=900, n_stocks=5)
    ms = _market_state(member, state)
    r1 = run_knn_state_conformal(df, ms, "m", alpha=0.10, warmup_days=150)
    # corrupt the future: scores after day 600 exploded
    df2 = df.copy()
    cut = member.index[600]
    df2.loc[df2.date > cut, "target"] += 50.0
    r2 = run_knn_state_conformal(df2, ms, "m", alpha=0.10, warmup_days=150)
    m1 = r1[r1.date <= cut].set_index(["ticker", "date"])["q_hi"]
    m2 = r2[r2.date <= cut].set_index(["ticker", "date"])["q_hi"]
    pd.testing.assert_series_equal(m1, m2)
