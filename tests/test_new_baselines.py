"""TCP-RM and cross-sectional panel conformal: marginal validity + causality."""

import numpy as np
import pandas as pd

from src.conformal.panel_xs import run_panel_xs
from src.conformal.tcp import run_tcp_rm


def test_tcp_rm_marginal_validity_iid():
    rng = np.random.default_rng(3)
    s = rng.standard_normal(5000)
    r = run_tcp_rm(s, alpha=0.10, warmup=100)
    ev = r[~r["warmup"]]
    assert abs(ev["covered"].mean() - 0.90) < 0.02


def test_tcp_rm_rm_offset_widens_after_misses():
    # a level shift in scores must push the threshold up via the RM offset
    rng = np.random.default_rng(4)
    s = np.concatenate([rng.standard_normal(500),
                        rng.standard_normal(500) + 4.0])
    r = run_tcp_rm(s, alpha=0.10, warmup=100)
    assert r["q_hi"].iloc[560] > r["q_hi"].iloc[499]


def test_panel_xs_marginal_validity_and_causality():
    rng = np.random.default_rng(5)
    dates = pd.date_range("2015-01-01", periods=600, freq="B")
    frames = []
    for tk in [f"S{i}" for i in range(20)]:
        m = rng.standard_normal(len(dates)).cumsum() * 0.01
        y = m + rng.standard_normal(len(dates))
        frames.append(pd.DataFrame({"ticker": tk, "date": dates,
                                    "target": y, "pool": m}))
    preds = pd.concat(frames, ignore_index=True)
    r = run_panel_xs(preds, "pool", alpha=0.10, warmup_days=100)
    ev = r[~r["warmup"]]
    assert abs(ev["covered"].mean() - 0.90) < 0.02
    # causality: the threshold issued on day t must not depend on day-t
    # outcomes — perturb the last day's targets and re-run; all issued
    # thresholds before the last day must be identical
    preds2 = preds.copy()
    last = preds2["date"] == dates[-1]
    preds2.loc[last, "target"] += 100.0
    r2 = run_panel_xs(preds2, "pool", alpha=0.10, warmup_days=100)
    early = r["date"] < dates[-1]
    assert np.allclose(r.loc[early, "q_hi"], r2.loc[early, "q_hi"],
                       equal_nan=True)
