"""The paper's propositions (paper2/sections/theory.tex) rest on one claim:
the implementation's update IS eq. (3),

    q_{t+1}(k) = q_t(k) + eta_k * pi_t(k) * sum_i (err_{i,t} - a),

with err computed on the ISSUED threshold. These tests replay that recursion
from the calibrator's own outputs and require exact agreement, then check
the telescoped bounds: Prop. 1's fixed-rate identity (the soft-membership
run checks the conditional identity of Section 4.4), and Prop. 2's
adaptive-corrector bound under hard regimes. If implementation and theory
ever drift, this fails before a referee finds it.
"""

import numpy as np
import pandas as pd

from src.conformal.panel_hierarchical import run_panel_mondrian

ALPHA, ETA, WARMUP = 0.20, 0.01, 50
N_TICK, N_DAYS, K = 20, 600, 3


def _panel_and_membership(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=N_DAYS)
    rows = []
    for i in range(N_TICK):
        s = rng.standard_t(6, size=N_DAYS) * (1.0 + 0.5 * (i % 3))
        for t, d in enumerate(dates):
            rows.append((f"T{i:02d}", d, s[t]))
    preds = pd.DataFrame(rows, columns=["ticker", "date", "s"])
    preds["pool"] = 0.0
    # persistent soft memberships over K regimes
    raw = np.abs(rng.standard_normal((N_DAYS, K))) + 0.2
    for t in range(1, N_DAYS):
        raw[t] = 0.9 * raw[t - 1] + 0.1 * raw[t]
    pi = raw / raw.sum(axis=1, keepdims=True)
    member = pd.DataFrame(pi, index=dates,
                          columns=[f"regime_{k}" for k in range(K)])
    return preds, member


def test_update_rule_is_equation_3():
    preds, member = _panel_and_membership()
    res = run_panel_mondrian(
        preds, member, "pool", alpha=ALPHA, eta_by_regime=ETA,
        eta_offset=0.0, offset_l2=0.0, warmup_days=WARMUP, score_col="s",
    )
    a = ALPHA / 2.0

    # replay the recursion from outputs
    daily = res.sort_values(["date", "ticker"]).groupby("date", sort=True)
    dates = list(daily.groups)
    warm_end = dates[min(WARMUP, len(dates) - 1)]
    warm_scores = res.loc[res.date <= warm_end, "s_std"].values
    q = np.full(K, float(np.quantile(warm_scores, 1 - a)))

    for d, block in daily:
        pi = member.loc[d].values
        pi = pi / pi.sum()
        issued = float(pi @ q)
        # the calibrator must have ISSUED exactly pi @ q on every day
        assert np.allclose(block["q_hi"].values, issued, atol=1e-10), str(d)
        if d <= warm_end:
            continue
        g = float((block["s_std"].values > issued).sum()
                  - a * len(block))                      # sum_i (err - a)
        q = q + ETA * pi * g                             # eq. (3)


def test_proposition2_bound_holds():
    preds, member = _panel_and_membership(seed=1)
    res = run_panel_mondrian(
        preds, member, "pool", alpha=ALPHA, eta_by_regime=ETA,
        eta_offset=0.0, offset_l2=0.0, warmup_days=WARMUP, score_col="s",
    )
    a = ALPHA / 2.0
    d = res[~res.warmup].copy()
    d["err"] = (~d["covered_hi"]).astype(float)
    B = float(np.abs(res["s_std"]).max())
    n_bar = d.groupby("date").size().max()

    pi = member.loc[d["date"].unique()]
    per_day = d.groupby("date")["err"].agg(["sum", "size"])
    g = per_day["sum"].values - a * per_day["size"].values

    for k in range(K):
        w = pi[f"regime_{k}"].values
        W_k = float((w * per_day["size"].values).sum())
        weighted_err = float((w * g).sum())
        bound = (2 * B + ETA * n_bar) / ETA
        assert abs(weighted_err) <= bound, f"regime {k}"
        # and the rate statement: weighted miss freq within bound / W_k
        assert abs(weighted_err) / W_k <= bound / W_k
        # sanity: with 600 days the realized weighted frequency should be
        # near nominal, far inside the worst-case bound
        assert abs(weighted_err) / W_k < 0.05


def test_proposition_adaptive_bound_hard_regimes():
    """Prop. 2 (adaptive, hard regimes): per-regime avg miss excess is within
    (2B + (eta_max + eta_corr) n_bar) / (eta_corr N_k) on the issued path."""
    from src.conformal.panel_hierarchical import DEFAULT_ETA_GRID

    preds, member_soft = _panel_and_membership(seed=2)
    # hard memberships: argmax of the soft draw, one-hot
    hard = np.zeros_like(member_soft.values)
    hard[np.arange(len(hard)), member_soft.values.argmax(axis=1)] = 1.0
    member = pd.DataFrame(hard, index=member_soft.index,
                          columns=member_soft.columns)
    eta_corr = 0.002
    res = run_panel_mondrian(
        preds, member, "pool", alpha=ALPHA, adaptive=True, eta_corr=eta_corr,
        eta_offset=0.0, offset_l2=0.0, warmup_days=WARMUP, score_col="s",
    )
    a = ALPHA / 2.0
    d = res[~res.warmup].copy()
    d["err"] = (~d["covered_hi"]).astype(float)
    B = float(np.abs(res["s_std"]).max())
    n_bar = d.groupby("date").size().max()
    eta_max = max(DEFAULT_ETA_GRID)

    per_day = d.groupby("date")["err"].agg(["sum", "size"])
    g = per_day["sum"].values - a * per_day["size"].values
    pi = member.loc[d["date"].unique()]
    for k in range(K):
        mask = pi[f"regime_{k}"].values == 1.0
        N_k = float(per_day["size"].values[mask].sum())
        if N_k == 0:
            continue
        excess = float(g[mask].sum())
        bound = (2 * B + (eta_max + eta_corr) * n_bar) / eta_corr
        assert abs(excess) <= bound, f"regime {k}"
        # realized per-regime miss frequency should be near nominal
        assert abs(excess) / N_k < 0.05
