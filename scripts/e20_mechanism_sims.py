"""E20: what pooling can and cannot buy under cross-sectional
dependence — empirical effective panel size + controlled simulations.

Part 1 (empirical): the paper's mechanism story previously claimed the
pooled update achieves per-stock speed at ~1/n of the gradient
variance. Under cross-sectional correlation rho that is false: for
exchangeable errors, Var(mean) = sigma^2 [rho + (1-rho)/n], flooring at
rho*sigma^2. We estimate rho of the daily upper-miss indicators (method
= rc_adaptive issued intervals re-derived from a fresh run) by the
moment identity rho = (n*Var(daily mean) - mean within-day var) /
((n-1)*mean within-day var), overall and on stress days, and report
n_eff = n / (1 + (n-1) rho).

Part 2 (simulation): synthetic panels isolate the channels the
empirical ablations confound. Scores s_it = sqrt(rho)*f_t +
sqrt(1-rho)*e_it, scaled by regime (stress x3, two-state Markov chain,
stationary stress prob .05, mean duration 15 days) and by per-stock
heterogeneity multipliers (lognormal, sigma_h). Methods: per-stock
tracker, pooled K=1, pooled K=2 hard-regime tracker (all fixed-rate,
canonical eta=.002/obs so rates are controlled). Grid: rho in
{0,.3,.6,.9} x n in {10,100} x sigma_h in {0,.5}. Readout: stress and
onset (first 2 days) coverage per cell.

Usage: .venv/bin/python scripts/e20_mechanism_sims.py
Output: reports/e20_rho_neff.csv, reports/e20_sim_grid.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal.panel_hierarchical import run_panel_mondrian
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.parallel import pmap
from src.utils.seeding import seed_everything

ETA = 0.002
T_SIM = 3000
WARM = 100
RHOS = [0.0, 0.3, 0.6, 0.9]
NS = [10, 100]
HETS = [0.0, 0.5]
P_STRESS = 0.05
DUR = 15                      # mean stress duration (days)


def estimate_rho(miss: pd.DataFrame) -> dict:
    """miss: columns [date, m] with m a 0/1 indicator per stock-day."""
    g = miss.groupby("date")["m"]
    daily_mean = g.mean()
    n_bar = g.size().mean()
    within = g.var(ddof=1).mean()
    between = daily_mean.var(ddof=1)
    # Var(mean_t) = sigma^2 [rho + (1-rho)/n]  with sigma^2 ~ within
    rho = (n_bar * between - within) / ((n_bar - 1) * within)
    rho = float(np.clip(rho, 0.0, 1.0))
    n_eff = n_bar / (1 + (n_bar - 1) * rho)
    return {"rho": rho, "n_bar": float(n_bar), "n_eff": float(n_eff)}


def sim_panel(rng, rho, n, sigma_h):
    dates = pd.bdate_range("2005-01-03", periods=T_SIM)
    # two-state Markov regime: stationary stress prob P_STRESS, mean
    # stress duration DUR days
    p_exit = 1.0 / DUR
    p_enter = p_exit * P_STRESS / (1 - P_STRESS)
    state = np.zeros(T_SIM, dtype=int)
    for t in range(1, T_SIM):
        if state[t - 1] == 1:
            state[t] = 0 if rng.random() < p_exit else 1
        else:
            state[t] = 1 if rng.random() < p_enter else 0
    scale_t = np.where(state == 1, 3.0, 1.0)
    h = np.exp(rng.normal(0, sigma_h, n))
    f = rng.standard_normal(T_SIM)
    e = rng.standard_normal((T_SIM, n))
    s = (np.sqrt(rho) * f[:, None] + np.sqrt(1 - rho) * e) \
        * scale_t[:, None] * h[None, :]
    rows = []
    for i in range(n):
        rows.append(pd.DataFrame({"ticker": f"S{i}", "date": dates,
                                  "target": s[:, i], "m": 0.0}))
    preds = pd.concat(rows, ignore_index=True)
    member2 = pd.DataFrame(
        {"calm": (state == 0).astype(float),
         "stress": (state == 1).astype(float)}, index=dates)
    member1 = pd.DataFrame(1.0, index=dates, columns=["all"])
    return preds, member1, member2, state, dates


def dse_of(state: np.ndarray) -> np.ndarray:
    run, c = np.zeros(len(state)), 0
    for i, flag in enumerate(state == 1):
        c = c + 1 if flag else 0
        run[i] = c
    return run


def _cell(args):
    seed, rho, n, sigma_h = args
    rng = np.random.default_rng(seed)
    preds, m1, m2, state, dates = sim_panel(rng, rho, n, sigma_h)
    dse = pd.Series(dse_of(state), index=dates)
    arms = {
        "pooled_k2": run_panel_mondrian(preds, m2, "m", alpha=0.10,
                                        eta_by_regime=ETA,
                                        warmup_days=WARM,
                                        scale_window=250),
        "pooled_k1": run_panel_mondrian(preds, m1, "m", alpha=0.10,
                                       eta_by_regime=ETA,
                                       warmup_days=WARM,
                                       scale_window=250),
    }
    # per-stock: same machinery, one stock at a time (rate controlled)
    parts = []
    for _, g in preds.groupby("ticker"):
        parts.append(run_panel_mondrian(g, m2, "m", alpha=0.10,
                                        eta_by_regime=ETA,
                                        warmup_days=WARM,
                                        scale_window=250))
    arms["per_stock_k2"] = pd.concat(parts, ignore_index=True)

    out = []
    for name, res in arms.items():
        res = res[~res.warmup].copy()
        res["stress"] = res["date"].map(
            pd.Series(state, index=dates)).astype(bool)
        res["dse"] = res["date"].map(dse)
        out.append({
            "rho": rho, "n": n, "sigma_h": sigma_h, "arm": name,
            "marginal": res["covered"].mean(),
            "cov_stress": res.loc[res.stress, "covered"].mean(),
            "cov_onset": res.loc[res.dse.between(1, 2),
                                 "covered"].mean()})
    return out


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    proc = PROJECT_ROOT / cfg["data"]["processed_path"]

    # ---- Part 1: empirical rho / n_eff of upper-miss indicators
    panel = pd.read_parquet(proc / "rv_panel.parquet")
    preds = pd.read_parquet(proc / "e0_predictions.parquet")
    from scripts.e6b_oracle_regimes import bin_membership
    market = panel.groupby("date")[["vix_pctl"]].first().sort_index()
    member = bin_membership(market, [0.5, 0.8, 0.95])
    res = run_panel_mondrian(preds, member, "pool", alpha=0.10,
                             adaptive=True, warmup_days=100)
    res = res[~res.warmup].merge(panel[["ticker", "date", "vix_pctl"]],
                                 on=["ticker", "date"], how="left")
    rows = []
    for lab, mask in [("all days", np.ones(len(res), bool)),
                      ("stress days", (res["vix_pctl"] > 0.95).values)]:
        sub = res[mask]
        miss = pd.DataFrame({"date": sub["date"],
                             "m": (~sub["covered_hi"]).astype(float)})
        r = estimate_rho(miss)
        r["slice"] = lab
        rows.append(r)
    rho_tab = pd.DataFrame(rows)[["slice", "rho", "n_bar", "n_eff"]]
    print("=== empirical intra-date correlation of upper-miss "
          "indicators ===")
    print(rho_tab.round(3).to_string(index=False))

    # ---- Part 2: simulation grid
    jobs = [(cfg["seed"] + 1000 * i, rho, n, het)
            for i, (rho, n, het) in enumerate(
                (r, n, h) for r in RHOS for n in NS for h in HETS)]
    all_rows = []
    for out in pmap(_cell, jobs):
        all_rows.extend(out)
    sim = pd.DataFrame(all_rows)
    print("\n=== simulation grid (fixed rates; stress coverage) ===")
    print(sim.pivot_table(index=["rho", "n", "sigma_h"], columns="arm",
                          values="cov_stress").round(3).to_string())
    print("\n(onset = first 2 days of stress)")
    print(sim.pivot_table(index=["rho", "n", "sigma_h"], columns="arm",
                          values="cov_onset").round(3).to_string())

    rep = PROJECT_ROOT / "reports"
    rho_tab.to_csv(rep / "e20_rho_neff.csv", index=False)
    sim.to_csv(rep / "e20_sim_grid.csv", index=False)
    print("\nsaved -> reports/e20_rho_neff.csv, reports/e20_sim_grid.csv")


if __name__ == "__main__":
    main()
