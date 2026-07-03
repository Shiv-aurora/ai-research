"""E2 round 2: pooled panel calibrator vs per-stock variants on real data.

Adds the hierarchical pooling layer (per-observation OGD steps over the
cross-section of MAD-standardized scores) that the diagnostics showed was
missing: per-stock stress trackers get ~150 visits in 15 years; pooled ones
get ~15,000.

Methods:
  aci             per-stock two-sided ACI (baseline)
  rc_panel_bins   pooled panel Mondrian, K=4 hard bins on vix_pctl aligned
                  with evaluation slices (.5/.8/.95)
  rc_panel_hmm    pooled panel soft-Mondrian, K=3 online-HMM filtered probs

Reports coverage by VIX regime, width, and the days-since-entry profile
(the transition diagnostic).

Usage: .venv/bin/python scripts/e2_panel.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal.aci import run_aci_panel
from src.conformal.panel_hierarchical import run_panel_mondrian
from src.eval.coverage import coverage_by_state, marginal_coverage
from src.regimes.online_hmm import online_hmm_memberships
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeding import seed_everything

FORECAST = "pool"
CUTS = [0.5, 0.8, 0.95]
ETAS_BINS = [0.001, 0.0015, 0.003, 0.01]   # per-observation, N~100
ETAS_HMM = [0.001, 0.002, 0.01]
HMM_FEATURES = ["vix_pctl", "mkt_rv_pctl", "xs_dispersion"]


def aligned_bin_membership_frame(market: pd.DataFrame) -> pd.DataFrame:
    k = np.searchsorted(CUTS, market["vix_pctl"].values, side="right")
    pi = np.zeros((len(market), len(CUTS) + 1))
    ok = ~np.isnan(market["vix_pctl"].values)
    pi[ok, k[ok]] = 1.0
    pi[~ok] = 1.0 / (len(CUTS) + 1)
    return pd.DataFrame(pi, index=market.index,
                        columns=[f"regime_{j}" for j in range(len(CUTS) + 1)])


def dse_profile(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    out = []
    for _, g in df.sort_values(["ticker", "date"]).groupby("ticker"):
        inside = (g["vix_pctl"] > threshold).values
        run, c = np.zeros(len(g)), 0
        for i, flag in enumerate(inside):
            c = c + 1 if flag else 0
            run[i] = c
        out.append(pd.Series(run, index=g.index))
    dse = pd.concat(out)
    d = df.loc[~df["warmup"]].copy()
    d["dse"] = dse.reindex(d.index).replace(0, np.nan)
    d = d.dropna(subset=["dse"])
    bins = pd.cut(d["dse"], [0, 1, 2, 3, 5, 10, 1000],
                  labels=["1", "2", "3", "4-5", "6-10", ">10"])
    return d.groupby(bins, observed=True)["covered"].agg(["mean", "size"])


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    alpha = cfg["evaluation"]["alpha_two_sided"]

    panel = pd.read_parquet(PROJECT_ROOT / cfg["data"]["processed_path"] / "rv_panel.parquet")
    preds = pd.read_parquet(PROJECT_ROOT / cfg["data"]["processed_path"] / "e0_predictions.parquet")
    state = panel[["ticker", "date", "vix_pctl"]]
    market = (panel.groupby("date")[HMM_FEATURES].first().sort_index())

    print("[regimes] memberships...")
    bins_member = aligned_bin_membership_frame(market)
    hmm_member = online_hmm_memberships(market, HMM_FEATURES, n_regimes=3,
                                        min_train=750, seed=cfg["seed"])

    results = {}
    print("[baseline] per-stock ACI...")
    aci = run_aci_panel(preds.dropna(subset=[FORECAST]), FORECAST, alpha=alpha,
                        eta=0.05, warmup=100)
    aci["width"] = aci["q_lo"] + aci["q_hi"]
    results["aci"] = aci.merge(state, on=["ticker", "date"], how="left")

    print("[method] pooled panel x aligned bins (K=4)...")
    results["rc_panel_bins"] = run_panel_mondrian(
        preds, bins_member, FORECAST, alpha=alpha,
        eta_by_regime=ETAS_BINS, warmup_days=100,
    ).merge(state, on=["ticker", "date"], how="left")

    print("[method] pooled panel x online HMM (K=3, soft)...")
    results["rc_panel_hmm"] = run_panel_mondrian(
        preds, hmm_member, FORECAST, alpha=alpha,
        eta_by_regime=ETAS_HMM, warmup_days=100,
    ).merge(state, on=["ticker", "date"], how="left")

    print(f"\n=== E2 round 2 (alpha={alpha}) ===")
    rows = []
    for name, df in results.items():
        by = coverage_by_state(df, "vix_pctl")
        row = {"method": name, "marginal": marginal_coverage(df),
               "width": df.loc[~df.warmup, "width"].mean()}
        for b in by.index:
            row[f"cov_{b}"] = by.loc[b, "coverage"]
        row["cov_stress_upper"] = by.loc["stress", "upper_coverage"]
        row["width_stress"] = df.loc[(~df.warmup) & (df.vix_pctl > 0.95),
                                     "width"].mean()
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("method")
    print(summary.round(4).to_string())

    print("\n=== days-since-entry profile (top slice) ===")
    for name, df in results.items():
        print(f"\n{name}:")
        print(dse_profile(df).round(4).to_string())

    out = PROJECT_ROOT / "reports"
    summary.to_csv(out / "e2_panel_summary.csv")
    print(f"\nsaved -> {out / 'e2_panel_summary.csv'}")


if __name__ == "__main__":
    main()
