"""E15: expanded rate grid — does the stress gap survive giving per-stock
adaptive baselines the fast rates that E13 showed pooling supplies?

E13 identified the pooling mechanism as effective-rate scaling: the
rate-matched per-stock tracker (fixed eta=0.2/obs) matches pooled average
stress coverage. But the ADAPTIVE per-stock arm's expert grid ends at
0.064, i.e. the aggregator was never offered the rates the fast arm needs.
This experiment closes that gap. Arms (alpha=0.10, warmup 100 days,
standardized scores, K=4 hard VIX bins unless noted):

  rc_std           pooled K=4, adaptive, standard grid (headline; reference)
  rc_exp           pooled K=4, adaptive, EXPANDED grid (+0.128/0.256/0.512)
  pooled_k1_exp    pooled K=1, adaptive, expanded grid
  pooled_avg_exp   pooled K=4, adaptive, AVERAGED errors (per-day step on
                   the mean error) with the expanded grid: same daily-rate
                   menu as the per-stock arms
  ps_k4_std        per-stock K=4, adaptive, standard grid (reference)
  ps_k4_exp        per-stock K=4, adaptive, expanded grid  <- THE TEST
  ps_k1_exp        per-stock K=1, adaptive, expanded grid (no regimes)
  dtaci_exp        per-stock DtACI on raw residual scores with its grid
                   extended from 0.2 to 0.8 (main-table baseline, widened)

Readout: if ps_k4_exp (or ps_k1_exp / dtaci_exp) closes the stress gap to
rc_std, the headline gain came from denying baselines fast rates; if the
gap persists, the aggregator cannot exploit fast rates on a single noisy
stream and panel pooling is doing real work.

Also reports the effective per-observation rate the aggregator issues in
stress for each adaptive arm (does ps_k4_exp actually reach ~0.2?), and
paired date-clustered stress-coverage tests of every arm against rc_std.

Usage: .venv/bin/python scripts/e15_expanded_grid.py
Output: reports/e15_expanded_grid.csv, reports/e15_expanded_grid_tests.csv,
        reports/e15_eff_eta.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.e6b_oracle_regimes import bin_membership
from src.conformal.dtaci import run_dtaci
from src.conformal.panel_hierarchical import (DEFAULT_ETA_GRID,
                                              run_panel_mondrian)
from src.eval.coverage import coverage_by_state, marginal_coverage
from src.eval.dm_hac import dm_test, hac_mean_se
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.parallel import pmap
from src.utils.seeding import seed_everything

FORECAST = "pool"
ALPHA = 0.10
WARMUP = 100
CUTS = [0.5, 0.8, 0.95]
GRID_STD = DEFAULT_ETA_GRID
GRID_EXP = DEFAULT_ETA_GRID + (0.128, 0.256, 0.512)
DTACI_ETAS_EXP = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8)


def _pooled(args):
    label, preds, member, kw = args
    res = run_panel_mondrian(preds, member, FORECAST, alpha=ALPHA,
                             warmup_days=WARMUP, adaptive=True, **kw)
    return label, res


def _per_stock(args):
    g, member, grid = args
    if len(g["date"].unique()) < 600:
        return None
    res = run_panel_mondrian(g, member, FORECAST, alpha=ALPHA,
                             warmup_days=WARMUP, adaptive=True,
                             eta_grid=grid)
    # keep the stress effective rate for the diagnostic before attrs are
    # lost in concat
    ad = res.attrs.get("adaptive", {})
    return res, ad


def _dtaci_stock(args):
    g, etas = args
    g = g.dropna(subset=["target", FORECAST]).sort_values("date").copy()
    if len(g) <= WARMUP + 50:
        return None
    s = (g["target"] - g[FORECAST]).values
    res = run_dtaci(s, alpha=ALPHA, warmup=WARMUP, etas=etas)
    g["covered"] = res["covered"].values
    g["covered_lo"] = res["covered_lo"].values
    g["covered_hi"] = res["covered_hi"].values
    g["warmup"] = res["warmup"].values
    g["width"] = (res["q_lo"] + res["q_hi"]).values
    return g[["ticker", "date", "covered", "covered_lo", "covered_hi",
              "warmup", "width"]]


def summarize(res, state, label):
    res = res.merge(state, on=["ticker", "date"], how="left")
    by = coverage_by_state(res, "vix_pctl")
    d = res[~res.warmup]
    return {"arm": label, "marginal": marginal_coverage(res),
            "cov_calm": by.loc["calm", "coverage"],
            "cov_stress": by.loc["stress", "coverage"],
            "cov_stress_upper": by.loc["stress", "upper_coverage"],
            "width": d["width"].mean(),
            "width_stress": d.loc[d.vix_pctl > 0.95, "width"].mean()}


def stress_eff_eta(ad: dict, member: pd.DataFrame) -> float:
    """Mean issued effective upper-side rate on stress days (regime 3)."""
    if not ad:
        return np.nan
    eff = pd.DataFrame(ad["eff_eta_hi"])
    eff.index = pd.to_datetime(ad["dates"])
    k_last = eff.columns[-1]
    stress_days = member.index[member.iloc[:, -1] > 0.5]
    sel = eff.index.intersection(stress_days)
    return float(eff.loc[sel, k_last].mean()) if len(sel) else np.nan


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    proc = PROJECT_ROOT / cfg["data"]["processed_path"]
    panel = pd.read_parquet(proc / "rv_panel.parquet")
    preds = pd.read_parquet(proc / "e0_predictions.parquet")
    state = panel[["ticker", "date", "vix_pctl"]]
    market = panel.groupby("date")[["vix_pctl"]].first().sort_index()
    member4 = bin_membership(market, CUTS)
    member1 = pd.DataFrame(1.0, index=member4.index, columns=["regime_0"])

    pooled_jobs = [
        ("rc_std", preds, member4, dict(eta_grid=GRID_STD)),
        ("rc_exp", preds, member4, dict(eta_grid=GRID_EXP)),
        ("pooled_k1_exp", preds, member1, dict(eta_grid=GRID_EXP)),
        ("pooled_avg_exp", preds, member4,
         dict(eta_grid=GRID_EXP, average_errors=True)),
    ]
    rows, eta_rows, cov_frames = [], [], {}
    print("[pooled arms]", flush=True)
    for label, res in pmap(_pooled, pooled_jobs):
        rows.append(summarize(res, state, label))
        eta_rows.append({"arm": label,
                         "eff_eta_stress":
                         stress_eff_eta(res.attrs.get("adaptive", {}),
                                        member4)})
        cov_frames[label] = res.loc[~res.warmup,
                                    ["ticker", "date", "covered"]]
        print(pd.Series(rows[-1]).drop("arm").rename(label)
              .round(4).to_string(), "\n", flush=True)

    for label, member, grid in [("ps_k4_std", member4, GRID_STD),
                                ("ps_k4_exp", member4, GRID_EXP),
                                ("ps_k1_exp", member1, GRID_EXP)]:
        print(f"[{label}] grid max={max(grid)}", flush=True)
        jobs = [(g, member, grid) for _, g in preds.groupby("ticker")]
        outs = [r for r in pmap(_per_stock, jobs) if r is not None]
        res = pd.concat([r[0] for r in outs], ignore_index=True)
        # per-stock effective stress rate: average the per-ticker series
        effs = [stress_eff_eta(ad, member4) for _, ad in outs]
        rows.append(summarize(res, state, label))
        eta_rows.append({"arm": label,
                         "eff_eta_stress": float(np.nanmean(effs))})
        cov_frames[label] = res.loc[~res.warmup,
                                    ["ticker", "date", "covered"]]
        print(pd.Series(rows[-1]).drop("arm").rename(label)
              .round(4).to_string(), "\n", flush=True)

    print("[dtaci_exp] raw-score DtACI, etas ->", DTACI_ETAS_EXP, flush=True)
    jobs = [(g, DTACI_ETAS_EXP) for _, g in preds.groupby("ticker")]
    parts = [r for r in pmap(_dtaci_stock, jobs) if r is not None]
    res = pd.concat(parts, ignore_index=True)
    rows.append(summarize(res, state, "dtaci_exp"))
    eta_rows.append({"arm": "dtaci_exp", "eff_eta_stress": np.nan})
    cov_frames["dtaci_exp"] = res.loc[~res.warmup,
                                      ["ticker", "date", "covered"]]
    print(pd.Series(rows[-1]).drop("arm").rename("dtaci_exp")
          .round(4).to_string(), "\n", flush=True)

    out = pd.DataFrame(rows).set_index("arm")
    print("\n=== E15 expanded-grid arms ===")
    print(out.round(4).to_string())

    # paired date-clustered stress tests vs rc_std
    stress_keys = state.loc[state["vix_pctl"] > 0.95, ["ticker", "date"]]
    ours = cov_frames["rc_std"].merge(stress_keys, on=["ticker", "date"])
    test_rows = []
    for label, d in cov_frames.items():
        if label == "rc_std":
            continue
        b = d.merge(stress_keys, on=["ticker", "date"])
        pair = b.merge(ours, on=["ticker", "date"], suffixes=("_b", "_a"))
        dm = dm_test(pair["covered_a"].astype(float),
                     pair["covered_b"].astype(float), pair["date"])
        st = hac_mean_se(b["covered"].astype(float), b["date"])
        test_rows.append({"arm": label, "cov_stress": st["mean"],
                          "se": st["se"], "n": len(b),
                          "diff_vs_rc": dm["mean_diff"], "t": dm["dm"],
                          "p": dm["p"]})
    tests = pd.DataFrame(test_rows)
    print("\n=== stress coverage vs rc_std (paired, date-clustered) ===")
    print(tests.round(4).to_string(index=False))

    etas = pd.DataFrame(eta_rows)
    print("\n=== issued effective upper rate on stress days ===")
    print(etas.round(4).to_string(index=False))

    rep = PROJECT_ROOT / "reports"
    out.to_csv(rep / "e15_expanded_grid.csv")
    tests.to_csv(rep / "e15_expanded_grid_tests.csv", index=False)
    etas.to_csv(rep / "e15_eff_eta.csv", index=False)
    print("\nsaved -> reports/e15_expanded_grid*.csv, reports/e15_eff_eta.csv")


if __name__ == "__main__":
    main()
