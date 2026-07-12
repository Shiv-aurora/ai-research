"""E13: pooling-mechanism identification — information sharing vs step size.

The pooled update sums errors over ~100 stocks, so a pooled tracker both
(a) shares one threshold across the cross-section and (b) mechanically
moves ~100x further per day than a per-stock tracker at the same
per-observation rate. This experiment separates the two channels with a
2x2 design over {pooled, per-stock} x {fast, slow} effective daily rate,
plus an averaged-error arm that isolates the summing convention itself.

All arms: fixed-rate (no adaptivity, to control rates exactly), K=4 hard
VIX bins, alpha=0.10, 100-day warmup, no offsets.

  pooled_fast      pooled, summed errors, eta=0.002/obs   (canonical rate)
  pooled_avg       pooled, averaged errors, eta=0.2/day   (= 0.002 x n_bar;
                   identical to pooled_fast up to n_t variation)
  pooled_slow      pooled, summed errors, eta=0.00002/obs (moves per day
                   like a per-stock tracker at the canonical rate)
  per_stock_slow   per-stock, eta=0.002/obs               (starved arm)
  per_stock_fast   per-stock, eta=0.2/obs                 (moves per day
                   like the pooled tracker: the rate-matched test)

Readout: if pooling works via step size, pooled_fast ~ per_stock_fast;
if via information sharing, pooled_fast ~ pooled_slow > per_stock_fast.

Usage: .venv/bin/python scripts/e13_pooling_mechanism.py
Output: reports/e13_pooling_mechanism.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.e6b_oracle_regimes import bin_membership
from src.conformal.panel_hierarchical import run_panel_mondrian
from src.eval.coverage import coverage_by_state, marginal_coverage
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.parallel import pmap
from src.utils.seeding import seed_everything

FORECAST = "pool"
CUTS = [0.5, 0.8, 0.95]
ETA_CANON = 0.002          # canonical per-observation rate
N_BAR = 100                # panel size scale for rate matching


def _pooled(args):
    label, preds, member, kw = args
    res = run_panel_mondrian(preds, member, FORECAST, alpha=0.10,
                             warmup_days=100, **kw)
    return label, res


def _per_stock(args):
    g, member, eta = args
    if len(g["date"].unique()) < 600:
        return None
    return run_panel_mondrian(g, member, FORECAST, alpha=0.10,
                              eta_by_regime=eta, warmup_days=100)


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


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    proc = PROJECT_ROOT / cfg["data"]["processed_path"]
    panel = pd.read_parquet(proc / "rv_panel.parquet")
    preds = pd.read_parquet(proc / "e0_predictions.parquet")
    state = panel[["ticker", "date", "vix_pctl"]]
    market = panel.groupby("date")[["vix_pctl"]].first().sort_index()
    member = bin_membership(market, CUTS)

    pooled_jobs = [
        ("pooled_fast", preds, member, dict(eta_by_regime=ETA_CANON)),
        ("pooled_avg", preds, member,
         dict(eta_by_regime=ETA_CANON * N_BAR, average_errors=True)),
        ("pooled_slow", preds, member,
         dict(eta_by_regime=ETA_CANON / N_BAR)),
    ]
    rows = []
    print("[pooled arms]", flush=True)
    for label, res in pmap(_pooled, pooled_jobs):
        row = summarize(res, state, label)
        print(pd.Series(row).drop("arm").rename(label).round(4).to_string(),
              "\n", flush=True)
        rows.append(row)

    for label, eta in [("per_stock_slow", ETA_CANON),
                       ("per_stock_fast", ETA_CANON * N_BAR)]:
        print(f"[{label}] eta={eta}", flush=True)
        jobs = [(g, member, eta) for _, g in preds.groupby("ticker")]
        parts = [r for r in pmap(_per_stock, jobs) if r is not None]
        res = pd.concat(parts, ignore_index=True)
        row = summarize(res, state, label)
        print(pd.Series(row).drop("arm").rename(label).round(4).to_string(),
              "\n", flush=True)
        rows.append(row)

    out = pd.DataFrame(rows).set_index("arm")
    print(out.round(4).to_string())
    out.to_csv(PROJECT_ROOT / "reports" / "e13_pooling_mechanism.csv")
    print("saved -> reports/e13_pooling_mechanism.csv")


if __name__ == "__main__":
    main()
