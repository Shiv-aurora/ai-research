"""E6-E8 ablations: which ingredients carry the method?

All runs use the adaptive-rate config (no tuning anywhere) on the pool
forecast unless stated. Three axes:

  (a) K: number of vix_pctl bin regimes, K=1..5. K=1 is the pooled
      marginal-ACI limit — the regime layer's value is the K>1 delta.
  (b) pooling: pooled thresholds + per-stock offsets (canonical) vs pooled
      only (offsets off) vs fully per-stock (each name calibrated alone —
      the starvation demonstration).
  (c) forecaster: conformalizing HAR / LGBM / pool — the layer should
      deliver its guarantee regardless of the point model underneath.

All independent runs execute across cores (pmap): 9 pooled configs in one
pool, then the 100 single-stock calibrations in another.

Usage: .venv/bin/python scripts/e6_ablations.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal.panel_hierarchical import run_panel_mondrian
from src.eval.coverage import coverage_by_state, marginal_coverage
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.parallel import pmap
from src.utils.seeding import seed_everything

K_CUTS = {1: [], 2: [0.95], 3: [0.8, 0.95],
          4: [0.5, 0.8, 0.95], 5: [0.25, 0.5, 0.8, 0.95]}


def bin_membership(market: pd.DataFrame, cuts: list[float]) -> pd.DataFrame:
    K = len(cuts) + 1
    v = market["vix_pctl"].values
    k = np.searchsorted(cuts, v, side="right")
    pi = np.zeros((len(v), K))
    ok = ~np.isnan(v)
    pi[ok, k[ok]] = 1.0
    pi[~ok] = 1.0 / K
    return pd.DataFrame(pi, index=market.index,
                        columns=[f"regime_{j}" for j in range(K)])


def _pooled_run(args):
    """Worker: one pooled calibration config (module-level for spawn)."""
    label, preds, member, fc, kw = args
    res = run_panel_mondrian(preds, member, fc, alpha=0.10,
                             adaptive=True, warmup_days=100, **kw)
    return label, res


def _single_stock_run(args):
    """Worker: one name calibrated alone (the starvation arm)."""
    g, member = args
    return run_panel_mondrian(g, member, "pool", alpha=0.10, adaptive=True,
                              eta_offset=0.0, offset_l2=0.0, warmup_days=100)


def summarize(res: pd.DataFrame, state: pd.DataFrame, label: str) -> dict:
    res = res.merge(state, on=["ticker", "date"], how="left")
    by = coverage_by_state(res, "vix_pctl")
    d = res[~res.warmup]
    row = {"config": label, "marginal": marginal_coverage(res),
           "cov_calm": by.loc["calm", "coverage"],
           "cov_stress": by.loc["stress", "coverage"],
           "cov_stress_upper": by.loc["stress", "upper_coverage"],
           "width": d["width"].mean(),
           "width_stress": d.loc[d.vix_pctl > 0.95, "width"].mean()}
    print(pd.Series(row).drop("config").rename(label).round(4).to_string(),
          "\n", flush=True)
    return row


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    proc = PROJECT_ROOT / cfg["data"]["processed_path"]
    panel = pd.read_parquet(proc / "rv_panel.parquet")
    preds = pd.read_parquet(proc / "e0_predictions.parquet")
    state = panel[["ticker", "date", "vix_pctl"]]
    market = panel.groupby("date")[["vix_pctl"]].first().sort_index()
    member4 = bin_membership(market, K_CUTS[4])

    jobs = [(f"K={K}", preds, bin_membership(market, cuts), "pool", {})
            for K, cuts in K_CUTS.items()]
    jobs.append(("pooled_no_offsets", preds, member4, "pool",
                 dict(eta_offset=0.0, offset_l2=0.0)))
    jobs += [(f"forecaster={fc}", preds.dropna(subset=[fc]), member4, fc, {})
             for fc in ["har", "lgbm", "pool"]]

    print(f"[pooled configs] {len(jobs)} runs across cores ...", flush=True)
    rows = []
    for label, res in pmap(_pooled_run, jobs):
        rows.append(summarize(res, state, label))

    print("[per-stock] 100 single-name calibrations across cores ...",
          flush=True)
    parts = pmap(_single_stock_run,
                 [(g, member4) for _, g in preds.groupby("ticker")])
    rows.append(summarize(pd.concat(parts, ignore_index=True), state,
                          "per_stock"))

    out = pd.DataFrame(rows).set_index("config")
    out.to_csv(PROJECT_ROOT / "reports" / "e6_ablations.csv")
    print(f"saved -> reports/e6_ablations.csv", flush=True)


if __name__ == "__main__":
    main()
