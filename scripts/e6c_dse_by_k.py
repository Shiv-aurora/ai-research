"""E6c: coverage by days-since-stress-entry for K=1 vs K=4 (transition axis).

Regenerates the paper's claim that regimes buy transition coverage: the
day-2 value under pooled marginal tracking (K=1) vs the canonical K=4
bins, identical adaptive calibrator otherwise. Artifact: the ablations
section's dse-by-K numbers trace here.

Usage: .venv/bin/python scripts/e6c_dse_by_k.py
Output: reports/e6_dse_by_k.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.e6b_oracle_regimes import bin_membership, dse_coverage
from src.conformal.panel_hierarchical import run_panel_mondrian
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.parallel import pmap
from src.utils.seeding import seed_everything

FORECAST = "pool"
K_CUTS = {1: [], 4: [0.5, 0.8, 0.95]}


def _run(args):
    label, preds, member = args
    res = run_panel_mondrian(preds, member, FORECAST, alpha=0.10,
                             adaptive=True, warmup_days=100)
    return label, res


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    proc = PROJECT_ROOT / cfg["data"]["processed_path"]
    panel = pd.read_parquet(proc / "rv_panel.parquet")
    preds = pd.read_parquet(proc / "e0_predictions.parquet")
    state = panel[["ticker", "date", "vix_pctl"]]
    market = panel.groupby("date")[["vix_pctl"]].first().sort_index()

    jobs = [(f"K={k}", preds, bin_membership(market, cuts))
            for k, cuts in K_CUTS.items()]
    cols = {}
    for label, res in pmap(_run, jobs):
        res = res.merge(state, on=["ticker", "date"], how="left")
        cols[label] = dse_coverage(res)
    out = pd.DataFrame(cols)
    print(out.round(4).to_string())
    out.to_csv(PROJECT_ROOT / "reports" / "e6_dse_by_k.csv")
    print("saved -> reports/e6_dse_by_k.csv")


if __name__ == "__main__":
    main()
