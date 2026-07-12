"""E6d: fixed intervals, two regime classifications (the conditioning gap).

E6b reruns the calibrator under hindsight memberships, which answers "what
if the algorithm had received a different state sequence". This experiment
answers the sharper evaluation question: holding the ISSUED intervals
fixed (the causal canonical run), how different does conditional coverage
look when stress days are classified causally (trailing VIX-percentile
bin) versus with hindsight (full-sample smoothed HMM, state ordered by
mean, P(stress) > 0.5)? If the two classifications agree on where the
method under-covers, the causal evaluation slices are not flattering us.

Usage: .venv/bin/python scripts/e6d_reclassified_eval.py
Output: reports/e6d_reclassified_eval.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.e6b_oracle_regimes import (bin_membership,
                                        oracle_smoothed_memberships)
from src.conformal.panel_hierarchical import run_panel_mondrian
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeding import seed_everything

FORECAST = "pool"
CUTS = [0.5, 0.8, 0.95]


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    proc = PROJECT_ROOT / cfg["data"]["processed_path"]
    panel = pd.read_parquet(proc / "rv_panel.parquet")
    preds = pd.read_parquet(proc / "e0_predictions.parquet")
    market = panel.groupby("date")[["vix_pctl", "mkt_rv_pctl",
                                    "xs_dispersion"]].first().sort_index()

    member = bin_membership(market, CUTS)
    res = run_panel_mondrian(preds, member, FORECAST, alpha=0.10,
                             adaptive=True, warmup_days=100)
    res = res[~res.warmup].merge(
        panel[["ticker", "date", "vix_pctl"]], on=["ticker", "date"],
        how="left")

    # hindsight stress label from the smoothed HMM (last state = stress)
    oracle = oracle_smoothed_memberships(market, seed=cfg["seed"])
    stress_prob = oracle.iloc[:, -1]
    res["stress_causal"] = res["vix_pctl"] > 0.95
    p = res["date"].map(stress_prob).fillna(0)
    res["stress_hindsight"] = p > 0.5
    # tail-comparable hindsight set: same unconditional size as the causal
    # VIX tail (top ~5% of days by smoothed stress probability)
    cut = stress_prob.quantile(1 - res["stress_causal"].mean())
    res["stress_hindsight_tail"] = p >= cut

    rows = []
    for label, mask in [("causal VIX bin", res["stress_causal"]),
                        ("hindsight HMM", res["stress_hindsight"]),
                        ("both", res["stress_causal"]
                                 & res["stress_hindsight"]),
                        ("hindsight only", ~res["stress_causal"]
                                           & res["stress_hindsight"]),
                        ("hindsight tail (size-matched)",
                         res["stress_hindsight_tail"])]:
        sub = res[mask]
        rows.append({"stress definition": label,
                     "stock_days": len(sub),
                     "coverage": sub["covered"].mean(),
                     "upper_coverage": sub["covered_hi"].mean()})
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    out.to_csv(PROJECT_ROOT / "reports" / "e6d_reclassified_eval.csv",
               index=False)
    print("saved -> reports/e6d_reclassified_eval.csv")


if __name__ == "__main__":
    main()
