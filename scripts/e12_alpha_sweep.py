"""E12: does the conclusion survive the choice of alpha?

Runs per-stock ACI vs pooled adaptive (ours) at alpha in {0.05, 0.10, 0.20},
reporting marginal/calm/stress coverage. The stress-undercoverage of
marginal methods and its repair should not be an artifact of alpha=0.10.

Usage: .venv/bin/python scripts/e12_alpha_sweep.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal.aci import run_aci_panel
from src.conformal.panel_hierarchical import run_panel_mondrian
from src.eval.coverage import coverage_by_state, marginal_coverage
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeding import seed_everything

CUTS = [0.5, 0.8, 0.95]


def aligned_bins(market: pd.DataFrame) -> pd.DataFrame:
    v = market["vix_pctl"].values
    k = np.searchsorted(CUTS, v, side="right")
    pi = np.zeros((len(v), 4))
    ok = ~np.isnan(v)
    pi[ok, k[ok]] = 1.0
    pi[~ok] = 0.25
    return pd.DataFrame(pi, index=market.index,
                        columns=[f"regime_{j}" for j in range(4)])


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    proc = PROJECT_ROOT / cfg["data"]["processed_path"]
    panel = pd.read_parquet(proc / "rv_panel.parquet")
    preds = pd.read_parquet(proc / "e0_predictions.parquet")
    state = panel[["ticker", "date", "vix_pctl"]]
    market = panel.groupby("date")[["vix_pctl"]].first().sort_index()
    member = aligned_bins(market)

    rows = []
    for alpha in (0.05, 0.10, 0.20):
        aci = run_aci_panel(preds.dropna(subset=["pool"]), "pool",
                            alpha=alpha, eta=0.05, warmup=100)
        aci["width"] = aci["q_lo"] + aci["q_hi"]
        aci = aci.merge(state, on=["ticker", "date"], how="left")
        rc = run_panel_mondrian(preds, member, "pool", alpha=alpha,
                                adaptive=True, warmup_days=100)
        rc = rc.merge(state, on=["ticker", "date"], how="left")
        for name, res in [("aci", aci), ("rc_adaptive", rc)]:
            by = coverage_by_state(res, "vix_pctl")
            row = {"alpha": alpha, "method": name,
                   "marginal": marginal_coverage(res),
                   "cov_calm": by.loc["calm", "coverage"],
                   "cov_stress": by.loc["stress", "coverage"],
                   "cov_stress_upper": by.loc["stress", "upper_coverage"]}
            rows.append(row)
            print(f"alpha={alpha} {name:<12} "
                  + "  ".join(f"{k}={v:.4f}" for k, v in row.items()
                              if k not in ("alpha", "method")))

    out = pd.DataFrame(rows)
    out.to_csv(PROJECT_ROOT / "reports" / "e12_alpha_sweep.csv", index=False)
    print(f"\nsaved -> reports/e12_alpha_sweep.csv")


if __name__ == "__main__":
    main()
