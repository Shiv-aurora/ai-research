"""E10: held-out generalization — does the method survive disjoint names?

Every design choice in this project was iterated against the full 100-name
panel. The method now has NO tuned parameters (adaptive rates, universal
defaults), so the honest generalization test is: split the universe into
two disjoint 50-name halves (alternating alphabetical — arbitrary and
pre-registered here), run the ENTIRE calibration pipeline independently on
each half with frozen defaults, and check that the headline conclusions
(regime-conditional repair vs ACI, coverage levels, width) reproduce on
both. Pooled thresholds learn only from their own half.

Usage: .venv/bin/python scripts/e10_holdout.py
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

    tickers = sorted(preds["ticker"].unique())
    halves = {"half_A": tickers[0::2], "half_B": tickers[1::2]}

    rows = []
    for half, names in halves.items():
        sub = preds[preds["ticker"].isin(names)]
        print(f"\n=== {half}: {len(names)} names, {len(sub):,} stock-days ===")

        aci = run_aci_panel(sub.dropna(subset=["pool"]), "pool", alpha=0.10,
                            eta=0.05, warmup=100)
        aci["width"] = aci["q_lo"] + aci["q_hi"]
        aci = aci.merge(state, on=["ticker", "date"], how="left")

        rc = run_panel_mondrian(sub, member, "pool", alpha=0.10,
                                adaptive=True, warmup_days=100)
        rc = rc.merge(state, on=["ticker", "date"], how="left")

        for name, res in [("aci", aci), ("rc_adaptive", rc)]:
            by = coverage_by_state(res, "vix_pctl")
            d = res[~res.warmup]
            row = {"half": half, "method": name,
                   "marginal": marginal_coverage(res),
                   "cov_stress": by.loc["stress", "coverage"],
                   "cov_stress_upper": by.loc["stress", "upper_coverage"],
                   "cov_calm": by.loc["calm", "coverage"],
                   "width": d["width"].mean(),
                   "width_stress": d.loc[d.vix_pctl > 0.95, "width"].mean()}
            rows.append(row)
            print(pd.Series(row).drop(["half", "method"])
                  .rename(name).round(4).to_string())

    out = pd.DataFrame(rows).set_index(["half", "method"])
    out.to_csv(PROJECT_ROOT / "reports" / "e10_holdout.csv")
    print(f"\nsaved -> reports/e10_holdout.csv")


if __name__ == "__main__":
    main()
