"""E2 round 4: forward-looking onset detection via the VIX term structure.

The day-1/2 onset pit survives every backward-looking conditioning scheme
(round 3). But the VIX term structure is forward-looking: when 9-day implied
vol exceeds 30-day (VIX9D/VIX > 1, near-term inversion), the market is
pricing acute stress AS IT HAPPENS — a signal that fires during the spike's
first day, not after it.

Groups: aligned K=4 bins on vix_pctl, PLUS an override group 'acute' when
VIX9D/VIX > 1 at the close of the issuance day. Evaluation restricted to
2012+ (VIX9D history starts 2011; warmup consumes the first year).

Usage: .venv/bin/python scripts/e2_onset.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal.panel_hierarchical import run_panel_mondrian
from src.data.cboe import fetch_cboe
from src.data.fred import fetch_series
from src.eval.coverage import coverage_by_state, marginal_coverage
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeding import seed_everything
from scripts.e2_transition import dse_profile

CUTS = [0.5, 0.8, 0.95]
ETAS = [0.001, 0.0015, 0.003, 0.01, 0.03]   # calm..stress, acute FAST
EVAL_START = "2012-01-01"


def onset_membership(market: pd.DataFrame) -> pd.DataFrame:
    v = market["vix_pctl"].values
    k = np.searchsorted(CUTS, v, side="right")
    acute = market["vix9d_vix"].values > 1.0
    groups = k.copy()
    groups[acute & ~np.isnan(market["vix9d_vix"].values)] = 4
    pi = np.zeros((len(v), 5))
    ok = ~np.isnan(v)
    pi[ok, groups[ok]] = 1.0
    pi[~ok] = 0.2
    return pd.DataFrame(pi, index=market.index,
                        columns=[f"regime_{j}" for j in range(5)])


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    proc = PROJECT_ROOT / cfg["data"]["processed_path"]
    panel = pd.read_parquet(proc / "rv_panel.parquet")
    preds = pd.read_parquet(proc / "e0_predictions.parquet")
    preds = preds[preds["date"] >= EVAL_START]

    vix = fetch_series("VIXCLS")
    vix9d = fetch_cboe("VIX9D")
    market = panel.groupby("date")[["vix_pctl"]].first().sort_index()
    ratio = (vix9d / vix).rename("vix9d_vix")
    market = market.join(ratio, how="left")
    print(f"acute (VIX9D>VIX) share of days since 2012: "
          f"{(market.loc[EVAL_START:, 'vix9d_vix'] > 1).mean():.3f}")

    member = onset_membership(market)
    state = panel[["ticker", "date", "vix_pctl"]]

    # baseline: same run WITHOUT the acute group (round-2 canonical config)
    base_member = member.iloc[:, :4].copy()
    row_sums = base_member.sum(axis=1)
    zero = row_sums == 0            # acute rows had their mass in group 4
    # put acute days back into the stress bin for the baseline
    base_member.loc[zero, "regime_3"] = 1.0

    results = {}
    for name, m, etas in [("rc_no_onset", base_member, ETAS[:4]),
                          ("rc_onset", member, ETAS)]:
        res = run_panel_mondrian(preds, m, "pool", alpha=0.10,
                                 eta_by_regime=etas, warmup_days=100)
        results[name] = res.merge(state, on=["ticker", "date"], how="left")

    for name, res in results.items():
        by = coverage_by_state(res, "vix_pctl")
        print(f"\n=== {name}: marginal={marginal_coverage(res):.4f} "
              f"width={res.loc[~res.warmup, 'width'].mean():.4f} ===")
        print(by[["coverage", "upper_coverage", "n"]].round(4).to_string())
        print("days-since-entry profile:")
        print(dse_profile(res).round(4).to_string())

    out = PROJECT_ROOT / "reports"
    for name, res in results.items():
        dse_profile(res).to_csv(out / f"e2onset_{name}_dse.csv")
    print(f"\nsaved -> reports/e2onset_*_dse.csv")


if __name__ == "__main__":
    main()
