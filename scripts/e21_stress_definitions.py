"""E21: stress definitions the algorithm does not use.

The canonical algorithm conditions on trailing VIX-percentile bins and
the canonical evaluation slices by the same partition — operationally
legitimate but partly self-referential. Here the ISSUED intervals of
the headline method and the ACI baseline (identical runs, evaluation
only re-sliced) are evaluated on stress definitions external to the
algorithm:

  vix_abs_30 / 25   absolute VIX level >= 30 / >= 25
  mkt_rv_tail       market realized-volatility trailing percentile > .95
  credit_tail       credit spread trailing-750d percentile > .95
  crisis_windows    the eight named episode windows of E9 (fixed dates)

Each slice reports both methods' coverage and the paired
date-clustered gap.

Usage: .venv/bin/python scripts/e21_stress_definitions.py
Output: reports/e21_stress_definitions.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.e6b_oracle_regimes import bin_membership
from src.conformal.aci import run_aci
from src.conformal.panel_hierarchical import run_panel_mondrian
from src.eval.dm_hac import dm_test, hac_mean_se
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.parallel import pmap
from src.utils.seeding import seed_everything

WARMUP = 100


def _aci_stock(g):
    g = g.dropna(subset=["target", "pool"]).sort_values("date").copy()
    if len(g) <= WARMUP + 50:
        return None
    s = (g["target"] - g["pool"]).values
    res = run_aci(s, alpha=0.10, eta=0.05, warmup=WARMUP)
    g["covered"] = res["covered"].values
    g["warmup"] = res["warmup"].values
    return g[["ticker", "date", "covered", "warmup"]]


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    proc = PROJECT_ROOT / cfg["data"]["processed_path"]
    panel = pd.read_parquet(proc / "rv_panel.parquet")
    preds = pd.read_parquet(proc / "e0_predictions.parquet")
    market = (panel.groupby("date")
              [["vix_pctl", "VIXCLS", "mkt_rv_pctl", "credit_spread"]]
              .first().sort_index())
    member = bin_membership(market[["vix_pctl"]], [0.5, 0.8, 0.95])

    rc = run_panel_mondrian(preds, member, "pool", alpha=0.10,
                            adaptive=True, warmup_days=WARMUP)
    rc = rc[~rc.warmup]
    # interval-validity diagnostics: sides are tracked independently, so
    # verify the issued two-sided interval is well ordered (empty iff
    # q_lo + q_hi < 0 in standardized units)
    diag = pd.DataFrame([{
        "min_q_hi": float(rc["q_hi"].min()),
        "min_q_lo": float(rc["q_lo"].min()),
        "n_negative_q_hi": int((rc["q_hi"] < 0).sum()),
        "n_negative_q_lo": int((rc["q_lo"] < 0).sum()),
        "n_crossed": int(((rc["q_lo"] + rc["q_hi"]) < 0).sum()),
        "n_total": len(rc)}])
    print("=== interval-validity diagnostics (headline run) ===")
    print(diag.round(4).to_string(index=False))
    diag.to_csv(PROJECT_ROOT / "reports" / "e21_interval_validity.csv",
                index=False)
    rc = rc[["ticker", "date", "covered"]]
    parts = [r for r in pmap(_aci_stock,
                             [g for _, g in preds.groupby("ticker")])
             if r is not None]
    aci = pd.concat(parts, ignore_index=True)
    aci = aci[~aci.warmup][["ticker", "date", "covered"]]
    pair = rc.merge(aci, on=["ticker", "date"],
                    suffixes=("_rc", "_aci"))

    # external stress definitions (market-level, date-indexed)
    credit_pctl = (market["credit_spread"]
                   .rolling(750, min_periods=250)
                   .rank(pct=True))
    win = pd.read_csv(PROJECT_ROOT / "reports" / "e9_stress_windows.csv")
    in_window = pd.Series(False, index=market.index)
    for w in win["window"]:
        a, b = w.split("..")
        in_window.loc[a:b] = True

    defs = {
        "vix_pctl>.95 (canonical)": market["vix_pctl"] > 0.95,
        "VIX>=30 (absolute)": market["VIXCLS"] >= 30,
        "VIX>=25 (absolute)": market["VIXCLS"] >= 25,
        "mkt RV pctl>.95": market["mkt_rv_pctl"] > 0.95,
        "credit spread pctl>.95": credit_pctl > 0.95,
        "named crisis windows": in_window,
    }
    rows = []
    for lab, flag in defs.items():
        d = pair[pair["date"].map(flag).fillna(False)]
        if len(d) == 0:
            continue
        st_rc = hac_mean_se(d["covered_rc"].astype(float), d["date"])
        st_ac = hac_mean_se(d["covered_aci"].astype(float), d["date"])
        dm = dm_test(d["covered_rc"].astype(float),
                     d["covered_aci"].astype(float), d["date"])
        rows.append({"definition": lab, "n": len(d),
                     "n_dates": st_rc["n_dates"],
                     "cov_rc": st_rc["mean"], "se_rc": st_rc["se"],
                     "cov_aci": st_ac["mean"],
                     "gap": dm["mean_diff"], "t": dm["dm"],
                     "p": dm["p"]})
    out = pd.DataFrame(rows)
    print("=== coverage under stress definitions external to the "
          "algorithm ===")
    print(out.round(4).to_string(index=False))
    out.to_csv(PROJECT_ROOT / "reports" / "e21_stress_definitions.csv",
               index=False)
    print("saved -> reports/e21_stress_definitions.csv")


if __name__ == "__main__":
    main()
