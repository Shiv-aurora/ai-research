"""E14: temporal robustness — subperiod stability and leave-one-crisis-out.

The method is fully online with constants fixed a priori (Appendix A), so
there is no fitting step to freeze; the frozen-design question is whether
the headline conclusions depend on any particular stretch of the
evaluation window that design iterations could have overfit to. Two
answers:

  (a) subperiod split: evaluate the SAME issued intervals separately on
      2010-2017 and 2018-2025 (the second half contains Volmageddon,
      COVID, the 2022 bear market, and the 2024-25 episodes, none of
      which existed when the HAR/conformal design space was fixed by the
      prior literature);
  (b) leave-one-crisis-out: recompute stress coverage dropping one
      stress episode's calendar year at a time — does the ACI deficit
      and its repair survive removing COVID? 2022? each other episode?

Methods: aci (the marginal baseline), pooled_k1, rc_adaptive (ours).
Usage: .venv/bin/python scripts/e14_temporal_holdout.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal.aci import run_aci_panel
from src.conformal.panel_hierarchical import run_panel_mondrian
from src.eval.dm_hac import dm_test
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeding import seed_everything

CUTS = [0.5, 0.8, 0.95]
SPLIT = pd.Timestamp("2018-01-01")
EPISODE_YEARS = [2011, 2015, 2018, 2020, 2022, 2024, 2025]


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
    member1 = pd.DataFrame(1.0, index=member.index, columns=["regime_0"])

    frames = {}
    aci = run_aci_panel(preds.dropna(subset=["pool"]), "pool", alpha=0.10,
                        eta=0.05, warmup=100)
    aci["width"] = aci["q_lo"] + aci["q_hi"]
    frames["aci"] = aci
    for name, mem in [("pooled_k1", member1), ("rc_adaptive", member)]:
        r = run_panel_mondrian(preds, mem, "pool", alpha=0.10,
                               adaptive=True, warmup_days=100)
        frames[name] = r

    for name in frames:
        d = frames[name].merge(state, on=["ticker", "date"], how="left")
        d = d[~d["warmup"]].copy()
        d["date"] = pd.to_datetime(d["date"])
        frames[name] = d

    # (a) subperiod stability
    sub_rows = []
    for name, d in frames.items():
        for period, mask in [("2010-2017", d["date"] < SPLIT),
                             ("2018-2025", d["date"] >= SPLIT)]:
            s = d.loc[mask]
            st = s[s["vix_pctl"] > 0.95]
            sub_rows.append({
                "method": name, "period": period,
                "marginal": s["covered"].mean(),
                "cov_stress": st["covered"].mean(),
                "cov_stress_upper": st["covered_hi"].mean(),
                "width": s["width"].mean(),
                "n": len(s), "n_stress": len(st)})
    sub = pd.DataFrame(sub_rows)
    print("=== E14a subperiod stability ===")
    print(sub.round(4).to_string(index=False))

    # paired stress-gap test per subperiod (rc_adaptive vs aci)
    gap_rows = []
    for period, lo, hi in [("2010-2017", None, SPLIT),
                           ("2018-2025", SPLIT, None)]:
        a = frames["aci"]
        r = frames["rc_adaptive"]
        am = (a["vix_pctl"] > 0.95)
        rm = (r["vix_pctl"] > 0.95)
        if lo is None:
            am &= a["date"] < hi
            rm &= r["date"] < hi
        else:
            am &= a["date"] >= lo
            rm &= r["date"] >= lo
        pair = r.loc[rm, ["ticker", "date", "covered"]].merge(
            a.loc[am, ["ticker", "date", "covered"]],
            on=["ticker", "date"], suffixes=("_rc", "_aci"))
        dm = dm_test(pair["covered_rc"].astype(float),
                     pair["covered_aci"].astype(float), pair["date"])
        gap_rows.append({"period": period, "stress_gap": dm["mean_diff"],
                         "t": dm["dm"], "p": dm["p"], "n": len(pair)})
    gaps = pd.DataFrame(gap_rows)
    print("\n=== E14a stress gap (rc_adaptive - aci), date-clustered ===")
    print(gaps.round(4).to_string(index=False))

    # (b) leave-one-crisis-out stress coverage
    loco_rows = []
    for name in ["aci", "rc_adaptive"]:
        d = frames[name]
        st = d[d["vix_pctl"] > 0.95]
        for yr in ["none"] + EPISODE_YEARS:
            s = st if yr == "none" else st[st["date"].dt.year != yr]
            loco_rows.append({"method": name, "dropped_year": yr,
                              "cov_stress": s["covered"].mean(),
                              "n_stress": len(s)})
    loco = pd.DataFrame(loco_rows)
    print("\n=== E14b leave-one-crisis-out stress coverage ===")
    print(loco.pivot(index="dropped_year", columns="method",
                     values="cov_stress").round(4).to_string())

    out = PROJECT_ROOT / "reports"
    sub.to_csv(out / "e14_subperiod.csv", index=False)
    gaps.to_csv(out / "e14_subperiod_gap.csv", index=False)
    loco.to_csv(out / "e14_loco.csv", index=False)
    print("\nsaved -> reports/e14_subperiod.csv, reports/e14_subperiod_gap.csv,"
          " reports/e14_loco.csv")


if __name__ == "__main__":
    main()
