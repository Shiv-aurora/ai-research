"""E18: uncertainty and identification for the day-2 onset finding.

The review's fair complaint: the onset limitation was stated without
sampling uncertainty and without showing what widening at onset would
cost. This experiment adds, for the headline RC K=4 run:

  (a) the number of independent market-level stress entries, with dates;
  (b) coverage by days-since-entry bucket with HAC (date-clustered) SEs
      and 95% CIs;
  (c) leave-one-entry-out day-2 coverage (drop each market-level stress
      episode in turn): range across episodes;
  (d) day 1-5 trajectories of the issued upper threshold, realized score,
      and upper-miss rate (does the tracker even move at onset?);
  (e) an onset-widening overlay Pareto: multiply the issued interval by
      m on days following a fresh market stress entry (previous day's
      days-in-stress 1-3, strictly causal, no feedback into the
      calibrator) for m in {1.0, 1.25, 1.5, 2.0}; report day-2 coverage
      against stress-interior interval score and width.

Usage: .venv/bin/python scripts/e18_onset_uncertainty.py
Output: reports/e18_entries.csv, reports/e18_dse_ci.csv,
        reports/e18_loo_day2.csv, reports/e18_trajectories.csv,
        reports/e18_pareto.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.e6b_oracle_regimes import bin_membership
from src.conformal.panel_hierarchical import run_panel_mondrian
from src.eval.dm_hac import hac_mean_se
from src.eval.mcs import interval_score
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeding import seed_everything

FORECAST = "pool"
ALPHA = 0.10
CUTS = [0.5, 0.8, 0.95]
BINS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"),
        (6, 10, "6-10"), (11, np.inf, ">10")]
MULTS = [1.0, 1.25, 1.5, 2.0]


def market_dse(market: pd.DataFrame) -> pd.Series:
    inside = (market["vix_pctl"] > 0.95).fillna(False).values
    run, c = np.zeros(len(market)), 0
    for i, flag in enumerate(inside):
        c = c + 1 if flag else 0
        run[i] = c
    return pd.Series(run, index=market.index, name="mkt_dse")


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    proc = PROJECT_ROOT / cfg["data"]["processed_path"]
    panel = pd.read_parquet(proc / "rv_panel.parquet")
    preds = pd.read_parquet(proc / "e0_predictions.parquet")
    state = panel[["ticker", "date", "vix_pctl"]]
    market = panel.groupby("date")[["vix_pctl"]].first().sort_index()
    member = bin_membership(market, CUTS)

    res = run_panel_mondrian(preds, member, FORECAST, alpha=ALPHA,
                             adaptive=True, warmup_days=100)
    res = res.merge(state, on=["ticker", "date"], how="left")
    res = res[~res.warmup].copy()

    # market-level entry bookkeeping (stock-day dse follows the MARKET
    # regime, identical across stocks on a day)
    mdse = market_dse(market)
    res = res.merge(mdse.rename("dse"), left_on="date", right_index=True,
                    how="left")
    entry_dates = mdse.index[(mdse == 1).values]
    # group entries into episodes: a new entry within 20 trading days of
    # the previous episode's last stress day is a re-entry, not a new one
    stress_days = mdse.index[(mdse > 0).values]
    ep_id = {}
    eid, last = -1, None
    all_days = market.index
    for d in stress_days:
        if last is None or (all_days.get_loc(d) - all_days.get_loc(last)
                            > 20):
            eid += 1
        ep_id[d] = eid
        last = d
    res["episode"] = res["date"].map(ep_id)
    n_entries = len(entry_dates)
    n_episodes = eid + 1
    entries = pd.DataFrame({
        "entry_date": entry_dates,
        "episode": [ep_id[d] for d in entry_dates]})
    print(f"independent stress entries (dse hits 1): {n_entries}; "
          f"distinct episodes (>20d apart): {n_episodes}")

    # (b) dse-bucket coverage with HAC CIs
    d = res[res["dse"] > 0]
    ci_rows = []
    for lo_b, hi_b, lab in BINS:
        s = d[(d["dse"] >= lo_b) & (d["dse"] <= hi_b)]
        st = hac_mean_se(s["covered"].astype(float), s["date"])
        ci_rows.append({"dse": lab, "n": len(s), "n_dates": st["n_dates"],
                        "coverage": st["mean"], "se": st["se"],
                        "ci_lo": st["mean"] - 1.96 * st["se"],
                        "ci_hi": st["mean"] + 1.96 * st["se"]})
    dse_ci = pd.DataFrame(ci_rows)
    print("\n=== coverage by days-since-entry, HAC CIs ===")
    print(dse_ci.round(4).to_string(index=False))

    # (c) leave-one-episode-out day-2 coverage
    day2 = d[d["dse"] == 2]
    loo_rows = [{"dropped": "none", "cov_day2": day2["covered"].mean(),
                 "n": len(day2)}]
    for e in sorted(day2["episode"].dropna().unique()):
        s = day2[day2["episode"] != e]
        loo_rows.append({"dropped": int(e), "cov_day2": s["covered"].mean(),
                         "n": len(s)})
    loo = pd.DataFrame(loo_rows)
    print("\n=== leave-one-episode-out day-2 coverage ===")
    print(loo.round(4).to_string(index=False))

    # (d) onset trajectories
    traj = (d[d["dse"] <= 5].groupby("dse")
            .agg(n=("covered", "size"),
                 q_hi=("q_hi", "mean"),
                 score=("s_std", "mean"),
                 miss_hi=("covered_hi", lambda x: 1 - x.mean()),
                 coverage=("covered", "mean"))
            .reset_index())
    print("\n=== day 1-5 trajectories (issued upper threshold vs realized "
          "score) ===")
    print(traj.round(4).to_string(index=False))

    # (e) onset-widening overlay Pareto (strictly causal: uses previous
    # day's market days-in-stress)
    prev_dse = mdse.shift(1)
    res["prev_dse"] = res["date"].map(prev_dse)
    onset_flag = res["prev_dse"].between(1, 3)
    stress_interior = res["dse"] > 3
    pareto_rows = []
    for m in MULTS:
        f = np.where(onset_flag, m, 1.0)
        lo_raw = res[FORECAST] - res["q_lo"] * res["sigma_hat"] * f
        hi_raw = res[FORECAST] + res["q_hi"] * res["sigma_hat"] * f
        cov = (res["target"] >= lo_raw) & (res["target"] <= hi_raw)
        is_ = interval_score(res["target"].values, lo_raw.values,
                             hi_raw.values, ALPHA)
        d2 = res["dse"] == 2
        stress_all = res["dse"] > 0
        pareto_rows.append({
            "mult": m,
            "cov_day2": cov[d2].mean(),
            "cov_stress": cov[stress_all].mean(),
            "is_stress_interior": is_[stress_interior.values].mean(),
            "is_stress": is_[stress_all.values].mean(),
            "width_onset": (hi_raw - lo_raw)[onset_flag].mean(),
            "marginal": cov.mean()})
    pareto = pd.DataFrame(pareto_rows)
    print("\n=== onset-widening overlay: day-2 coverage vs stress "
          "interval score ===")
    print(pareto.round(4).to_string(index=False))

    rep = PROJECT_ROOT / "reports"
    entries.to_csv(rep / "e18_entries.csv", index=False)
    dse_ci.to_csv(rep / "e18_dse_ci.csv", index=False)
    loo.to_csv(rep / "e18_loo_day2.csv", index=False)
    traj.to_csv(rep / "e18_trajectories.csv", index=False)
    pareto.to_csv(rep / "e18_pareto.csv", index=False)
    print("\nsaved -> reports/e18_*.csv")


if __name__ == "__main__":
    main()
