"""E2 round 3: transition-augmented regime groups.

Round 2 isolated the residual failure to spike ONSET (day 1-2 of a stress
spell: 79-88% coverage vs ~92% mature). 'Entered the top slice within the
last d days' is observable at issuance time, so transition freshness is a
legitimate Mondrian group: split stress into fresh (dse <= 2) and mature,
and likewise give the elevated regime a fresh subgroup.

Groups (hard, from market-level vix_pctl, cutpoints .5/.8/.95):
  0 calm | 1 normal | 2 elevated | 3 stress-fresh (dse<=2) | 4 stress-mature

Usage: .venv/bin/python scripts/e2_transition.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal.panel_hierarchical import run_panel_mondrian
from src.eval.coverage import coverage_by_state, marginal_coverage
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeding import seed_everything

FORECAST = "pool"
CUTS = [0.5, 0.8, 0.95]
FRESH_DAYS = 5
# per-observation etas: calm..elevated moderate; fresh-stress FAST (short
# spells, must move immediately); mature-stress moderate
ETAS = [0.001, 0.0015, 0.003, 0.02, 0.01]


def transition_membership_frame(market: pd.DataFrame) -> pd.DataFrame:
    v = market["vix_pctl"].values
    k = np.searchsorted(CUTS, v, side="right")           # 0..3
    dse = np.zeros(len(v))
    c = 0
    for i in range(len(v)):
        c = c + 1 if (not np.isnan(v[i]) and v[i] > CUTS[-1]) else 0
        dse[i] = c
    groups = k.copy()
    groups[(k == 3) & (dse > FRESH_DAYS)] = 4            # stress-mature
    # k==3 & dse<=FRESH_DAYS stays 3 = stress-fresh
    K = 5
    pi = np.zeros((len(v), K))
    ok = ~np.isnan(v)
    pi[ok, groups[ok]] = 1.0
    pi[~ok] = 1.0 / K
    return pd.DataFrame(pi, index=market.index,
                        columns=[f"regime_{j}" for j in range(K)])


def dse_profile(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    out = []
    for _, g in df.sort_values(["ticker", "date"]).groupby("ticker"):
        inside = (g["vix_pctl"] > threshold).values
        run, c = np.zeros(len(g)), 0
        for i, flag in enumerate(inside):
            c = c + 1 if flag else 0
            run[i] = c
        out.append(pd.Series(run, index=g.index))
    dse = pd.concat(out)
    d = df.loc[~df["warmup"]].copy()
    d["dse"] = dse.reindex(d.index).replace(0, np.nan)
    d = d.dropna(subset=["dse"])
    bins = pd.cut(d["dse"], [0, 1, 2, 3, 5, 10, 1000],
                  labels=["1", "2", "3", "4-5", "6-10", ">10"])
    return d.groupby(bins, observed=True)["covered"].agg(["mean", "size"])


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    alpha = cfg["evaluation"]["alpha_two_sided"]

    panel = pd.read_parquet(PROJECT_ROOT / cfg["data"]["processed_path"] / "rv_panel.parquet")
    preds = pd.read_parquet(PROJECT_ROOT / cfg["data"]["processed_path"] / "e0_predictions.parquet")
    state = panel[["ticker", "date", "vix_pctl"]]
    market = panel.groupby("date")[["vix_pctl"]].first().sort_index()

    member = transition_membership_frame(market)
    print("group day-counts:", member.sum().round(0).to_dict())

    res = run_panel_mondrian(preds, member, FORECAST, alpha=alpha,
                             eta_by_regime=ETAS, warmup_days=100)
    res = res.merge(state, on=["ticker", "date"], how="left")

    print(f"\n=== E2 round 3: transition-augmented groups (alpha={alpha}) ===")
    by = coverage_by_state(res, "vix_pctl")
    print(f"marginal={marginal_coverage(res):.4f}  "
          f"width={res.loc[~res.warmup, 'width'].mean():.4f}")
    print(by[["coverage", "upper_coverage", "n"]].round(4).to_string())

    print("\ndays-since-entry profile:")
    print(dse_profile(res).round(4).to_string())

    out = PROJECT_ROOT / "reports"
    by.to_csv(out / "e2_transition_by_state.csv")
    dse_profile(res).to_csv(out / "e2_transition_dse.csv")
    print(f"\nsaved -> reports/e2_transition_*.csv")


if __name__ == "__main__":
    main()
