"""E6b: oracle vs estimated regimes in the calibrator (queue item B).

The coverage propositions condition on the algorithm's own filtered state,
not the latent truth. This ablation quantifies what that costs: run the
identical pooled adaptive calibrator under three membership sources —

  bins_k4          aligned hard VIX-percentile bins (canonical, causal)
  hmm_filtered     online Gaussian HMM, expanding refits, FILTERED probs
                   (causal — what the paper's method actually uses)
  hmm_oracle       ONE Gaussian HMM fit on the FULL sample, SMOOTHED probs
                   P(state_t | all data). LEAKY BY DESIGN: this is the
                   infeasible upper bound of perfect regime knowledge.

If oracle ~ filtered, regime-estimation error is not the binding constraint
(and in particular cannot explain the day-2 transition pit). Reports
coverage by VIX state plus the days-since-stress-entry profile.

Usage: .venv/bin/python scripts/e6b_oracle_regimes.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal.panel_hierarchical import run_panel_mondrian
from src.eval.coverage import coverage_by_state, marginal_coverage
from src.regimes.online_hmm import online_hmm_memberships
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.parallel import pmap
from src.utils.seeding import seed_everything

FORECAST = "pool"
HMM_FEATURES = ["vix_pctl", "mkt_rv_pctl", "xs_dispersion"]
N_REGIMES = 3


def oracle_smoothed_memberships(market: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Full-sample smoothed HMM posteriors — deliberately leaky (oracle)."""
    from hmmlearn.hmm import GaussianHMM

    X = market[HMM_FEATURES].values
    ok = ~np.isnan(X).any(axis=1)
    X_imp = np.where(ok[:, None], X, np.nanmean(X, axis=0))
    m = GaussianHMM(n_components=N_REGIMES, covariance_type="diag",
                    n_iter=200, random_state=seed)
    m.fit(X_imp[ok])
    probs = m.predict_proba(X_imp)          # smoothed: conditions on ALL data
    order = np.argsort(m.means_[:, 0])      # regime 0 = calm ... K-1 = stress
    return pd.DataFrame(probs[:, order], index=market.index,
                        columns=[f"regime_{k}" for k in range(N_REGIMES)])


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


def dse_coverage(res: pd.DataFrame, threshold: float = 0.95) -> pd.Series:
    """Coverage by days-since-entry into the stress regime (transition axis)."""
    d = res[~res.warmup].sort_values(["ticker", "date"])
    runs = []
    for _, g in d.groupby("ticker"):
        inside = (g["vix_pctl"] > threshold).values
        run, c = np.zeros(len(g)), 0
        for i, flag in enumerate(inside):
            c = c + 1 if flag else 0
            run[i] = c
        runs.append(pd.Series(run, index=g.index))
    d["dse"] = pd.concat(runs)
    d = d[d.dse > 0]
    d["dse_bin"] = pd.cut(d.dse, [0, 1, 2, 3, 5, 10, np.inf],
                          labels=["1", "2", "3", "4-5", "6-10", ">10"])
    return d.groupby("dse_bin", observed=True)["covered"].mean()


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

    market = (panel.groupby("date")[HMM_FEATURES].first().sort_index())

    print("[memberships] filtered (online) + oracle (smoothed) HMM ...", flush=True)
    filtered = online_hmm_memberships(market, HMM_FEATURES,
                                      n_regimes=N_REGIMES, seed=cfg["seed"])
    oracle = oracle_smoothed_memberships(market, cfg["seed"])
    bins4 = bin_membership(market, [0.5, 0.8, 0.95])

    # Diagnostic: how different are the memberships where it matters?
    stress_days = market["vix_pctl"] > 0.95
    dis = (filtered.values[:, -1] - oracle.values[:, -1])
    print(f"filtered-vs-oracle stress-prob MAD: all days "
          f"{np.nanmean(np.abs(dis)):.3f}, stress days "
          f"{np.nanmean(np.abs(dis[stress_days.values])):.3f}", flush=True)

    jobs = [("bins_k4", preds, bins4),
            ("hmm_filtered", preds, filtered),
            ("hmm_oracle", preds, oracle)]
    rows, dse_rows = [], {}
    for label, res in pmap(_run, jobs):
        res = res.merge(state, on=["ticker", "date"], how="left")
        by = coverage_by_state(res, "vix_pctl")
        d = res[~res.warmup]
        rows.append({
            "membership": label,
            "marginal": marginal_coverage(res),
            "cov_calm": by.loc["calm", "coverage"],
            "cov_stress": by.loc["stress", "coverage"],
            "cov_stress_upper": by.loc["stress", "upper_coverage"],
            "width": d["width"].mean(),
            "width_stress": d.loc[d.vix_pctl > 0.95, "width"].mean(),
        })
        dse_rows[label] = dse_coverage(res)
        print(pd.Series(rows[-1]).drop("membership").rename(label)
              .astype(float).round(4).to_string(), "\n", flush=True)

    out = pd.DataFrame(rows).set_index("membership")
    dse = pd.DataFrame(dse_rows)
    out.to_csv(PROJECT_ROOT / "reports" / "e6b_oracle_regimes.csv")
    dse.to_csv(PROJECT_ROOT / "reports" / "e6b_oracle_dse.csv")
    print("days-since-stress-entry coverage:")
    print(dse.round(4).to_string())
    print("saved -> reports/e6b_oracle_regimes.csv, reports/e6b_oracle_dse.csv",
          flush=True)


if __name__ == "__main__":
    main()
