"""E2 prototype: soft-Mondrian regime-conditional conformal vs marginal
baselines (ACI, DtACI, SF-OGD) on the real panel.

Uses the pool forecasts from E0 (data/processed_v2/e0_predictions.parquet).
Regimes: online Gaussian HMM (soft, filtered) and quantile bins (hard).
Reports marginal + per-VIX-regime coverage and width for every method.

Usage: .venv/bin/python scripts/e2_prototype.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal.aci import run_aci
from src.conformal.dtaci import run_dtaci
from src.conformal.mondrian_soft import run_soft_mondrian
from src.conformal.sfogd import run_sfogd
from src.eval.coverage import coverage_by_state, marginal_coverage
from src.regimes.online_hmm import online_hmm_memberships
from src.regimes.quantile_bins import quantile_bin_memberships
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeding import seed_everything

WARMUP = 100
FORECAST = "pool"
HMM_FEATURES = ["vix_pctl", "mkt_rv_pctl", "xs_dispersion"]


def per_stock_streams(preds, panel, member, alpha, method_fn, name):
    """Run a per-stock conformal stream with per-date memberships."""
    frames = []
    for ticker, g in preds.sort_values("date").groupby("ticker"):
        g = g.dropna(subset=["target", FORECAST]).copy()
        if len(g) <= WARMUP + 50:
            continue
        scores = (g["target"] - g[FORECAST]).values
        if member is None:
            res = method_fn(scores)
        else:
            pi = member.reindex(g["date"]).values
            pi = np.where(np.isnan(pi), 1.0 / member.shape[1], pi)
            pi = pi / pi.sum(axis=1, keepdims=True)
            res = method_fn(scores, pi)
        res.index = g.index
        frames.append(pd.concat([g, res], axis=1))
    out = pd.concat(frames)
    out["method"] = name
    out["width"] = out["q_lo"] + out["q_hi"]
    return out


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    alpha = cfg["evaluation"]["alpha_two_sided"]
    K = cfg["regimes"]["n_regimes"]

    panel = pd.read_parquet(PROJECT_ROOT / cfg["data"]["processed_path"] / "rv_panel.parquet")
    preds = pd.read_parquet(PROJECT_ROOT / cfg["data"]["processed_path"] / "e0_predictions.parquet")

    market = (panel.groupby("date")[["vix_pctl", "mkt_rv_pctl", "xs_dispersion"]]
              .first().sort_index())
    print("[regimes] online HMM (filtered) and quantile bins...")
    hmm = online_hmm_memberships(market, HMM_FEATURES, n_regimes=K,
                                 min_train=750, seed=cfg["seed"])
    bins_ = quantile_bin_memberships(market, n_regimes=K)

    # per-regime learning rates: calm small -> stress fast
    etas = list(np.geomspace(0.02, 0.10, K))

    methods = {
        "aci": lambda s: run_aci(s, alpha=alpha, eta=0.05, warmup=WARMUP),
        "dtaci": lambda s: run_dtaci(s, alpha=alpha, warmup=WARMUP),
        "sfogd": lambda s: run_sfogd(s, alpha=alpha, warmup=WARMUP),
    }

    results = []
    for name, fn in methods.items():
        print(f"[baseline] {name}...")
        results.append(per_stock_streams(preds, panel, None, alpha, fn, name))
    print("[method] soft-Mondrian x online HMM...")
    results.append(per_stock_streams(
        preds, panel, hmm, alpha,
        lambda s, pi: run_soft_mondrian(s, pi, alpha=alpha,
                                        eta_by_regime=etas, warmup=WARMUP),
        "rc_hmm"))
    print("[method] Mondrian x quantile bins (hard)...")
    results.append(per_stock_streams(
        preds, panel, bins_, alpha,
        lambda s, pi: run_soft_mondrian(s, pi, alpha=alpha,
                                        eta_by_regime=etas, warmup=WARMUP),
        "rc_bins"))

    state = panel[["ticker", "date", "vix_pctl"]]
    print(f"\n=== E2 prototype (alpha={alpha}, target {1-alpha:.0%}) ===")
    summary_rows = []
    for res in results:
        res = res.merge(state, on=["ticker", "date"], how="left")
        name = res["method"].iloc[0]
        m = marginal_coverage(res)
        by = coverage_by_state(res, "vix_pctl")
        row = {"method": name, "marginal": m,
               "width": res.loc[~res.warmup, "width"].mean()}
        for b in by.index:
            row[f"cov_{b}"] = by.loc[b, "coverage"]
            row[f"width_{b}"] = by.loc[b, "mean_width"]
        row["cov_stress_upper"] = by.loc["stress", "upper_coverage"]
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).set_index("method")
    cols = ["marginal", "cov_calm", "cov_normal", "cov_elevated", "cov_stress",
            "cov_stress_upper", "width", "width_calm", "width_stress"]
    print(summary[cols].round(4).to_string())

    out = PROJECT_ROOT / "reports"
    summary.to_csv(out / "e2_prototype_summary.csv")
    print(f"\nsaved -> {out / 'e2_prototype_summary.csv'}")


if __name__ == "__main__":
    main()
