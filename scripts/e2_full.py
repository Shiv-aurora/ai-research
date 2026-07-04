"""E2 full: the paper's main interval table + Model Confidence Set (E11).

Methods (all two-sided at alpha=0.10, walk-forward, common sample):
  aci / dtaci / sfogd   per-stock marginal online conformal baselines
  har_qreg              per-stock quantile regression on HAR lags (direct)
  knn_state             similarity-weighted conformal (continuous-state
                        rival to discrete regimes; HopCPT/NexCP spirit)
  rc_hand               pooled regime-conditional, hand-tuned rates
  rc_adaptive           pooled regime-conditional, adaptive rates (ours)

Outputs: per-regime coverage/width table; MCS over daily cross-sectional
mean interval scores (Winkler, alpha=0.10). All intervals converted to raw
log-RV units before scoring.

Usage: .venv/bin/python scripts/e2_full.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal.aci import run_aci
from src.conformal.dtaci import run_dtaci
from src.conformal.panel_hierarchical import run_panel_mondrian
from src.conformal.sfogd import run_sfogd
from src.conformal.similarity import run_knn_state_conformal
from src.eval.coverage import coverage_by_state, marginal_coverage
from src.eval.mcs import interval_score, mcs
from src.forecasters.quantile_baselines import har_qreg
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeding import seed_everything

ALPHA = 0.10
WARMUP = 100
CUTS = [0.5, 0.8, 0.95]
ETAS_HAND = [0.001, 0.0015, 0.003, 0.01]


def aligned_bins(market: pd.DataFrame) -> pd.DataFrame:
    v = market["vix_pctl"].values
    k = np.searchsorted(CUTS, v, side="right")
    pi = np.zeros((len(v), 4))
    ok = ~np.isnan(v)
    pi[ok, k[ok]] = 1.0
    pi[~ok] = 0.25
    return pd.DataFrame(pi, index=market.index,
                        columns=[f"regime_{j}" for j in range(4)])


def per_stock_bounds(preds: pd.DataFrame, method_fn) -> pd.DataFrame:
    """Run a per-stock stream on raw residuals; return raw-unit bounds."""
    frames = []
    for _, g in preds.sort_values("date").groupby("ticker"):
        g = g.dropna(subset=["target", "pool"]).copy()
        if len(g) <= WARMUP + 50:
            continue
        res = method_fn((g["target"] - g["pool"]).values)
        g["lo"] = g["pool"].values - res["q_lo"].values
        g["hi"] = g["pool"].values + res["q_hi"].values
        g["warm"] = res["warmup"].values
        frames.append(g[["ticker", "date", "target", "lo", "hi", "warm"]])
    return pd.concat(frames, ignore_index=True)


def panel_bounds(preds, member, **kw) -> pd.DataFrame:
    res = run_panel_mondrian(preds, member, "pool", alpha=ALPHA,
                             warmup_days=WARMUP, **kw)
    out = res[["ticker", "date", "target"]].copy()
    out["lo"] = res["pool"] - res["q_lo"] * res["sigma_hat"]
    out["hi"] = res["pool"] + res["q_hi"] * res["sigma_hat"]
    out["warm"] = res["warmup"]
    return out


def qreg_bounds(preds: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, g in preds.sort_values("date").groupby("ticker"):
        g = g.dropna(subset=["target"]).copy()
        if len(g) <= 800:
            continue
        q = har_qreg(g["target"].values, [ALPHA / 2, 1 - ALPHA / 2],
                     min_train=750)
        g["lo"], g["hi"] = q[ALPHA / 2], q[1 - ALPHA / 2]
        g["warm"] = np.isnan(g["lo"])
        frames.append(g[["ticker", "date", "target", "lo", "hi", "warm"]])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    proc = PROJECT_ROOT / cfg["data"]["processed_path"]
    panel = pd.read_parquet(proc / "rv_panel.parquet")
    preds = pd.read_parquet(proc / "e0_predictions.parquet")
    state = panel[["ticker", "date", "vix_pctl"]]
    market = panel.groupby("date")[["vix_pctl"]].first().sort_index()
    member = aligned_bins(market)

    print("[1/6] per-stock ACI / DtACI / SF-OGD ...")
    bounds = {
        "aci": per_stock_bounds(preds, lambda s: run_aci(
            s, alpha=ALPHA, eta=0.05, warmup=WARMUP)),
        "dtaci": per_stock_bounds(preds, lambda s: run_dtaci(
            s, alpha=ALPHA, warmup=WARMUP)),
        "sfogd": per_stock_bounds(preds, lambda s: run_sfogd(
            s, alpha=ALPHA, warmup=WARMUP)),
    }
    print("[4/7] HAR-QREG ...")
    bounds["har_qreg"] = qreg_bounds(preds)
    print("[5/7] KNN-state similarity conformal ...")
    ms = panel.groupby("date")[["vix_pctl", "mkt_rv_pctl",
                                "xs_dispersion"]].first().sort_index()
    knn = run_knn_state_conformal(preds, ms, "pool", alpha=ALPHA,
                                  k=250, warmup_days=WARMUP)
    knn_b = knn[["ticker", "date", "target"]].copy()
    knn_b["lo"] = knn["pool"] - knn["q_lo"] * knn["sigma_hat"]
    knn_b["hi"] = knn["pool"] + knn["q_hi"] * knn["sigma_hat"]
    knn_b["warm"] = knn["warmup"]
    bounds["knn_state"] = knn_b
    print("[6/7] rc_hand ...")
    bounds["rc_hand"] = panel_bounds(preds, member, eta_by_regime=ETAS_HAND)
    print("[7/7] rc_adaptive ...")
    bounds["rc_adaptive"] = panel_bounds(preds, member, adaptive=True)

    # common evaluation sample: (ticker, date) present & non-warm everywhere
    keys = None
    for name, b in bounds.items():
        k = b.loc[~b["warm"] & b["lo"].notna(), ["ticker", "date"]]
        keys = k if keys is None else keys.merge(k, on=["ticker", "date"])
    print(f"common sample: {len(keys):,} stock-days")

    rows, daily_is = [], {}
    for name, b in bounds.items():
        d = b.merge(keys, on=["ticker", "date"])
        d = d.merge(state, on=["ticker", "date"], how="left")
        d["covered"] = (d.target >= d.lo) & (d.target <= d.hi)
        d["covered_lo"] = d.target >= d.lo
        d["covered_hi"] = d.target <= d.hi
        d["warmup"] = False
        d["width"] = d.hi - d.lo
        by = coverage_by_state(d, "vix_pctl")
        rows.append({
            "method": name, "marginal": marginal_coverage(d),
            "cov_calm": by.loc["calm", "coverage"],
            "cov_normal": by.loc["normal", "coverage"],
            "cov_elevated": by.loc["elevated", "coverage"],
            "cov_stress": by.loc["stress", "coverage"],
            "cov_stress_upper": by.loc["stress", "upper_coverage"],
            "width": d.width.mean(),
            "width_stress": d.loc[d.vix_pctl > 0.95, "width"].mean(),
        })
        d["is_"] = interval_score(d.target.values, d.lo.values, d.hi.values,
                                  ALPHA)
        daily_is[name] = d.groupby("date")["is_"].mean()

    summary = pd.DataFrame(rows).set_index("method")
    print("\n=== E2 main table (alpha=0.10, common sample) ===")
    print(summary.round(4).to_string())

    losses = pd.DataFrame(daily_is).dropna()
    m = mcs(losses, alpha=0.10, n_boot=2000, block=20, seed=cfg["seed"])
    print("\n=== MCS over daily interval scores ===")
    print(m.round(4).to_string())

    out = PROJECT_ROOT / "reports"
    summary.to_csv(out / "e2_full_summary.csv")
    m.to_csv(out / "e2_full_mcs.csv")
    print(f"\nsaved -> reports/e2_full_summary.csv, reports/e2_full_mcs.csv")


if __name__ == "__main__":
    main()
