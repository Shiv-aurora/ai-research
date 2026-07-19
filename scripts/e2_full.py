"""E2 full: the paper's main interval table + Model Confidence Set (E11).

Methods (all two-sided at alpha=0.10, walk-forward, common sample):
  aci / dtaci / sfogd   per-stock marginal online conformal baselines
  tcp_rm                per-stock rolling conformal + Robbins-Monro offset
                        (Aich et al. 2025, ported to our score scale)
  har_qreg              per-stock quantile regression on HAR lags (direct)
  knn_state             similarity-weighted conformal (continuous-state
                        rival to discrete regimes; HopCPT/NexCP spirit)
  xs_panel              cross-sectional split-conformal + adaptive level
                        (Tu-Giesecke 2026 spirit; panel pooling, no regimes)
  pogo                  parameter-free group-conditional coin-betting
                        (Bharti et al. 2026), per stock, our hard VIX bins
  cpid                  conformal PID control (Angelopoulos et al. 2023),
                        per stock: quantile tracking + saturated
                        integrator + trailing-quantile scorecaster
  rkr                   no-regret FTRL group-conditional (Ramalingam,
                        Kiyani & Roth 2025), per stock, overlapping
                        groups = marginal + our hard VIX bins
  pooled_k1             our machinery with K=1 (pooling without regimes)
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
from src.conformal.panel_xs import run_panel_xs
from src.conformal.pogo import run_pogo_panel
from src.conformal.pid import run_conformal_pid
from src.conformal.rkr import marginal_plus_bins, run_rkr
from src.conformal.similarity import run_knn_state_conformal
from src.conformal.tcp import run_tcp_rm
from src.eval.coverage import coverage_by_state, marginal_coverage
from src.eval.dm_hac import dm_test, hac_mean_se
from src.eval.mcs import interval_score, mcs
from src.forecasters.quantile_baselines import har_qreg
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.parallel import pmap
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


COLS = ["ticker", "date", "target", "lo", "hi", "warm"]


def _stock_worker(args):
    """Worker: one (method, name) per-stock stream in raw-unit bounds
    (module-level for spawn; the 400 jobs parallelize across cores)."""
    method, g = args
    g = g.dropna(subset=["target", "pool"]).sort_values("date").copy()
    if method == "har_qreg":
        if len(g) <= 800:
            return None
        q = har_qreg(g["target"].values, [ALPHA / 2, 1 - ALPHA / 2],
                     min_train=750)
        g["lo"], g["hi"] = q[ALPHA / 2], q[1 - ALPHA / 2]
        g["warm"] = np.isnan(g["lo"])
        return method, g[COLS]
    if len(g) <= WARMUP + 50:
        return None
    s = (g["target"] - g["pool"]).values
    res = {"aci": lambda: run_aci(s, alpha=ALPHA, eta=0.05, warmup=WARMUP),
           "dtaci": lambda: run_dtaci(s, alpha=ALPHA, warmup=WARMUP),
           "sfogd": lambda: run_sfogd(s, alpha=ALPHA, warmup=WARMUP),
           "tcp_rm": lambda: run_tcp_rm(s, alpha=ALPHA, warmup=WARMUP),
           "cpid": lambda: run_conformal_pid(s, alpha=ALPHA, warmup=WARMUP),
           "rkr": lambda: run_rkr(
               s, marginal_plus_bins(g["vix_pctl"].values), alpha=ALPHA,
               warmup=WARMUP),
           }[method]()
    g["lo"] = g["pool"].values - res["q_lo"].values
    g["hi"] = g["pool"].values + res["q_hi"].values
    g["warm"] = res["warmup"].values
    return method, g[COLS]


def _pogo_worker(args):
    """One POGO instance per stock (module-level for spawn)."""
    g, member = args
    g = g.dropna(subset=["target", "pool"]).sort_values("date")
    if len(g) <= WARMUP + 50:
        return None
    return run_pogo_panel(g, member, "pool", alpha=ALPHA,
                          warmup_days=WARMUP)


def panel_bounds(preds, member, **kw) -> pd.DataFrame:
    res = run_panel_mondrian(preds, member, "pool", alpha=ALPHA,
                             warmup_days=WARMUP, **kw)
    out = res[["ticker", "date", "target"]].copy()
    out["lo"] = res["pool"] - res["q_lo"] * res["sigma_hat"]
    out["hi"] = res["pool"] + res["q_hi"] * res["sigma_hat"]
    out["warm"] = res["warmup"]
    return out


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    proc = PROJECT_ROOT / cfg["data"]["processed_path"]
    panel = pd.read_parquet(proc / "rv_panel.parquet")
    preds = pd.read_parquet(proc / "e0_predictions.parquet")
    state = panel[["ticker", "date", "vix_pctl"]]
    market = panel.groupby("date")[["vix_pctl"]].first().sort_index()
    member = aligned_bins(market)

    print("[1/3] per-stock ACI / DtACI / SF-OGD / HAR-QREG across cores ...")
    preds = preds.merge(state, on=["ticker", "date"], how="left")
    jobs = [(m, g) for m in ["aci", "dtaci", "sfogd", "tcp_rm", "cpid",
                             "rkr", "har_qreg"]
            for _, g in preds.groupby("ticker")]
    bounds: dict[str, list | pd.DataFrame] = {}
    for r in pmap(_stock_worker, jobs):
        if r is not None:
            bounds.setdefault(r[0], []).append(r[1])
    bounds = {m: pd.concat(fs, ignore_index=True) for m, fs in bounds.items()}
    print("[2/3] KNN-state similarity conformal ...")
    ms = panel.groupby("date")[["vix_pctl", "mkt_rv_pctl",
                                "xs_dispersion"]].first().sort_index()
    knn = run_knn_state_conformal(preds, ms, "pool", alpha=ALPHA,
                                  k=250, warmup_days=WARMUP)
    knn_b = knn[["ticker", "date", "target"]].copy()
    knn_b["lo"] = knn["pool"] - knn["q_lo"] * knn["sigma_hat"]
    knn_b["hi"] = knn["pool"] + knn["q_hi"] * knn["sigma_hat"]
    knn_b["warm"] = knn["warmup"]
    bounds["knn_state"] = knn_b
    print("[3/3] rc_hand + rc_adaptive ...")
    bounds["rc_hand"] = panel_bounds(preds, member, eta_by_regime=ETAS_HAND)
    bounds["rc_adaptive"] = panel_bounds(preds, member, adaptive=True)
    # pooled marginal baseline (K=1): gives the per-stock conformal
    # baselines' closest pooled counterpart a seat in the MAIN table
    member1 = pd.DataFrame(1.0, index=member.index, columns=["regime_0"])
    bounds["pooled_k1"] = panel_bounds(preds, member1, adaptive=True)
    # cross-sectional split-conformal (Tu-Giesecke-style panel baseline)
    xs = run_panel_xs(preds, "pool", alpha=ALPHA, warmup_days=WARMUP)
    xs_b = xs[["ticker", "date", "target"]].copy()
    xs_b["lo"] = xs["pool"] - xs["q_lo"] * xs["sigma_hat"]
    xs_b["hi"] = xs["pool"] + xs["q_hi"] * xs["sigma_hat"]
    xs_b["warm"] = xs["warmup"]
    bounds["xs_panel"] = xs_b
    # parameter-free group-conditional (Bharti et al.), given OUR bins:
    # the closest published rival on the group-conditional axis. POGO is
    # a single-stream algorithm, so the faithful port runs one instance
    # per stock (its per-group guarantee then holds per stock-stream).
    pg_jobs = [(g, member) for _, g in preds.groupby("ticker")]
    pg_parts = [r for r in pmap(_pogo_worker, pg_jobs) if r is not None]
    pg = pd.concat(pg_parts, ignore_index=True)
    pg_b = pg[["ticker", "date", "target"]].copy()
    pg_b["lo"] = pg["pool"] - pg["q_lo"] * pg["sigma_hat"]
    pg_b["hi"] = pg["pool"] + pg["q_hi"] * pg["sigma_hat"]
    pg_b["warm"] = pg["warmup"]
    bounds["pogo"] = pg_b

    # common evaluation sample: (ticker, date) present & non-warm everywhere
    keys, flow = None, []
    flow.append(("full panel 2005-2025", len(panel)))
    flow.append(("with pool forecast and target",
                 int(preds.dropna(subset=["target", "pool"]).shape[0])))
    for name, b in bounds.items():
        k = b.loc[~b["warm"] & b["lo"].notna(), ["ticker", "date"]]
        flow.append((f"usable by {name}", len(k)))
        keys = k if keys is None else keys.merge(k, on=["ticker", "date"])
    flow.append(("common sample (intersection)", len(keys)))
    sample_flow = pd.DataFrame(flow, columns=["stage", "stock_days"])
    print(f"common sample: {len(keys):,} stock-days")
    print(sample_flow.to_string(index=False))

    rows, daily_is, cov_frames = [], {}, {}
    daily_cov_rows: list[pd.DataFrame] = []
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
        cov_frames[name] = d[["ticker", "date", "vix_pctl", "covered"]]
        daily_cov_rows.append(
            d.groupby("date")
            .agg(covered=("covered", "mean"),
                 covered_hi=("covered_hi", "mean"),
                 is_=("is_", "mean"), width=("width", "mean"),
                 n=("covered", "size"), vix_pctl=("vix_pctl", "first"))
            .assign(method=name).reset_index())

    summary = pd.DataFrame(rows).set_index("method")
    print("\n=== E2 main table (alpha=0.10, common sample) ===")
    print(summary.round(4).to_string())

    losses = pd.DataFrame(daily_is).dropna()
    m = mcs(losses, alpha=0.10, n_boot=2000, block=20, seed=cfg["seed"])
    print("\n=== MCS over daily interval scores ===")
    print(m.round(4).to_string())

    # date-clustered inference: HAC (Newey-West over daily means) SEs for
    # each method's coverage, plus paired coverage-difference tests of
    # rc_adaptive against every baseline (clustering removes the
    # cross-sectional dependence that raw stock-day counts would ignore)
    sig_rows = []
    ours = cov_frames["rc_adaptive"]
    for name, d in cov_frames.items():
        for slice_name, mask in [("marginal", d["vix_pctl"].notna()),
                                 ("stress", d["vix_pctl"] > 0.95)]:
            s = d.loc[mask]
            st = hac_mean_se(s["covered"].astype(float), s["date"])
            row = {"method": name, "slice": slice_name,
                   "coverage": st["mean"], "se": st["se"],
                   "n_dates": st["n_dates"]}
            if name != "rc_adaptive":
                pair = s.merge(ours, on=["ticker", "date"],
                               suffixes=("_b", "_a"))
                dm = dm_test(pair["covered_a"].astype(float),
                             pair["covered_b"].astype(float), pair["date"])
                row.update({"diff_vs_rc": dm["mean_diff"],
                            "t_vs_rc": dm["dm"], "p_vs_rc": dm["p"]})
            sig_rows.append(row)
    sig = pd.DataFrame(sig_rows)
    print("\n=== Date-clustered coverage inference ===")
    print(sig.round(4).to_string(index=False))

    # per-stock coverage distribution (pooling is panel-level; show the
    # cross-sectional dispersion a pooled threshold leaves behind)
    ps_rows = []
    for name in ["aci", "rc_adaptive"]:
        d = cov_frames[name]
        for slice_name, sub in [("marginal", d),
                                ("stress", d[d["vix_pctl"] > 0.95])]:
            per = sub.groupby("ticker")["covered"].agg(["mean", "size"])
            per = per[per["size"] >= 25]     # need enough obs per stock
            qs = per["mean"].quantile([0.05, 0.10, 0.25, 0.50, 0.75])
            ps_rows.append({
                "method": name, "slice": slice_name,
                "n_stocks": len(per),
                "q05": qs[0.05], "q10": qs[0.10], "q25": qs[0.25],
                "median": qs[0.50], "q75": qs[0.75],
                "share_below_85": float((per["mean"] < 0.85).mean()),
            })
    per_stock = pd.DataFrame(ps_rows)
    print("\n=== Per-stock coverage distribution ===")
    print(per_stock.round(4).to_string(index=False))

    out = PROJECT_ROOT / "reports"
    summary.to_csv(out / "e2_full_summary.csv")
    m.to_csv(out / "e2_full_mcs.csv")
    sig.to_csv(out / "e2_clustered_se.csv", index=False)
    sample_flow.to_csv(out / "e2_sample_flow.csv", index=False)
    per_stock.to_csv(out / "e2_per_stock_coverage.csv", index=False)
    # daily cross-sectional coverage means per method (common sample):
    # input for episode-block bootstrap / HAC-lag sensitivity (e16)
    pd.concat(daily_cov_rows, ignore_index=True).to_parquet(
        out / "e2_daily_coverage.parquet", index=False)
    print("\nsaved -> reports/e2_full_summary.csv, reports/e2_full_mcs.csv,"
          " reports/e2_clustered_se.csv, reports/e2_sample_flow.csv,"
          " reports/e2_per_stock_coverage.csv")


if __name__ == "__main__":
    main()
