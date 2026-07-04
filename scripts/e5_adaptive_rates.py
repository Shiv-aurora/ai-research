"""E5: adaptive per-regime learning rates vs hand-tuned (the last knob).

Every headline run so far used hand-tuned per-regime rates. The referee
question is obvious: how were they chosen, and would results survive without
that freedom? This experiment replaces them with DtACI-style per-regime rate
aggregation over a fixed dyadic grid (no tuning anywhere) and re-runs

  (a) the E2 two-sided panel config (aligned K=4 bins, alpha=0.10)
  (b) the E3 one-sided VaR heads on standardized returns (alpha 0.05, 0.01)

reporting coverage/exceedance by regime plus the effective rate the
aggregator selects per regime — it should rediscover both hand-tuned
findings: faster rates in stress, larger steps for more extreme quantiles.

Usage: .venv/bin/python scripts/e5_adaptive_rates.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal.panel_hierarchical import run_panel_mondrian
from src.data.ohlc import load_returns
from src.data.universe import get_universe
from src.eval.coverage import coverage_by_state, marginal_coverage
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeding import seed_everything

CUTS = [0.5, 0.8, 0.95]
ETAS_BINS = [0.001, 0.0015, 0.003, 0.01]          # E2 hand-tuned
ETAS_VAR = {0.05: [0.001, 0.0015, 0.003, 0.01],   # E3 hand-tuned
            0.01: [0.002, 0.003, 0.006, 0.02]}
REGIME_NAMES = ["calm", "normal", "elevated", "stress"]


def aligned_bins(market: pd.DataFrame) -> pd.DataFrame:
    v = market["vix_pctl"].values
    k = np.searchsorted(CUTS, v, side="right")
    pi = np.zeros((len(v), 4))
    ok = ~np.isnan(v)
    pi[ok, k[ok]] = 1.0
    pi[~ok] = 0.25
    return pd.DataFrame(pi, index=market.index,
                        columns=[f"regime_{j}" for j in range(4)])


def eff_eta_report(res: pd.DataFrame, label: str) -> None:
    diag = res.attrs.get("adaptive")
    if diag is None:
        return
    eff = np.asarray(diag["eff_eta_hi"])
    late = eff[len(eff) // 2:].mean(axis=0)
    print(f"  {label} effective eta_hi (late-sample mean): "
          + "  ".join(f"{n}={v:.4f}" for n, v in zip(REGIME_NAMES, late)))


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    proc = PROJECT_ROOT / cfg["data"]["processed_path"]
    panel = pd.read_parquet(proc / "rv_panel.parquet")
    preds = pd.read_parquet(proc / "e0_predictions.parquet")
    state = panel[["ticker", "date", "vix_pctl"]]
    market = panel.groupby("date")[["vix_pctl"]].first().sort_index()
    member = aligned_bins(market)

    rows = []

    # ---------- (a) two-sided E2 config ----------
    print("=== (a) two-sided, alpha=0.10, aligned K=4 bins ===")
    for name, kw in [("hand_tuned", dict(eta_by_regime=ETAS_BINS)),
                     ("adaptive", dict(adaptive=True))]:
        res = run_panel_mondrian(preds, member, "pool", alpha=0.10,
                                 warmup_days=100, **kw)
        diag = res.attrs.get("adaptive")          # merge() drops attrs
        res = res.merge(state, on=["ticker", "date"], how="left")
        res.attrs["adaptive"] = diag
        by = coverage_by_state(res, "vix_pctl")
        d = res[~res.warmup]
        row = {"experiment": "two_sided_0.10", "method": name,
               "marginal": marginal_coverage(res),
               "width": d["width"].mean(),
               "width_stress": d.loc[d.vix_pctl > 0.95, "width"].mean(),
               "cov_stress_upper": by.loc["stress", "upper_coverage"]}
        for b in by.index:
            row[f"cov_{b}"] = by.loc[b, "coverage"]
        rows.append(row)
        print(pd.Series(row).drop(["experiment", "method"])
              .rename(name).round(4).to_string())
        eff_eta_report(res, name)

    # ---------- (b) one-sided VaR heads ----------
    rets = load_returns(get_universe(cfg), cfg["data"]["start_date"],
                        "2026-01-01", verbose=False)
    rets = rets.sort_values(["ticker", "date"])
    rets["pred_date"] = rets.groupby("ticker")["date"].shift(1)
    nxt = rets.dropna(subset=["pred_date"])[["ticker", "pred_date", "ret"]]
    nxt = nxt.rename(columns={"pred_date": "date", "ret": "ret_next"})
    df = preds.merge(nxt, on=["ticker", "date"], how="inner")
    df["sigma_pred"] = np.exp(df["pool"] / 2.0)
    df["z"] = -df["ret_next"] / df["sigma_pred"]
    df = df.dropna(subset=["z"])

    for alpha in (0.05, 0.01):
        print(f"\n=== (b) one-sided VaR, alpha={alpha} ===")
        for name, kw in [("hand_tuned", dict(eta_by_regime=ETAS_VAR[alpha])),
                         ("adaptive", dict(adaptive=True))]:
            res = run_panel_mondrian(df, member, "pool", alpha=alpha,
                                     warmup_days=250, one_sided=True,
                                     score_col="z", **kw)
            diag = res.attrs.get("adaptive")      # merge() drops attrs
            res = res.merge(state, on=["ticker", "date"], how="left")
            res.attrs["adaptive"] = diag
            d = res[~res.warmup]
            exc = ~d["covered_hi"]
            row = {"experiment": f"var_{alpha}", "method": name,
                   "rate": exc.mean(),
                   "rate_calm": exc[d.vix_pctl <= 0.5].mean(),
                   "rate_stress": exc[d.vix_pctl > 0.95].mean()}
            rows.append(row)
            print(pd.Series(row).drop(["experiment", "method"])
                  .rename(name).round(4).to_string())
            eff_eta_report(res, name)

    out = PROJECT_ROOT / "reports"
    pd.DataFrame(rows).to_csv(out / "e5_adaptive_rates.csv", index=False)
    print(f"\nsaved -> {out / 'e5_adaptive_rates.csv'}")


if __name__ == "__main__":
    main()
