"""E2 diagnostics: why did regime conditioning barely improve stress coverage?

H1 (granularity): K=3 regimes are too coarse — the method's 'stress' bucket
    is ~top-third while evaluation slices the top 5%. Test: condition on bins
    ALIGNED with the evaluation slices (cutpoints .5/.8/.95 on vix_pctl).
    Mondrian guarantee => aligned conditioning must repair each slice, unless
H2 (transition lag): miscoverage concentrates in the first days after a
    regime is entered (vol spike precedes the signal). Test: coverage as a
    function of days-since-entry into the top slice, for ACI vs aligned bins.

Usage: .venv/bin/python scripts/e2_diagnostics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal.aci import run_aci
from src.conformal.mondrian_soft import run_soft_mondrian
from src.eval.coverage import coverage_by_state, marginal_coverage
from src.utils.config import PROJECT_ROOT, load_config

WARMUP = 100
FORECAST = "pool"
CUTS = [0.5, 0.8, 0.95]          # aligned with evaluation slices
ETAS = [0.02, 0.03, 0.06, 0.15]  # calm ... stress


def aligned_bin_memberships(vix_pctl: pd.Series) -> np.ndarray:
    k = np.searchsorted(CUTS, vix_pctl.values, side="right")
    pi = np.zeros((len(vix_pctl), len(CUTS) + 1))
    pi[np.arange(len(vix_pctl)), k] = 1.0
    return pi


def run_methods(preds, state):
    frames = {"aci": [], "rc_aligned": []}
    for ticker, g in preds.sort_values("date").groupby("ticker"):
        g = g.dropna(subset=["target", FORECAST]).copy()
        if len(g) <= WARMUP + 50:
            continue
        g = g.merge(state, on=["ticker", "date"], how="left")
        g["vix_pctl"] = g["vix_pctl"].fillna(0.5)
        scores = (g["target"] - g[FORECAST]).values

        a = run_aci(scores, alpha=0.10, eta=0.05, warmup=WARMUP)
        a.index = g.index
        frames["aci"].append(pd.concat([g, a], axis=1))

        pi = aligned_bin_memberships(g["vix_pctl"])
        r = run_soft_mondrian(scores, pi, alpha=0.10, eta_by_regime=ETAS,
                              warmup=WARMUP)
        r.index = g.index
        frames["rc_aligned"].append(pd.concat([g, r], axis=1))
    return {k: pd.concat(v) for k, v in frames.items()}


def days_since_entry(df: pd.DataFrame, threshold: float = 0.95) -> pd.Series:
    """Per ticker: consecutive days (1-based) the stock has been in the top
    slice; NaN outside it."""
    out = []
    for _, g in df.groupby("ticker"):
        inside = (g["vix_pctl"] > threshold).values
        run = np.zeros(len(g))
        c = 0
        for i, flag in enumerate(inside):
            c = c + 1 if flag else 0
            run[i] = c
        out.append(pd.Series(run, index=g.index))
    return pd.concat(out).replace(0, np.nan)


def main() -> None:
    cfg = load_config()
    panel = pd.read_parquet(PROJECT_ROOT / cfg["data"]["processed_path"] / "rv_panel.parquet")
    preds = pd.read_parquet(PROJECT_ROOT / cfg["data"]["processed_path"] / "e0_predictions.parquet")
    state = panel[["ticker", "date", "vix_pctl"]]

    res = run_methods(preds, state)

    print("=== H1 test: conditioning ALIGNED with evaluation slices ===")
    for name, df in res.items():
        df = df.sort_values(["ticker", "date"])
        by = coverage_by_state(df, "vix_pctl")
        print(f"\n{name}: marginal={marginal_coverage(df):.4f}")
        print(by[["coverage", "upper_coverage", "n"]].round(4).to_string())

    print("\n=== H2 test: coverage by days since entering top slice ===")
    for name, df in res.items():
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        dse = days_since_entry(df)
        d = df[~df["warmup"]].copy()
        d["dse"] = dse[~df["warmup"]].values
        d = d.dropna(subset=["dse"])
        bins = pd.cut(d["dse"], [0, 1, 2, 3, 5, 10, 1000],
                      labels=["1", "2", "3", "4-5", "6-10", ">10"])
        tab = d.groupby(bins, observed=True)["covered"].agg(["mean", "size"])
        print(f"\n{name} (top-slice days only):")
        print(tab.round(4).to_string())

    out = PROJECT_ROOT / "reports"
    for name, df in res.items():
        coverage_by_state(df, "vix_pctl").to_csv(out / f"e2diag_{name}_by_state.csv")
    print(f"\nsaved -> reports/e2diag_*_by_state.csv")


if __name__ == "__main__":
    main()
