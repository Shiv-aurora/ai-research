"""E17: what exactly do regimes add over pooled K=1? Inferential package.

The main table shows pooled K=1 at 87.3% stress vs RC K=4 at 88.1% with a
nonsignificant average difference. The paper's claim is that the regime
layer buys CONDITIONAL structure. This experiment gives every distinct
claim its own estimate + uncertainty, on the identical adaptive
calibrator with only K varied:

  (a) per-regime coverage and daily interval-score differences with
      date-clustered DM tests (the prespecified conditional objectives);
  (b) days-since-stress-entry profile differences with HAC CIs per
      bucket (incl. the day-2 claim), and the count of independent
      market-level stress entries behind them;
  (c) regime-calibration dispersion: weighted RMS and max absolute
      deviation of per-regime coverage from 0.90, with a 20-day
      moving-block bootstrap CI on the K=4 - K=1 difference.

Usage: .venv/bin/python scripts/e17_k1_vs_k4.py
Output: reports/e17_regime_slices.csv, reports/e17_dse_diff.csv,
        reports/e17_dispersion.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.e6b_oracle_regimes import bin_membership
from src.conformal.panel_hierarchical import run_panel_mondrian
from src.eval.dm_hac import dm_test, hac_mean_se
from src.eval.mcs import interval_score
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.parallel import pmap
from src.utils.seeding import seed_everything

FORECAST = "pool"
ALPHA = 0.10
CUTS = [0.5, 0.8, 0.95]
REGIMES = ["calm", "normal", "elevated", "stress"]
N_BOOT = 5_000
BLOCK = 20


def _run(args):
    label, preds, member = args
    res = run_panel_mondrian(preds, member, FORECAST, alpha=ALPHA,
                             adaptive=True, warmup_days=100)
    return label, res


def add_dse(d: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    d = d.sort_values(["ticker", "date"]).copy()
    runs = []
    for _, g in d.groupby("ticker"):
        inside = (g["vix_pctl"] > threshold).values
        run, c = np.zeros(len(g)), 0
        for i, flag in enumerate(inside):
            c = c + 1 if flag else 0
            run[i] = c
        runs.append(pd.Series(run, index=g.index))
    d["dse"] = pd.concat(runs)
    return d


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    proc = PROJECT_ROOT / cfg["data"]["processed_path"]
    panel = pd.read_parquet(proc / "rv_panel.parquet")
    preds = pd.read_parquet(proc / "e0_predictions.parquet")
    state = panel[["ticker", "date", "vix_pctl"]]
    market = panel.groupby("date")[["vix_pctl"]].first().sort_index()

    jobs = [("k4", preds, bin_membership(market, CUTS)),
            ("k1", preds, bin_membership(market, []))]
    res = {}
    for label, r in pmap(_run, jobs):
        r = r.merge(state, on=["ticker", "date"], how="left")
        r = r[~r.warmup].copy()
        r["lo_raw"] = r[FORECAST] - r["q_lo"] * r["sigma_hat"]
        r["hi_raw"] = r[FORECAST] + r["q_hi"] * r["sigma_hat"]
        r["is_"] = interval_score(r["target"].values, r["lo_raw"].values,
                                  r["hi_raw"].values, ALPHA)
        res[label] = add_dse(r)

    k1 = res["k1"][["ticker", "date", "covered", "is_"]].rename(
        columns={"covered": "covered_k1", "is_": "is_k1"})
    pair = res["k4"].merge(k1, on=["ticker", "date"])
    pair["k"] = np.searchsorted(CUTS, pair["vix_pctl"].values, side="right")
    pair = pair.dropna(subset=["vix_pctl"])

    # (a) per-regime coverage and interval-score differences
    rows = []
    for k, name in enumerate(REGIMES):
        s = pair[pair["k"] == k]
        dm_cov = dm_test(s["covered"].astype(float),
                         s["covered_k1"].astype(float), s["date"])
        dm_is = dm_test(s["is_"], s["is_k1"], s["date"])
        rows.append({
            "regime": name, "n": len(s),
            "cov_k4": s["covered"].mean(), "cov_k1": s["covered_k1"].mean(),
            "cov_diff": dm_cov["mean_diff"], "cov_t": dm_cov["dm"],
            "cov_p": dm_cov["p"],
            "is_k4": s["is_"].mean(), "is_k1": s["is_k1"].mean(),
            "is_diff": dm_is["mean_diff"], "is_t": dm_is["dm"],
            "is_p": dm_is["p"]})
    slices = pd.DataFrame(rows)
    print("=== per-regime K=4 vs K=1 (coverage and interval score; "
          "negative is_diff favors K=4) ===")
    print(slices.round(4).to_string(index=False))

    # (b) days-since-entry differences with HAC CIs
    ent = market.copy()
    stress_flag = (ent["vix_pctl"] > 0.95).fillna(False).values
    entries = int(np.sum(np.diff(stress_flag.astype(int)) == 1)
                  + int(stress_flag[0]))
    print(f"\nindependent market-level stress entries: {entries}")
    dse_rows = []
    bins = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"),
            (6, 10, "6-10"), (11, np.inf, ">10")]
    for lo_b, hi_b, lab in bins:
        s = pair[(pair["dse"] >= lo_b) & (pair["dse"] <= hi_b)]
        dm = dm_test(s["covered"].astype(float),
                     s["covered_k1"].astype(float), s["date"])
        st = hac_mean_se(
            (s["covered"].astype(float) - s["covered_k1"].astype(float)),
            s["date"])
        dse_rows.append({
            "dse": lab, "n": len(s), "n_dates": st["n_dates"],
            "cov_k4": s["covered"].mean(), "cov_k1": s["covered_k1"].mean(),
            "diff": st["mean"], "se": st["se"],
            "ci_lo": st["mean"] - 1.96 * st["se"],
            "ci_hi": st["mean"] + 1.96 * st["se"], "p": dm["p"]})
    dse = pd.DataFrame(dse_rows)
    print("\n=== days-since-stress-entry: K=4 - K=1 coverage ===")
    print(dse.round(4).to_string(index=False))

    # (c) dispersion of per-regime calibration
    daily = (pair.groupby(["date", "k"])
             .agg(c4=("covered", "mean"), c1=("covered_k1", "mean"),
                  n=("covered", "size")).reset_index())

    def disp(d: pd.DataFrame) -> dict:
        out = {}
        for col, tag in [("c4", "k4"), ("c1", "k1")]:
            cov = d.groupby("k").apply(
                lambda g: np.average(g[col], weights=g["n"]),
                include_groups=False)
            w = d.groupby("k")["n"].sum() / d["n"].sum()
            dev = cov - (1 - ALPHA)
            out[f"rms_{tag}"] = float(np.sqrt((w * dev ** 2).sum()))
            out[f"max_{tag}"] = float(dev.abs().max())
        return out

    obs = disp(daily)
    dates_u = np.sort(daily["date"].unique())
    n_days = len(dates_u)
    rng = np.random.default_rng(cfg["seed"])
    by_date = {d: g for d, g in daily.groupby("date")}
    diffs_rms, diffs_max = np.empty(N_BOOT), np.empty(N_BOOT)
    n_blocks = int(np.ceil(n_days / BLOCK))
    for i in range(N_BOOT):
        starts = rng.integers(0, n_days, n_blocks)
        take = np.concatenate([np.arange(s, min(s + BLOCK, n_days))
                               for s in starts])[:n_days]
        bs = pd.concat([by_date[dates_u[j]] for j in take],
                       ignore_index=True)
        r = disp(bs)
        diffs_rms[i] = r["rms_k4"] - r["rms_k1"]
        diffs_max[i] = r["max_k4"] - r["max_k1"]
    disp_rows = []
    for key, d_boot in [("rms", diffs_rms), ("max", diffs_max)]:
        lo_ci, hi_ci = np.quantile(d_boot, [0.025, 0.975])
        p = 2 * min((d_boot <= 0).mean(), (d_boot >= 0).mean())
        disp_rows.append({"measure": key,
                          "k4": obs[f"{key}_k4"], "k1": obs[f"{key}_k1"],
                          "diff": obs[f"{key}_k4"] - obs[f"{key}_k1"],
                          "ci_lo": lo_ci, "ci_hi": hi_ci, "p_boot": p})
    dispersion = pd.DataFrame(disp_rows)
    print("\n=== regime-calibration dispersion (negative diff favors "
          "K=4) ===")
    print(dispersion.round(4).to_string(index=False))

    rep = PROJECT_ROOT / "reports"
    slices.to_csv(rep / "e17_regime_slices.csv", index=False)
    dse.to_csv(rep / "e17_dse_diff.csv", index=False)
    dispersion.to_csv(rep / "e17_dispersion.csv", index=False)
    print("\nsaved -> reports/e17_regime_slices.csv, "
          "reports/e17_dse_diff.csv, reports/e17_dispersion.csv")


if __name__ == "__main__":
    main()
