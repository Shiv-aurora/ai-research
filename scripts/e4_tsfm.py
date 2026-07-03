"""E4: do time-series foundation models know what they don't know in crises?

Chronos-Bolt zero-shot forecasts of next-day log-RV with its NATIVE 80%
interval (q10, q90 — Bolt cannot express quantiles beyond [.1,.9]).

  (a) point sanity: QLIKE of the Chronos median vs HAR / pool on same rows
  (b) raw calibration: coverage of the native 80% band, marginal + by regime
  (c) repair: regime-conditional conformal expansion of the band, using the
      scale-free CQR score s = max(q10 - y, y - q90) / halfwidth so pooling
      across stocks is legitimate.

Usage: .venv/bin/python scripts/e4_tsfm.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal.panel_hierarchical import run_panel_mondrian
from src.forecasters.base import qlike
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeding import seed_everything

ALPHA = 0.20   # Bolt's native band
CUTS = [0.5, 0.8, 0.95]
ETAS = [0.001, 0.0015, 0.003, 0.01]


def aligned_bins(market: pd.DataFrame) -> pd.DataFrame:
    v = market["vix_pctl"].values
    k = np.searchsorted(CUTS, v, side="right")
    pi = np.zeros((len(v), 4))
    ok = ~np.isnan(v)
    pi[ok, k[ok]] = 1.0
    pi[~ok] = 0.25
    return pd.DataFrame(pi, index=market.index,
                        columns=[f"regime_{j}" for j in range(4)])


def cov_by_regime(df, cov_col="covered"):
    d = df[~df.get("warmup", pd.Series(False, index=df.index))]
    bins = pd.cut(d["vix_pctl"], [0, 0.5, 0.8, 0.95, 1.0],
                  labels=["calm", "normal", "elevated", "stress"],
                  include_lowest=True)
    out = d.groupby(bins, observed=True)[cov_col].agg(["mean", "size"])
    out.loc["marginal"] = [d[cov_col].mean(), len(d)]
    return out


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    proc = PROJECT_ROOT / cfg["data"]["processed_path"]
    panel = pd.read_parquet(proc / "rv_panel.parquet")
    tsfm = pd.read_parquet(proc / "tsfm_predictions.parquet")
    e0 = pd.read_parquet(proc / "e0_predictions.parquet")

    df = tsfm.merge(e0[["ticker", "date", "target", "har", "pool"]],
                    on=["ticker", "date"], how="inner")
    df = df.merge(panel[["ticker", "date", "vix_pctl"]],
                  on=["ticker", "date"], how="left").dropna(subset=["target"])
    print(f"TSFM eval rows: {len(df):,}")

    # (a) point sanity
    print("\n[a] point-forecast sanity (QLIKE):")
    for m in ["q50", "har", "pool"]:
        print(f"  {m:<5} {qlike(df['target'].values, df[m].values).mean():.4f}")

    # (b) raw native-band calibration
    df["covered"] = (df["target"] >= df["q10"]) & (df["target"] <= df["q90"])
    print(f"\n[b] RAW Chronos 80% band coverage (target 0.80):")
    print(cov_by_regime(df).round(4).to_string())

    # directional misses: symmetric coverage can mask one-sided failure
    df["exceed_up"] = df["target"] > df["q90"]
    df["exceed_dn"] = df["target"] < df["q10"]
    bins = pd.cut(df["vix_pctl"], [0, 0.5, 0.8, 0.95, 1.0],
                  labels=["calm", "normal", "elevated", "stress"],
                  include_lowest=True)
    direc = df.groupby(bins, observed=True)[["exceed_up", "exceed_dn"]].mean()
    direc.loc["marginal"] = df[["exceed_up", "exceed_dn"]].mean()
    print("\n[b'] directional miss rates (nominal 10/10):")
    print((direc * 100).round(2).to_string())

    # (c) regime-conditional conformal repair
    df["halfwidth"] = (df["q90"] - df["q10"]) / 2.0
    df = df[df["halfwidth"] > 1e-8]
    df["cqr_scale_free"] = (np.maximum(df["q10"] - df["target"],
                                       df["target"] - df["q90"])
                            / df["halfwidth"])
    market = panel.groupby("date")[["vix_pctl"]].first().sort_index()
    member = aligned_bins(market)
    rc = run_panel_mondrian(df, member, "q50", alpha=ALPHA,
                            eta_by_regime=ETAS, warmup_days=100,
                            one_sided=True, score_col="cqr_scale_free")
    rc = rc.merge(panel[["ticker", "date", "vix_pctl"]],
                  on=["ticker", "date"], how="left", suffixes=("", "_p"))
    print(f"\n[c] regime-conditional conformal repair (coverage target 0.80):")
    print(cov_by_regime(rc).round(4).to_string())
    # expansion factor: q_hi is the tracked threshold on the scale-free score;
    # issued band = model band expanded by (1 + q_hi) * halfwidth per side
    exp_stress = rc.loc[(~rc.warmup) & (rc.vix_pctl > 0.95), "q_hi"].mean()
    exp_calm = rc.loc[(~rc.warmup) & (rc.vix_pctl <= 0.5), "q_hi"].mean()
    print(f"\nmean band expansion (fraction of model halfwidth): "
          f"calm {exp_calm:+.3f}, stress {exp_stress:+.3f}")

    out = PROJECT_ROOT / "reports"
    cov_by_regime(df).to_csv(out / "e4_tsfm_raw_coverage.csv")
    cov_by_regime(rc).to_csv(out / "e4_tsfm_repaired_coverage.csv")
    print(f"\nsaved -> reports/e4_tsfm_*.csv")


if __name__ == "__main__":
    main()
