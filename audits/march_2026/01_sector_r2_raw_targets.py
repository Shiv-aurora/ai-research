"""
March Audit — Test 01: Sector-level R² on raw targets
======================================================
Recomputes GICS-55 sector-level R² using the same raw-targets pipeline
(targets.parquet) that produces the 23.04% aggregate headline.

This resolves the inconsistency where sector values in the paper came from
the deseasonalized pipeline (22.44% aggregate) while the headline R² came
from the raw-targets pipeline (23.04% aggregate).

Author: March 2026 Audit
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.scale_up.config_universes import GICS_BALANCED_55, SECTOR_MAP_SP500


def load_data():
    dp = PROJECT_ROOT / "data" / "processed"
    targets = pd.read_parquet(dp / "targets.parquet")
    residuals = pd.read_parquet(dp / "residuals.parquet")

    for d in [targets, residuals]:
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)

    try:
        news = pd.read_parquet(dp / "news_predictions.parquet")
        news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(None)
        if "news_pred" in news.columns:
            news = news.rename(columns={"news_pred": "news_risk_score"})
    except Exception:
        news = None

    try:
        retail = pd.read_parquet(dp / "retail_predictions.parquet")
        retail["date"] = pd.to_datetime(retail["date"]).dt.tz_localize(None)
    except Exception:
        retail = None

    return targets, residuals, news, retail


def build_rive_df(targets, residuals, news, retail):
    """Merge all features — mirrors february tests/01_garch_mle_benchmark.py exactly."""
    df = targets.copy()
    df = df.merge(residuals[["date", "ticker", "pred_tech"]],
                  on=["date", "ticker"], how="left")
    df["tech_pred"] = df["pred_tech"]

    if news is not None and "news_risk_score" in news.columns:
        df = df.merge(news[["date", "ticker", "news_risk_score"]],
                      on=["date", "ticker"], how="left")
    df["news_risk_score"] = df.get("news_risk_score",
                                    pd.Series(0.2, index=df.index)).fillna(0.2)

    if retail is not None and "retail_risk_score" in retail.columns:
        df = df.merge(retail[["date", "ticker", "retail_risk_score"]],
                      on=["date", "ticker"], how="left")
    df["retail_risk_score"] = df.get("retail_risk_score",
                                      pd.Series(0.2, index=df.index)).fillna(0.2)

    dow = df["date"].dt.dayofweek
    df["is_monday"] = (dow == 0).astype(int)
    df["is_friday"] = (dow == 4).astype(int)
    df["is_q4"] = (df["date"].dt.quarter == 4).astype(int)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    for tk in df["ticker"].unique():
        m = df["ticker"] == tk
        tv = df.loc[m, "target_log_var"]
        df.loc[m, "vol_ma5"]  = tv.rolling(5,  min_periods=1).mean().shift(1)
        df.loc[m, "vol_ma10"] = tv.rolling(10, min_periods=1).mean().shift(1)
        df.loc[m, "vol_std5"] = tv.rolling(5,  min_periods=2).std().shift(1)

    mu = df["target_log_var"].mean()
    df["vol_ma5"]  = df["vol_ma5"].fillna(mu)
    df["vol_ma10"] = df["vol_ma10"].fillna(mu)
    df["vol_std5"] = df["vol_std5"].fillna(0)
    df["news_x_retail"] = df["news_risk_score"] * df["retail_risk_score"]
    return df


def main():
    print("\n" + "=" * 72)
    print("  MARCH AUDIT — TEST 01: SECTOR R² ON RAW TARGETS")
    print("  (Consistent with 23.04% headline pipeline)")
    print("=" * 72)

    targets, residuals, news, retail = load_data()

    # Filter to GICS-55 tickers
    gics_tickers = [t for t in GICS_BALANCED_55 if t in targets["ticker"].unique()]
    targets = targets[targets["ticker"].isin(gics_tickers)]
    print(f"\n  GICS-55 tickers available: {len(gics_tickers)}")

    # Build features
    rive_df = build_rive_df(targets, residuals, news, retail)
    rive_feats = ["tech_pred", "news_risk_score", "retail_risk_score",
                  "is_friday", "is_monday", "is_q4",
                  "vol_ma5", "vol_ma10", "vol_std5", "news_x_retail"]

    cutoff = pd.to_datetime("2023-01-01")
    r_tr = rive_df[rive_df["date"] < cutoff].dropna(subset=rive_feats + ["target_log_var"])
    r_te = rive_df[rive_df["date"] >= cutoff].dropna(subset=rive_feats + ["target_log_var"])
    r_tr = r_tr[np.isfinite(r_tr["target_log_var"])]
    r_te = r_te[np.isfinite(r_te["target_log_var"])]

    # Winsorize train targets (2/98 percentile) — matches production pipeline
    y_tr = r_tr["target_log_var"].values
    lo, hi = np.percentile(y_tr, 2), np.percentile(y_tr, 98)
    y_tr_w = np.clip(y_tr, lo, hi)

    # Fit RIVE coordinator
    rive_model = Ridge(alpha=100.0)
    rive_model.fit(r_tr[rive_feats].fillna(0), y_tr_w)
    r_te = r_te.copy()
    r_te["rive_pred"] = rive_model.predict(r_te[rive_feats].fillna(0))

    # Also fit HAR-RV-X baseline for comparison
    har_feats = ["prev_day_rv", "rv_5d_mean", "rv_20d_mean",
                 "returns_sq_lag_1", "VIX_close", "rsi_14"]
    har_feats_avail = [f for f in har_feats if f in targets.columns]

    if "returns_sq_lag_1" not in targets.columns and "close" in targets.columns:
        targets["returns_sq_lag_1"] = (targets.groupby("ticker")["close"].pct_change() * 100) ** 2

    har_tr = targets[targets["date"] < cutoff].dropna(subset=har_feats_avail + ["target_log_var"])
    har_te = targets[targets["date"] >= cutoff].dropna(subset=har_feats_avail + ["target_log_var"])
    har_tr = har_tr[np.isfinite(har_tr["target_log_var"])]
    har_te = har_te[np.isfinite(har_te["target_log_var"])]

    har_model = Ridge(alpha=1.0)
    har_model.fit(har_tr[har_feats_avail].fillna(0), har_tr["target_log_var"])

    # Aggregate R² verification
    agg_r2 = r2_score(r_te["target_log_var"], r_te["rive_pred"])
    print(f"\n  Aggregate RIVE R² (raw targets): {agg_r2*100:.2f}%")
    print(f"  (Should be ~23.04% to confirm pipeline match)")

    # Sector-level R²
    r_te_sec = r_te.copy()
    r_te_sec["sector"] = r_te_sec["ticker"].map(SECTOR_MAP_SP500)

    print(f"\n{'Sector':<22s} {'RIVE R²':>10s} {'N':>8s}")
    print("-" * 44)

    sector_rows = []
    for sector in sorted(r_te_sec["sector"].dropna().unique()):
        mask = r_te_sec["sector"] == sector
        sub = r_te_sec[mask]
        if len(sub) > 50:
            sr2 = r2_score(sub["target_log_var"], sub["rive_pred"])
            print(f"  {sector:<20s} {sr2*100:>8.2f}%  {len(sub):>6,}")
            sector_rows.append({"sector": sector, "r2": sr2, "n": len(sub)})

    # Save results
    out_path = Path(__file__).parent / "sector_r2_raw_targets.csv"
    pd.DataFrame(sector_rows).to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")

    # Also save aggregate for cross-check
    summary_path = Path(__file__).parent / "sector_aggregate_check.csv"
    pd.DataFrame([{
        "metric": "aggregate_r2_pct",
        "value": agg_r2 * 100,
    }]).to_csv(summary_path, index=False)

    print(f"  Aggregate check saved → {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
