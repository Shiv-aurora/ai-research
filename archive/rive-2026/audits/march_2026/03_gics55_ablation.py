"""
March Audit — Test 03: GICS-55 Architecture-Level Ablation Study
================================================================
Runs architecture-level ablation on GICS-55 (consistent with 23.04% pipeline):
  1. HAR-RV-X baseline (tech_pred only, Ridge α=1.0)
  2. HAR-RV-X + News (linear augmentation)
  3. HAR-RV-X + Retail (linear augmentation)
  4. HAR-RV-X + News + Retail (linear, no calendar/momentum/interaction)
  5. RIVE full (all 10 features, Ridge α=100)
  6. RIVE without News Agent
  7. RIVE without Retail Agent
  8. News + Retail only (no tech_pred, no momentum, no calendar)

Author: March 2026 Audit
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.scale_up.config_universes import GICS_BALANCED_55


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


def build_features(targets, residuals, news, retail):
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


def run_config(df, feature_list, alpha, cutoff):
    avail = [f for f in feature_list if f in df.columns]
    if not avail:
        return None, None, 0

    tr = df[df["date"] < cutoff].dropna(subset=avail + ["target_log_var"]).copy()
    te = df[df["date"] >= cutoff].dropna(subset=avail + ["target_log_var"]).copy()
    tr = tr[np.isfinite(tr["target_log_var"])]
    te = te[np.isfinite(te["target_log_var"])]

    if len(tr) < 100 or len(te) < 50:
        return None, None, 0

    y_tr = tr["target_log_var"].values
    lo, hi = np.percentile(y_tr, 2), np.percentile(y_tr, 98)
    y_tr_w = np.clip(y_tr, lo, hi)

    model = Ridge(alpha=alpha)
    model.fit(tr[avail].fillna(0), y_tr_w)
    pred = model.predict(te[avail].fillna(0))
    actual = te["target_log_var"].values

    r2 = r2_score(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return r2, rmse, len(te)


def main():
    print("\n" + "=" * 72)
    print("  MARCH AUDIT — TEST 03: GICS-55 ARCHITECTURE-LEVEL ABLATION")
    print("=" * 72)

    targets, residuals, news, retail = load_data()

    gics_tickers = [t for t in GICS_BALANCED_55 if t in targets["ticker"].unique()]
    targets = targets[targets["ticker"].isin(gics_tickers)]
    print(f"\n  GICS-55 tickers available: {len(gics_tickers)}")

    df = build_features(targets, residuals, news, retail)
    cutoff = pd.to_datetime("2023-01-01")

    RIVE_FULL = ["tech_pred", "news_risk_score", "retail_risk_score",
                 "is_friday", "is_monday", "is_q4",
                 "vol_ma5", "vol_ma10", "vol_std5", "news_x_retail"]

    configs = [
        # Upper panel: linear integration strategies
        ("HAR-RV-X baseline", ["tech_pred"], 1.0,
         "Technical dynamics only"),
        ("HAR-RV-X + News", ["tech_pred", "news_risk_score"], 1.0,
         "Linear news augmentation"),
        ("HAR-RV-X + Retail", ["tech_pred", "retail_risk_score"], 1.0,
         "Linear attention proxy"),
        ("HAR-RV-X + News + Retail (linear)",
         ["tech_pred", "news_risk_score", "retail_risk_score"], 1.0,
         "All signals, no regime logic"),
        ("RIVE (full)", RIVE_FULL, 100.0,
         "Regime-aware ensemble"),

        # Lower panel: agent removal
        ("RIVE without News Agent",
         [f for f in RIVE_FULL if f not in ["news_risk_score", "news_x_retail"]],
         100.0, "No news-driven tail detection"),
        ("RIVE without Retail Agent",
         [f for f in RIVE_FULL if f not in ["retail_risk_score", "news_x_retail"]],
         100.0, "No attention regime detection"),
        ("News + Retail only (no HAR)",
         ["news_risk_score", "retail_risk_score", "news_x_retail"],
         100.0, "No volatility memory"),
    ]

    print(f"\n{'Configuration':<38s} {'R²(%)':>8s} {'RMSE':>8s} {'N':>8s}  Interpretation")
    print("-" * 100)

    results = []
    for name, feats, alpha, interp in configs:
        r2, rmse, n = run_config(df, feats, alpha, cutoff)
        if r2 is not None:
            print(f"  {name:<36s} {r2*100:>7.1f}% {rmse:>8.3f} {n:>7,}  {interp}")
            results.append({
                "configuration": name,
                "features": ", ".join(feats),
                "alpha": alpha,
                "r2_pct": round(r2 * 100, 1),
                "rmse": round(rmse, 3),
                "n": n,
                "interpretation": interp,
            })
        else:
            print(f"  {name:<36s}  FAILED (insufficient data)")

    out_path = Path(__file__).parent / "gics55_ablation_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
