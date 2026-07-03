"""
March Audit — Generate publication figures for ESWA submission
==============================================================
Creates three figures:
  1. Actual vs Predicted time series (2 representative stocks)
  2. Calibration scatter plot (Actual vs Predicted with 45° line)
  3. Sector R² horizontal bar chart

All use the raw-targets pipeline (consistent with 23.04% headline).

Author: March 2026
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.scale_up.config_universes import GICS_BALANCED_55, SECTOR_MAP_SP500

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RIVE_COLOR = "#1f77b4"
HAR_COLOR = "#ff7f0e"
ACTUAL_COLOR = "#333333"


def load_and_build():
    """Load data and build RIVE features (mirrors pipeline exactly)."""
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

    # Filter to GICS-55
    gics_tickers = [t for t in GICS_BALANCED_55 if t in targets["ticker"].unique()]
    targets = targets[targets["ticker"].isin(gics_tickers)]

    # Build features
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

    return df, gics_tickers


def fit_models(df):
    """Fit RIVE and HAR-RV-X, return test set with predictions."""
    rive_feats = ["tech_pred", "news_risk_score", "retail_risk_score",
                  "is_friday", "is_monday", "is_q4",
                  "vol_ma5", "vol_ma10", "vol_std5", "news_x_retail"]

    # Full 6-feature HAR-RV-X (matches main results table)
    har_feats_full = ["prev_day_rv", "rv_5d_mean", "rv_20d_mean",
                      "returns_sq_lag_1", "VIX_close", "rsi_14"]
    har_feats = [f for f in har_feats_full if f in df.columns]

    cutoff = pd.to_datetime("2023-01-01")
    all_feats = list(set(rive_feats + har_feats))
    tr = df[df["date"] < cutoff].dropna(subset=rive_feats + ["target_log_var"]).copy()
    te = df[df["date"] >= cutoff].dropna(subset=rive_feats + ["target_log_var"]).copy()
    tr = tr[np.isfinite(tr["target_log_var"])]
    te = te[np.isfinite(te["target_log_var"])]

    # Winsorize
    y_tr = tr["target_log_var"].values
    lo, hi = np.percentile(y_tr, 2), np.percentile(y_tr, 98)
    y_tr_w = np.clip(y_tr, lo, hi)

    # RIVE
    rive_model = Ridge(alpha=100.0)
    rive_model.fit(tr[rive_feats].fillna(0), y_tr_w)
    te["rive_pred"] = rive_model.predict(te[rive_feats].fillna(0))

    # HAR-RV-X (full 6-feature, consistent with Table 4)
    if har_feats:
        har_tr = tr.dropna(subset=har_feats)
        har_model = Ridge(alpha=1.0)
        har_model.fit(har_tr[har_feats].fillna(0), har_tr["target_log_var"].values)
        te["har_pred"] = har_model.predict(te[har_feats].fillna(0))
        print(f"  HAR-RV-X features used: {har_feats}")
    else:
        # Fallback to tech_pred only
        har_model = Ridge(alpha=1.0)
        har_model.fit(tr[["tech_pred"]].fillna(0), y_tr_w)
        te["har_pred"] = har_model.predict(te[["tech_pred"]].fillna(0))
        print("  HAR-RV-X: fallback to tech_pred only")

    return te


def figure1_timeseries(te):
    """Actual vs Predicted time series for 2 representative stocks."""
    # Pick one high-R² stock and one moderate-R² stock
    ticker_r2 = {}
    for tk in te["ticker"].unique():
        sub = te[te["ticker"] == tk]
        if len(sub) > 100:
            ticker_r2[tk] = r2_score(sub["target_log_var"], sub["rive_pred"])

    sorted_tickers = sorted(ticker_r2.items(), key=lambda x: x[1], reverse=True)

    # Pick: one from top ~5 (high R²), one from middle
    high_tk = sorted_tickers[2][0]  # 3rd best to avoid cherry-picking the very best
    mid_idx = len(sorted_tickers) // 2
    mid_tk = sorted_tickers[mid_idx][0]

    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=False)

    for ax, tk, panel_label in zip(axes, [high_tk, mid_tk], ["(a)", "(b)"]):
        sub = te[te["ticker"] == tk].sort_values("date").copy()
        sector = SECTOR_MAP_SP500.get(tk, "")
        tk_r2 = r2_score(sub["target_log_var"], sub["rive_pred"])

        ax.plot(sub["date"], sub["target_log_var"], color=ACTUAL_COLOR,
                linewidth=0.6, alpha=0.7, label="Realized (actual)")
        ax.plot(sub["date"], sub["rive_pred"], color=RIVE_COLOR,
                linewidth=0.8, alpha=0.85, label=f"RIVE ($R^2$ = {tk_r2*100:.1f}%)")

        ax.set_ylabel("Log realized variance")
        ax.set_title(f"{panel_label} {tk} ({sector})", loc="left", fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.tick_params(axis="x", rotation=30)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Out-of-sample RIVE forecasts vs. realized log variance (2023\u20132024)",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = OUT_DIR / "fig_actual_vs_predicted.png"
    fig.savefig(path)
    fig.savefig(OUT_DIR / "fig_actual_vs_predicted.pdf")
    plt.close(fig)
    print(f"  Figure 1 saved → {path}")
    print(f"    Stocks chosen: {high_tk} (R²={ticker_r2[high_tk]*100:.1f}%), "
          f"{mid_tk} (R²={ticker_r2[mid_tk]*100:.1f}%)")


def figure2_calibration(te):
    """Calibration scatter: actual vs predicted with 45° line."""
    # Subsample for visual clarity (full dataset is ~28k points)
    np.random.seed(42)
    sample_idx = np.random.choice(len(te), size=min(5000, len(te)), replace=False)
    sub = te.iloc[sample_idx]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    for ax, pred_col, color, label in [
        (axes[0], "har_pred", HAR_COLOR, "HAR-RV-X"),
        (axes[1], "rive_pred", RIVE_COLOR, "RIVE"),
    ]:
        actual = sub["target_log_var"].values
        pred = sub[pred_col].values
        # R² from FULL dataset, not subsample
        r2 = r2_score(te["target_log_var"].values, te[pred_col].values)

        ax.scatter(pred, actual, alpha=0.08, s=4, color=color, rasterized=True)

        # 45° line
        lo = min(actual.min(), pred.min())
        hi = max(actual.max(), pred.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", linewidth=0.8, alpha=0.5, label="Perfect calibration")

        # OLS fit line
        from numpy.polynomial.polynomial import polyfit
        b, m_slope = polyfit(pred, actual, 1)
        x_line = np.linspace(pred.min(), pred.max(), 100)
        ax.plot(x_line, b + m_slope * x_line, color="red", linewidth=1.0,
                alpha=0.7, label=f"OLS fit ($\\beta$={m_slope:.2f})")

        ax.set_xlabel("Predicted log variance")
        ax.set_ylabel("Realized log variance")
        ax.set_title(f"{label} ($R^2$ = {r2*100:.1f}%)", fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle("Forecast calibration: predicted vs. realized log variance",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = OUT_DIR / "fig_calibration_scatter.png"
    fig.savefig(path)
    fig.savefig(OUT_DIR / "fig_calibration_scatter.pdf")
    plt.close(fig)
    print(f"  Figure 2 saved → {path}")


def figure3_sector_bar(te):
    """Sector R² horizontal bar chart."""
    te_sec = te.copy()
    te_sec["sector"] = te_sec["ticker"].map(SECTOR_MAP_SP500)

    sector_r2 = []
    for sector in te_sec["sector"].dropna().unique():
        sub = te_sec[te_sec["sector"] == sector]
        if len(sub) > 50:
            r2 = r2_score(sub["target_log_var"], sub["rive_pred"])
            sector_r2.append({"sector": sector, "r2": r2 * 100})

    sdf = pd.DataFrame(sector_r2).sort_values("r2", ascending=True)

    fig, ax = plt.subplots(figsize=(5.5, 4))

    colors = [RIVE_COLOR if r > 0 else "#d62728" for r in sdf["r2"]]
    bars = ax.barh(sdf["sector"], sdf["r2"], color=colors, edgecolor="white",
                   linewidth=0.5, height=0.65)

    # Add value labels
    for bar, val in zip(bars, sdf["r2"]):
        offset = 0.5 if val > 0 else -0.5
        ha = "left" if val > 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha=ha, fontsize=8)

    # Aggregate line
    agg_r2 = r2_score(te["target_log_var"], te["rive_pred"]) * 100
    ax.axvline(agg_r2, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(agg_r2 + 0.5, len(sdf) - 0.5, f"Aggregate: {agg_r2:.1f}%",
            color="red", fontsize=8, va="top")

    ax.set_xlabel("Out-of-sample $R^2$ (%)")
    ax.set_title("RIVE sector-level performance (GICS-55, 2023\u20132024)",
                 fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)

    plt.tight_layout()

    path = OUT_DIR / "fig_sector_r2_bar.png"
    fig.savefig(path)
    fig.savefig(OUT_DIR / "fig_sector_r2_bar.pdf")
    plt.close(fig)
    print(f"  Figure 3 saved → {path}")


def main():
    print("\n" + "=" * 60)
    print("  GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    print("\n  Loading data and fitting models...")
    df, tickers = load_and_build()
    print(f"  GICS-55 tickers: {len(tickers)}")

    te = fit_models(df)
    agg_r2 = r2_score(te["target_log_var"], te["rive_pred"])
    print(f"  Aggregate RIVE R²: {agg_r2*100:.2f}%")
    print(f"  Test observations: {len(te):,}\n")

    figure1_timeseries(te)
    figure2_calibration(te)
    figure3_sector_bar(te)

    print(f"\n  All figures saved to {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
