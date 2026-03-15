"""
Assemble the FULL feature set available to RIVE into a single flat DataFrame.

This module loads and merges ALL raw end-of-day features that the RIVE agents
individually consume, producing one unified feature matrix suitable for
monolithic baseline models (LightGBM, Elastic Net).

Feature groups:
  1. Technical (HAR-RV): rv_lag_1, rv_lag_5, rv_lag_22, returns_sq_lag_1,
                          VIX_close, rsi_14
  2. News:               news_memory, shock_memory, sentiment_memory,
                          shock_vix_memory, sentiment_avg, novelty_score,
                          shock_index, news_count, news_pca_0..19
  3. Retail/Attention:    volume_shock, volume_shock_roll3, hype_signal,
                          hype_signal_roll3, hype_signal_roll7, hype_zscore,
                          price_acceleration
  4. Calendar:            is_friday, is_monday, is_q4
  5. Short-Horizon Vol:   vol_ma5, vol_ma10, vol_std5

Target: target_log_var  (log next-day realized variance)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_CUTOFF = pd.Timestamp("2023-01-01")

SECTOR_MAP = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech',
    'JPM': 'Finance', 'BAC': 'Finance', 'V': 'Finance',
    'CAT': 'Industrial', 'GE': 'Industrial', 'BA': 'Industrial',
    'WMT': 'Consumer', 'MCD': 'Consumer', 'COST': 'Consumer',
    'XOM': 'Energy', 'CVX': 'Energy', 'SLB': 'Energy',
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
}


def _norm_date(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    if df["ticker"].dtype.name == "category":
        df["ticker"] = df["ticker"].astype(str)
    return df


def _decay_kernel(series: pd.Series, group: pd.Series) -> pd.Series:
    weights = [0.50, 0.25, 0.15, 0.10]
    tmp = pd.DataFrame({"value": series, "group": group})
    result = pd.Series(0.0, index=series.index)
    for i, w in enumerate(weights, 1):
        result += w * tmp.groupby("group")["value"].shift(i).fillna(0)
    return result


def assemble_full_features() -> pd.DataFrame:
    """
    Load, merge, and feature-engineer the full information set.

    Returns
    -------
    pd.DataFrame with columns:
        date, ticker, target_log_var, sector,
        <all technical features>,
        <all news features>,
        <all retail features>,
        <calendar features>,
        <short-horizon vol features>
    """
    print("=" * 70)
    print("ASSEMBLING FULL FEATURE SET FOR MONOLITHIC BASELINES")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. TARGETS + TECHNICAL FEATURES
    # ------------------------------------------------------------------
    targets = _norm_date(pd.read_parquet(DATA_DIR / "targets.parquet"))
    targets = targets.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Engineer HAR-RV features per ticker
    parts = []
    for ticker in targets["ticker"].unique():
        t = targets[targets["ticker"] == ticker].copy().sort_values("date")
        t["rv_lag_1"] = t["realized_vol"].shift(1)
        t["rv_lag_5"] = t["realized_vol"].rolling(5, min_periods=1).mean().shift(1)
        t["rv_lag_22"] = t["realized_vol"].rolling(22, min_periods=1).mean().shift(1)
        t["daily_return"] = t["close"] / t["close"].shift(1) - 1
        t["returns_sq_lag_1"] = (t["daily_return"] ** 2).shift(1)
        # Short-horizon vol features
        t["vol_ma5"] = t["target_log_var"].rolling(5, min_periods=1).mean().shift(1)
        t["vol_ma10"] = t["target_log_var"].rolling(10, min_periods=1).mean().shift(1)
        t["vol_std5"] = t["target_log_var"].rolling(5, min_periods=2).std().shift(1)
        # Drop warm-up rows
        if len(t) > 22:
            t = t.iloc[22:]
        parts.append(t)

    df = pd.concat(parts, ignore_index=True)

    # VIX + RSI
    df["VIX_close"] = df["VIX_close"].ffill().fillna(20)
    if "rsi_14" in df.columns:
        df["rsi_14"] = df["rsi_14"].ffill().fillna(50)
    else:
        df["rsi_14"] = 50.0

    # Fill vol features
    mean_tlv = df["target_log_var"].mean()
    df["vol_ma5"] = df["vol_ma5"].fillna(mean_tlv)
    df["vol_ma10"] = df["vol_ma10"].fillna(mean_tlv)
    df["vol_std5"] = df["vol_std5"].fillna(0)

    print(f"  Technical features: {len(df):,} rows")

    # ------------------------------------------------------------------
    # 2. NEWS FEATURES
    # ------------------------------------------------------------------
    news = _norm_date(pd.read_parquet(DATA_DIR / "news_features.parquet"))
    df = pd.merge(df, news, on=["date", "ticker"], how="left")

    # Decay kernel features (same as NewsAgent)
    for col in ["news_count", "shock_index", "sentiment_avg"]:
        if col in df.columns:
            mem_col = col.split("_")[0] + "_memory" if col != "sentiment_avg" else "sentiment_memory"
            if col == "news_count":
                mem_col = "news_memory"
            elif col == "shock_index":
                mem_col = "shock_memory"
            df[mem_col] = _decay_kernel(df[col].fillna(0), df["ticker"])

    # Shock-VIX interaction
    if "shock_memory" in df.columns:
        df["shock_vix_memory"] = df["shock_memory"] * df["VIX_close"] / 20

    # Fill missing news features with 0
    news_cols = [c for c in df.columns if c.startswith("news_pca_")]
    for c in news_cols + ["sentiment_avg", "novelty_score", "shock_index",
                          "news_count", "news_memory", "shock_memory",
                          "sentiment_memory", "shock_vix_memory"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    print(f"  + News features merged")

    # ------------------------------------------------------------------
    # 3. RETAIL / ATTENTION FEATURES
    # ------------------------------------------------------------------
    reddit = _norm_date(pd.read_parquet(DATA_DIR / "reddit_proxy.parquet"))
    retail_cols = ["date", "ticker", "volume_shock", "volume_shock_roll3",
                   "hype_signal", "hype_signal_roll3", "hype_signal_roll7",
                   "hype_zscore", "price_acceleration"]
    retail_cols = [c for c in retail_cols if c in reddit.columns]
    df = pd.merge(df, reddit[retail_cols], on=["date", "ticker"], how="left")

    for c in retail_cols:
        if c not in ("date", "ticker") and c in df.columns:
            df[c] = df[c].fillna(0)

    print(f"  + Retail features merged")

    # ------------------------------------------------------------------
    # 4. CALENDAR FEATURES
    # ------------------------------------------------------------------
    dow = df["date"].dt.dayofweek
    df["is_friday"] = (dow == 4).astype(int)
    df["is_monday"] = (dow == 0).astype(int)
    df["is_q4"] = (df["date"].dt.quarter == 4).astype(int)
    print(f"  + Calendar features added")

    # ------------------------------------------------------------------
    # 5. SECTOR
    # ------------------------------------------------------------------
    df["sector"] = df["ticker"].map(SECTOR_MAP).fillna("Other")

    # ------------------------------------------------------------------
    # CLEAN UP
    # ------------------------------------------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["target_log_var", "rv_lag_1"])

    # Define the final feature list
    technical_feats = ["rv_lag_1", "rv_lag_5", "rv_lag_22",
                       "returns_sq_lag_1", "VIX_close", "rsi_14"]

    news_feats = (["news_memory", "shock_memory", "sentiment_memory",
                   "shock_vix_memory", "sentiment_avg", "novelty_score",
                   "shock_index", "news_count"]
                  + sorted([c for c in df.columns if c.startswith("news_pca_")]))

    retail_feats = ["volume_shock", "volume_shock_roll3",
                    "hype_signal", "hype_signal_roll3", "hype_signal_roll7",
                    "hype_zscore", "price_acceleration"]

    calendar_feats = ["is_friday", "is_monday", "is_q4"]

    vol_feats = ["vol_ma5", "vol_ma10", "vol_std5"]

    all_features = []
    for group in [technical_feats, news_feats, retail_feats,
                  calendar_feats, vol_feats]:
        all_features.extend([f for f in group if f in df.columns])

    # Remove duplicates preserving order
    seen = set()
    feature_cols = []
    for f in all_features:
        if f not in seen:
            feature_cols.append(f)
            seen.add(f)

    print(f"\n  Total features: {len(feature_cols)}")
    print(f"    Technical:  {len([f for f in technical_feats if f in df.columns])}")
    print(f"    News:       {len([f for f in news_feats if f in df.columns])}")
    print(f"    Retail:     {len([f for f in retail_feats if f in df.columns])}")
    print(f"    Calendar:   {len(calendar_feats)}")
    print(f"    Vol/Mom:    {len(vol_feats)}")
    print(f"  Final rows: {len(df):,}")
    print("=" * 70)

    return df, feature_cols


def split_train_test(df, cutoff=TRAIN_CUTOFF):
    """Time-series split at cutoff date."""
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    return train, test


if __name__ == "__main__":
    df, feature_cols = assemble_full_features()
    train, test = split_train_test(df)
    print(f"\nTrain: {len(train):,}  Test: {len(test):,}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
