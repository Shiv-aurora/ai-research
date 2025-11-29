"""
Reddit Proxy Pipeline: Volume/Price Anomaly Detection

Creates a proxy for retail hype using volume and price anomalies.
This serves as a substitute for actual Reddit/social media sentiment data.

Features:
- Volume Shock: Volume / 20-day rolling mean
- Price Acceleration: Second derivative of log price
- Hype Signal: Volume Shock * |Price Acceleration|
- Smoothed signals: 3-day and 7-day rolling means
- Z-Score normalization

Output: data/processed/reddit_proxy.parquet

Usage:
    python -m src.pipeline.create_reddit_proxy
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_reddit_proxy(input_path: str = "data/processed/targets.parquet",
                        output_path: str = "data/processed/reddit_proxy.parquet") -> pd.DataFrame:
    """
    Create retail hype proxy from volume/price anomalies.
    
    Args:
        input_path: Path to targets.parquet with price/volume data
        output_path: Path to save reddit_proxy.parquet
        
    Returns:
        DataFrame with hype signals
    """
    print("\n" + "=" * 65)
    print("📊 REDDIT PROXY PIPELINE")
    print("   Creating retail hype proxy from volume/price anomalies")
    print("=" * 65)
    
    # Load targets data
    print("\n📂 Loading targets data...")
    df = pd.read_parquet(input_path)
    
    # Normalize date
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    if df["ticker"].dtype.name == "category":
        df["ticker"] = df["ticker"].astype(str)
    
    # Sort for proper calculations
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    print(f"   Loaded {len(df):,} rows, {df['ticker'].nunique()} tickers")
    
    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================
    print("\n🔧 Engineering hype features...")
    
    # Check what columns we have
    print(f"   Available columns: {list(df.columns)}")
    
    # Volume column (might be 'volume' or 'Volume')
    vol_col = "volume" if "volume" in df.columns else "Volume"
    close_col = "close" if "close" in df.columns else "Close"
    
    if vol_col not in df.columns:
        print(f"   ⚠️ Volume column not found! Using realized_vol as proxy.")
        vol_col = "realized_vol"
    
    if close_col not in df.columns:
        print(f"   ⚠️ Close column not found! Creating from realized_vol.")
        # Create a synthetic price from cumulative returns
        df[close_col] = 100 * (1 + df.groupby("ticker")["realized_vol"].cumsum() * 0.01)
    
    # 1. Volume Shock: Volume / 20-day rolling mean
    print("   1. Calculating Volume Shock...")
    df["volume_ma20"] = df.groupby("ticker")[vol_col].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    df["volume_shock"] = df[vol_col] / df["volume_ma20"]
    df["volume_shock"] = df["volume_shock"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    # 2. Price Acceleration: Second derivative of log price
    print("   2. Calculating Price Acceleration...")
    df["log_close"] = np.log(df[close_col].clip(lower=0.01))
    df["log_return"] = df.groupby("ticker")["log_close"].diff()
    df["price_acceleration"] = df.groupby("ticker")["log_return"].diff()
    df["price_acceleration"] = df["price_acceleration"].fillna(0)
    
    # 3. Hype Signal: Volume Shock * |Price Acceleration|
    print("   3. Calculating Hype Signal...")
    df["hype_signal"] = df["volume_shock"] * df["price_acceleration"].abs()
    
    # 4. Smoothing: Rolling means
    print("   4. Creating smoothed signals...")
    df["hype_signal_roll3"] = df.groupby("ticker")["hype_signal"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df["hype_signal_roll7"] = df.groupby("ticker")["hype_signal"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    
    # 5. Z-Score normalization (per ticker)
    print("   5. Z-Score normalization...")
    df["hype_zscore"] = df.groupby("ticker")["hype_signal"].transform(
        lambda x: (x - x.mean()) / max(x.std(), 0.001)
    )
    df["hype_zscore"] = df["hype_zscore"].clip(-5, 5).fillna(0)
    
    # 6. Volume Shock rolling
    df["volume_shock_roll3"] = df.groupby("ticker")["volume_shock"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    
    # =====================================================
    # OUTPUT
    # =====================================================
    print("\n📊 Output statistics:")
    print(f"   Volume Shock:      mean={df['volume_shock'].mean():.3f}, std={df['volume_shock'].std():.3f}")
    print(f"   Hype Signal:       mean={df['hype_signal'].mean():.5f}, std={df['hype_signal'].std():.5f}")
    print(f"   Hype Z-Score:      mean={df['hype_zscore'].mean():.3f}, std={df['hype_zscore'].std():.3f}")
    
    # Select output columns
    output_cols = [
        "date", "ticker",
        "volume_shock", "volume_shock_roll3",
        "hype_signal", "hype_signal_roll3", "hype_signal_roll7",
        "hype_zscore", "price_acceleration"
    ]
    
    output_df = df[output_cols].copy()
    
    # Drop NaN rows
    before = len(output_df)
    output_df = output_df.dropna()
    after = len(output_df)
    print(f"\n   Dropped {before - after} NaN rows")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    
    print(f"\n✅ Saved reddit proxy to: {output_path}")
    print(f"   Shape: {output_df.shape}")
    print(f"   Columns: {list(output_df.columns)}")
    
    return output_df


def main():
    """Run the reddit proxy pipeline."""
    create_reddit_proxy()
    
    print("\n" + "=" * 65)
    print("🚀 REDDIT PROXY PIPELINE COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()

