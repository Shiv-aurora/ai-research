"""
Retail Risk Signal Ingestion Pipeline

Fetches external market proxies to detect systemic retail mania:
- BTC-USD: Bitcoin volatility (Crypto/Degen risk appetite)
- GME: GameStop volume shock (Meme stock mania)
- IWM: Small Cap strength vs SPY (Risk-On sentiment)

These are GLOBAL signals - same values for all tickers on a given day.

Output: data/processed/retail_signals.parquet

Usage:
    python -m src.pipeline.ingest_retail
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def fetch_retail_signals(
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31",
    output_path: str = "data/processed/retail_signals.parquet"
) -> pd.DataFrame:
    """
    Fetch and engineer retail risk signals.
    
    Signals:
    - btc_vol_5d: Bitcoin 5-day volatility (Crypto risk)
    - btc_ret_5d: Bitcoin 5-day return (Crypto momentum)
    - gme_vol_shock: GameStop volume anomaly (Meme mania)
    - small_cap_excess: IWM/SPY ratio (Risk-on sentiment)
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        output_path: Path to save output
        
    Returns:
        DataFrame with retail risk signals
    """
    print("\n" + "=" * 65)
    print("📊 RETAIL RISK SIGNAL INGESTION")
    print("   Fetching Crypto, Meme, and Small Cap proxies")
    print("=" * 65)
    
    # Tickers to fetch
    tickers = {
        "BTC-USD": "Bitcoin",
        "GME": "GameStop",
        "IWM": "Small Caps",
        "SPY": "S&P 500 (for relative strength)"
    }
    
    print(f"\n📥 Fetching data from {start_date} to {end_date}...")
    
    # Fetch all data
    data = {}
    for ticker, name in tickers.items():
        print(f"   Fetching {ticker} ({name})...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if len(df) > 0:
                data[ticker] = df
                print(f"      ✓ {len(df)} rows")
            else:
                print(f"      ⚠️ No data returned")
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================
    print("\n🔧 Engineering retail risk features...")
    
    # Initialize master dataframe with SPY dates
    if "SPY" not in data:
        raise ValueError("SPY data required but not available")
    
    master = pd.DataFrame(index=data["SPY"].index)
    master.index.name = "date"
    
    # ----- BITCOIN SIGNALS -----
    if "BTC-USD" in data:
        btc = data["BTC-USD"]
        
        # Log returns
        btc["log_ret"] = np.log(btc["Close"] / btc["Close"].shift(1))
        
        # 5-day volatility (annualized)
        master["btc_vol_5d"] = btc["log_ret"].rolling(5).std() * np.sqrt(252)
        
        # 5-day return
        master["btc_ret_5d"] = btc["Close"].pct_change(5)
        
        # 20-day momentum
        master["btc_mom_20d"] = btc["Close"].pct_change(20)
        
        print("   ✓ Bitcoin signals: btc_vol_5d, btc_ret_5d, btc_mom_20d")
    else:
        print("   ⚠️ Bitcoin data unavailable - filling with 0")
        master["btc_vol_5d"] = 0
        master["btc_ret_5d"] = 0
        master["btc_mom_20d"] = 0
    
    # ----- GAMESTOP SIGNALS -----
    if "GME" in data:
        gme = data["GME"]
        
        # Volume shock (vs 50-day average)
        gme_vol_ma50 = gme["Volume"].rolling(50).mean()
        master["gme_vol_shock"] = gme["Volume"] / gme_vol_ma50
        
        # GME volatility
        gme["log_ret"] = np.log(gme["Close"] / gme["Close"].shift(1))
        master["gme_vol_5d"] = gme["log_ret"].rolling(5).std() * np.sqrt(252)
        
        # GME momentum
        master["gme_ret_5d"] = gme["Close"].pct_change(5)
        
        print("   ✓ GameStop signals: gme_vol_shock, gme_vol_5d, gme_ret_5d")
    else:
        print("   ⚠️ GME data unavailable - filling with 1")
        master["gme_vol_shock"] = 1
        master["gme_vol_5d"] = 0
        master["gme_ret_5d"] = 0
    
    # ----- SMALL CAP SIGNALS -----
    if "IWM" in data and "SPY" in data:
        iwm = data["IWM"]
        spy = data["SPY"]
        
        # Small cap relative strength (IWM / SPY)
        master["small_cap_excess"] = iwm["Close"] / spy["Close"]
        
        # Small cap relative momentum
        iwm_ret = iwm["Close"].pct_change(5)
        spy_ret = spy["Close"].pct_change(5)
        master["small_cap_mom"] = iwm_ret - spy_ret
        
        print("   ✓ Small cap signals: small_cap_excess, small_cap_mom")
    else:
        print("   ⚠️ IWM/SPY data unavailable")
        master["small_cap_excess"] = 1
        master["small_cap_mom"] = 0
    
    # ----- COMPOSITE SIGNALS -----
    print("\n🔧 Creating composite signals...")
    
    # Retail Mania Index: High BTC vol + High GME volume
    btc_std = max(master["btc_vol_5d"].std(), 0.001)
    gme_std = max(master["gme_vol_shock"].std(), 0.001)
    
    master["retail_mania"] = (
        master["btc_vol_5d"].fillna(0) / btc_std +
        master["gme_vol_shock"].fillna(1) / gme_std
    ) / 2
    
    # Risk-On Composite
    master["risk_on_signal"] = (
        master["btc_ret_5d"].fillna(0) + 
        master["small_cap_mom"].fillna(0)
    )
    
    print("   ✓ Composite signals: retail_mania, risk_on_signal")
    
    # =====================================================
    # CLEANUP
    # =====================================================
    print("\n🧹 Cleaning up...")
    
    # Handle infinities
    master = master.replace([np.inf, -np.inf], np.nan)
    
    # Forward fill then backfill for any gaps
    master = master.ffill().bfill()
    
    # Drop any remaining NaN rows
    before = len(master)
    master = master.dropna()
    after = len(master)
    print(f"   Dropped {before - after} NaN rows")
    
    # Reset index to make date a column
    master = master.reset_index()
    master["date"] = pd.to_datetime(master["date"]).dt.tz_localize(None)
    
    # =====================================================
    # OUTPUT
    # =====================================================
    print("\n📊 Output statistics:")
    for col in master.columns:
        if col != "date":
            print(f"   {col:20s}: mean={master[col].mean():>8.3f}, std={master[col].std():>8.3f}")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(output_path, index=False)
    
    print(f"\n✅ Saved retail signals to: {output_path}")
    print(f"   Shape: {master.shape}")
    print(f"   Date range: {master['date'].min()} to {master['date'].max()}")
    
    return master


def main():
    """Run the retail signal ingestion pipeline."""
    fetch_retail_signals()
    
    print("\n" + "=" * 65)
    print("🚀 RETAIL SIGNAL INGESTION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()

