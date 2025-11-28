"""
Core Data Ingestion Pipeline for Titan V8
Multi-Agent Volatility Prediction Research Project

This module fetches historical price data from Polygon.io and generates
the "Ground Truth" target variable (next-day realized volatility).

Usage:
    python -m src.pipeline.ingest
"""

import os
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from polygon import RESTClient
from tqdm import tqdm


def load_config(config_path: str = "conf/base/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_polygon_client() -> RESTClient:
    """Initialize Polygon.io REST client from environment variable."""
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise ValueError(
            "POLYGON_API_KEY not found in environment. "
            "Please set it in your .env file."
        )
    return RESTClient(api_key)


def fetch_5min_bars(
    client: RESTClient,
    ticker: str,
    start_date: str,
    end_date: str,
    sleep_time: float = 0.15
) -> pd.DataFrame:
    """
    Fetch 5-minute OHLCV bars for a single ticker.
    
    Args:
        client: Polygon REST client
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        sleep_time: Delay between API calls to respect rate limits
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume, vwap, n_trades
    """
    bars = []
    
    try:
        # Polygon's list_aggs returns an iterator
        for bar in client.list_aggs(
            ticker=ticker,
            multiplier=5,
            timespan="minute",
            from_=start_date,
            to=end_date,
            limit=50000
        ):
            bars.append({
                "timestamp": pd.Timestamp(bar.timestamp, unit="ms", tz="UTC"),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "vwap": getattr(bar, "vwap", None),
                "n_trades": getattr(bar, "transactions", None)
            })
        
        # Respect rate limits
        time.sleep(sleep_time)
        
    except Exception as e:
        print(f"  ⚠️ Error fetching {ticker}: {e}")
        return pd.DataFrame()
    
    if not bars:
        return pd.DataFrame()
    
    df = pd.DataFrame(bars)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df


def calculate_realized_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily realized volatility from 5-minute bars.
    
    Math:
        1. Log Returns: r_t = ln(Close_t / Close_{t-1})
        2. Realized Variance: RV_daily = sum(r_5min^2) for each day
        3. Realized Volatility: sqrt(RV_daily)
    
    Args:
        df: DataFrame with 5-minute bars (must have 'timestamp' and 'close')
    
    Returns:
        DataFrame with daily realized volatility metrics
    """
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Calculate 5-minute log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    
    # Extract date for grouping
    df["date"] = df["timestamp"].dt.date
    
    # Group by date and calculate daily metrics
    daily = df.groupby("date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        # Realized Variance: sum of squared log returns
        realized_var=("log_return", lambda x: (x ** 2).sum()),
        # Number of 5-min bars (for data quality check)
        n_bars=("close", "count")
    ).reset_index()
    
    # Realized Volatility = sqrt(Realized Variance)
    daily["realized_vol"] = np.sqrt(daily["realized_var"])
    
    # Annualized RV (252 trading days)
    daily["realized_vol_annual"] = daily["realized_vol"] * np.sqrt(252)
    
    # Convert date to datetime
    daily["date"] = pd.to_datetime(daily["date"])
    
    return daily


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the target variable: Next Day's Log Variance.
    
    Target: target_log_var = ln(RV_{t+1}^2)
    
    CRUCIAL: No look-ahead bias. Row T's target is day T+1's volatility.
    We shift RV backward by 1 day to align properly.
    
    Args:
        df: DataFrame with daily realized volatility
    
    Returns:
        DataFrame with target variable added
    """
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    # Next day's realized variance (shift -1 brings future value to current row)
    df["next_day_rv"] = df["realized_vol"].shift(-1)
    
    # Target: log of next day's variance
    # RV^2 = Variance, then take log for better distribution
    df["target_log_var"] = np.log(df["next_day_rv"] ** 2)
    
    # Also create lagged features for potential use
    df["prev_day_rv"] = df["realized_vol"].shift(1)
    df["rv_5d_mean"] = df["realized_vol"].rolling(window=5, min_periods=1).mean()
    df["rv_20d_mean"] = df["realized_vol"].rolling(window=20, min_periods=1).mean()
    
    return df


# =============================================================================
# EVENT DATA: Dividends
# =============================================================================

def fetch_dividends(
    client: RESTClient,
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Fetch dividend data for a ticker and create dividend-related features.
    
    Features created:
        - is_ex_dividend: 1 if date is an ex-dividend date, 0 otherwise
        - days_to_ex_div: Days until the next ex-dividend date (forward looking)
    
    Args:
        client: Polygon REST client
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with date, is_ex_dividend, days_to_ex_div
    """
    try:
        dividends = []
        for div in client.list_dividends(ticker=ticker, limit=1000):
            ex_date = getattr(div, "ex_dividend_date", None)
            if ex_date:
                dividends.append({"ex_dividend_date": pd.to_datetime(ex_date)})
        
        time.sleep(0.12)
        
        if not dividends:
            return pd.DataFrame()
        
        div_df = pd.DataFrame(dividends)
        div_df = div_df.drop_duplicates(subset=["ex_dividend_date"])
        
        # Create date range for the full period
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        result_df = pd.DataFrame({"date": date_range})
        
        # Mark ex-dividend dates
        ex_dates_set = set(div_df["ex_dividend_date"].dt.normalize())
        result_df["is_ex_dividend"] = result_df["date"].apply(
            lambda x: 1 if x in ex_dates_set else 0
        )
        
        # Calculate days to next ex-dividend (forward looking)
        sorted_ex_dates = sorted(div_df["ex_dividend_date"].tolist())
        
        def days_to_next_ex_div(current_date):
            for ex_date in sorted_ex_dates:
                if ex_date > current_date:
                    return (ex_date - current_date).days
            return np.nan  # No future ex-div date
        
        result_df["days_to_ex_div"] = result_df["date"].apply(days_to_next_ex_div)
        
        return result_df
        
    except Exception as e:
        # Silently handle - some tickers don't have dividends
        return pd.DataFrame()


def fetch_macro_context(
    client: RESTClient,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Fetch SPY and VIX data for macro context features.
    
    - SPY: Fetched from Polygon (5-min bars → realized volatility)
    - VIX: Fetched from yfinance (free, no premium required)
    
    Args:
        client: Polygon REST client
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with SPY_close, SPY_vol, VIX_close indexed by date
    """
    macro_data = {}
    
    # =========================================
    # 1. Fetch SPY from Polygon (5-min bars)
    # =========================================
    print("  📊 Fetching SPY from Polygon...")
    spy_5min = fetch_5min_bars(client, "SPY", start_date, end_date)
    if not spy_5min.empty:
        spy_daily = calculate_realized_volatility(spy_5min)
        macro_data["SPY_close"] = spy_daily.set_index("date")["close"]
        macro_data["SPY_vol"] = spy_daily.set_index("date")["realized_vol"]
        print(f"    ✓ SPY: {len(spy_daily)} days")
    else:
        print("    ⚠️ No SPY data fetched")
    
    # =========================================
    # 2. Fetch VIX from yfinance (free!)
    # =========================================
    print("  📊 Fetching VIX from yfinance...")
    try:
        # yfinance uses ^VIX for the CBOE Volatility Index
        vix = yf.Ticker("^VIX")
        vix_df = vix.history(start=start_date, end=end_date, auto_adjust=True)
        
        if not vix_df.empty:
            # Reset index to get date as column
            vix_df = vix_df.reset_index()
            
            # Rename columns
            vix_df = vix_df.rename(columns={"Date": "date", "Close": "VIX_close"})
            
            # Keep only date and VIX_close
            vix_df = vix_df[["date", "VIX_close"]]
            
            # Ensure date is datetime (remove timezone if present)
            vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.tz_localize(None)
            
            # Remove duplicates
            vix_df = vix_df.drop_duplicates(subset=["date"])
            
            # Add to macro_data
            macro_data["VIX_close"] = vix_df.set_index("date")["VIX_close"]
            print(f"    ✓ VIX: {len(vix_df)} days")
        else:
            print("    ⚠️ No VIX data returned from yfinance")
            
    except Exception as e:
        print(f"    ⚠️ Could not fetch VIX from yfinance: {e}")
    
    # =========================================
    # 3. Combine into single DataFrame
    # =========================================
    if macro_data:
        macro_df = pd.DataFrame(macro_data)
        macro_df.index.name = "date"
        return macro_df.reset_index()
    
    return pd.DataFrame()


def fetch_market_data(config: dict) -> pd.DataFrame:
    """
    Main function to fetch and process all market data.
    
    This function:
    1. Fetches 5-minute bars for all tickers
    2. Calculates realized volatility
    3. Creates target variable (next-day log variance)
    4. Merges macro context (SPY, VIX)
    5. Returns a global DataFrame with Ticker column
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Global DataFrame with all tickers and features
    """
    client = get_polygon_client()
    
    tickers = config["data"]["tickers"]
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    
    print(f"\n{'='*60}")
    print("🚀 TITAN V8 DATA INGESTION PIPELINE")
    print(f"{'='*60}")
    print(f"Tickers: {len(tickers)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}\n")
    
    # Fetch macro context first (SPY, VIX)
    print("📈 Fetching Macro Context...")
    macro_df = fetch_macro_context(client, start_date, end_date)
    print(f"  ✓ Macro data: {len(macro_df)} days\n")
    
    # Process each ticker
    all_data = []
    
    print("📊 Processing Individual Tickers...")
    for ticker in tqdm(tickers, desc="Fetching tickers"):
        # Skip SPY if already in macro (we'll use it for context)
        # But still process it as a target ticker
        
        # Fetch 5-minute bars
        df_5min = fetch_5min_bars(client, ticker, start_date, end_date)
        
        if df_5min.empty:
            print(f"  ⚠️ No data for {ticker}, skipping...")
            continue
        
        # Calculate daily realized volatility
        df_daily = calculate_realized_volatility(df_5min)
        
        if df_daily.empty:
            continue
        
        # Create target variable
        df_daily = create_target_variable(df_daily)
        
        # Add ticker column
        df_daily["ticker"] = ticker
        
        all_data.append(df_daily)
    
    if not all_data:
        raise ValueError("No data fetched for any ticker!")
    
    # Combine all tickers
    print("\n🔗 Combining all ticker data...")
    global_df = pd.concat(all_data, ignore_index=True)
    
    # Merge macro context
    if not macro_df.empty:
        print("🔗 Merging macro context (SPY, VIX)...")
        global_df = global_df.merge(
            macro_df,
            on="date",
            how="left"
        )
    
    # Make ticker categorical for memory efficiency
    global_df["ticker"] = global_df["ticker"].astype("category")
    
    # Sort by ticker and date
    global_df = global_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # Drop rows where target is NaN (last day for each ticker)
    n_before = len(global_df)
    global_df = global_df.dropna(subset=["target_log_var"])
    n_after = len(global_df)
    print(f"  ✓ Dropped {n_before - n_after} rows with missing target")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("📊 DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows: {len(global_df):,}")
    print(f"Unique tickers: {global_df['ticker'].nunique()}")
    print(f"Date range: {global_df['date'].min()} to {global_df['date'].max()}")
    print(f"Columns: {list(global_df.columns)}")
    print(f"{'='*60}\n")
    
    return global_df


def save_to_parquet(df: pd.DataFrame, output_path: str) -> None:
    """Save DataFrame to Parquet format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, index=False, engine="pyarrow")
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"💾 Saved to: {output_path} ({file_size_mb:.2f} MB)")


def main():
    """Main entry point for the ingestion pipeline."""
    # Load configuration
    config = load_config()
    
    # Fetch and process all data
    global_df = fetch_market_data(config)
    
    # Save to parquet
    output_path = Path(config["data"]["processed_path"]) / "targets.parquet"
    save_to_parquet(global_df, output_path)
    
    print("\n✅ Data ingestion complete!")
    print("Next step: Run feature engineering pipeline.")


if __name__ == "__main__":
    main()

