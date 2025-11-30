"""
Target De-Seasonalization Pipeline (Phase 7)

Transforms the target variable to model "Excess Volatility" (anomalies)
instead of "Total Volatility" to remove calendar effects.

Key Insight:
- The Friday effect dominates the model at 40.6%
- By removing the predictable calendar component, we force the model
  to focus on the NEWS-driven deviations

Approach:
1. Calculate Median volatility per (ticker, day_of_week) - robust to crashes
2. target_excess = target_log_var - seasonal_component
3. Train agents on target_excess (anomalies only)
4. Re-seasonalize predictions for final evaluation

Output:
- data/processed/targets_deseasonalized.parquet
- models/seasonality_map.json

Usage:
    python -m src.pipeline.deseasonalize
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def deseasonalize_targets(
    input_path: str = "data/processed/targets.parquet",
    output_path: str = "data/processed/targets_deseasonalized.parquet",
    map_path: str = "models/seasonality_map.json"
) -> pd.DataFrame:
    """
    Remove calendar seasonality from target variable.
    
    Creates target_excess = target_log_var - seasonal_component
    where seasonal_component is the median volatility for each (ticker, day_of_week).
    
    Args:
        input_path: Path to original targets.parquet
        output_path: Path to save deseasonalized targets
        map_path: Path to save seasonality map JSON
        
    Returns:
        DataFrame with deseasonalized targets
    """
    print("\n" + "=" * 70)
    print("🔧 TARGET DE-SEASONALIZATION (Phase 7)")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading targets...")
    df = pd.read_parquet(input_path)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    if df['ticker'].dtype.name == 'category':
        df['ticker'] = df['ticker'].astype(str)
    
    # Clean infinity values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['target_log_var'])
    
    print(f"   Loaded {len(df):,} rows, {df['ticker'].nunique()} tickers")
    
    # Add day of week
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_name'] = df['date'].dt.day_name()
    
    # =====================================================
    # CALCULATE SEASONALITY (Median per ticker/day)
    # =====================================================
    print("\n📊 Calculating seasonal components...")
    print("   Using MEDIAN (robust to crash outliers)")
    
    # Calculate median for each (ticker, day_of_week)
    seasonal_stats = df.groupby(['ticker', 'day_of_week'])['target_log_var'].agg([
        'median', 'mean', 'std', 'count'
    ]).reset_index()
    
    # Create the seasonality map
    seasonal_map = {}
    for _, row in seasonal_stats.iterrows():
        ticker = row['ticker']
        dow = int(row['day_of_week'])
        median_val = row['median']
        
        if ticker not in seasonal_map:
            seasonal_map[ticker] = {}
        seasonal_map[ticker][dow] = median_val
    
    # Show example
    print(f"\n   Sample Seasonal Map (AAPL):")
    if 'AAPL' in seasonal_map:
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        for dow in range(5):
            val = seasonal_map['AAPL'].get(dow, 0)
            print(f"      {day_names[dow]:12}: {val:.4f}")
    
    # =====================================================
    # TRANSFORM: Create target_excess
    # =====================================================
    print("\n🔧 Transforming targets...")
    
    # Map seasonal component to each row
    def get_seasonal(row):
        ticker = row['ticker']
        dow = row['day_of_week']
        return seasonal_map.get(ticker, {}).get(dow, row['target_log_var'])
    
    df['seasonal_component'] = df.apply(get_seasonal, axis=1)
    df['target_excess'] = df['target_log_var'] - df['seasonal_component']
    
    # Stats
    print(f"\n   📊 Target Statistics:")
    print(f"      Original (target_log_var):")
    print(f"         Mean: {df['target_log_var'].mean():.4f}")
    print(f"         Std:  {df['target_log_var'].std():.4f}")
    print(f"      Seasonal Component:")
    print(f"         Mean: {df['seasonal_component'].mean():.4f}")
    print(f"         Std:  {df['seasonal_component'].std():.4f}")
    print(f"      Excess (target_excess):")
    print(f"         Mean: {df['target_excess'].mean():.4f}")
    print(f"         Std:  {df['target_excess'].std():.4f}")
    
    # Variance decomposition
    var_original = df['target_log_var'].var()
    var_seasonal = df['seasonal_component'].var()
    var_excess = df['target_excess'].var()
    
    pct_seasonal = (var_seasonal / var_original) * 100
    pct_excess = (var_excess / var_original) * 100
    
    print(f"\n   📊 Variance Decomposition:")
    print(f"      Total Variance:    {var_original:.4f} (100%)")
    print(f"      Seasonal Variance: {var_seasonal:.4f} ({pct_seasonal:.1f}%)")
    print(f"      Excess Variance:   {var_excess:.4f} ({pct_excess:.1f}%)")
    
    # =====================================================
    # SAVE OUTPUTS
    # =====================================================
    
    # Save seasonality map
    map_dir = Path(map_path).parent
    map_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert int keys to strings for JSON
    json_map = {ticker: {str(k): v for k, v in days.items()} 
                for ticker, days in seasonal_map.items()}
    
    with open(map_path, 'w') as f:
        json.dump(json_map, f, indent=2)
    print(f"\n💾 Saved seasonality map to: {map_path}")
    
    # Save deseasonalized targets
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"💾 Saved to: {output_path} ({file_size_mb:.2f} MB)")
    
    return df


def load_seasonality_map(map_path: str = "models/seasonality_map.json") -> dict:
    """Load the seasonality map from JSON."""
    with open(map_path, 'r') as f:
        json_map = json.load(f)
    
    # Convert string keys back to int
    return {ticker: {int(k): v for k, v in days.items()} 
            for ticker, days in json_map.items()}


def reseasonalize_predictions(
    predictions: pd.Series,
    df: pd.DataFrame,
    seasonal_map: dict = None,
    map_path: str = "models/seasonality_map.json"
) -> pd.Series:
    """
    Add seasonal component back to predictions.
    
    Args:
        predictions: Excess predictions from model
        df: DataFrame with ticker and date columns
        seasonal_map: Pre-loaded seasonal map (optional)
        map_path: Path to load seasonal map from
        
    Returns:
        Re-seasonalized predictions
    """
    if seasonal_map is None:
        seasonal_map = load_seasonality_map(map_path)
    
    # Get day of week
    dow = df['date'].dt.dayofweek
    
    # Map seasonal component
    def get_seasonal(row):
        ticker = row['ticker']
        d = row['day_of_week']
        return seasonal_map.get(ticker, {}).get(d, 0)
    
    temp_df = df.copy()
    temp_df['day_of_week'] = dow
    seasonal = temp_df.apply(get_seasonal, axis=1)
    
    return predictions + seasonal


def main():
    """Run the de-seasonalization pipeline."""
    print("\n" + "=" * 70)
    print("🚀 PHASE 7: TARGET DE-SEASONALIZATION")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Run transformation
    df = deseasonalize_targets()
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ DE-SEASONALIZATION COMPLETE")
    print("=" * 70)
    
    print(f"\n   📊 Output Summary:")
    print(f"      Rows: {len(df):,}")
    print(f"      Tickers: {df['ticker'].nunique()}")
    print(f"      New columns: seasonal_component, target_excess")
    
    print("\n   📋 Next Steps:")
    print("      1. Update TechnicalAgent to train on target_excess")
    print("      2. Update NewsAgent with de-seasonalized features")
    print("      3. Re-seasonalize coordinator predictions for eval")
    
    print("\n" + "=" * 70)
    
    return df


if __name__ == "__main__":
    main()

