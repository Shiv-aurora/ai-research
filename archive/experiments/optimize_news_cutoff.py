"""
Intraday Cutoff Optimization

Grid search to find the optimal hour to split "Today's News" from "Tomorrow's News".

THEORY:
- Earlier cutoff (10 AM) = more news counts as "next-day" (more causal)
- Later cutoff (4 PM) = standard market close definition
- The optimal cutoff balances information availability vs causality

We test cutoffs from 10 AM to 4 PM ET and measure which
produces the best predictive signal for next-day volatility residuals.

Usage:
    python scripts/optimize_news_cutoff.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytz
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from tqdm import tqdm


# Cutoff hours to test (Eastern Time)
CUTOFF_HOURS = [10, 11, 12, 13, 14, 15, 16, 17, 18]

# Timezone
ET = pytz.timezone('US/Eastern')


def calculate_effective_date_for_cutoff(df: pd.DataFrame, cutoff_hour: int) -> pd.DataFrame:
    """
    Calculate effective date based on a specific cutoff hour.
    
    Args:
        df: DataFrame with 'date' column (UTC datetime)
        cutoff_hour: Hour in Eastern Time to split today/tomorrow
    
    Returns:
        DataFrame with 'effective_date' column
    """
    df = df.copy()
    
    # Ensure date is timezone-aware UTC
    if df['date'].dt.tz is None:
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize('UTC')
    else:
        df['date'] = pd.to_datetime(df['date']).dt.tz_convert('UTC')
    
    # Convert to Eastern Time
    df['datetime_et'] = df['date'].dt.tz_convert(ET)
    df['hour_et'] = df['datetime_et'].dt.hour
    
    # Calculate effective date based on cutoff
    def get_effective_date(row):
        dt_et = row['datetime_et']
        date_part = dt_et.date()
        
        if dt_et.hour >= cutoff_hour:
            # After cutoff → next trading day
            next_day = date_part + timedelta(days=1)
            # Skip Saturday → Monday
            if next_day.weekday() == 5:
                next_day = next_day + timedelta(days=2)
            # Skip Sunday → Monday
            elif next_day.weekday() == 6:
                next_day = next_day + timedelta(days=1)
            return pd.Timestamp(next_day)
        else:
            # Before cutoff → same day
            return pd.Timestamp(date_part)
    
    df['effective_date'] = df.apply(get_effective_date, axis=1)
    
    # Clean up
    df = df.drop(columns=['datetime_et', 'hour_et'])
    
    return df


def aggregate_news_fast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast aggregation using only scalar signals.
    
    Aggregates:
    - news_count: Number of articles
    - shock_index: Sum of shock scores
    - sentiment_sum: Sum of sentiment (proxy for net sentiment)
    """
    # Ensure date is date-only
    df = df.copy()
    df['agg_date'] = pd.to_datetime(df['effective_date']).dt.date
    df['agg_date'] = pd.to_datetime(df['agg_date'])
    
    # Aggregate
    agg_dict = {
        'raw_text': 'count',  # news_count
    }
    
    # Add shock_score if available
    if 'shock_score' in df.columns:
        agg_dict['shock_score'] = 'sum'
    
    # Add sentiment if available
    if 'sentiment_score' in df.columns:
        agg_dict['sentiment_score'] = ['sum', 'mean']
    
    daily = df.groupby(['agg_date', 'ticker']).agg(agg_dict).reset_index()
    
    # Flatten column names
    daily.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                     for col in daily.columns]
    
    # Rename
    rename_dict = {
        'agg_date': 'date',
        'raw_text_count': 'news_count',
        'raw_text': 'news_count',
        'shock_score_sum': 'shock_index',
        'shock_score': 'shock_index',
        'sentiment_score_sum': 'sentiment_sum',
        'sentiment_score_mean': 'sentiment_avg'
    }
    
    for old, new in rename_dict.items():
        if old in daily.columns:
            daily = daily.rename(columns={old: new})
    
    return daily


def train_and_evaluate(features_df: pd.DataFrame, residuals_df: pd.DataFrame) -> float:
    """
    Train Ridge regression and return Test R².
    """
    # Merge
    df = pd.merge(features_df, residuals_df[['date', 'ticker', 'resid_tech']], 
                  on=['date', 'ticker'], how='inner')
    
    df = df.dropna(subset=['resid_tech'])
    
    if len(df) < 100:
        return np.nan
    
    # Features
    feature_cols = ['news_count']
    if 'shock_index' in df.columns:
        feature_cols.append('shock_index')
    if 'sentiment_sum' in df.columns:
        feature_cols.append('sentiment_sum')
    if 'sentiment_avg' in df.columns:
        feature_cols.append('sentiment_avg')
    
    # Fill NaN
    for col in feature_cols:
        df[col] = df[col].fillna(0)
    
    # Time split
    cutoff = pd.to_datetime("2023-01-01")
    train_mask = df['date'] < cutoff
    test_mask = df['date'] >= cutoff
    
    if train_mask.sum() < 50 or test_mask.sum() < 50:
        return np.nan
    
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, 'resid_tech']
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, 'resid_tech']
    
    # Train Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    return r2


def main():
    """Run the cutoff optimization."""
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("⏰ INTRADAY CUTOFF OPTIMIZATION")
    print("   Finding the Optimal Hour to Split Today/Tomorrow News")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n📂 Loading data...")
    
    # Load news base
    news_base_path = Path("data/processed/news_base.parquet")
    if not news_base_path.exists():
        print(f"   ❌ {news_base_path} not found!")
        return
    
    news_df = pd.read_parquet(news_base_path)
    print(f"   ✓ news_base.parquet: {len(news_df):,} rows")
    
    # Load residuals
    residuals_path = Path("data/processed/residuals.parquet")
    if not residuals_path.exists():
        print(f"   ❌ {residuals_path} not found!")
        return
    
    residuals_df = pd.read_parquet(residuals_path)
    print(f"   ✓ residuals.parquet: {len(residuals_df):,} rows")
    
    # Normalize dates in residuals
    residuals_df['date'] = pd.to_datetime(residuals_df['date']).dt.tz_localize(None)
    if residuals_df['ticker'].dtype.name == 'category':
        residuals_df['ticker'] = residuals_df['ticker'].astype(str)
    
    # Calculate shock scores if not present
    if 'shock_score' not in news_df.columns:
        print("\n   📊 Calculating shock scores...")
        from src.pipeline.process_news import calculate_shock_score
        news_df['shock_score'] = news_df['raw_text'].apply(calculate_shock_score)
    
    # =========================================================================
    # ANALYZE HOUR DISTRIBUTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 HOUR DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # Convert to ET for analysis
    if news_df['date'].dt.tz is None:
        news_df['date'] = pd.to_datetime(news_df['date']).dt.tz_localize('UTC')
    
    news_df['hour_et'] = news_df['date'].dt.tz_convert(ET).dt.hour
    
    hour_counts = news_df['hour_et'].value_counts().sort_index()
    
    print(f"\n   {'Hour (ET)':<12} {'Count':>10} {'% Total':>10} {'Cumulative':>12}")
    print("   " + "-" * 46)
    
    cumulative = 0
    for hour in range(6, 24):
        count = hour_counts.get(hour, 0)
        pct = 100 * count / len(news_df)
        cumulative += count
        cum_pct = 100 * cumulative / len(news_df)
        marker = " ◀ Market Open" if hour == 9 else " ◀ Market Close" if hour == 16 else ""
        print(f"   {hour:02d}:00        {count:>10,} {pct:>9.1f}% {cum_pct:>10.1f}%{marker}")
    
    # =========================================================================
    # GRID SEARCH
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔍 GRID SEARCH: Testing Cutoff Hours")
    print("=" * 70)
    
    results = []
    
    for cutoff_hour in tqdm(CUTOFF_HOURS, desc="Testing cutoffs"):
        # Calculate effective dates for this cutoff
        news_with_dates = calculate_effective_date_for_cutoff(news_df.copy(), cutoff_hour)
        
        # Count how much news goes to "next day"
        next_day_pct = (news_with_dates['effective_date'] > 
                        pd.to_datetime(news_with_dates['date']).dt.date.apply(pd.Timestamp)).mean()
        
        # Aggregate
        features = aggregate_news_fast(news_with_dates)
        
        # Train and evaluate
        test_r2 = train_and_evaluate(features, residuals_df)
        
        results.append({
            'cutoff_hour': cutoff_hour,
            'cutoff_label': f"{cutoff_hour:02d}:00 ET",
            'next_day_pct': next_day_pct,
            'test_r2': test_r2,
            'test_r2_pct': test_r2 * 100 if not np.isnan(test_r2) else np.nan
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_r2', ascending=False)
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 RESULTS: Cutoff Hour vs Predictive Power")
    print("=" * 70)
    
    print(f"\n   {'Cutoff':<12} {'→Next Day':>12} {'Test R²':>12} {'Status':>10}")
    print("   " + "-" * 48)
    
    best_r2 = results_df['test_r2'].max()
    
    for _, row in results_df.sort_values('cutoff_hour').iterrows():
        is_best = row['test_r2'] == best_r2
        status = "🏆 BEST" if is_best else ""
        print(f"   {row['cutoff_label']:<12} {row['next_day_pct']:>11.1%} {row['test_r2_pct']:>11.2f}% {status:>10}")
    
    # Winner
    winner = results_df.iloc[0]
    
    print("\n" + "=" * 70)
    print("🏆 OPTIMAL CUTOFF FOUND")
    print("=" * 70)
    
    print(f"""
   Winner: {winner['cutoff_label']}
   Test R²: {winner['test_r2_pct']:.4f}%
   
   This means:
   - News published AFTER {winner['cutoff_hour']:02d}:00 ET → impacts NEXT day
   - News published BEFORE {winner['cutoff_hour']:02d}:00 ET → impacts SAME day
   - {winner['next_day_pct']:.1%} of news shifts to next-day
    """)
    
    # Comparison with baseline (4 PM)
    baseline_row = results_df[results_df['cutoff_hour'] == 16].iloc[0] if 16 in results_df['cutoff_hour'].values else None
    
    if baseline_row is not None:
        delta = winner['test_r2'] - baseline_row['test_r2']
        print(f"   Improvement over 16:00 baseline: {delta*100:+.4f}%")
    
    # =========================================================================
    # VISUALIZATION (ASCII)
    # =========================================================================
    print("\n" + "=" * 70)
    print("📈 R² BY CUTOFF HOUR (ASCII Chart)")
    print("=" * 70)
    
    # Normalize for chart
    sorted_results = results_df.sort_values('cutoff_hour')
    min_r2 = sorted_results['test_r2'].min()
    max_r2 = sorted_results['test_r2'].max()
    range_r2 = max_r2 - min_r2 if max_r2 != min_r2 else 1
    
    print()
    for _, row in sorted_results.iterrows():
        # Normalize to 0-40 bar width
        if np.isnan(row['test_r2']):
            bar_len = 0
        else:
            bar_len = int(40 * (row['test_r2'] - min_r2) / range_r2) if range_r2 > 0 else 20
        
        bar = "█" * max(bar_len, 1)
        marker = " ★" if row['test_r2'] == best_r2 else ""
        print(f"   {row['cutoff_hour']:02d}:00 |{bar} {row['test_r2_pct']:+.3f}%{marker}")
    
    print()
    
    # =========================================================================
    # RECOMMENDATION
    # =========================================================================
    print("=" * 70)
    print("📝 RECOMMENDATION")
    print("=" * 70)
    
    print(f"""
   To use the optimal cutoff, update:
   
   src/pipeline/ingest_news.py:
   
       MARKET_CLOSE_HOUR = {winner['cutoff_hour']}  # Optimal cutoff (was 16)
   
   Or run scripts/run_chronos_split.py with the new cutoff.
    """)
    
    # Timing
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n   Duration: {duration}")
    print(f"   Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return results_df


if __name__ == "__main__":
    main()

