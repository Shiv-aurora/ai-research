"""
Run Phase 10: Overnight News Split (Chronos Split)

This script implements the "Effective Date" strategy to fix causality issues
in the News Agent.

THEORY:
- News released during trading hours is priced in immediately
- Only AFTER-HOURS news (>= 4 PM ET) drives NEXT DAY volatility
- Pre-market news (< 9:30 AM ET) impacts CURRENT day

IMPLEMENTATION:
1. Run ingestion with effective_date calculation
2. Process news with grouping by effective_date
3. Train NewsAgent (LSTM + LightGBM) on causal data
4. Compare R² against non-causal baseline

Usage:
    python scripts/run_chronos_split.py
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor


def main():
    """Run the Chronos Split experiment."""
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🌙 PHASE 10: OVERNIGHT NEWS SPLIT (CHRONOS)")
    print("   Fixing Causality in News → Volatility Prediction")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: CHECK EXISTING DATA
    # =========================================================================
    print("\n" + "=" * 70)
    print("📂 STEP 1: Check Existing Data")
    print("=" * 70)
    
    news_base_path = Path("data/processed/news_base.parquet")
    
    if news_base_path.exists():
        news_base = pd.read_parquet(news_base_path)
        has_effective_date = 'effective_date' in news_base.columns
        print(f"\n   news_base.parquet: {len(news_base):,} rows")
        print(f"   Has effective_date: {has_effective_date}")
        
        if has_effective_date:
            print(f"\n   ✅ Effective dates already calculated")
            print(f"   Date range: {news_base['date'].min()} to {news_base['date'].max()}")
            print(f"   Effective date range: {news_base['effective_date'].min()} to {news_base['effective_date'].max()}")
        else:
            print("\n   ⚠️ No effective_date column found!")
            print("   Need to re-run ingestion with calculate_effective_date()")
            
            # Calculate effective dates
            print("\n   🔄 Calculating effective dates...")
            from src.pipeline.ingest_news import calculate_effective_date
            news_base = calculate_effective_date(news_base)
            
            # Save updated file
            news_base.to_parquet(news_base_path, index=False)
            print(f"   ✅ Updated news_base.parquet with effective_date")
    else:
        print(f"\n   ❌ news_base.parquet not found!")
        print(f"   Please run: python -m src.pipeline.ingest_news")
        return
    
    # =========================================================================
    # STEP 2: PROCESS NEWS WITH EFFECTIVE DATE
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔄 STEP 2: Process News with Effective Date")
    print("=" * 70)
    
    from src.pipeline.process_news import main as process_news_main
    process_news_main(mode="lite", use_effective_date=True)
    
    # Load the processed features
    news_features_path = Path("data/processed/news_features.parquet")
    news_features = pd.read_parquet(news_features_path)
    print(f"\n   ✅ news_features.parquet: {len(news_features):,} rows")
    
    # =========================================================================
    # STEP 3: PREPARE DATA FOR TRAINING
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 STEP 3: Prepare Training Data")
    print("=" * 70)
    
    # Load residuals (contains resid_tech - our target)
    residuals_path = Path("data/processed/residuals.parquet")
    if not residuals_path.exists():
        print("\n   ⚠️ residuals.parquet not found!")
        print("   Please run Technical Agent first.")
        return
    
    residuals = pd.read_parquet(residuals_path)
    print(f"\n   residuals.parquet: {len(residuals):,} rows")
    
    # Normalize dates
    news_features['date'] = pd.to_datetime(news_features['date']).dt.tz_localize(None)
    residuals['date'] = pd.to_datetime(residuals['date']).dt.tz_localize(None)
    
    if news_features['ticker'].dtype.name == 'category':
        news_features['ticker'] = news_features['ticker'].astype(str)
    if residuals['ticker'].dtype.name == 'category':
        residuals['ticker'] = residuals['ticker'].astype(str)
    
    # Merge
    df = pd.merge(news_features, residuals[['date', 'ticker', 'resid_tech']], 
                  on=['date', 'ticker'], how='inner')
    df = df.dropna(subset=['resid_tech'])
    
    print(f"   Merged dataset: {len(df):,} rows")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Define features
    pca_cols = [c for c in df.columns if c.startswith('news_pca_')]
    feature_cols = ['news_count', 'shock_index', 'sentiment_avg', 'novelty_score'] + pca_cols
    
    # Fill NaN in features
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    print(f"   Features: {len(feature_cols)} columns")
    
    # =========================================================================
    # STEP 4: TRAIN LIGHTGBM (BASELINE)
    # =========================================================================
    print("\n" + "=" * 70)
    print("🎯 STEP 4: Train LightGBM (Causal Data)")
    print("=" * 70)
    
    # Time-based split
    cutoff = pd.to_datetime("2023-01-01")
    train_mask = df['date'] < cutoff
    test_mask = df['date'] >= cutoff
    
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, 'resid_tech']
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, 'resid_tech']
    
    print(f"\n   Train: {len(X_train):,} rows ({train_mask.sum()/len(df)*100:.1f}%)")
    print(f"   Test:  {len(X_test):,} rows ({test_mask.sum()/len(df)*100:.1f}%)")
    
    # Train LightGBM
    lgbm = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.01,
        max_depth=3,
        num_leaves=8,
        random_state=42,
        verbose=-1
    )
    
    print("\n   Training LightGBM...")
    lgbm.fit(X_train.fillna(0), y_train)
    
    y_train_pred = lgbm.predict(X_train.fillna(0))
    y_test_pred = lgbm.predict(X_test.fillna(0))
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n   LightGBM Results (Causal):")
    print(f"      Train R²: {train_r2:.4f} ({train_r2*100:.2f}%)")
    print(f"      Test R²:  {test_r2:.4f} ({test_r2*100:.2f}%)")
    
    # =========================================================================
    # STEP 5: TRAIN LSTM (CAUSAL)
    # =========================================================================
    print("\n" + "=" * 70)
    print("🧠 STEP 5: Train Global LSTM (Causal Data)")
    print("=" * 70)
    
    from src.agents.news_lstm import NewsLSTMAgent
    
    lstm_agent = NewsLSTMAgent(
        seq_length=5,
        hidden_size=64,
        num_layers=2,
        embedding_dim=8,
        dropout=0.2,
        learning_rate=0.001,
        weight_decay=1e-5,
        batch_size=64,
        epochs=50,
        patience=10
    )
    
    # The LSTM agent needs to reload the processed data
    lstm_df = lstm_agent.load_and_process_data()
    lstm_metrics = lstm_agent.train(lstm_df)
    
    lstm_test_r2 = lstm_metrics['test']['R2']
    
    # =========================================================================
    # STEP 6: RESULTS COMPARISON
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 PHASE 10 RESULTS: CAUSALITY FIX")
    print("=" * 70)
    
    # Baseline results (non-causal)
    lgbm_baseline = -0.0138  # From Phase 7 (original date)
    lstm_baseline = 0.0048   # From Phase 9 (original date)
    
    print(f"\n   {'Model':<40} {'Before':<12} {'After':<12} {'Delta':>10}")
    print("   " + "-" * 76)
    print(f"   {'LightGBM News Agent':<40} {lgbm_baseline:>12.4f} {test_r2:>12.4f} {test_r2 - lgbm_baseline:>+10.4f}")
    print(f"   {'Global LSTM':<40} {lstm_baseline:>12.4f} {lstm_test_r2:>12.4f} {lstm_test_r2 - lstm_baseline:>+10.4f}")
    print("   " + "-" * 76)
    
    # Best result
    best_model = "LSTM" if lstm_test_r2 > test_r2 else "LightGBM"
    best_r2 = max(test_r2, lstm_test_r2)
    
    print(f"\n   Best Model: {best_model}")
    print(f"   Best Test R²: {best_r2:.4f} ({best_r2*100:.2f}%)")
    
    # Verdict
    print("\n" + "=" * 70)
    print("🏆 VERDICT")
    print("=" * 70)
    
    if best_r2 > 0.15:
        print(f"""
   ✅ SUCCESS! Causality fix achieves R² > 15%!
   
   Best Model: {best_model}
   Test R²: {best_r2:.4f} ({best_r2*100:.2f}%)
   
   The "Overnight News Split" strategy works:
   - After-hours news predicts next-day volatility
   - Same-day news is already priced in
        """)
    elif best_r2 > 0.05:
        print(f"""
   ⚠️ PARTIAL SUCCESS: R² improved but below 15%
   
   Best Model: {best_model}
   Test R²: {best_r2:.4f} ({best_r2*100:.2f}%)
   
   The causality fix helps, but news signal is weak.
        """)
    elif best_r2 > 0:
        print(f"""
   ⚠️ MARGINAL: Positive R² but very weak
   
   Best Model: {best_model}
   Test R²: {best_r2:.4f} ({best_r2*100:.2f}%)
   
   News has limited predictive power for next-day residuals.
        """)
    else:
        print(f"""
   ❌ FAILED: Causality fix did not help
   
   Best Model: {best_model}
   Test R²: {best_r2:.4f} ({best_r2*100:.2f}%)
   
   Even with causal data, news cannot predict HAR residuals.
   
   CONCLUSION:
   - News affects SAME-DAY volatility (40% R²)
   - But does NOT predict NEXT-DAY residuals
   - HAR already captures most predictable volatility
        """)
    
    # Timing
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n   Duration: {duration}")
    print(f"   Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return {
        'lgbm_r2': test_r2,
        'lstm_r2': lstm_test_r2,
        'best_model': best_model,
        'best_r2': best_r2
    }


if __name__ == "__main__":
    main()

