"""
Phase 7: Target De-Seasonalization Pipeline

Fixes NewsAgent underperformance by:
1. Removing calendar seasonality from targets
2. Training agents on "Excess Volatility" (anomalies)
3. Re-seasonalizing predictions for final evaluation

Expected Outcomes:
- NewsAgent R² should turn positive (was -2.72%)
- is_friday importance should decrease significantly
- Final R² should remain competitive or improve

Usage:
    python scripts/run_phase7.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import r2_score, mean_squared_error

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')


# Sector mapping
SECTOR_MAP = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech',
    'JPM': 'Finance', 'BAC': 'Finance', 'V': 'Finance',
    'CAT': 'Industrial', 'GE': 'Industrial', 'BA': 'Industrial',
    'WMT': 'Consumer', 'MCD': 'Consumer', 'COST': 'Consumer',
    'XOM': 'Energy', 'CVX': 'Energy', 'SLB': 'Energy',
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
}


def step_1_deseasonalize():
    """Run the de-seasonalization pipeline."""
    print("\n" + "=" * 70)
    print("📊 STEP 1: TARGET DE-SEASONALIZATION")
    print("=" * 70)
    
    from src.pipeline.deseasonalize import deseasonalize_targets
    
    df = deseasonalize_targets()
    
    return df


def step_2_train_technical(use_deseasonalized=True):
    """Train TechnicalAgent on excess volatility."""
    print("\n" + "=" * 70)
    print("🔧 STEP 2: TRAIN TECHNICAL AGENT (EXCESS MODE)")
    print("=" * 70)
    
    # End any stale runs
    mlflow.end_run()
    
    from src.agents.technical_agent import TechnicalAgent
    
    agent = TechnicalAgent(
        experiment_name="titan_v8_phase7",
        use_deseasonalized=use_deseasonalized
    )
    
    df = agent.load_and_process_data()
    metrics = agent.train(df)
    agent.save_residuals()
    
    print(f"\n   ✅ TechnicalAgent Test R²: {metrics['test']['R2']:.4f}")
    
    mlflow.end_run()
    
    return agent, metrics


def step_3_train_news(use_deseasonalized=True):
    """Train NewsAgent on clean residuals with de-seasonalized features."""
    print("\n" + "=" * 70)
    print("🔧 STEP 3: TRAIN NEWS AGENT (DE-SEASONALIZED)")
    print("=" * 70)
    
    # End any stale runs
    mlflow.end_run()
    
    from src.agents.news_agent import NewsAgent
    
    agent = NewsAgent(
        experiment_name="titan_v8_phase7",
        use_deseasonalized=use_deseasonalized
    )
    
    df = agent.load_and_merge_data()
    metrics = agent.train(df)
    
    print(f"\n   ✅ NewsAgent Test R²: {metrics['test']['R2']:.4f}")
    
    # Save predictions
    df['news_pred'] = agent.predict(df)
    news_pred_path = Path("data/processed/news_predictions.parquet")
    df[['date', 'ticker', 'news_pred']].to_parquet(news_pred_path, index=False)
    
    mlflow.end_run()
    
    return agent, metrics


def step_4_train_fundamental():
    """Train FundamentalAgent on clean residuals."""
    print("\n" + "=" * 70)
    print("🔧 STEP 4: TRAIN FUNDAMENTAL AGENT")
    print("=" * 70)
    
    # End any stale runs
    mlflow.end_run()
    
    try:
        from src.agents.alpha_agents import FundamentalAgent
        
        agent = FundamentalAgent()
        agent.train()
        
        print(f"\n   ✅ FundamentalAgent Test R²: {agent.test_metrics['R2']:.4f}")
        
        mlflow.end_run()
        return agent
    except Exception as e:
        print(f"\n   ⚠️ FundamentalAgent skipped: {e}")
        mlflow.end_run()
        return None


def step_5_train_coordinator(agents):
    """Train coordinator and evaluate with re-seasonalization."""
    print("\n" + "=" * 70)
    print("🎯 STEP 5: TRAIN COORDINATOR (WITH RE-SEASONALIZATION)")
    print("=" * 70)
    
    # End any stale runs
    mlflow.end_run()
    
    from src.coordinator.fusion import TitanCoordinator
    from src.pipeline.deseasonalize import load_seasonality_map
    
    # Load seasonality map
    seasonal_map = load_seasonality_map()
    print(f"   ✓ Loaded seasonality map for {len(seasonal_map)} tickers")
    
    # Initialize coordinator
    coordinator = TitanCoordinator(experiment_name="titan_v8_phase7")
    
    # Load de-seasonalized targets
    targets = pd.read_parquet("data/processed/targets_deseasonalized.parquet")
    targets['date'] = pd.to_datetime(targets['date']).dt.tz_localize(None)
    if targets['ticker'].dtype.name == 'category':
        targets['ticker'] = targets['ticker'].astype(str)
    
    # Load residuals
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    residuals['date'] = pd.to_datetime(residuals['date']).dt.tz_localize(None)
    if 'pred_tech' in residuals.columns:
        residuals = residuals.rename(columns={'pred_tech': 'tech_pred'})
    
    # Merge
    df = pd.merge(targets, residuals[['date', 'ticker', 'tech_pred', 'resid_tech']], 
                  on=['date', 'ticker'], how='left')
    
    # Load news predictions
    news_path = Path("data/processed/news_predictions.parquet")
    if news_path.exists():
        news_pred = pd.read_parquet(news_path)
        news_pred['date'] = pd.to_datetime(news_pred['date']).dt.tz_localize(None)
        df = pd.merge(df, news_pred, on=['date', 'ticker'], how='left')
    
    # Fill missing
    df['tech_pred'] = df['tech_pred'].fillna(0)
    df['news_pred'] = df['news_pred'].fillna(0)
    df['fund_pred'] = 0  # Simplified
    df['retail_pred'] = 0
    
    # Add calendar features (should have less importance now!)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_q4'] = (df['date'].dt.quarter == 4).astype(int)
    
    # VIX
    df['VIX_close'] = df['VIX_close'].ffill().fillna(15)
    
    # Drop NaN
    df = df.dropna(subset=['target_excess'])
    
    print(f"   ✓ Prepared {len(df):,} rows")
    
    # Features - train on EXCESS target
    feature_cols = ['tech_pred', 'news_pred', 'VIX_close', 'is_friday', 'is_monday', 'is_q4']
    target_col = 'target_excess'  # Train on excess!
    
    # Split
    cutoff = pd.to_datetime("2023-01-01")
    train_mask = df['date'] < cutoff
    test_mask = df['date'] >= cutoff
    
    X_train = df.loc[train_mask, feature_cols].fillna(0)
    y_train = df.loc[train_mask, target_col]
    X_test = df.loc[test_mask, feature_cols].fillna(0)
    y_test_excess = df.loc[test_mask, target_col]
    
    # Get original target for final evaluation
    y_test_original = df.loc[test_mask, 'target_log_var']
    seasonal_test = df.loc[test_mask, 'seasonal_component']
    
    print(f"\n   Split: Train={len(X_train):,}, Test={len(X_test):,}")
    
    # Train on excess
    from sklearn.linear_model import ElasticNet
    model = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000)
    model.fit(X_train, y_train)
    
    # Predict excess
    pred_excess = model.predict(X_test)
    
    # =============================================
    # CRUCIAL: RE-SEASONALIZE FOR FINAL EVAL
    # =============================================
    print("\n   🔧 Re-seasonalizing predictions...")
    
    # pred_total = pred_excess + seasonal_component
    pred_total = pred_excess + seasonal_test.values
    
    # Calculate metrics
    r2_excess = r2_score(y_test_excess, pred_excess)
    r2_total = r2_score(y_test_original, pred_total)
    rmse_total = np.sqrt(mean_squared_error(y_test_original, pred_total))
    
    print(f"\n   📊 Results:")
    print(f"      R² on Excess (internal):    {r2_excess:.4f} ({r2_excess*100:.2f}%)")
    print(f"      R² on Total (re-seasonal):  {r2_total:.4f} ({r2_total*100:.2f}%)")
    print(f"      RMSE on Total:              {rmse_total:.4f}")
    
    # Feature importance
    print(f"\n   📊 Feature Importance:")
    total_abs = sum(abs(c) for c in model.coef_)
    for feat, coef in zip(feature_cols, model.coef_):
        pct = abs(coef) / total_abs * 100 if total_abs > 0 else 0
        print(f"      {feat:15}: {coef:+.4f} ({pct:.1f}%)")
    
    # Store for return
    coordinator.model = model
    coordinator.feature_cols = feature_cols
    coordinator.test_metrics = {'R2': r2_total, 'RMSE': rmse_total}
    coordinator.excess_metrics = {'R2': r2_excess}
    coordinator.df = df
    coordinator.test_mask = test_mask
    coordinator.pred_total = pred_total
    
    mlflow.end_run()
    
    return coordinator, r2_total


def print_comparison(baseline_r2, phase7_r2, news_before, news_after, 
                     friday_before, friday_after):
    """Print before/after comparison."""
    print("\n" + "=" * 70)
    print("📊 PHASE 7 RESULTS COMPARISON")
    print("=" * 70)
    
    print(f"\n   {'Metric':<30} {'Before':>12} {'After':>12} {'Change':>12}")
    print("   " + "-" * 68)
    
    # Ensemble R²
    delta_ensemble = phase7_r2 - baseline_r2
    print(f"   {'Ensemble Test R²':<30} {baseline_r2:>12.4f} {phase7_r2:>12.4f} {delta_ensemble:>+12.4f}")
    
    # NewsAgent R²
    delta_news = news_after - news_before
    print(f"   {'NewsAgent Test R²':<30} {news_before:>12.4f} {news_after:>12.4f} {delta_news:>+12.4f}")
    
    # Friday importance
    delta_friday = friday_after - friday_before
    print(f"   {'is_friday Importance':<30} {friday_before:>11.1f}% {friday_after:>11.1f}% {delta_friday:>+11.1f}%")
    
    print("   " + "-" * 68)
    
    # Verdict
    print("\n   🏆 VERDICT:")
    
    if news_after > 0:
        print(f"      ✅ NewsAgent FIXED: {news_before:.2%} → {news_after:.2%}")
    else:
        print(f"      ⚠️ NewsAgent still negative: {news_after:.2%}")
    
    if friday_after < friday_before * 0.5:
        print(f"      ✅ is_friday reduced: {friday_before:.1f}% → {friday_after:.1f}%")
    else:
        print(f"      ⚠️ is_friday still dominant: {friday_after:.1f}%")
    
    if phase7_r2 >= baseline_r2 * 0.9:
        print(f"      ✅ Ensemble performance maintained: {phase7_r2:.2%}")
    else:
        print(f"      ⚠️ Ensemble performance dropped: {phase7_r2:.2%}")


def main():
    """Run Phase 7: Target De-Seasonalization."""
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🚀 PHASE 7: TARGET DE-SEASONALIZATION")
    print("   Fixing NewsAgent by removing calendar effects from targets")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Baseline values from previous run
    BASELINE_R2 = 0.1736
    NEWS_BEFORE = -0.0272
    FRIDAY_BEFORE = 40.6
    
    # Set MLflow
    mlflow.set_experiment("titan_v8_phase7")
    
    # Step 1: De-seasonalize targets
    step_1_deseasonalize()
    
    # Step 2: Train TechnicalAgent on excess
    tech_agent, tech_metrics = step_2_train_technical(use_deseasonalized=True)
    
    # Step 3: Train NewsAgent on clean residuals
    news_agent, news_metrics = step_3_train_news(use_deseasonalized=True)
    news_r2 = news_metrics['test']['R2']
    
    # Step 4: Train FundamentalAgent
    fund_agent = step_4_train_fundamental()
    
    # Step 5: Train Coordinator with re-seasonalization
    agents = {'tech': tech_agent, 'news': news_agent, 'fund': fund_agent}
    coordinator, final_r2 = step_5_train_coordinator(agents)
    
    # Calculate Friday importance
    total_abs = sum(abs(c) for c in coordinator.model.coef_)
    friday_idx = coordinator.feature_cols.index('is_friday')
    friday_after = abs(coordinator.model.coef_[friday_idx]) / total_abs * 100 if total_abs > 0 else 0
    
    # Print comparison
    print_comparison(
        baseline_r2=BASELINE_R2,
        phase7_r2=final_r2,
        news_before=NEWS_BEFORE,
        news_after=news_r2,
        friday_before=FRIDAY_BEFORE,
        friday_after=friday_after
    )
    
    # Timing
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("✅ PHASE 7 COMPLETE")
    print(f"   Duration: {duration}")
    print(f"   Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return {
        'final_r2': final_r2,
        'news_r2': news_r2,
        'friday_importance': friday_after
    }


if __name__ == "__main__":
    main()

