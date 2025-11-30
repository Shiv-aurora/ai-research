"""
Titan V8 Full Pipeline: 18-Ticker Universe Baseline

Establishes baseline performance for the diversified 18-ticker portfolio.
Uses ElasticNet Linear Stacking with systematic features.

Steps:
A. Verification - Check config and data
B. Train Anchors - TechnicalAgent, NewsAgent, FundamentalAgent
C. Train Coordinator - ElasticNet with systematic features
D. Report - Full universe R², per-sector breakdown, feature importance

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --skip-ingest  # Use existing data
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
import mlflow

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# Sector mapping for 18-ticker universe
SECTOR_MAP = {
    # Tech
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech',
    # Finance
    'JPM': 'Finance', 'BAC': 'Finance', 'V': 'Finance',
    # Industrial
    'CAT': 'Industrial', 'GE': 'Industrial', 'BA': 'Industrial',
    # Consumer
    'WMT': 'Consumer', 'MCD': 'Consumer', 'COST': 'Consumer',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'SLB': 'Energy',
    # Healthcare
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
}


def load_config():
    """Load config WITHOUT overwriting - preserve 18-ticker universe."""
    config_path = Path("conf/base/config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def verify_data():
    """Step A: Verify configuration and data integrity."""
    print("\n" + "=" * 70)
    print("📋 STEP A: VERIFICATION")
    print("=" * 70)
    
    # Check config
    config = load_config()
    tickers = config["data"]["tickers"]
    
    print(f"\n   📊 Configuration:")
    print(f"      Tickers: {len(tickers)}")
    print(f"      Date range: {config['data']['start_date']} to {config['data']['end_date']}")
    
    # Check data
    targets_path = Path("data/processed/targets.parquet")
    if not targets_path.exists():
        print(f"\n   ❌ targets.parquet not found!")
        print(f"      Run: python scripts/run_universe_expansion.py")
        return None, None
    
    df = pd.read_parquet(targets_path)
    
    # Clean infinity values in target
    inf_count = np.isinf(df['target_log_var']).sum()
    if inf_count > 0:
        print(f"\n   ⚠️ Found {inf_count} infinity values in target - cleaning...")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['target_log_var'])
    
    print(f"\n   📊 Data Summary:")
    print(f"      Rows: {len(df):,}")
    print(f"      Tickers: {df['ticker'].nunique()}")
    print(f"      Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Sector breakdown
    if 'ticker' in df.columns:
        df['sector'] = df['ticker'].map(SECTOR_MAP).fillna('Other')
        sector_counts = df.groupby('sector').size()
        print(f"\n   📊 Sector Breakdown:")
        for sector in ['Tech', 'Finance', 'Industrial', 'Consumer', 'Energy', 'Healthcare']:
            if sector in sector_counts.index:
                print(f"      {sector:12}: {sector_counts[sector]:,} rows")
    
    return config, df


def train_technical_agent():
    """Train TechnicalAgent (HAR-RV) and save residuals."""
    print("\n" + "-" * 70)
    print("🔧 Training TechnicalAgent (HAR-RV Baseline)")
    print("-" * 70)
    
    # End any existing runs first
    mlflow.end_run()
    
    from src.agents.technical_agent import TechnicalAgent
    
    agent = TechnicalAgent(experiment_name="titan_v8_universe_18")
    df = agent.load_and_process_data()
    
    # Clean infinity values before training
    inf_count = np.isinf(df['target_log_var']).sum()
    if inf_count > 0:
        print(f"\n   ⚠️ Cleaning {inf_count} infinity values...")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['target_log_var'])
    
    agent.train(df)
    
    print(f"\n   Train R²: {agent.train_metrics['R2']:.4f} ({agent.train_metrics['R2']*100:.2f}%)")
    print(f"   Test R²:  {agent.test_metrics['R2']:.4f} ({agent.test_metrics['R2']*100:.2f}%)")
    
    # Save residuals
    residuals_path = Path("data/processed/residuals.parquet")
    print(f"\n   Residuals saved to: {residuals_path}")
    
    # End run after agent
    mlflow.end_run()
    
    return agent


def train_news_agent():
    """Train NewsAgent (Decay Kernel) on residuals."""
    print("\n" + "-" * 70)
    print("🔧 Training NewsAgent (Decay Kernel)")
    print("-" * 70)
    
    # End any existing runs
    mlflow.end_run()
    
    from src.agents.news_agent import NewsAgent
    
    agent = NewsAgent(experiment_name="titan_v8_universe_18")
    df = agent.load_and_merge_data()
    metrics = agent.train(df)
    
    print(f"\n   Train R²: {agent.train_metrics['R2']:.4f} ({agent.train_metrics['R2']*100:.2f}%)")
    print(f"   Test R²:  {agent.test_metrics['R2']:.4f} ({agent.test_metrics['R2']*100:.2f}%)")
    
    # Save news predictions
    df['news_pred'] = agent.predict(df)
    news_pred_path = Path("data/processed/news_predictions.parquet")
    df[['date', 'ticker', 'news_pred']].to_parquet(news_pred_path, index=False)
    print(f"\n   News predictions saved to: {news_pred_path}")
    
    # End run after agent
    mlflow.end_run()
    
    return agent


def train_fundamental_agent():
    """Train FundamentalAgent on residuals."""
    print("\n" + "-" * 70)
    print("🔧 Training FundamentalAgent")
    print("-" * 70)
    
    # End any existing runs
    mlflow.end_run()
    
    try:
        from src.agents.alpha_agents import FundamentalAgent
        
        agent = FundamentalAgent()
        agent.train()
        
        print(f"\n   Train R²: {agent.train_metrics['R2']:.4f} ({agent.train_metrics['R2']*100:.2f}%)")
        print(f"   Test R²:  {agent.test_metrics['R2']:.4f} ({agent.test_metrics['R2']*100:.2f}%)")
        
        # End run
        mlflow.end_run()
        
        return agent
    except Exception as e:
        print(f"\n   ⚠️ FundamentalAgent skipped: {e}")
        mlflow.end_run()
        return None


def train_coordinator(agents):
    """Train TitanCoordinator with systematic features."""
    print("\n" + "=" * 70)
    print("🎯 STEP C: TRAIN TITAN COORDINATOR (ElasticNet)")
    print("=" * 70)
    
    from src.coordinator.fusion import TitanCoordinator
    from sklearn.metrics import r2_score, mean_squared_error
    
    # End any stale runs
    mlflow.end_run()
    
    # Initialize
    coordinator = TitanCoordinator(experiment_name="titan_v8_universe_18")
    
    # Load data
    print("\n   Loading data...")
    targets = pd.read_parquet("data/processed/targets.parquet")
    targets['date'] = pd.to_datetime(targets['date']).dt.tz_localize(None)
    if targets['ticker'].dtype.name == 'category':
        targets['ticker'] = targets['ticker'].astype(str)
    
    # Clean infinity values
    targets = targets.replace([np.inf, -np.inf], np.nan)
    targets = targets.dropna(subset=['target_log_var'])
    
    # Load news features if exists
    news_features = None
    news_path = Path("data/processed/news_features.parquet")
    if news_path.exists():
        news_features = pd.read_parquet(news_path)
    
    # Prepare predictions dataset using coordinator's method
    print("\n   Preparing predictions dataset...")
    df = coordinator.prepare_predictions_dataset(
        tech_agent=agents.get('tech'),
        news_agent=agents.get('news'),
        fund_agent=agents.get('fund'),
        retail_agent=agents.get('retail'),
        targets_df=targets,
        news_features_df=news_features
    )
    
    # Add calendar features
    df = coordinator.add_calendar_features(df)
    
    print(f"   Prepared {len(df):,} rows for training")
    
    # Train using coordinator's method
    metrics = coordinator.train(df)
    
    print(f"\n   ✅ TITAN V8 ENSEMBLE PERFORMANCE")
    print(f"      Test R²:  {coordinator.test_metrics['R2']:.4f} ({coordinator.test_metrics['R2']*100:.2f}%)")
    print(f"      Test RMSE: {coordinator.test_metrics['RMSE']:.4f}")
    
    # Store for later
    coordinator.df = df
    
    # End run
    mlflow.end_run()
    
    return coordinator


def print_feature_importance(coordinator):
    """Print feature importance / coefficients."""
    print("\n" + "-" * 70)
    print("📊 FEATURE IMPORTANCE (ElasticNet Coefficients)")
    print("-" * 70)
    
    try:
        importance = coordinator.get_feature_importance()
        
        print(f"\n   {'Feature':<20} {'Coefficient':>12} {'Importance':>12}")
        print("   " + "-" * 46)
        
        for _, row in importance.iterrows():
            pct = row.get('pct', abs(row['coefficient']))
            print(f"   {row['feature']:<20} {row['coefficient']:>+12.4f} {pct:>11.1f}%")
    except Exception as e:
        print(f"   ⚠️ Could not get importance: {e}")


def print_per_ticker_breakdown(coordinator):
    """Print per-ticker and per-sector R² breakdown."""
    print("\n" + "-" * 70)
    print("📊 PER-TICKER R² BREAKDOWN")
    print("-" * 70)
    
    from sklearn.metrics import r2_score
    
    df = coordinator.df
    cutoff = pd.to_datetime("2023-01-01")
    test_df = df[df['date'] >= cutoff].copy()
    
    feature_cols = coordinator.feature_cols
    test_df['y_pred'] = coordinator.model.predict(test_df[feature_cols].fillna(0))
    test_df['sector'] = test_df['ticker'].map(SECTOR_MAP).fillna('Other')
    
    # Per-ticker
    ticker_r2 = []
    for ticker in sorted(test_df['ticker'].unique()):
        mask = test_df['ticker'] == ticker
        if mask.sum() > 10:
            y_true = test_df.loc[mask, 'target_log_var']
            y_pred = test_df.loc[mask, 'y_pred']
            r2 = r2_score(y_true, y_pred)
            sector = SECTOR_MAP.get(ticker, 'Other')
            ticker_r2.append({'ticker': ticker, 'sector': sector, 'r2': r2, 'n': mask.sum()})
    
    ticker_df = pd.DataFrame(ticker_r2).sort_values('r2', ascending=False)
    
    print(f"\n   {'Ticker':<8} {'Sector':<12} {'R²':>10} {'N':>8}")
    print("   " + "-" * 42)
    
    for _, row in ticker_df.iterrows():
        r2_str = f"{row['r2']:.4f}" if row['r2'] > 0 else f"{row['r2']:.4f}"
        status = "✅" if row['r2'] > 0.10 else "⚠️" if row['r2'] > 0 else "❌"
        print(f"   {row['ticker']:<8} {row['sector']:<12} {r2_str:>10} {row['n']:>8} {status}")
    
    # Per-sector summary
    print("\n" + "-" * 70)
    print("📊 PER-SECTOR R² SUMMARY")
    print("-" * 70)
    
    sector_r2 = []
    for sector in ['Tech', 'Finance', 'Industrial', 'Consumer', 'Energy', 'Healthcare']:
        mask = test_df['sector'] == sector
        if mask.sum() > 10:
            y_true = test_df.loc[mask, 'target_log_var']
            y_pred = test_df.loc[mask, 'y_pred']
            r2 = r2_score(y_true, y_pred)
            sector_r2.append({'sector': sector, 'r2': r2, 'n': mask.sum()})
    
    sector_df = pd.DataFrame(sector_r2).sort_values('r2', ascending=False)
    
    print(f"\n   {'Sector':<15} {'R²':>10} {'N':>8} {'Status':<10}")
    print("   " + "-" * 45)
    
    for _, row in sector_df.iterrows():
        status = "BEST" if row['r2'] == sector_df['r2'].max() else ""
        if row['r2'] == sector_df['r2'].min():
            status = "HARDEST"
        print(f"   {row['sector']:<15} {row['r2']:>10.4f} {row['n']:>8} {status:<10}")
    
    return ticker_df, sector_df


def main():
    """Run the full Titan V8 baseline pipeline."""
    parser = argparse.ArgumentParser(description="Titan V8 Full Pipeline - 18 Ticker Universe")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip data verification details")
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🚀 TITAN V8: 18-TICKER UNIVERSE BASELINE")
    print("   Establishing baseline performance across 6 sectors")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # End any stale MLflow runs
    mlflow.end_run()
    
    # MLflow tracking
    mlflow.set_experiment("titan_v8_universe_18")
    
    # Step A: Verify
    config, df = verify_data()
    if df is None:
        print("\n❌ Data verification failed. Exiting.")
        return
    
    # Step B: Train Anchors
    print("\n" + "=" * 70)
    print("🎯 STEP B: TRAIN ANCHOR AGENTS")
    print("=" * 70)
    
    agents = {}
    
    # Technical Agent
    try:
        agents['tech'] = train_technical_agent()
    except Exception as e:
        print(f"\n   ⚠️ TechnicalAgent error: {e}")
        import traceback
        traceback.print_exc()
    
    # News Agent
    try:
        agents['news'] = train_news_agent()
    except Exception as e:
        print(f"\n   ⚠️ NewsAgent error: {e}")
        import traceback
        traceback.print_exc()
    
    # Fundamental Agent
    agents['fund'] = train_fundamental_agent()
    
    # Retail Agent (optional)
    agents['retail'] = None
    
    # Step C: Train Coordinator
    coordinator = train_coordinator(agents)
    
    # Step D: Reporting
    print("\n" + "=" * 70)
    print("📊 STEP D: BASELINE PERFORMANCE REPORT")
    print("=" * 70)
    
    # Feature importance
    print_feature_importance(coordinator)
    
    # Per-ticker breakdown
    ticker_df, sector_df = print_per_ticker_breakdown(coordinator)
    
    # Final summary
    print("\n" + "=" * 70)
    print("💰 FINAL BASELINE SUMMARY")
    print("=" * 70)
    
    print(f"\n   {'Model':<30} {'Test R²':>12}")
    print("   " + "-" * 44)
    
    if agents.get('tech') and hasattr(agents['tech'], 'test_metrics'):
        print(f"   {'TechnicalAgent (HAR-RV)':<30} {agents['tech'].test_metrics['R2']:>12.4f}")
    if agents.get('news') and hasattr(agents['news'], 'test_metrics'):
        print(f"   {'NewsAgent (Decay Kernel)':<30} {agents['news'].test_metrics['R2']:>12.4f}")
    if agents.get('fund') and hasattr(agents['fund'], 'test_metrics'):
        print(f"   {'FundamentalAgent':<30} {agents['fund'].test_metrics['R2']:>12.4f}")
    
    print("   " + "-" * 44)
    print(f"   {'🏆 TITAN V8 ENSEMBLE':<30} {coordinator.test_metrics['R2']:>12.4f}")
    print(f"   {'   (as percentage)':<30} {coordinator.test_metrics['R2']*100:>11.2f}%")
    
    # Best/worst sectors
    if len(sector_df) > 0:
        best = sector_df.iloc[0]
        worst = sector_df.iloc[-1]
        print(f"\n   Best Sector:  {best['sector']} ({best['r2']:.4f})")
        print(f"   Hardest Sector: {worst['sector']} ({worst['r2']:.4f})")
    
    # Is Friday still dominant?
    print("\n" + "-" * 70)
    print("🔍 KEY QUESTION: Does is_friday still dominate?")
    print("-" * 70)
    
    try:
        importance = coordinator.get_feature_importance()
        friday_row = importance[importance['feature'] == 'is_friday']
        if len(friday_row) > 0:
            friday_pct = friday_row['pct'].values[0]
            if friday_pct > 30:
                print(f"   ✅ YES: is_friday = {friday_pct:.1f}% (still dominant across all sectors)")
            elif friday_pct > 15:
                print(f"   ⚠️ PARTIAL: is_friday = {friday_pct:.1f}% (significant but not dominant)")
            else:
                print(f"   ❌ NO: is_friday = {friday_pct:.1f}% (other features now more important)")
    except:
        pass
    
    # Timing
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("✅ TITAN V8 BASELINE COMPLETE")
    print(f"   Duration: {duration}")
    print(f"   Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
