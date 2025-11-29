"""
Titan V8 Full Pipeline: The Final Full-Scale Run (Attempt 2)

Phase 5: Orchestrates the entire training loop on the full 7-year dataset.
Uses Ridge Linear Stacking to prevent overfitting.

Steps:
A. Config Update - Force full ticker list and date range
B. Data Engine - Force fresh data ingestion
C. Train Ensemble - Train all agents
D. Train Coordinator - Ridge Linear Stacking
E. Money Shot - Print final comparison table with ensemble weights

Usage:
    python scripts/run_full_pipeline.py
    
Options:
    --skip-ingest    Skip data ingestion if already done
    --quick          Use current data without full ingestion
"""

import sys
import os
import argparse
import shutil
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Full dataset configuration
FULL_CONFIG = {
    "project_name": "titan_v8",
    "seed": 42,
    "data": {
        "tickers": [
            'SPY', 'QQQ', 'IWM', 'AAPL', 'NVDA', 'MSFT',
            'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'JPM'
        ],
        "start_date": "2018-01-01",
        "end_date": "2024-12-31",
        "raw_path": "data/raw",
        "processed_path": "data/processed"
    },
    "mlflow": {
        "experiment_name": "titan_v8_full"
    }
}


def update_config():
    """Step A: Force update config with full dataset parameters."""
    print("\n" + "=" * 70)
    print("📋 STEP A: CONFIG UPDATE (FORCED)")
    print("=" * 70)
    
    config_path = Path("conf/base/config.yaml")
    
    # Force write config
    with open(config_path, 'w') as f:
        yaml.dump(FULL_CONFIG, f, default_flow_style=False)
    
    print(f"   ✓ Tickers: {len(FULL_CONFIG['data']['tickers'])} tickers")
    print(f"   ✓ List: {FULL_CONFIG['data']['tickers']}")
    print(f"   ✓ Date range: {FULL_CONFIG['data']['start_date']} to {FULL_CONFIG['data']['end_date']}")
    print(f"   ✓ Config saved to: {config_path}")
    
    return FULL_CONFIG


def force_fresh_ingestion():
    """Force delete existing data and run fresh ingestion."""
    print("\n" + "=" * 70)
    print("📊 STEP B: DATA ENGINE (FORCED FRESH)")
    print("=" * 70)
    
    # Delete existing processed data
    targets_path = Path("data/processed/targets.parquet")
    if targets_path.exists():
        print(f"   🗑️ Deleting {targets_path}")
        targets_path.unlink()
    
    residuals_path = Path("data/processed/residuals.parquet")
    if residuals_path.exists():
        print(f"   🗑️ Deleting {residuals_path}")
        residuals_path.unlink()
    
    # Run fresh ingestion
    print("\n   📥 Running FULL market data ingestion...")
    print("   ⚠️ This may take 15-30 minutes for 12 tickers × 7 years")
    
    try:
        from src.pipeline.ingest import fetch_market_data
        fetch_market_data()
        print("   ✓ Market data ingested successfully")
    except Exception as e:
        print(f"   ❌ Market data ingestion error: {e}")
        return False
    
    # Check if data is valid
    if targets_path.exists():
        df = pd.read_parquet(targets_path)
        print(f"\n   📊 Data Check:")
        print(f"      Rows: {len(df):,}")
        print(f"      Tickers: {df['ticker'].nunique()}")
        print(f"      Size: {targets_path.stat().st_size / (1024*1024):.2f} MB")
        return True
    else:
        print("   ❌ targets.parquet not created!")
        return False


def run_news_pipeline():
    """Run news ingestion and processing."""
    print("\n   📰 Running news pipeline...")
    
    # Ingest news
    try:
        from src.pipeline.ingest_news import ingest_all_news
        ingest_all_news()
        print("   ✓ News data ingested")
    except Exception as e:
        print(f"   ⚠️ News ingestion error: {e}")
    
    # Process news
    try:
        from src.pipeline.process_news import process_news_features
        process_news_features()
        print("   ✓ News features processed")
    except Exception as e:
        print(f"   ⚠️ News processing error: {e}")


def run_retail_pipeline():
    """Run retail signal ingestion."""
    print("\n   💹 Running retail signals pipeline...")
    
    try:
        from src.pipeline.ingest_retail import fetch_retail_signals
        fetch_retail_signals()
        print("   ✓ Retail signals ingested")
    except Exception as e:
        print(f"   ⚠️ Retail signals error: {e}")


def train_technical_agent():
    """Train TechnicalAgent and save residuals."""
    print("\n" + "-" * 70)
    print("🔧 Training TechnicalAgent (HAR-RV Baseline)")
    print("-" * 70)
    
    from src.agents.technical_agent import TechnicalAgent
    
    agent = TechnicalAgent()
    df = agent.load_and_process_data()
    agent.train(df)
    
    print(f"   Train R²: {agent.train_metrics['R2']:.4f}")
    print(f"   Test R²:  {agent.test_metrics['R2']:.4f}")
    
    return agent


def train_news_agent():
    """Train NewsAgent on residuals."""
    print("\n" + "-" * 70)
    print("🔧 Training NewsAgent (Residual Corrector)")
    print("-" * 70)
    
    from src.agents.news_agent import NewsAgent
    
    agent = NewsAgent()
    df = agent.load_and_merge_data()
    agent.train(df)
    
    print(f"   Train R²: {agent.train_metrics['R2']:.4f}")
    print(f"   Test R²:  {agent.test_metrics['R2']:.4f}")
    
    return agent


def train_fundamental_agent():
    """Train FundamentalAgent on residuals."""
    print("\n" + "-" * 70)
    print("🔧 Training FundamentalAgent (Fundamental Corrector)")
    print("-" * 70)
    
    from src.agents.alpha_agents import FundamentalAgent
    
    agent = FundamentalAgent()
    agent.train()
    
    print(f"   Train R²: {agent.train_metrics['R2']:.4f}")
    print(f"   Test R²:  {agent.test_metrics['R2']:.4f}")
    
    return agent


def train_retail_agent():
    """Train RetailRiskAgent."""
    print("\n" + "-" * 70)
    print("🔧 Training RetailRiskAgent (Retail Signal)")
    print("-" * 70)
    
    from src.agents.alpha_agents import RetailRiskAgent
    
    agent = RetailRiskAgent()
    agent.train()
    
    print(f"   Train R²: {agent.train_metrics['R2']:.4f}")
    print(f"   Test R²:  {agent.test_metrics['R2']:.4f}")
    
    return agent


def train_ensemble(skip_ingest=False):
    """Step C: Train all ensemble agents."""
    print("\n" + "=" * 70)
    print("🎯 STEP C: TRAIN ENSEMBLE")
    print("=" * 70)
    
    agents = {}
    
    # 1. Technical Agent (Baseline)
    try:
        agents['tech'] = train_technical_agent()
    except Exception as e:
        print(f"   ⚠️ TechnicalAgent error: {e}")
        agents['tech'] = None
    
    # 2. News Agent
    try:
        agents['news'] = train_news_agent()
    except Exception as e:
        print(f"   ⚠️ NewsAgent error: {e}")
        agents['news'] = None
    
    # 3. Fundamental Agent
    try:
        agents['fund'] = train_fundamental_agent()
    except Exception as e:
        print(f"   ⚠️ FundamentalAgent error: {e}")
        agents['fund'] = None
    
    # 4. Retail Agent
    try:
        agents['retail'] = train_retail_agent()
    except Exception as e:
        print(f"   ⚠️ RetailRiskAgent error: {e}")
        agents['retail'] = None
    
    return agents


def train_coordinator(agents):
    """Step D: Train the TitanCoordinator (Ridge Linear Stacking)."""
    print("\n" + "=" * 70)
    print("🎯 STEP D: TRAIN TITAN COORDINATOR (Ridge Linear Stacking)")
    print("=" * 70)
    
    from src.coordinator.fusion import TitanCoordinator
    
    # Load data
    targets = pd.read_parquet("data/processed/targets.parquet")
    
    # Try to load news features
    news_features = None
    news_path = Path("data/processed/news_features.parquet")
    if news_path.exists():
        news_features = pd.read_parquet(news_path)
    
    # Initialize coordinator
    coordinator = TitanCoordinator()
    
    # Prepare unified predictions dataset
    df = coordinator.prepare_predictions_dataset(
        tech_agent=agents.get('tech'),
        news_agent=agents.get('news'),
        fund_agent=agents.get('fund'),
        retail_agent=agents.get('retail'),
        targets_df=targets,
        news_features_df=news_features
    )
    
    # Train coordinator
    metrics = coordinator.train(df)
    
    # Print feature importance (Ridge coefficients)
    print("\n   📊 Ridge Coefficients (Ensemble Weights):")
    importance = coordinator.get_feature_importance()
    for _, row in importance.iterrows():
        print(f"      - {row['feature']}: {row['coefficient']:+.4f} ({row['pct']:.1f}%)")
    
    return coordinator, metrics


def print_money_shot(coordinator, agents):
    """Step E: Print the final comparison table."""
    print("\n" + "=" * 70)
    print("💰 STEP E: THE MONEY SHOT")
    print("=" * 70)
    
    coordinator.print_comparison_table()
    
    # Agent summary
    print("\n📊 AGENT CONTRIBUTION SUMMARY")
    print("-" * 70)
    print(f"{'Agent':<25} {'Target':<20} {'Test R²':>10}")
    print("-" * 70)
    
    if agents.get('tech') and agents['tech'].test_metrics:
        print(f"{'TechnicalAgent':<25} {'target_log_var':<20} {agents['tech'].test_metrics['R2']:>10.4f}")
    
    if agents.get('news') and agents['news'].test_metrics:
        print(f"{'NewsAgent':<25} {'resid_tech':<20} {agents['news'].test_metrics['R2']:>10.4f}")
    
    if agents.get('fund') and agents['fund'].test_metrics:
        print(f"{'FundamentalAgent':<25} {'resid_tech':<20} {agents['fund'].test_metrics['R2']:>10.4f}")
    
    if agents.get('retail') and agents['retail'].test_metrics:
        target = agents['retail'].target_col
        print(f"{'RetailRiskAgent':<25} {target:<20} {agents['retail'].test_metrics['R2']:>10.4f}")
    
    if coordinator.test_metrics:
        print("-" * 70)
        print(f"{'TITAN V8 (Ensemble)':<25} {'target_log_var':<20} {coordinator.test_metrics['R2']:>10.4f}")
    
    print("-" * 70)


def main():
    """Run the full Titan V8 pipeline."""
    parser = argparse.ArgumentParser(description="Titan V8 Full Pipeline")
    parser.add_argument("--skip-ingest", action="store_true", 
                        help="Skip data ingestion")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with current data")
    parser.add_argument("--force", action="store_true",
                        help="Force fresh data ingestion")
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🚀 TITAN V8: FULL-SCALE PIPELINE (Linear Ensemble)")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Step A: Config
    config = update_config()
    
    # Step B: Data Engine
    if args.force:
        success = force_fresh_ingestion()
        if success:
            run_news_pipeline()
            run_retail_pipeline()
    elif args.skip_ingest or args.quick:
        print("\n" + "=" * 70)
        print("📊 STEP B: DATA ENGINE (SKIPPED)")
        print("=" * 70)
        print("   ⏭️ Using existing data")
        
        targets_path = Path("data/processed/targets.parquet")
        if targets_path.exists():
            df = pd.read_parquet(targets_path)
            print(f"   Rows: {len(df):,}")
            print(f"   Tickers: {df['ticker'].nunique()}")
    else:
        # Check if we need to ingest
        targets_path = Path("data/processed/targets.parquet")
        if not targets_path.exists() or targets_path.stat().st_size < 1024 * 1024:
            print("\n   ⚠️ Data missing or too small - running ingestion")
            force_fresh_ingestion()
            run_news_pipeline()
            run_retail_pipeline()
        else:
            print("\n   ✓ Data exists, skipping ingestion")
    
    # Step C: Train Ensemble
    agents = train_ensemble()
    
    # Step D: Train Coordinator
    coordinator, metrics = train_coordinator(agents)
    
    # Step E: Money Shot
    print_money_shot(coordinator, agents)
    
    # Timing
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("✅ TITAN V8 PIPELINE COMPLETE")
    print(f"   Duration: {duration}")
    print(f"   Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
