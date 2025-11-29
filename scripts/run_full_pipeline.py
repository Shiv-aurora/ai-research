"""
Titan V8 Full Pipeline: The Final Full-Scale Run

Phase 5: Orchestrates the entire training loop on the full 7-year dataset.

Steps:
A. Config Update - Set full ticker list and date range
B. Data Engine - Ingest all data (price, news, retail signals)
C. Train Ensemble - Train all agents
D. Train Coordinator - Fuse predictions with XGBoost
E. Money Shot - Print final comparison table

Usage:
    python scripts/run_full_pipeline.py
    
Options:
    --skip-ingest    Skip data ingestion if already done
    --quick          Use current data without full ingestion
"""

import sys
import os
import argparse
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


def update_config():
    """Step A: Update config with full dataset parameters."""
    print("\n" + "=" * 70)
    print("📋 STEP A: CONFIG UPDATE")
    print("=" * 70)
    
    config_path = Path("conf/base/config.yaml")
    
    # Full dataset configuration
    full_config = {
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
    
    # Write config
    with open(config_path, 'w') as f:
        yaml.dump(full_config, f, default_flow_style=False)
    
    print(f"   ✓ Tickers: {len(full_config['data']['tickers'])} ({full_config['data']['tickers'][:3]}...)")
    print(f"   ✓ Date range: {full_config['data']['start_date']} to {full_config['data']['end_date']}")
    print(f"   ✓ Config saved to: {config_path}")
    
    return full_config


def check_data_freshness():
    """Check if data needs to be re-ingested."""
    targets_path = Path("data/processed/targets.parquet")
    
    if not targets_path.exists():
        return False, "targets.parquet not found"
    
    size_mb = targets_path.stat().st_size / (1024 * 1024)
    
    if size_mb < 1:  # Less than 1MB is probably incomplete
        return False, f"targets.parquet too small ({size_mb:.2f} MB)"
    
    # Check row count
    df = pd.read_parquet(targets_path)
    n_tickers = df["ticker"].nunique() if "ticker" in df.columns else 0
    
    if n_tickers < 10:  # Expect at least 10 tickers for full run
        return False, f"Only {n_tickers} tickers in data"
    
    return True, f"Data OK: {len(df):,} rows, {n_tickers} tickers, {size_mb:.2f} MB"


def run_data_engine(skip_ingest=False):
    """Step B: Run the data ingestion pipeline."""
    print("\n" + "=" * 70)
    print("📊 STEP B: DATA ENGINE")
    print("=" * 70)
    
    if skip_ingest:
        print("   ⏭️ Skipping ingestion (--skip-ingest flag)")
        data_ok, msg = check_data_freshness()
        print(f"   {msg}")
        return data_ok
    
    # Check if we need fresh data
    data_ok, msg = check_data_freshness()
    print(f"   Data check: {msg}")
    
    if not data_ok:
        print("\n   📥 Running full data ingestion...")
        print("   ⚠️ This may take 10-15 minutes for 7 years of data")
        
        # Run ingest pipeline
        try:
            from src.pipeline.ingest import fetch_market_data
            fetch_market_data()
            print("   ✓ Market data ingested")
        except Exception as e:
            print(f"   ⚠️ Market data ingestion error: {e}")
    
    # Ingest news
    print("\n   📰 Ingesting news data...")
    try:
        from src.pipeline.ingest_news import ingest_all_news
        ingest_all_news()
        print("   ✓ News data ingested")
    except Exception as e:
        print(f"   ⚠️ News ingestion error: {e}")
    
    # Process news
    print("\n   🔧 Processing news features...")
    try:
        from src.pipeline.process_news import process_news_features
        process_news_features()
        print("   ✓ News features processed")
    except Exception as e:
        print(f"   ⚠️ News processing error: {e}")
    
    # Ingest retail signals
    print("\n   💹 Ingesting retail signals...")
    try:
        from src.pipeline.ingest_retail import fetch_retail_signals
        fetch_retail_signals()
        print("   ✓ Retail signals ingested")
    except Exception as e:
        print(f"   ⚠️ Retail signals error: {e}")
    
    # Create reddit proxy
    print("\n   📊 Creating reddit proxy...")
    try:
        from src.pipeline.create_reddit_proxy import create_reddit_proxy
        create_reddit_proxy()
        print("   ✓ Reddit proxy created")
    except Exception as e:
        print(f"   ⚠️ Reddit proxy error: {e}")
    
    return True


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
    """Step D: Train the TitanCoordinator."""
    print("\n" + "=" * 70)
    print("🎯 STEP D: TRAIN TITAN COORDINATOR")
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
    
    # Print feature importance
    print("\n   📊 Coordinator Feature Importance:")
    importance = coordinator.get_feature_importance()
    for _, row in importance.head(5).iterrows():
        print(f"      - {row['feature']}: {row['pct']:.1f}%")
    
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
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🚀 TITAN V8: FULL-SCALE PIPELINE")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Step A: Config
    config = update_config()
    
    # Step B: Data Engine
    skip_ingest = args.skip_ingest or args.quick
    run_data_engine(skip_ingest=skip_ingest)
    
    # Step C: Train Ensemble
    agents = train_ensemble(skip_ingest=skip_ingest)
    
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

