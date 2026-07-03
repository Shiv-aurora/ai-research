"""
Universe Expansion Pipeline: Strategic Diversification

Expands from 3 tickers to 18-ticker universe across 6 sectors:
- Tech: AAPL, MSFT, NVDA
- Finance: JPM, BAC, V
- Industrial: CAT, GE, BA
- Consumer: WMT, MCD, COST
- Energy: XOM, CVX, SLB
- Healthcare: JNJ, PFE, UNH

Pipeline Steps:
A. Price Ingestion - Full refresh for all 18 tickers
B. News Ingestion - Fetch news for all tickers
C. Feature Engineering - Re-run vectorization and PCA
D. Proxy Generation - Generate volume shocks

Expected Output: ~30,000+ rows in targets.parquet

Usage:
    python scripts/run_universe_expansion.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
import mlflow

# Ensure environment variables are loaded
from dotenv import load_dotenv
load_dotenv()


def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = Path("conf/base/config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"🚀 {title}")
    print("=" * 70)


def print_sector_breakdown(tickers: list):
    """Print sector breakdown of the universe."""
    sectors = {
        "Tech": ["AAPL", "MSFT", "NVDA"],
        "Finance": ["JPM", "BAC", "V"],
        "Industrial": ["CAT", "GE", "BA"],
        "Consumer": ["WMT", "MCD", "COST"],
        "Energy": ["XOM", "CVX", "SLB"],
        "Healthcare": ["JNJ", "PFE", "UNH"]
    }
    
    print("\n   📊 Universe Composition:")
    for sector, sector_tickers in sectors.items():
        matching = [t for t in sector_tickers if t in tickers]
        if matching:
            print(f"      {sector:12} {matching}")
    
    print(f"\n   Total: {len(tickers)} tickers")


def step_a_price_ingestion():
    """
    Step A: Full price data ingestion for all 18 tickers.
    
    Uses Polygon.io API to fetch 5-min bars and calculate realized volatility.
    """
    print_header("STEP A: PRICE INGESTION")
    
    try:
        from src.pipeline.ingest import main as ingest_main
        
        print("\n   Running full price ingestion...")
        print("   (This may take 15-30 minutes for 18 tickers)")
        
        ingest_main()
        
        # Verify output
        targets_path = Path("data/processed/targets.parquet")
        if targets_path.exists():
            df = pd.read_parquet(targets_path)
            print(f"\n   ✅ targets.parquet: {len(df):,} rows, {df['ticker'].nunique()} tickers")
            return True
        else:
            print("\n   ❌ targets.parquet not created!")
            return False
            
    except Exception as e:
        print(f"\n   ❌ Price ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step_b_news_ingestion():
    """
    Step B: News data ingestion for all tickers.
    
    Fetches news from Kaggle (2018-2020) and Polygon.io (2020-2025).
    """
    print_header("STEP B: NEWS INGESTION")
    
    try:
        from src.pipeline.ingest_news import main as news_main
        
        print("\n   Running news ingestion...")
        print("   (Fetching Kaggle + Polygon news for all tickers)")
        
        news_main()
        
        # Verify output
        news_path = Path("data/processed/news_base.parquet")
        if news_path.exists():
            df = pd.read_parquet(news_path)
            print(f"\n   ✅ news_base.parquet: {len(df):,} articles, {df['ticker'].nunique()} tickers")
            return True
        else:
            print("\n   ⚠️ news_base.parquet not created (continuing anyway)")
            return True  # Non-fatal
            
    except Exception as e:
        print(f"\n   ⚠️ News ingestion warning: {e}")
        return True  # Non-fatal


def step_c_feature_engineering():
    """
    Step C: Re-run vectorization and PCA.
    
    CRUCIAL: PCA needs to learn new market themes:
    - Energy patterns (XOM, CVX, SLB)
    - Healthcare/Pharma patterns (JNJ, PFE, UNH)
    - Industrial/Manufacturing patterns (CAT, GE, BA)
    """
    print_header("STEP C: FEATURE ENGINEERING")
    
    try:
        from src.pipeline.process_news import main as process_main
        
        print("\n   Re-running TF-IDF vectorization and PCA...")
        print("   (Learning new sector themes: Energy, Healthcare, Industrial)")
        
        # Use lite mode for speed
        process_main()
        
        # Verify output
        features_path = Path("data/processed/news_features.parquet")
        if features_path.exists():
            df = pd.read_parquet(features_path)
            pca_cols = [c for c in df.columns if "pca" in c.lower()]
            print(f"\n   ✅ news_features.parquet: {len(df):,} rows")
            print(f"      PCA components: {len(pca_cols)}")
            return True
        else:
            print("\n   ⚠️ news_features.parquet not created")
            return True  # Non-fatal
            
    except Exception as e:
        print(f"\n   ⚠️ Feature engineering warning: {e}")
        return True  # Non-fatal


def step_d_proxy_generation():
    """
    Step D: Generate Reddit proxy (volume shocks) for all tickers.
    """
    print_header("STEP D: PROXY GENERATION")
    
    try:
        from src.pipeline.create_reddit_proxy import main as proxy_main
        
        print("\n   Generating volume shock proxy...")
        
        proxy_main()
        
        # Verify output
        proxy_path = Path("data/processed/reddit_proxy.parquet")
        if proxy_path.exists():
            df = pd.read_parquet(proxy_path)
            print(f"\n   ✅ reddit_proxy.parquet: {len(df):,} rows, {df['ticker'].nunique()} tickers")
            return True
        else:
            print("\n   ⚠️ reddit_proxy.parquet not created")
            return True  # Non-fatal
            
    except Exception as e:
        print(f"\n   ⚠️ Proxy generation warning: {e}")
        return True  # Non-fatal


def generate_summary():
    """Generate final summary of the expanded universe."""
    print_header("EXPANSION SUMMARY")
    
    # Check all data files
    files_to_check = {
        "targets.parquet": "data/processed/targets.parquet",
        "news_base.parquet": "data/processed/news_base.parquet",
        "news_features.parquet": "data/processed/news_features.parquet",
        "reddit_proxy.parquet": "data/processed/reddit_proxy.parquet",
    }
    
    results = {}
    
    print("\n   📊 Data Files Status:")
    print("   " + "-" * 55)
    
    for name, path in files_to_check.items():
        path = Path(path)
        if path.exists():
            df = pd.read_parquet(path)
            rows = len(df)
            tickers = df['ticker'].nunique() if 'ticker' in df.columns else 0
            size_mb = path.stat().st_size / (1024 * 1024)
            
            results[name] = {"rows": rows, "tickers": tickers, "size_mb": size_mb}
            print(f"   ✅ {name:25} {rows:>8,} rows | {tickers:>3} tickers | {size_mb:>6.2f} MB")
        else:
            results[name] = None
            print(f"   ❌ {name:25} NOT FOUND")
    
    # Final verdict
    targets_result = results.get("targets.parquet")
    
    print("\n   " + "-" * 55)
    
    if targets_result and targets_result["rows"] >= 10000:
        print(f"\n   🏆 EXPANSION SUCCESSFUL!")
        print(f"      Total price data: {targets_result['rows']:,} rows")
        print(f"      Tickers: {targets_result['tickers']}")
        print(f"      Expected: ~30,000+ rows (18 tickers × ~1,700 days)")
        
        if targets_result["rows"] >= 25000:
            print(f"      Status: ✅ EXCELLENT (>25K rows)")
        elif targets_result["rows"] >= 15000:
            print(f"      Status: ✅ GOOD (>15K rows)")
        else:
            print(f"      Status: ⚠️ PARTIAL (some tickers may have limited data)")
    else:
        print(f"\n   ❌ EXPANSION INCOMPLETE")
        print(f"      Check API credentials and try again")
    
    return results


def main():
    """Main entry point for universe expansion."""
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🌍 UNIVERSE EXPANSION: STRATEGIC DIVERSIFICATION")
    print("   Expanding from 3 → 18 tickers across 6 sectors")
    print("=" * 70)
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and display config
    config = load_config()
    tickers = config["data"]["tickers"]
    
    print_sector_breakdown(tickers)
    
    # MLflow tracking
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    with mlflow.start_run(run_name="universe_expansion_18"):
        mlflow.log_param("n_tickers", len(tickers))
        mlflow.log_param("tickers", ",".join(tickers))
        mlflow.log_param("start_date", config["data"]["start_date"])
        mlflow.log_param("end_date", config["data"]["end_date"])
        
        # Run pipeline steps
        success = True
        
        # Step A: Price Ingestion (CRITICAL)
        if not step_a_price_ingestion():
            print("\n❌ Critical failure in Step A. Aborting.")
            success = False
        
        if success:
            # Step B: News Ingestion (Non-critical)
            step_b_news_ingestion()
            
            # Step C: Feature Engineering (Non-critical)
            step_c_feature_engineering()
            
            # Step D: Proxy Generation (Non-critical)
            step_d_proxy_generation()
        
        # Summary
        results = generate_summary()
        
        # Log to MLflow
        if results.get("targets.parquet"):
            mlflow.log_metric("final_rows", results["targets.parquet"]["rows"])
            mlflow.log_metric("final_tickers", results["targets.parquet"]["tickers"])
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        mlflow.log_metric("duration_minutes", duration)
    
    print("\n" + "=" * 70)
    print(f"   Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Duration: {duration:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()

