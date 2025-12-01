"""
Scale-Up Phase: Multi-Universe Experiment Orchestrator

Runs RIVE on two stock universes:
1. High Octane 50: Most active/volatile stocks
2. SP500 Sector Leaders 55: Blue-chip stocks across 11 sectors

Features:
- Safe config management (backup/restore)
- Full pipeline execution per universe
- Comparative results reporting

Usage:
    python scripts/scale_up/run_scale_up_experiments.py
"""

import sys
from pathlib import Path
from datetime import datetime
import copy
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
print(f"✓ Loaded environment from {PROJECT_ROOT / '.env'}")

import yaml
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMClassifier
import mlflow

# Import universe configurations
from scripts.scale_up.config_universes import (
    TOP_50_ACTIVE,
    GICS_BALANCED_55,
    SECTOR_MAP_ACTIVE,
    SECTOR_MAP_SP500,
    UNIVERSE_METADATA
)


class ScaleUpOrchestrator:
    """
    Orchestrates scale-up experiments across multiple stock universes.
    
    Key features:
    - Safe config management with backup/restore
    - Full pipeline execution (ingest -> process -> train)
    - Sector-aware evaluation
    - Comparative reporting
    """
    
    def __init__(self, base_config_path: str = "conf/base/config.yaml"):
        """Initialize orchestrator with config path."""
        self.base_config_path = PROJECT_ROOT / base_config_path
        self.original_config = None
        self.results = {}
        
    def setup_experiment(self, experiment_name: str, tickers: list):
        """
        Safely update config for an experiment.
        
        Args:
            experiment_name: Name for MLflow experiment
            tickers: List of ticker symbols
        """
        print(f"\n   📝 Setting up experiment: {experiment_name}")
        print(f"      Tickers: {len(tickers)}")
        
        # Load current config
        with open(self.base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Backup original on first run
        if self.original_config is None:
            self.original_config = copy.deepcopy(config)
            print(f"      ✓ Backed up original config")
        
        # Modify config
        config['data']['tickers'] = tickers
        config['mlflow']['experiment_name'] = f"titan_v15_{experiment_name}"
        
        # Write back cleanly
        with open(self.base_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"      ✓ Updated config with {len(tickers)} tickers")
        
    def restore_original_config(self):
        """Restore original config after experiments."""
        if self.original_config:
            with open(self.base_config_path, 'w') as f:
                yaml.dump(self.original_config, f, default_flow_style=False, sort_keys=False)
            print("\n   ✓ Restored original config")
    
    def run_data_pipeline(self) -> bool:
        """
        Run the full data ingestion pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n   📊 Running Data Pipeline...")
        
        try:
            # Step 1: Ingest price data
            print("      [1/5] Ingesting price data...")
            from src.pipeline.ingest import main as ingest_main
            ingest_main()
            
            # Step 2: Ingest news data
            print("      [2/5] Ingesting news data...")
            from src.pipeline.ingest_news import main as ingest_news_main
            ingest_news_main()
            
            # Step 3: Process news features
            print("      [3/5] Processing news features (Lite Mode)...")
            from src.pipeline.process_news import main as process_news_main
            process_news_main(mode='lite', use_effective_date=True)
            
            # Step 4: Create retail proxy
            print("      [4/5] Creating retail proxy signals...")
            from src.pipeline.create_reddit_proxy import main as reddit_proxy_main
            reddit_proxy_main()
            
            # Step 5: De-seasonalize targets
            print("      [5/5] De-seasonalizing targets...")
            from src.pipeline.deseasonalize import main as deseasonalize_main
            deseasonalize_main()
            
            print("      ✓ Data pipeline complete!")
            return True
            
        except Exception as e:
            print(f"      ❌ Data pipeline failed: {e}")
            return False
    
    def run_training_pipeline(self, sector_map: dict) -> dict:
        """
        Run the full training pipeline.
        
        Args:
            sector_map: Dictionary mapping tickers to sectors
            
        Returns:
            Dictionary with training results
        """
        print("\n   🎯 Running Training Pipeline...")
        
        mlflow.end_run()  # End any stale runs
        
        try:
            # Step 1: Train Technical Agent
            print("      [1/4] Training Technical Agent (HAR-RV)...")
            from src.agents.technical_agent import TechnicalAgent
            
            tech_agent = TechnicalAgent(use_deseasonalized=True)
            tech_df = tech_agent.load_and_process_data()
            tech_metrics = tech_agent.train(tech_df)
            tech_agent.save_residuals()
            tech_r2 = tech_metrics['test']['R2']
            print(f"            Test R²: {tech_r2:.4f}")
            mlflow.end_run()
            
            # Step 2: Train News Classifier
            print("      [2/4] Training News Classifier...")
            from src.agents.news_agent import NewsAgent
            
            news_agent = NewsAgent(extreme_percentile=0.80)
            news_df = news_agent.load_and_merge_data()
            news_metrics = news_agent.train(news_df)
            news_auc = news_metrics['test']['AUC']
            print(f"            Test AUC: {news_auc:.4f}")
            mlflow.end_run()
            
            # Step 3: Train Retail Agent
            print("      [3/4] Training Retail Regime Agent...")
            from src.agents.retail_agent import RetailRegimeAgent
            
            retail_agent = RetailRegimeAgent()
            retail_df = retail_agent.load_and_process_data()
            retail_metrics = retail_agent.train(retail_df)
            
            # Save retail predictions
            retail_risk = retail_agent.predict_proba(retail_df)
            retail_df["retail_risk_score"] = retail_risk
            retail_df[["date", "ticker", "retail_risk_score"]].to_parquet(
                PROJECT_ROOT / "data/processed/retail_predictions.parquet", index=False
            )
            retail_auc = retail_metrics['test']['AUC']
            print(f"            Test AUC: {retail_auc:.4f}")
            mlflow.end_run()
            
            # Step 4: Train Coordinator
            print("      [4/4] Training RIVE Coordinator...")
            from src.coordinator.fusion import RiveCoordinator
            
            targets_df = pd.read_parquet(PROJECT_ROOT / "data/processed/targets_deseasonalized.parquet")
            residuals_df = pd.read_parquet(PROJECT_ROOT / "data/processed/residuals.parquet")
            
            coordinator = RiveCoordinator(alpha=100.0, winsorize_pct=0.02)
            coord_df = coordinator.prepare_predictions_dataset(
                tech_agent=tech_agent,
                news_agent=news_agent,
                retail_agent=retail_agent,
                targets_df=targets_df,
                residuals_df=residuals_df
            )
            
            # Add sector column
            coord_df["sector"] = coord_df["ticker"].map(sector_map)
            
            coord_metrics = coordinator.train(coord_df)
            coord_r2 = coord_metrics['test']['R2']
            print(f"            Test R²: {coord_r2:.4f}")
            mlflow.end_run()
            
            # Calculate sector breakdown
            sector_r2 = coord_metrics.get('sector', {})
            
            return {
                'tech_r2': tech_r2,
                'news_auc': news_auc,
                'retail_auc': retail_auc,
                'coordinator_r2': coord_r2,
                'baseline_r2': coord_metrics.get('baseline', {}).get('R2', 0),
                'sector_r2': sector_r2,
                'rmse': coord_metrics['test']['RMSE'],
                'n_tickers': len(coord_df['ticker'].unique()),
                'n_samples': len(coord_df)
            }
            
        except Exception as e:
            print(f"      ❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_results(self, experiment_name: str, results: dict):
        """
        Save experiment results to CSV.
        
        Args:
            experiment_name: Name of the experiment
            results: Dictionary with results
        """
        output_dir = PROJECT_ROOT / "reports/scale_up"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary DataFrame
        summary = pd.DataFrame([{
            'experiment': experiment_name,
            'n_tickers': results['n_tickers'],
            'n_samples': results['n_samples'],
            'tech_r2': results['tech_r2'],
            'news_auc': results['news_auc'],
            'retail_auc': results['retail_auc'],
            'coordinator_r2': results['coordinator_r2'],
            'baseline_r2': results['baseline_r2'],
            'rmse': results['rmse'],
            'timestamp': datetime.now().isoformat()
        }])
        
        summary_path = output_dir / f"{experiment_name}_results.csv"
        summary.to_csv(summary_path, index=False)
        print(f"      ✓ Saved results to {summary_path}")
        
        # Save sector breakdown
        if results.get('sector_r2'):
            sector_df = pd.DataFrame([
                {'sector': sector, 'r2': r2}
                for sector, r2 in results['sector_r2'].items()
            ]).sort_values('r2', ascending=False)
            
            sector_path = output_dir / f"{experiment_name}_sectors.csv"
            sector_df.to_csv(sector_path, index=False)
            print(f"      ✓ Saved sector breakdown to {sector_path}")
    
    def run_experiment(self, name: str, tickers: list, sector_map: dict) -> dict:
        """
        Run a complete experiment for one universe.
        
        Args:
            name: Experiment name
            tickers: List of ticker symbols
            sector_map: Sector mapping dictionary
            
        Returns:
            Results dictionary
        """
        print("\n" + "=" * 70)
        print(f"🚀 RUNNING EXPERIMENT: {name}")
        print(f"   Universe: {len(tickers)} tickers")
        print("=" * 70)
        
        start_time = datetime.now()
        
        # Setup
        self.setup_experiment(name, tickers)
        
        # Run data pipeline
        data_success = self.run_data_pipeline()
        if not data_success:
            print(f"   ❌ Experiment {name} failed at data pipeline")
            return None
        
        # Run training
        results = self.run_training_pipeline(sector_map)
        if results is None:
            print(f"   ❌ Experiment {name} failed at training")
            return None
        
        # Save results
        self.save_results(name, results)
        
        # Store in memory
        self.results[name] = results
        
        duration = datetime.now() - start_time
        
        print(f"\n   ✅ Experiment {name} complete!")
        print(f"      Duration: {duration}")
        print(f"      Final R²: {results['coordinator_r2']:.4f} ({results['coordinator_r2']*100:.2f}%)")
        
        return results
    
    def print_comparison(self):
        """Print comparison table of all experiments."""
        if len(self.results) < 2:
            print("\n   ⚠️ Need at least 2 experiments to compare")
            return
        
        print("\n" + "=" * 80)
        print("📊 SCALE-UP EXPERIMENT COMPARISON")
        print("=" * 80)
        
        # Header
        print(f"\n   {'Universe':<25} {'Tickers':>8} {'Test R²':>10} {'RMSE':>10} {'Baseline':>10}")
        print("   " + "-" * 65)
        
        for name, results in self.results.items():
            r2_pct = results['coordinator_r2'] * 100
            baseline_pct = results['baseline_r2'] * 100
            print(f"   {name:<25} {results['n_tickers']:>8} {r2_pct:>9.2f}% {results['rmse']:>10.4f} {baseline_pct:>9.2f}%")
        
        # Sector comparison
        print(f"\n   📊 SECTOR BREAKDOWN:")
        print("   " + "-" * 65)
        
        # Get all unique sectors
        all_sectors = set()
        for results in self.results.values():
            all_sectors.update(results.get('sector_r2', {}).keys())
        
        for sector in sorted(all_sectors):
            row = f"   {sector:<20}"
            for name, results in self.results.items():
                sector_r2 = results.get('sector_r2', {}).get(sector, float('nan'))
                if pd.notna(sector_r2):
                    row += f" {sector_r2*100:>10.2f}%"
                else:
                    row += f" {'N/A':>10}"
            print(row)
        
        # Best/Worst analysis
        print(f"\n   📈 BEST PERFORMERS:")
        for name, results in self.results.items():
            sector_r2 = results.get('sector_r2', {})
            if sector_r2:
                best = max(sector_r2.items(), key=lambda x: x[1])
                worst = min(sector_r2.items(), key=lambda x: x[1])
                print(f"      {name}:")
                print(f"         Best:  {best[0]} ({best[1]*100:.2f}%)")
                print(f"         Worst: {worst[0]} ({worst[1]*100:.2f}%)")
        
        print("\n" + "=" * 80)


def main():
    """Main entry point for scale-up experiments."""
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🚀 TITAN V15 SCALE-UP EXPERIMENTS")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    orchestrator = ScaleUpOrchestrator()
    
    try:
        # Experiment 1: Top 50 Most Active U.S. Stocks
        results_a = orchestrator.run_experiment(
            name="Top_50_Active",
            tickers=TOP_50_ACTIVE,
            sector_map=SECTOR_MAP_ACTIVE
        )
        
        # Experiment 2: S&P 500 GICS Sector-Balanced 55
        results_b = orchestrator.run_experiment(
            name="GICS_Balanced_55",
            tickers=GICS_BALANCED_55,
            sector_map=SECTOR_MAP_SP500
        )
        
        # Print comparison
        orchestrator.print_comparison()
        
    finally:
        # Always restore original config
        orchestrator.restore_original_config()
    
    duration = datetime.now() - start_time
    print(f"\n   Total Duration: {duration}")
    print(f"   Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return orchestrator.results


if __name__ == "__main__":
    main()

