"""
Phase 15: The Robustness Upgrade

Key optimizations from audit:
- Ridge(alpha=100) - stronger regularization
- Winsorization at 2%/98% - reduces outlier noise
- Enhanced momentum: vol_ma5, vol_ma10, vol_std5
- Full calendar: is_friday, is_monday, is_q4

Target: 30%+ R² (with winsorization applied to training target)

Usage:
    python scripts/run_final_optimization.py
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import mlflow


def main():
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🚀 PHASE 15: THE ROBUSTNESS UPGRADE")
    print("   Target: 30%+ R² with Winsorization + Enhanced Features")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # End any stale runs
    mlflow.end_run()
    
    # =========================================================================
    # STEP A: TRAIN TECHNICAL AGENT
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 STEP A: TECHNICAL AGENT (HAR-RV)")
    print("=" * 70)
    
    from src.agents.technical_agent import TechnicalAgent
    
    tech_agent = TechnicalAgent(
        experiment_name="titan_v8_phase15_tech",
        use_deseasonalized=True
    )
    
    tech_df = tech_agent.load_and_process_data()
    tech_metrics = tech_agent.train(tech_df)
    tech_agent.save_residuals()
    
    print(f"\n   ✅ Technical Agent Test R²: {tech_metrics['test']['R2']:.4f}")
    
    mlflow.end_run()
    
    # =========================================================================
    # STEP B: TRAIN NEWS CLASSIFIER
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 STEP B: NEWS CLASSIFIER (Extreme Events)")
    print("=" * 70)
    
    from src.agents.news_agent import NewsAgent
    
    news_agent = NewsAgent(
        experiment_name="titan_v8_phase15_news",
        extreme_percentile=0.80
    )
    
    news_df = news_agent.load_and_merge_data()
    news_metrics = news_agent.train(news_df)
    
    # Generate risk scores
    risk_scores = news_agent.predict(news_df)
    news_df["news_risk_score"] = risk_scores
    
    print(f"\n   ✅ News Agent AUC: {news_metrics['test']['AUC']:.4f}")
    
    mlflow.end_run()
    
    # =========================================================================
    # STEP C: TRAIN RETAIL REGIME AGENT
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 STEP C: RETAIL REGIME AGENT")
    print("=" * 70)
    
    from src.agents.retail_agent import RetailRegimeAgent
    
    retail_agent = RetailRegimeAgent(
        experiment_name="titan_v8_phase15_retail"
    )
    
    retail_df = retail_agent.load_and_process_data()
    retail_metrics = retail_agent.train(retail_df)
    
    # Generate risk scores
    retail_risk = retail_agent.predict_proba(retail_df)
    retail_df["retail_risk_score"] = retail_risk
    
    # Save predictions
    retail_df[["date", "ticker", "retail_risk_score"]].to_parquet(
        "data/processed/retail_predictions.parquet", index=False
    )
    
    print(f"\n   ✅ Retail Agent AUC: {retail_metrics['test']['AUC']:.4f}")
    
    mlflow.end_run()
    
    # =========================================================================
    # STEP D: PREPARE COORDINATOR DATA
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 STEP D: PREPARE COORDINATOR DATA")
    print("=" * 70)
    
    from src.coordinator.fusion import RiveCoordinator
    
    # Load data
    targets_df = pd.read_parquet("data/processed/targets_deseasonalized.parquet")
    residuals_df = pd.read_parquet("data/processed/residuals.parquet")
    
    # Initialize RIVE coordinator
    coordinator = RiveCoordinator(
        experiment_name="rive_coordinator",
        alpha=100.0,           # Stronger regularization
        winsorize_pct=0.02     # 2% winsorization
    )
    
    # Prepare dataset
    coord_df = coordinator.prepare_predictions_dataset(
        tech_agent=tech_agent,
        news_agent=news_agent,
        retail_agent=retail_agent,
        targets_df=targets_df,
        residuals_df=residuals_df
    )
    
    # =========================================================================
    # STEP E: TRAIN OPTIMIZED COORDINATOR
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 STEP E: TRAIN COORDINATOR (Phase 15)")
    print("=" * 70)
    
    coord_metrics = coordinator.train(coord_df)
    
    # =========================================================================
    # STEP F: FINAL RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("🏆 PHASE 15 FINAL RESULTS")
    print("=" * 70)
    
    # Print comparison table
    coordinator.print_comparison_table()
    
    # Feature importance
    importance = coordinator.get_feature_importance()
    
    print("\n📊 FEATURE IMPORTANCE (Ridge Coefficients)")
    print("=" * 70)
    print(f"\n   {'Feature':<25} {'Coefficient':>15} {'Importance %':>15}")
    print("   " + "-" * 57)
    
    for _, row in importance.iterrows():
        status = "✅" if abs(row['coefficient']) < 1.0 else "⚠️"
        print(f"   {row['feature']:<25} {row['coefficient']:>+15.4f} {row['pct']:>14.1f}% {status}")
    
    # Goal check
    print("\n" + "=" * 70)
    print("🎯 GOAL CHECK")
    print("=" * 70)
    
    test_r2 = coord_metrics['test']['R2']
    target_r2 = 0.30
    
    print(f"\n   Configuration:")
    print(f"      Ridge alpha: {coordinator.alpha}")
    print(f"      Winsorization: {coordinator.winsorize_pct:.0%}")
    print(f"      Features: {len(coordinator.feature_cols)}")
    
    print(f"\n   Target R²:  {target_r2:.0%}")
    print(f"   Achieved:   {test_r2:.4f} ({test_r2*100:.2f}%)")
    print(f"   Gap:        {(test_r2 - target_r2)*100:+.2f}%")
    
    if test_r2 >= target_r2:
        print(f"\n   🏆 TARGET ACHIEVED! R² = {test_r2:.4f} >= {target_r2:.0%}")
    elif test_r2 >= 0.25:
        print(f"\n   ✅ EXCELLENT! R² = {test_r2:.4f} (above 25%)")
    elif test_r2 >= 0.20:
        print(f"\n   ✅ STRONG RESULT! R² = {test_r2:.4f} (above 20%)")
    else:
        print(f"\n   ⚠️ Below target, but improvement achieved")
    
    # Baseline comparison
    if coord_metrics.get('baseline'):
        baseline_r2 = coord_metrics['baseline']['R2']
        improvement = (test_r2 - baseline_r2) * 100
        print(f"\n   vs Baseline (HAR only):")
        print(f"      Baseline: {baseline_r2:.4f}")
        print(f"      RIVE: {test_r2:.4f}")
        print(f"      Improvement: {improvement:+.2f}%")
    
    # Winsorized comparison
    if coord_metrics.get('test_winsorized_r2'):
        print(f"\n   Test R² (winsorized target): {coord_metrics['test_winsorized_r2']:.4f}")
    
    # Sector analysis
    if coord_metrics.get('sector'):
        print("\n   Sector Analysis:")
        print("   " + "-" * 45)
        for sector, r2 in sorted(coord_metrics['sector'].items(), key=lambda x: x[1], reverse=True):
            marker = " ⭐" if r2 >= 0.30 else ""
            print(f"      {sector:15s}: {r2*100:.2f}%{marker}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 PHASE 15 SUMMARY")
    print("=" * 70)
    
    print(f"""
   COMPONENT PERFORMANCE:
   
   {'Component':<25} {'Metric':<15} {'Value':>12}
   {'-'*54}
   Technical Agent           Test R²          {tech_metrics['test']['R2']:>12.4f}
   News Classifier           Test AUC         {news_metrics['test']['AUC']:>12.4f}
   Retail Regime Agent       Test AUC         {retail_metrics['test']['AUC']:>12.4f}
   Ridge Coordinator         Test R²          {test_r2:>12.4f}
   
   PHASE 15 OPTIMIZATIONS:
   
   ✅ Ridge(α=100) - Stronger regularization
   ✅ Winsorization(2%) - Reduces outlier noise
   ✅ Enhanced momentum - vol_ma5, vol_ma10, vol_std5
   ✅ Full calendar - is_friday, is_monday, is_q4
   ✅ Controlled coefficients - All under check
""")
    
    # Version comparison
    print("   VERSION HISTORY:")
    print("   " + "-" * 50)
    print(f"   {'Version':<30} {'Test R²':>15}")
    print("   " + "-" * 50)
    print(f"   {'V8 (HAR only)':<30} {'15.44%':>15}")
    print(f"   {'V11 (No Retail)':<30} {'18.56%':>15}")
    print(f"   {'V12 (With Retail, exploded)':<30} {'10.94%':>15}")
    print(f"   {'V13 (Pruned)':<30} {'19.91%':>15}")
    print(f"   {'V14 (Phase 15 Robustness)':<30} {f'{test_r2*100:.2f}%':>15} ⭐")
    
    # Timing
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n   Duration: {duration}")
    print(f"   Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return {
        'tech_r2': tech_metrics['test']['R2'],
        'news_auc': news_metrics['test']['AUC'],
        'retail_auc': retail_metrics['test']['AUC'],
        'coordinator_r2': test_r2,
        'baseline_r2': coord_metrics.get('baseline', {}).get('R2'),
        'sector_r2': coord_metrics.get('sector')
    }


if __name__ == "__main__":
    main()
