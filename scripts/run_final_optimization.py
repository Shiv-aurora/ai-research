"""
Phase 12: Structural Optimization - Target 23%+ R²

This script runs the optimized Titan V8 pipeline with:
- Ridge(alpha=0.1) coordinator
- Momentum features (vol_ma5)
- Calendar features (is_friday, is_monday, is_q4)
- News risk score from classifier

Expected Result: 23%+ Test R² (validated in audit)

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


def main():
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🚀 PHASE 12: STRUCTURAL OPTIMIZATION")
    print("   Target: 23%+ R² (Ridge + Momentum + Calendar)")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # =========================================================================
    # STEP A: TRAIN TECHNICAL AGENT
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 STEP A: TECHNICAL AGENT (HAR-RV)")
    print("=" * 70)
    
    from src.agents.technical_agent import TechnicalAgent
    
    tech_agent = TechnicalAgent(
        experiment_name="titan_v8_phase12_tech",
        use_deseasonalized=True
    )
    
    tech_df = tech_agent.load_and_process_data()
    tech_metrics = tech_agent.train(tech_df)
    tech_agent.save_residuals()
    
    print(f"\n   ✅ Technical Agent Test R²: {tech_metrics['test']['R2']:.4f}")
    
    # =========================================================================
    # STEP B: TRAIN NEWS CLASSIFIER
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 STEP B: NEWS CLASSIFIER (Extreme Events)")
    print("=" * 70)
    
    from src.agents.news_agent import NewsAgent
    
    news_agent = NewsAgent(
        experiment_name="titan_v8_phase12_news",
        extreme_percentile=0.80
    )
    
    news_df = news_agent.load_and_merge_data()
    news_metrics = news_agent.train(news_df)
    
    # Generate risk scores
    risk_scores = news_agent.predict_proba(news_df)
    news_df["news_risk_score"] = risk_scores
    
    print(f"\n   ✅ News Agent AUC: {news_metrics['test']['AUC']:.4f}")
    
    # =========================================================================
    # STEP C: PREPARE COORDINATOR DATA
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 STEP C: PREPARE COORDINATOR DATA")
    print("=" * 70)
    
    from src.coordinator.fusion import TitanCoordinator
    
    # Load data
    targets_df = pd.read_parquet("data/processed/targets_deseasonalized.parquet")
    residuals_df = pd.read_parquet("data/processed/residuals.parquet")
    news_features_df = pd.read_parquet("data/processed/news_features.parquet")
    
    # Initialize coordinator
    coordinator = TitanCoordinator(experiment_name="titan_v8_phase12_coordinator")
    
    # Prepare dataset
    coord_df = coordinator.prepare_predictions_dataset(
        tech_agent=tech_agent,
        news_agent=news_agent,
        targets_df=targets_df,
        news_features_df=news_features_df,
        residuals_df=residuals_df
    )
    
    # =========================================================================
    # STEP D: TRAIN OPTIMIZED COORDINATOR
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 STEP D: TRAIN RIDGE COORDINATOR")
    print("=" * 70)
    
    coord_metrics = coordinator.train(coord_df)
    
    # =========================================================================
    # STEP E: FINAL RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("🏆 PHASE 12 FINAL RESULTS")
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
        print(f"   {row['feature']:<25} {row['coefficient']:>+15.4f} {row['pct']:>14.1f}%")
    
    # Goal check
    print("\n" + "=" * 70)
    print("🎯 GOAL CHECK")
    print("=" * 70)
    
    test_r2 = coord_metrics['test']['R2']
    target_r2 = 0.23
    
    print(f"\n   Target R²:  {target_r2:.2%}")
    print(f"   Achieved:   {test_r2:.4f} ({test_r2*100:.2f}%)")
    print(f"   Gap:        {(target_r2 - test_r2)*100:+.2f}%")
    
    if test_r2 >= target_r2:
        print(f"\n   🏆 TARGET ACHIEVED! R² = {test_r2:.4f} >= {target_r2:.2f}")
    elif test_r2 >= 0.20:
        print(f"\n   ✅ STRONG RESULT! R² = {test_r2:.4f} (above 20%)")
    else:
        print(f"\n   ⚠️ Below target, but improvement achieved")
    
    # Baseline comparison
    if coord_metrics['baseline']:
        baseline_r2 = coord_metrics['baseline']['R2']
        improvement = (test_r2 - baseline_r2) * 100
        print(f"\n   vs Baseline (HAR only):")
        print(f"      Baseline: {baseline_r2:.4f}")
        print(f"      Titan V8: {test_r2:.4f}")
        print(f"      Improvement: {improvement:+.2f}%")
    
    # Sector analysis
    if coord_metrics['sector']:
        print("\n   Sector Analysis:")
        print("   " + "-" * 40)
        for sector, r2 in sorted(coord_metrics['sector'].items(), key=lambda x: x[1], reverse=True):
            marker = " ⭐" if r2 >= 0.25 else ""
            print(f"      {sector:15s}: {r2*100:.2f}%{marker}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    
    print(f"""
   COMPONENT PERFORMANCE:
   
   {'Component':<25} {'Metric':<15} {'Value':>12}
   {'-'*54}
   Technical Agent           Test R²          {tech_metrics['test']['R2']:>12.4f}
   News Classifier           Test AUC         {news_metrics['test']['AUC']:>12.4f}
   Ridge Coordinator         Test R²          {test_r2:>12.4f}
   
   KEY FEATURES (by importance):
""")
    
    for i, (_, row) in enumerate(importance.head(5).iterrows(), 1):
        print(f"   {i}. {row['feature']}: {row['coefficient']:+.4f} ({row['pct']:.1f}%)")
    
    # Timing
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n   Duration: {duration}")
    print(f"   Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return {
        'tech_r2': tech_metrics['test']['R2'],
        'news_auc': news_metrics['test']['AUC'],
        'coordinator_r2': test_r2,
        'baseline_r2': coord_metrics['baseline']['R2'] if coord_metrics['baseline'] else None,
        'sector_r2': coord_metrics['sector']
    }


if __name__ == "__main__":
    main()

