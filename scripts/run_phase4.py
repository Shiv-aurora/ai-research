"""
Phase 4: Alpha Agents Execution Script

This script:
1. Creates the Reddit Proxy (volume/price anomalies)
2. Trains FundamentalAgent on residuals
3. Trains RedditAgent on residuals
4. Prints feature importance for each
5. Summarizes all agent performance

Usage:
    python scripts/run_phase4.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.create_reddit_proxy import create_reddit_proxy
from src.agents.alpha_agents import FundamentalAgent, RedditAgent


def main():
    """Run Phase 4: Alpha Agents."""
    print("\n" + "=" * 70)
    print("🚀 PHASE 4: ALPHA AGENTS")
    print("   Building FundamentalAgent and RedditAgent")
    print("=" * 70)
    
    # =========================================================
    # STEP 1: Create Reddit Proxy
    # =========================================================
    print("\n" + "=" * 70)
    print("📊 STEP 1: CREATE REDDIT PROXY")
    print("=" * 70)
    
    try:
        create_reddit_proxy()
        reddit_proxy_ready = True
    except Exception as e:
        print(f"   ⚠️ Error creating Reddit proxy: {e}")
        reddit_proxy_ready = False
    
    # =========================================================
    # STEP 2: Train FundamentalAgent
    # =========================================================
    print("\n" + "=" * 70)
    print("📊 STEP 2: FUNDAMENTAL AGENT")
    print("=" * 70)
    
    fund_agent = FundamentalAgent()
    fund_metrics = fund_agent.train()
    
    print(f"\n📈 FundamentalAgent Results:")
    print(f"   {'Metric':<20} {'Train':>10} {'Test':>10}")
    print("   " + "-" * 42)
    print(f"   {'RMSE':<20} {fund_metrics['train']['RMSE']:>10.4f} {fund_metrics['test']['RMSE']:>10.4f}")
    print(f"   {'MAE':<20} {fund_metrics['train']['MAE']:>10.4f} {fund_metrics['test']['MAE']:>10.4f}")
    print(f"   {'R²':<20} {fund_metrics['train']['R2']:>10.4f} {fund_metrics['test']['R2']:>10.4f}")
    
    print("\n   Top 3 Features:")
    fund_importance = fund_agent.get_feature_importance()
    for i, row in fund_importance.head(3).iterrows():
        print(f"      {i+1}. {row['feature']:<25} {row['pct']:>5.1f}%")
    
    # =========================================================
    # STEP 3: Train RedditAgent
    # =========================================================
    print("\n" + "=" * 70)
    print("📊 STEP 3: REDDIT AGENT")
    print("=" * 70)
    
    reddit_metrics = None
    reddit_importance = None
    
    if reddit_proxy_ready:
        try:
            reddit_agent = RedditAgent()
            reddit_metrics = reddit_agent.train()
            
            print(f"\n📈 RedditAgent Results:")
            print(f"   {'Metric':<20} {'Train':>10} {'Test':>10}")
            print("   " + "-" * 42)
            print(f"   {'RMSE':<20} {reddit_metrics['train']['RMSE']:>10.4f} {reddit_metrics['test']['RMSE']:>10.4f}")
            print(f"   {'MAE':<20} {reddit_metrics['train']['MAE']:>10.4f} {reddit_metrics['test']['MAE']:>10.4f}")
            print(f"   {'R²':<20} {reddit_metrics['train']['R2']:>10.4f} {reddit_metrics['test']['R2']:>10.4f}")
            
            print("\n   Top 3 Features:")
            reddit_importance = reddit_agent.get_feature_importance()
            for i, row in reddit_importance.head(3).iterrows():
                print(f"      {i+1}. {row['feature']:<25} {row['pct']:>5.1f}%")
        except Exception as e:
            print(f"   ⚠️ Error training RedditAgent: {e}")
    else:
        print("   ⚠️ Skipping RedditAgent (proxy not ready)")
    
    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("📊 PHASE 4 SUMMARY")
    print("=" * 70)
    
    print("\n   Agent Performance (Test R²):")
    print("   " + "-" * 40)
    print(f"   FundamentalAgent:   {fund_metrics['test']['R2']:.4f} ({fund_metrics['test']['R2']*100:.2f}%)")
    
    if reddit_metrics:
        print(f"   RedditAgent:        {reddit_metrics['test']['R2']:.4f} ({reddit_metrics['test']['R2']*100:.2f}%)")
    
    # Assessment
    print("\n   Assessment:")
    
    fund_r2 = fund_metrics['test']['R2']
    if fund_r2 > 0.05:
        print("   ✅ FundamentalAgent: Strong signal!")
    elif fund_r2 > 0:
        print("   ⚠️ FundamentalAgent: Weak positive signal")
    else:
        print("   ❌ FundamentalAgent: No predictive power")
    
    if reddit_metrics:
        reddit_r2 = reddit_metrics['test']['R2']
        if reddit_r2 > 0.05:
            print("   ✅ RedditAgent: Strong signal!")
        elif reddit_r2 > 0:
            print("   ⚠️ RedditAgent: Weak positive signal")
        else:
            print("   ❌ RedditAgent: No predictive power")
    
    print("\n   Top Features Across Agents:")
    print("   " + "-" * 40)
    print(f"   Fundamental #1: {fund_importance.iloc[0]['feature']}")
    if reddit_importance is not None:
        print(f"   Reddit #1:      {reddit_importance.iloc[0]['feature']}")
    
    print("\n" + "=" * 70)
    print("✅ PHASE 4 COMPLETE")
    print("   Next: Phase 5 - Hybrid Ensemble combining all agents")
    print("=" * 70)


if __name__ == "__main__":
    main()

