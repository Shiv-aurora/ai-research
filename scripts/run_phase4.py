"""
Phase 4.5: Alpha Agents Execution Script

This script:
1. Ingests retail risk signals (BTC, GME, IWM)
2. Trains FundamentalAgent on residuals
3. Trains RetailRiskAgent on residuals
4. Prints feature importance for each
5. Summarizes all agent performance

Usage:
    python scripts/run_phase4.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.ingest_retail import fetch_retail_signals
from src.agents.alpha_agents import FundamentalAgent, RetailRiskAgent


def main():
    """Run Phase 4.5: Alpha Agents."""
    print("\n" + "=" * 70)
    print("🚀 PHASE 4.5: ALPHA AGENTS")
    print("   FundamentalAgent + RetailRiskAgent")
    print("=" * 70)
    
    # =========================================================
    # STEP 1: Ingest Retail Risk Signals
    # =========================================================
    print("\n" + "=" * 70)
    print("📊 STEP 1: INGEST RETAIL RISK SIGNALS")
    print("=" * 70)
    
    try:
        fetch_retail_signals()
        retail_signals_ready = True
    except Exception as e:
        print(f"   ⚠️ Error fetching retail signals: {e}")
        retail_signals_ready = False
    
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
    for i, (_, row) in enumerate(fund_importance.head(3).iterrows()):
        print(f"      {i+1}. {row['feature']:<25} {row['pct']:>5.1f}%")
    
    # =========================================================
    # STEP 3: Train RetailRiskAgent
    # =========================================================
    print("\n" + "=" * 70)
    print("📊 STEP 3: RETAIL RISK AGENT")
    print("=" * 70)
    
    retail_metrics = None
    retail_importance = None
    
    if retail_signals_ready:
        try:
            retail_agent = RetailRiskAgent()
            retail_metrics = retail_agent.train()
            
            print(f"\n📈 RetailRiskAgent Results:")
            print(f"   {'Metric':<20} {'Train':>10} {'Test':>10}")
            print("   " + "-" * 42)
            print(f"   {'RMSE':<20} {retail_metrics['train']['RMSE']:>10.4f} {retail_metrics['test']['RMSE']:>10.4f}")
            print(f"   {'MAE':<20} {retail_metrics['train']['MAE']:>10.4f} {retail_metrics['test']['MAE']:>10.4f}")
            print(f"   {'R²':<20} {retail_metrics['train']['R2']:>10.4f} {retail_metrics['test']['R2']:>10.4f}")
            
            print("\n   Top 3 Features:")
            retail_importance = retail_agent.get_feature_importance()
            for i, (_, row) in enumerate(retail_importance.head(3).iterrows()):
                print(f"      {i+1}. {row['feature']:<25} {row['pct']:>5.1f}%")
        except Exception as e:
            print(f"   ⚠️ Error training RetailRiskAgent: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   ⚠️ Skipping RetailRiskAgent (signals not ready)")
    
    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("📊 PHASE 4.5 SUMMARY")
    print("=" * 70)
    
    print("\n   Agent Performance (Test R²):")
    print("   " + "-" * 50)
    print(f"   FundamentalAgent:   {fund_metrics['test']['R2']:>7.4f} ({fund_metrics['test']['R2']*100:.2f}%)")
    
    if retail_metrics:
        print(f"   RetailRiskAgent:    {retail_metrics['test']['R2']:>7.4f} ({retail_metrics['test']['R2']*100:.2f}%)")
    
    # Assessment
    print("\n   Assessment:")
    
    fund_r2 = fund_metrics['test']['R2']
    if fund_r2 > 0.05:
        print("   ✅ FundamentalAgent: Strong signal!")
    elif fund_r2 > 0:
        print("   ⚠️ FundamentalAgent: Weak positive signal")
    else:
        print("   ❌ FundamentalAgent: No predictive power")
    
    if retail_metrics:
        retail_r2 = retail_metrics['test']['R2']
        if retail_r2 > 0.05:
            print("   ✅ RetailRiskAgent: Strong signal!")
        elif retail_r2 > 0.02:
            print("   ⚠️ RetailRiskAgent: Moderate signal")
        elif retail_r2 > 0:
            print("   ⚠️ RetailRiskAgent: Weak positive signal")
        else:
            print("   ❌ RetailRiskAgent: No predictive power")
    
    print("\n   Top Features Across Agents:")
    print("   " + "-" * 50)
    print(f"   Fundamental #1: {fund_importance.iloc[0]['feature']}")
    if retail_importance is not None and len(retail_importance) > 0:
        print(f"   Retail #1:      {retail_importance.iloc[0]['feature']}")
    
    # Agent Portfolio Summary
    print("\n   Current Agent Portfolio:")
    print("   " + "-" * 50)
    print("   Agent                Target            R²")
    print("   " + "-" * 50)
    print(f"   TechnicalAgent       target_log_var    ~40%")
    print(f"   NewsAgent            resid_tech        ~9%")
    print(f"   FundamentalAgent     resid_tech        {fund_r2*100:.1f}%")
    if retail_metrics:
        print(f"   RetailRiskAgent      resid_tech        {retail_r2*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ PHASE 4.5 COMPLETE")
    print("   Next: Phase 5 - Hybrid Ensemble combining all agents")
    print("=" * 70)


if __name__ == "__main__":
    main()
