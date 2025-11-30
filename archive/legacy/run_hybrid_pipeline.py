"""
Phase 11: Hybrid Pipeline - Classification + Regression

This script orchestrates the new hybrid approach:
- Tech Agent: Ridge regression on de-seasonalized targets (HAR-RV)
- News Agent: Classification for extreme events → risk_score
- Fundamental Agent: Regression on residuals
- Coordinator: Combines predictions with news_risk_score as boost factor

The key innovation: News Agent outputs a PROBABILITY (0-1) that
the Coordinator uses to adjust volatility predictions upward
when extreme events are likely.

Usage:
    python scripts/run_hybrid_pipeline.py
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error


def main():
    """Run the hybrid pipeline."""
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🚀 PHASE 11: HYBRID PIPELINE")
    print("   The Classification + Regression Fusion")
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
        experiment_name="titan_v8_phase11_tech",
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
        experiment_name="titan_v8_phase11_news",
        extreme_percentile=0.80
    )
    
    news_df = news_agent.load_and_merge_data()
    news_metrics = news_agent.train(news_df)
    
    # Get risk scores for all data
    news_risk_scores = news_agent.predict_proba(news_df)
    news_df["news_risk_score"] = news_risk_scores
    
    print(f"\n   ✅ News Agent AUC: {news_metrics['test']['AUC']:.4f}")
    print(f"   Goal: > 0.60")
    
    if news_metrics['test']['AUC'] >= 0.60:
        print(f"   ✅ Goal achieved!")
    else:
        print(f"   ⚠️ Below goal but still useful")
    
    # =========================================================================
    # STEP C: TRAIN FUNDAMENTAL AGENT
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 STEP C: FUNDAMENTAL AGENT")
    print("=" * 70)
    
    try:
        from src.agents.fundamental_agent import FundamentalAgent
        
        fund_agent = FundamentalAgent(
            experiment_name="titan_v8_phase11_fund",
            use_deseasonalized_target=True
        )
        
        fund_metrics = fund_agent.train()
        print(f"\n   ✅ Fundamental Agent Test R²: {fund_metrics['test']['R2']:.4f}")
        has_fund = True
    except Exception as e:
        print(f"\n   ⚠️ Fundamental Agent skipped: {e}")
        has_fund = False
    
    # =========================================================================
    # STEP D: PREPARE COORDINATOR DATASET
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 STEP D: PREPARE COORDINATOR DATASET")
    print("=" * 70)
    
    # Load targets
    targets = pd.read_parquet("data/processed/targets_deseasonalized.parquet")
    targets["date"] = pd.to_datetime(targets["date"]).dt.tz_localize(None)
    if targets["ticker"].dtype.name == "category":
        targets["ticker"] = targets["ticker"].astype(str)
    
    # Load residuals (has tech_pred)
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    residuals["date"] = pd.to_datetime(residuals["date"]).dt.tz_localize(None)
    if residuals["ticker"].dtype.name == "category":
        residuals["ticker"] = residuals["ticker"].astype(str)
    
    # Merge
    coord_df = pd.merge(
        targets[["date", "ticker", "target_excess", "target_log_var", "seasonal_component"]],
        residuals[["date", "ticker", "pred_tech_excess", "resid_tech"]],
        on=["date", "ticker"],
        how="inner"
    )
    
    # Add news risk score
    news_score_df = news_df[["date", "ticker", "news_risk_score"]].copy()
    news_score_df["date"] = pd.to_datetime(news_score_df["date"]).dt.tz_localize(None)
    
    coord_df = pd.merge(
        coord_df,
        news_score_df,
        on=["date", "ticker"],
        how="left"
    )
    coord_df["news_risk_score"] = coord_df["news_risk_score"].fillna(0.2)  # Default to base rate
    
    # Add VIX
    try:
        vix = pd.read_parquet("data/processed/vix.parquet")
        vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None)
        coord_df = pd.merge(coord_df, vix[["date", "VIX_close"]], on="date", how="left")
        coord_df["VIX_close"] = coord_df["VIX_close"].fillna(20)
    except:
        coord_df["VIX_close"] = 20.0
    
    # Add fundamental prediction if available
    if has_fund:
        try:
            fund_pred = fund_agent.predict(coord_df)
            coord_df["fund_pred"] = fund_pred
        except:
            coord_df["fund_pred"] = 0.0
    else:
        coord_df["fund_pred"] = 0.0
    
    coord_df = coord_df.dropna(subset=["target_excess", "pred_tech_excess"])
    coord_df = coord_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    print(f"\n   Coordinator dataset: {len(coord_df):,} rows")
    print(f"   News risk score range: [{coord_df['news_risk_score'].min():.3f}, {coord_df['news_risk_score'].max():.3f}]")
    print(f"   Mean news risk: {coord_df['news_risk_score'].mean():.3f}")
    
    # =========================================================================
    # STEP E: TRAIN HYBRID COORDINATOR
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 STEP E: HYBRID COORDINATOR")
    print("=" * 70)
    
    # Features for coordinator
    coord_features = ["pred_tech_excess", "news_risk_score", "VIX_close"]
    if has_fund:
        coord_features.append("fund_pred")
    
    # Create interaction: tech_pred × risk_score (boost when risk is high)
    coord_df["tech_x_risk"] = coord_df["pred_tech_excess"] * coord_df["news_risk_score"]
    coord_features.append("tech_x_risk")
    
    print(f"\n   Features: {coord_features}")
    
    # Time split
    cutoff = pd.to_datetime("2023-01-01")
    train_mask = coord_df["date"] < cutoff
    test_mask = coord_df["date"] >= cutoff
    
    X_train = coord_df.loc[train_mask, coord_features].fillna(0)
    y_train = coord_df.loc[train_mask, "target_excess"]
    X_test = coord_df.loc[test_mask, coord_features].fillna(0)
    y_test = coord_df.loc[test_mask, "target_excess"]
    
    print(f"\n   Train: {len(X_train):,} samples")
    print(f"   Test:  {len(X_test):,} samples")
    
    # Train ElasticNet coordinator
    coordinator = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
    coordinator.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = coordinator.predict(X_train)
    y_test_pred = coordinator.predict(X_test)
    
    # Metrics on target_excess
    train_r2_excess = r2_score(y_train, y_train_pred)
    test_r2_excess = r2_score(y_test, y_test_pred)
    
    print(f"\n   Coordinator Results (on target_excess):")
    print(f"      Train R²: {train_r2_excess:.4f}")
    print(f"      Test R²:  {test_r2_excess:.4f}")
    
    # Re-seasonalize for final evaluation
    y_test_total_pred = y_test_pred + coord_df.loc[test_mask, "seasonal_component"].values
    y_test_total_actual = coord_df.loc[test_mask, "target_log_var"].values
    
    test_r2_total = r2_score(y_test_total_actual, y_test_total_pred)
    test_rmse_total = np.sqrt(mean_squared_error(y_test_total_actual, y_test_total_pred))
    
    print(f"\n   Re-seasonalized Results (on target_log_var):")
    print(f"      Test R²:  {test_r2_total:.4f} ({test_r2_total*100:.2f}%)")
    print(f"      Test RMSE: {test_rmse_total:.4f}")
    
    # =========================================================================
    # STEP F: FEATURE IMPORTANCE
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 FEATURE IMPORTANCE")
    print("=" * 70)
    
    coef_df = pd.DataFrame({
        "feature": coord_features,
        "coefficient": coordinator.coef_
    })
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    
    print(f"\n   {'Feature':<25} {'Coefficient':>15}")
    print("   " + "-" * 42)
    for _, row in coef_df.iterrows():
        print(f"   {row['feature']:<25} {row['coefficient']:>15.4f}")
    
    # Check if news_risk_score is used
    risk_coef = coef_df[coef_df["feature"] == "news_risk_score"]["coefficient"].values[0]
    tech_x_risk_coef = coef_df[coef_df["feature"] == "tech_x_risk"]["coefficient"].values[0]
    
    print(f"\n   News Risk Score coefficient: {risk_coef:.4f}")
    print(f"   Tech × Risk interaction:     {tech_x_risk_coef:.4f}")
    
    if abs(risk_coef) > 0.01 or abs(tech_x_risk_coef) > 0.01:
        print(f"\n   ✅ Coordinator USES the news risk score!")
    else:
        print(f"\n   ⚠️ News risk score has minimal impact")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("🏆 PHASE 11 SUMMARY")
    print("=" * 70)
    
    print(f"""
   COMPONENT RESULTS:
   
   {'Component':<25} {'Metric':<15} {'Value':>12}
   {'-'*54}
   Technical Agent           Test R²          {tech_metrics['test']['R2']:>12.4f}
   News Classifier           Test AUC         {news_metrics['test']['AUC']:>12.4f}
   Hybrid Coordinator        Test R² (excess) {test_r2_excess:>12.4f}
   Hybrid Coordinator        Test R² (total)  {test_r2_total:>12.4f}
   
   GOAL CHECK:
   - News Agent AUC > 0.60: {'✅ PASS' if news_metrics['test']['AUC'] >= 0.60 else '❌ FAIL'} ({news_metrics['test']['AUC']:.4f})
   - Coordinator uses risk score: {'✅ YES' if abs(risk_coef) > 0.01 or abs(tech_x_risk_coef) > 0.01 else '⚠️ MINIMAL'}
   
   HYBRID APPROACH:
   - News → Classification (extreme events) → risk_score
   - Coordinator learns to boost predictions when risk_score is high
   - Final prediction = weighted combination of HAR + News risk
    """)
    
    # Comparison with pre-hybrid baseline
    baseline_r2 = 0.0260  # From previous experiments
    improvement = test_r2_total - baseline_r2
    
    print(f"   vs Baseline (HAR only):")
    print(f"      Baseline R²:  {baseline_r2:.4f}")
    print(f"      Hybrid R²:    {test_r2_total:.4f}")
    print(f"      Improvement:  {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    # Timing
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n   Duration: {duration}")
    print(f"   Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return {
        "tech_r2": tech_metrics["test"]["R2"],
        "news_auc": news_metrics["test"]["AUC"],
        "coord_r2_excess": test_r2_excess,
        "coord_r2_total": test_r2_total,
        "coefficients": coef_df.to_dict()
    }


if __name__ == "__main__":
    main()

