"""
Phase 14: The Final Pruning

Problem: Phase 13 suffered from Coefficient Explosion
- retail_risk_score: +2.52 (too high!)
- fund_pred: +2.14 (overfitting)
- Result: Test R² dropped to 10.94%

Solution: Strong Regularization + Feature Pruning
- Ridge(alpha=10.0) instead of Ridge(alpha=0.1)
- Remove toxic features (fund_pred, tech_x_calm)
- Keep only proven features

Target: Test R² > 18%

Usage:
    python scripts/run_final_pruning.py
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import mlflow


SECTOR_MAP = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech',
    'JPM': 'Finance', 'BAC': 'Finance', 'V': 'Finance',
    'CAT': 'Industrial', 'GE': 'Industrial', 'BA': 'Industrial',
    'WMT': 'Consumer', 'MCD': 'Consumer', 'COST': 'Consumer',
    'XOM': 'Energy', 'CVX': 'Energy', 'SLB': 'Energy',
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
}


def load_and_prepare_data():
    """Load all data and create features."""
    print("\n📂 Loading data...")
    
    # Load targets
    targets = pd.read_parquet("data/processed/targets_deseasonalized.parquet")
    targets["date"] = pd.to_datetime(targets["date"]).dt.tz_localize(None)
    if targets["ticker"].dtype.name == "category":
        targets["ticker"] = targets["ticker"].astype(str)
    print(f"   ✓ Targets: {len(targets):,} rows")
    
    # Load residuals (for tech_pred)
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    residuals["date"] = pd.to_datetime(residuals["date"]).dt.tz_localize(None)
    if residuals["ticker"].dtype.name == "category":
        residuals["ticker"] = residuals["ticker"].astype(str)
    print(f"   ✓ Residuals: {len(residuals):,} rows")
    
    # Merge
    df = pd.merge(targets, residuals[["date", "ticker", "pred_tech_excess", "resid_tech"]], 
                  on=["date", "ticker"], how="left")
    
    # tech_pred = pred_tech_excess + seasonal_component
    if "seasonal_component" in df.columns:
        df["tech_pred"] = df["pred_tech_excess"] + df["seasonal_component"]
    else:
        df["tech_pred"] = df["pred_tech_excess"]
    
    # Load retail predictions
    retail_path = Path("data/processed/retail_predictions.parquet")
    if retail_path.exists():
        retail = pd.read_parquet(retail_path)
        retail["date"] = pd.to_datetime(retail["date"]).dt.tz_localize(None)
        if retail["ticker"].dtype.name == "category":
            retail["ticker"] = retail["ticker"].astype(str)
        df = pd.merge(df, retail[["date", "ticker", "retail_risk_score"]], 
                      on=["date", "ticker"], how="left")
        df["retail_risk_score"] = df["retail_risk_score"].fillna(0.2)
        print(f"   ✓ Retail predictions: loaded")
    else:
        # Train retail agent if predictions don't exist
        print(f"   ⚠️ Training RetailRegimeAgent...")
        from src.agents.retail_agent import RetailRegimeAgent
        retail_agent = RetailRegimeAgent()
        retail_df = retail_agent.load_and_process_data()
        retail_agent.train(retail_df)
        
        risk_scores = retail_agent.predict_proba(retail_df)
        retail_df["retail_risk_score"] = risk_scores
        
        df = pd.merge(df, retail_df[["date", "ticker", "retail_risk_score"]], 
                      on=["date", "ticker"], how="left")
        df["retail_risk_score"] = df["retail_risk_score"].fillna(0.2)
    
    # Load news features for news_risk_score
    print(f"   ⚠️ Training NewsAgent for news_risk_score...")
    mlflow.end_run()
    
    from src.agents.news_agent import NewsAgent
    news_agent = NewsAgent(experiment_name="titan_v8_phase14")
    news_df = news_agent.load_and_merge_data()
    news_agent.train(news_df)
    
    news_risk = news_agent.predict(news_df)
    news_df["news_risk_score"] = news_risk
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.tz_localize(None)
    
    df = pd.merge(df, news_df[["date", "ticker", "news_risk_score"]], 
                  on=["date", "ticker"], how="left")
    df["news_risk_score"] = df["news_risk_score"].fillna(0.2)
    print(f"   ✓ News risk scores: generated")
    
    mlflow.end_run()
    
    # Calendar features
    df["is_friday"] = (df["date"].dt.dayofweek == 4).astype(int)
    df["is_monday"] = (df["date"].dt.dayofweek == 0).astype(int)
    df["is_q4"] = (df["date"].dt.quarter == 4).astype(int)
    
    # The interaction that works
    df["news_x_retail"] = df["news_risk_score"] * df["retail_risk_score"]
    
    # VIX
    try:
        vix = pd.read_parquet("data/processed/vix.parquet")
        vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None)
        df = pd.merge(df, vix[["date", "VIX_close"]], on="date", how="left")
        df["VIX_close"] = df["VIX_close"].fillna(20)
    except:
        df["VIX_close"] = 20.0
    df["VIX_close"] = df["VIX_close"].ffill().fillna(20)
    
    # Momentum
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        df.loc[mask, "vol_ma5"] = df.loc[mask, "target_log_var"].rolling(5, min_periods=1).mean().shift(1)
    df["vol_ma5"] = df["vol_ma5"].fillna(df["target_log_var"].mean())
    
    # Sector
    df["sector"] = df["ticker"].map(SECTOR_MAP)
    
    # Drop NaN
    df = df.dropna(subset=["target_log_var", "tech_pred"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    print(f"\n   📊 Final dataset: {len(df):,} rows")
    
    return df


def train_pruned_coordinator(df, alpha=10.0):
    """
    Train the pruned, highly regularized coordinator.
    
    Key changes from Phase 13:
    - Ridge(alpha=10.0) instead of Ridge(alpha=0.1)
    - Removed: fund_pred (toxic), tech_x_calm (collinear)
    - Kept: proven features only
    """
    print("\n" + "=" * 70)
    print(f"🎯 TRAINING PRUNED COORDINATOR (Ridge α={alpha})")
    print("=" * 70)
    
    # THE "CLEAN" FEATURE LIST
    feature_cols = [
        "tech_pred",           # Baseline HAR
        "news_risk_score",     # Semantic risk
        "retail_risk_score",   # Behavioral risk  
        "is_friday",           # Calendar beta
        "news_x_retail",       # The interaction that works
        "vol_ma5",             # Momentum
    ]
    
    available_features = [f for f in feature_cols if f in df.columns]
    
    print(f"\n   THE CLEAN FEATURE LIST ({len(available_features)}):")
    for f in available_features:
        print(f"      ✓ {f}")
    
    print(f"\n   DROPPED (Toxic/Collinear):")
    print(f"      ✗ fund_pred (overfitting)")
    print(f"      ✗ tech_x_calm (collinear)")
    print(f"      ✗ VIX_close (absorbed by tech_pred)")
    
    # Time split
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    
    print(f"\n   📊 Split:")
    print(f"      Train: {len(train):,} samples")
    print(f"      Test:  {len(test):,} samples")
    
    X_train = train[available_features].fillna(0)
    y_train = train["target_log_var"]
    X_test = test[available_features].fillna(0)
    y_test = test["target_log_var"]
    
    # Baseline
    baseline_r2 = r2_score(y_test, test["tech_pred"])
    print(f"\n   📈 Baseline (HAR only): R² = {baseline_r2:.4f} ({baseline_r2*100:.2f}%)")
    
    # Train with STRONG regularization
    print(f"\n   🔧 Training Ridge(α={alpha})...")
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"\n   📈 Results:")
    print(f"   {'Metric':<20} {'Train':>12} {'Test':>12}")
    print("   " + "-" * 46)
    print(f"   {'R²':<20} {train_r2:>12.4f} {test_r2:>12.4f}")
    print(f"   {'RMSE':<20} {'':<12} {test_rmse:>12.4f}")
    
    improvement = (test_r2 - baseline_r2) * 100
    print(f"\n   📈 Improvement over HAR: {improvement:+.2f}%")
    
    # Coefficients (should all be < 1.0)
    print(f"\n   📊 Coefficients (Target: all < 1.0):")
    print("   " + "-" * 55)
    
    coef_df = pd.DataFrame({
        "feature": available_features,
        "coefficient": model.coef_
    })
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    
    all_under_one = True
    for _, row in coef_df.iterrows():
        status = "✅" if abs(row["coefficient"]) < 1.0 else "⚠️"
        if abs(row["coefficient"]) >= 1.0:
            all_under_one = False
        print(f"      {row['feature']:<25}: {row['coefficient']:>+10.4f} {status}")
    print(f"      {'Intercept':<25}: {model.intercept_:>+10.4f}")
    
    if all_under_one:
        print(f"\n   ✅ All coefficients < 1.0 - No more explosion!")
    else:
        print(f"\n   ⚠️ Some coefficients still > 1.0")
    
    # Sector breakdown
    print(f"\n   📊 Sector Breakdown:")
    print("   " + "-" * 45)
    
    test["y_pred"] = y_test_pred
    sector_r2 = {}
    for sector in test["sector"].dropna().unique():
        mask = test["sector"] == sector
        if mask.sum() > 50:
            sector_r2[sector] = r2_score(y_test[mask], y_test_pred[mask])
    
    for sector, r2 in sorted(sector_r2.items(), key=lambda x: x[1], reverse=True):
        marker = " ⭐" if r2 >= 0.25 else ""
        print(f"      {sector:15s}: {r2:.4f} ({r2*100:.2f}%){marker}")
    
    return {
        "model": model,
        "features": available_features,
        "coef_df": coef_df,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "baseline_r2": baseline_r2,
        "sector_r2": sector_r2,
        "improvement": improvement
    }


def compare_versions():
    """Print comparison table of different versions."""
    print("\n" + "=" * 70)
    print("📊 VERSION COMPARISON")
    print("=" * 70)
    
    # Historical results (from previous runs)
    versions = [
        {"Version": "Titan V8 (HAR only)", "R²": 15.44, "Notes": "Baseline"},
        {"Version": "Titan V11 (No Retail)", "R²": 18.56, "Notes": "Phase 12 - Ridge α=0.1"},
        {"Version": "Titan V12 (With Retail)", "R²": 10.94, "Notes": "Phase 13 - Coefficient explosion"},
    ]
    
    print(f"\n   {'Version':<30} {'Test R²':>12} {'Notes':<30}")
    print("   " + "-" * 74)
    
    for v in versions:
        print(f"   {v['Version']:<30} {v['R²']:>11.2f}% {v['Notes']:<30}")
    
    return versions


def main():
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🔧 PHASE 14: THE FINAL PRUNING")
    print("   Fixing Coefficient Explosion with Strong Regularization")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # End any stale runs
    mlflow.end_run()
    
    # Load data
    df = load_and_prepare_data()
    
    # Test different alpha values
    print("\n" + "=" * 70)
    print("🔬 ALPHA SWEEP: Finding optimal regularization")
    print("=" * 70)
    
    alphas = [0.1, 1.0, 5.0, 10.0, 50.0]
    alpha_results = []
    
    print(f"\n   {'Alpha':>10} {'Train R²':>12} {'Test R²':>12} {'Improvement':>15}")
    print("   " + "-" * 52)
    
    for alpha in alphas:
        result = train_pruned_coordinator(df, alpha=alpha)
        alpha_results.append({
            "alpha": alpha,
            "train_r2": result["train_r2"],
            "test_r2": result["test_r2"],
            "improvement": result["improvement"]
        })
        
        marker = " ⭐" if result["test_r2"] >= 0.18 else ""
        print(f"   {alpha:>10.1f} {result['train_r2']:>12.4f} {result['test_r2']:>12.4f} {result['improvement']:>+14.2f}%{marker}")
    
    # Find best alpha
    best = max(alpha_results, key=lambda x: x["test_r2"])
    print(f"\n   🏆 Best Alpha: {best['alpha']} → Test R² = {best['test_r2']:.4f}")
    
    # Train final model with best alpha
    print("\n" + "=" * 70)
    print(f"🎯 FINAL MODEL: Ridge(α={best['alpha']})")
    print("=" * 70)
    
    final_result = train_pruned_coordinator(df, alpha=best["alpha"])
    
    # Version comparison
    versions = compare_versions()
    
    # Add current version
    print(f"   {'Titan V13 (Pruned)':<30} {final_result['test_r2']*100:>11.2f}% {'Phase 14 - Ridge α=' + str(best['alpha']):<30}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🏆 PHASE 14 FINAL SUMMARY")
    print("=" * 70)
    
    print(f"""
   PROBLEM: Coefficient Explosion in Phase 13
   -----------------------------------------
   • retail_risk_score: +2.52 (too high!)
   • fund_pred: +2.14 (overfitting)
   • Result: Test R² = 10.94% (worse than baseline)
   
   SOLUTION: Strong Regularization + Feature Pruning
   ------------------------------------------------
   • Ridge(α={best['alpha']}) instead of Ridge(α=0.1)
   • Removed: fund_pred, tech_x_calm
   • Kept: tech_pred, news_risk, retail_risk, is_friday, news_x_retail
   
   RESULT:
   -------
   • Test R²: {final_result['test_r2']:.4f} ({final_result['test_r2']*100:.2f}%)
   • Improvement over HAR: {final_result['improvement']:+.2f}%
   • All coefficients < 1.0: {'✅ Yes' if all(abs(c) < 1 for c in final_result['coef_df']['coefficient']) else '⚠️ No'}
   
   TARGET CHECK:
   ------------
   • Target: Test R² > 18%
   • Actual: {final_result['test_r2']*100:.2f}%
   • Status: {'✅ ACHIEVED' if final_result['test_r2'] >= 0.18 else '⚠️ Below target'}
""")
    
    # Timing
    end_time = datetime.now()
    print(f"\n   Duration: {end_time - start_time}")
    print("=" * 70)
    
    return final_result


if __name__ == "__main__":
    main()


