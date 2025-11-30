"""
Titan V8 Full Pipeline: 18-Ticker Universe with Retail Regime

Phase 13: The Retail Regime Agent Integration

Steps:
A. Verification - Check config and data
B. Train Anchors - TechnicalAgent, NewsAgent, RetailRegimeAgent
C. Feature Engineering - Create regime interactions (tech_x_calm)
D. Train Coordinator - Ridge with regime-adjusted features
E. Report - Full universe R², per-sector breakdown, feature importance

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --skip-ingest  # Use existing data
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
import mlflow

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# Sector mapping for 18-ticker universe
SECTOR_MAP = {
    # Tech
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech',
    # Finance
    'JPM': 'Finance', 'BAC': 'Finance', 'V': 'Finance',
    # Industrial
    'CAT': 'Industrial', 'GE': 'Industrial', 'BA': 'Industrial',
    # Consumer
    'WMT': 'Consumer', 'MCD': 'Consumer', 'COST': 'Consumer',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'SLB': 'Energy',
    # Healthcare
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
}


def load_config():
    """Load config WITHOUT overwriting - preserve 18-ticker universe."""
    config_path = Path("conf/base/config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def verify_data():
    """Step A: Verify configuration and data integrity."""
    print("\n" + "=" * 70)
    print("📋 STEP A: VERIFICATION")
    print("=" * 70)
    
    # Check config
    config = load_config()
    tickers = config["data"]["tickers"]
    
    print(f"\n   📊 Configuration:")
    print(f"      Tickers: {len(tickers)}")
    print(f"      Date range: {config['data']['start_date']} to {config['data']['end_date']}")
    
    # Check data
    targets_path = Path("data/processed/targets.parquet")
    if not targets_path.exists():
        print(f"\n   ❌ targets.parquet not found!")
        print(f"      Run: python scripts/run_universe_expansion.py")
        return None, None
    
    df = pd.read_parquet(targets_path)
    
    # Clean infinity values in target
    inf_count = np.isinf(df['target_log_var']).sum()
    if inf_count > 0:
        print(f"\n   ⚠️ Found {inf_count} infinity values in target - cleaning...")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['target_log_var'])
    
    print(f"\n   📊 Data Summary:")
    print(f"      Rows: {len(df):,}")
    print(f"      Tickers: {df['ticker'].nunique()}")
    print(f"      Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Sector breakdown
    if 'ticker' in df.columns:
        df['sector'] = df['ticker'].map(SECTOR_MAP).fillna('Other')
        sector_counts = df.groupby('sector').size()
        print(f"\n   📊 Sector Breakdown:")
        for sector in ['Tech', 'Finance', 'Industrial', 'Consumer', 'Energy', 'Healthcare']:
            if sector in sector_counts.index:
                print(f"      {sector:12}: {sector_counts[sector]:,} rows")
    
    return config, df


def train_technical_agent():
    """Train TechnicalAgent (HAR-RV) and save residuals."""
    print("\n" + "-" * 70)
    print("🔧 Training TechnicalAgent (HAR-RV Baseline)")
    print("-" * 70)
    
    # End any existing runs first
    mlflow.end_run()
    
    from src.agents.technical_agent import TechnicalAgent
    
    agent = TechnicalAgent(experiment_name="titan_v8_phase13", use_deseasonalized=True)
    df = agent.load_and_process_data()
    
    # Clean infinity values before training
    inf_count = np.isinf(df['target_excess']).sum() if 'target_excess' in df.columns else 0
    if inf_count > 0:
        print(f"\n   ⚠️ Cleaning {inf_count} infinity values...")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['target_excess'])
    
    agent.train(df)
    agent.save_residuals()
    
    print(f"\n   Train R²: {agent.train_metrics['R2']:.4f} ({agent.train_metrics['R2']*100:.2f}%)")
    print(f"   Test R²:  {agent.test_metrics['R2']:.4f} ({agent.test_metrics['R2']*100:.2f}%)")
    
    # End run after agent
    mlflow.end_run()
    
    return agent


def train_news_agent():
    """Train NewsAgent (Classifier) on residuals."""
    print("\n" + "-" * 70)
    print("🔧 Training NewsAgent (Extreme Event Classifier)")
    print("-" * 70)
    
    # End any existing runs
    mlflow.end_run()
    
    from src.agents.news_agent import NewsAgent
    
    agent = NewsAgent(experiment_name="titan_v8_phase13")
    df = agent.load_and_merge_data()
    metrics = agent.train(df)
    
    # Get predictions
    risk_scores = agent.predict(df)
    df['news_risk_score'] = risk_scores
    
    print(f"\n   Train AUC: {agent.train_metrics['AUC']:.4f}")
    print(f"   Test AUC:  {agent.test_metrics['AUC']:.4f}")
    
    # End run after agent
    mlflow.end_run()
    
    return agent


def train_retail_agent():
    """Train RetailRegimeAgent and save predictions."""
    print("\n" + "-" * 70)
    print("🔧 Training RetailRegimeAgent (Regime Classifier)")
    print("-" * 70)
    
    # End any existing runs
    mlflow.end_run()
    
    from src.agents.retail_agent import RetailRegimeAgent
    
    agent = RetailRegimeAgent(experiment_name="titan_v8_phase13_retail")
    df = agent.load_and_process_data()
    metrics = agent.train(df)
    
    # Get predictions
    risk_scores = agent.predict_proba(df)
    regime_flags = agent.predict_class(df)
    
    df['retail_risk_score'] = risk_scores
    df['is_high_attention'] = regime_flags
    
    print(f"\n   Train AUC: {agent.train_metrics['AUC']:.4f}")
    print(f"   Test AUC:  {agent.test_metrics['AUC']:.4f}")
    
    # Save predictions
    retail_pred_path = Path("data/processed/retail_predictions.parquet")
    df[['date', 'ticker', 'retail_risk_score', 'is_high_attention']].to_parquet(
        retail_pred_path, index=False
    )
    print(f"\n   Retail predictions saved to: {retail_pred_path}")
    
    # End run
    mlflow.end_run()
    
    return agent


def train_fundamental_agent():
    """Train FundamentalAgent on residuals."""
    print("\n" + "-" * 70)
    print("🔧 Training FundamentalAgent")
    print("-" * 70)
    
    # End any existing runs
    mlflow.end_run()
    
    try:
        from src.agents.alpha_agents import FundamentalAgent
        
        agent = FundamentalAgent()
        agent.train()
        
        print(f"\n   Train R²: {agent.train_metrics['R2']:.4f} ({agent.train_metrics['R2']*100:.2f}%)")
        print(f"   Test R²:  {agent.test_metrics['R2']:.4f} ({agent.test_metrics['R2']*100:.2f}%)")
        
        # End run
        mlflow.end_run()
        
        return agent
    except Exception as e:
        print(f"\n   ⚠️ FundamentalAgent skipped: {e}")
        mlflow.end_run()
        return None


def prepare_coordinator_features(agents):
    """
    Prepare features for coordinator with regime interactions.
    
    The "Gear Shift" Logic:
    - tech_x_calm = tech_pred × (1 - retail_risk_score)
    - Hypothesis: Technical Agent works best when retail risk is LOW
    """
    print("\n" + "=" * 70)
    print("🔧 FEATURE ENGINEERING: REGIME INTERACTIONS")
    print("=" * 70)
    
    # Load base data
    targets = pd.read_parquet("data/processed/targets_deseasonalized.parquet")
    targets["date"] = pd.to_datetime(targets["date"]).dt.tz_localize(None)
    if targets["ticker"].dtype.name == "category":
        targets["ticker"] = targets["ticker"].astype(str)
    
    # Load residuals (for tech_pred)
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    residuals["date"] = pd.to_datetime(residuals["date"]).dt.tz_localize(None)
    if residuals["ticker"].dtype.name == "category":
        residuals["ticker"] = residuals["ticker"].astype(str)
    
    # Merge
    df = pd.merge(targets, residuals[["date", "ticker", "pred_tech_excess"]], 
                  on=["date", "ticker"], how="left")
    
    # Convert excess to total prediction
    if "seasonal_component" in df.columns:
        df["tech_pred"] = df["pred_tech_excess"] + df["seasonal_component"]
    else:
        df["tech_pred"] = df["pred_tech_excess"]
    
    print(f"   ✓ Base data: {len(df):,} rows")
    
    # Add news_risk_score
    if agents.get('news') and hasattr(agents['news'], 'df'):
        news_df = agents['news'].df.copy()
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.tz_localize(None)
        news_risk = agents['news'].predict(news_df)
        news_df["news_risk_score"] = news_risk
        
        df = pd.merge(df, news_df[["date", "ticker", "news_risk_score"]], 
                      on=["date", "ticker"], how="left")
        df["news_risk_score"] = df["news_risk_score"].fillna(0.2)
        print(f"   ✓ Added news_risk_score")
    else:
        df["news_risk_score"] = 0.2
        print(f"   ⚠️ news_risk_score: using default 0.2")
    
    # Add retail_risk_score and is_high_attention
    if agents.get('retail') and hasattr(agents['retail'], 'df'):
        retail_df = agents['retail'].df.copy()
        retail_df["date"] = pd.to_datetime(retail_df["date"]).dt.tz_localize(None)
        
        retail_risk = agents['retail'].predict_proba(retail_df)
        retail_regime = agents['retail'].predict_class(retail_df)
        
        retail_df["retail_risk_score"] = retail_risk
        retail_df["is_high_attention"] = retail_regime
        
        df = pd.merge(df, retail_df[["date", "ticker", "retail_risk_score", "is_high_attention"]], 
                      on=["date", "ticker"], how="left")
        df["retail_risk_score"] = df["retail_risk_score"].fillna(0.2)
        df["is_high_attention"] = df["is_high_attention"].fillna(0)
        print(f"   ✓ Added retail_risk_score and is_high_attention")
    else:
        df["retail_risk_score"] = 0.2
        df["is_high_attention"] = 0
        print(f"   ⚠️ retail signals: using defaults")
    
    # Add fund_pred
    if agents.get('fund') and hasattr(agents['fund'], 'model'):
        try:
            fund_df = agents['fund'].load_and_process_data()
            fund_df["date"] = pd.to_datetime(fund_df["date"]).dt.tz_localize(None)
            fund_preds = agents['fund'].model.predict(fund_df[agents['fund'].feature_cols].fillna(0))
            fund_df["fund_pred"] = fund_preds
            df = pd.merge(df, fund_df[["date", "ticker", "fund_pred"]], 
                          on=["date", "ticker"], how="left")
            df["fund_pred"] = df["fund_pred"].fillna(0)
            print(f"   ✓ Added fund_pred")
        except:
            df["fund_pred"] = 0
            print(f"   ⚠️ fund_pred: using default 0")
    else:
        df["fund_pred"] = 0
    
    # Load VIX
    try:
        vix = pd.read_parquet("data/processed/vix.parquet")
        vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None)
        df = pd.merge(df, vix[["date", "VIX_close"]], on="date", how="left")
        df["VIX_close"] = df["VIX_close"].fillna(20)
    except:
        df["VIX_close"] = 20.0
    df["VIX_close"] = df["VIX_close"].ffill().fillna(20)
    print(f"   ✓ Added VIX_close")
    
    # =========================================================================
    # THE "GEAR SHIFT" INTERACTION FEATURES
    # =========================================================================
    print("\n   🔧 Creating Regime Interaction Features...")
    
    # tech_x_calm: Technical prediction weighted by calm market (1 - retail_risk)
    # Hypothesis: HAR works better when retail activity is low
    df["tech_x_calm"] = df["tech_pred"] * (1 - df["retail_risk_score"])
    print(f"      ✓ tech_x_calm = tech_pred × (1 - retail_risk_score)")
    
    # tech_x_attention: Technical prediction when attention is high
    df["tech_x_attention"] = df["tech_pred"] * df["is_high_attention"]
    print(f"      ✓ tech_x_attention = tech_pred × is_high_attention")
    
    # news_x_retail: Interaction between news and retail signals
    df["news_x_retail"] = df["news_risk_score"] * df["retail_risk_score"]
    print(f"      ✓ news_x_retail = news_risk × retail_risk")
    
    # Calendar features
    df["is_friday"] = (df["date"].dt.dayofweek == 4).astype(int)
    df["is_monday"] = (df["date"].dt.dayofweek == 0).astype(int)
    df["is_q4"] = (df["date"].dt.quarter == 4).astype(int)
    print(f"   ✓ Added calendar features")
    
    # Momentum
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        df.loc[mask, "vol_ma5"] = df.loc[mask, "target_log_var"].rolling(5, min_periods=1).mean().shift(1)
    df["vol_ma5"] = df["vol_ma5"].fillna(df["target_log_var"].mean())
    print(f"   ✓ Added vol_ma5 (momentum)")
    
    # Sector
    df["sector"] = df["ticker"].map(SECTOR_MAP)
    
    # Drop NaN
    df = df.dropna(subset=["target_log_var", "tech_pred"])
    
    print(f"\n   📊 Final dataset: {len(df):,} rows")
    print(f"   Features: tech_pred, tech_x_calm, tech_x_attention, news_risk_score,")
    print(f"             retail_risk_score, news_x_retail, VIX_close, is_friday, etc.")
    
    return df


def train_coordinator(df):
    """Train Coordinator with regime-adjusted features."""
    print("\n" + "=" * 70)
    print("🎯 TRAINING TITAN COORDINATOR (Ridge + Regime Features)")
    print("=" * 70)
    
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    # End any stale runs
    mlflow.end_run()
    
    # Define feature columns
    feature_cols = [
        "tech_pred",           # HAR baseline
        "tech_x_calm",         # Regime-adjusted HAR (KEY FEATURE)
        "news_risk_score",     # News classifier output
        "retail_risk_score",   # Retail classifier output (contrarian)
        "news_x_retail",       # News × Retail interaction
        "fund_pred",           # Fundamental signal
        "VIX_close",           # Market context
        "is_friday",           # Calendar
        "is_monday",           # Calendar
        "vol_ma5",             # Momentum
    ]
    
    # Filter to available features
    available_features = [f for f in feature_cols if f in df.columns]
    
    print(f"\n   Input Features ({len(available_features)}):")
    for f in available_features:
        print(f"      - {f}")
    
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
    
    # Baseline (tech_pred only)
    baseline_r2 = r2_score(y_test, test["tech_pred"])
    print(f"\n   📈 Baseline (HAR only): R² = {baseline_r2:.4f} ({baseline_r2*100:.2f}%)")
    
    # Train Ridge
    print("\n   🔧 Training Ridge Coordinator...")
    model = Ridge(alpha=0.1)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\n   📈 Coordinator Results:")
    print(f"   {'Metric':<25} {'Train':>10} {'Test':>10}")
    print("   " + "-" * 47)
    print(f"   {'R²':<25} {train_r2:>10.4f} {test_r2:>10.4f}")
    print(f"   {'RMSE':<25} {'':<10} {test_rmse:>10.4f}")
    print(f"   {'MAE':<25} {'':<10} {test_mae:>10.4f}")
    
    # Improvement
    improvement = (test_r2 - baseline_r2) * 100
    print(f"\n   📈 Improvement over HAR baseline: {improvement:+.2f}%")
    
    # Coefficients
    print(f"\n   📊 Ridge Coefficients:")
    print("   " + "-" * 50)
    
    coef_df = pd.DataFrame({
        "feature": available_features,
        "coefficient": model.coef_
    })
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    
    total_importance = coef_df["abs_coef"].sum()
    
    for _, row in coef_df.iterrows():
        pct = (row["abs_coef"] / total_importance * 100) if total_importance > 0 else 0
        marker = " ⭐" if "x_calm" in row["feature"] or "retail" in row["feature"] else ""
        print(f"      {row['feature']:<25}: {row['coefficient']:>+10.4f} ({pct:>5.1f}%){marker}")
    print(f"      {'Intercept':<25}: {model.intercept_:>+10.4f}")
    
    # Sector breakdown
    print(f"\n   📊 Sector Breakdown:")
    print("   " + "-" * 50)
    
    test["y_pred"] = y_test_pred
    sector_r2 = {}
    for sector in test["sector"].dropna().unique():
        mask = test["sector"] == sector
        if mask.sum() > 50:
            sector_r2[sector] = r2_score(y_test[mask], y_test_pred[mask])
    
    for sector, r2 in sorted(sector_r2.items(), key=lambda x: x[1], reverse=True):
        marker = " ⭐" if r2 >= 0.25 else ""
        print(f"      {sector:15s}: {r2:.4f} ({r2*100:.2f}%){marker}")
    
    # Store results
    results = {
        "model": model,
        "feature_cols": available_features,
        "coef_df": coef_df,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "baseline_r2": baseline_r2,
        "sector_r2": sector_r2,
        "df": df
    }
    
    # MLflow logging
    mlflow.set_experiment("titan_v8_phase13")
    with mlflow.start_run(run_name="coordinator_phase13"):
        mlflow.log_params({
            "model": "Ridge",
            "alpha": 0.1,
            "n_features": len(available_features),
            "features": available_features,
            "n_train": len(train),
            "n_test": len(test)
        })
        mlflow.log_metrics({
            "train_r2": train_r2,
            "test_r2": test_r2,
            "baseline_r2": baseline_r2,
            "improvement": improvement
        })
        mlflow.sklearn.log_model(model, "coordinator_phase13")
    mlflow.end_run()
    
    return results


def main():
    """Run the full Titan V8 Phase 13 pipeline."""
    parser = argparse.ArgumentParser(description="Titan V8 Phase 13 - Retail Regime Integration")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip data verification details")
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🚀 TITAN V8 PHASE 13: RETAIL REGIME INTEGRATION")
    print("   Adding Regime-Aware Features to the Ensemble")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # End any stale MLflow runs
    mlflow.end_run()
    
    # MLflow tracking
    mlflow.set_experiment("titan_v8_phase13")
    
    # Step A: Verify
    config, df = verify_data()
    if df is None:
        print("\n❌ Data verification failed. Exiting.")
        return
    
    # Step B: Train Anchor Agents
    print("\n" + "=" * 70)
    print("🎯 STEP B: TRAIN ANCHOR AGENTS")
    print("=" * 70)
    
    agents = {}
    
    # Technical Agent
    try:
        agents['tech'] = train_technical_agent()
    except Exception as e:
        print(f"\n   ⚠️ TechnicalAgent error: {e}")
        import traceback
        traceback.print_exc()
    
    # News Agent
    try:
        agents['news'] = train_news_agent()
    except Exception as e:
        print(f"\n   ⚠️ NewsAgent error: {e}")
        import traceback
        traceback.print_exc()
    
    # Retail Agent (NEW)
    try:
        agents['retail'] = train_retail_agent()
    except Exception as e:
        print(f"\n   ⚠️ RetailAgent error: {e}")
        import traceback
        traceback.print_exc()
    
    # Fundamental Agent
    agents['fund'] = train_fundamental_agent()
    
    # Step C: Feature Engineering (Regime Interactions)
    coord_df = prepare_coordinator_features(agents)
    
    # Step D: Train Coordinator
    results = train_coordinator(coord_df)
    
    # Step E: Final Report
    print("\n" + "=" * 70)
    print("🏆 PHASE 13 FINAL REPORT")
    print("=" * 70)
    
    print(f"""
   AGENT PERFORMANCE:
   ------------------
   {'Agent':<30} {'Metric':<15} {'Test Value':>12}
   {'-'*60}""")
    
    if agents.get('tech') and hasattr(agents['tech'], 'test_metrics'):
        print(f"   {'TechnicalAgent (HAR-RV)':<30} {'R²':<15} {agents['tech'].test_metrics['R2']:>12.4f}")
    if agents.get('news') and hasattr(agents['news'], 'test_metrics'):
        print(f"   {'NewsAgent (Classifier)':<30} {'AUC':<15} {agents['news'].test_metrics['AUC']:>12.4f}")
    if agents.get('retail') and hasattr(agents['retail'], 'test_metrics'):
        print(f"   {'RetailRegimeAgent':<30} {'AUC':<15} {agents['retail'].test_metrics['AUC']:>12.4f}")
    if agents.get('fund') and hasattr(agents['fund'], 'test_metrics'):
        print(f"   {'FundamentalAgent':<30} {'R²':<15} {agents['fund'].test_metrics['R2']:>12.4f}")
    
    print(f"""
   COORDINATOR PERFORMANCE:
   ------------------------
   Baseline (HAR only):   R² = {results['baseline_r2']:.4f} ({results['baseline_r2']*100:.2f}%)
   Titan V8 Phase 13:     R² = {results['test_r2']:.4f} ({results['test_r2']*100:.2f}%)
   
   Improvement: {(results['test_r2'] - results['baseline_r2'])*100:+.2f}%
""")
    
    # Check regime interaction effectiveness
    print("   KEY FEATURES (Regime Integration Check):")
    print("   " + "-" * 50)
    
    for _, row in results['coef_df'].head(5).iterrows():
        marker = ""
        if "x_calm" in row["feature"]:
            marker = " 🔧 REGIME INTERACTION"
        elif "retail" in row["feature"]:
            marker = " 🔧 RETAIL SIGNAL"
        print(f"      {row['feature']:<25}: {row['coefficient']:>+10.4f}{marker}")
    
    # Goal check
    print("\n" + "=" * 70)
    print("🎯 GOAL CHECK")
    print("=" * 70)
    
    retail_auc = agents['retail'].test_metrics['AUC'] if agents.get('retail') else 0
    
    print(f"\n   Retail Agent AUC Target: 0.60")
    print(f"   Retail Agent AUC Actual: {retail_auc:.4f}")
    print(f"   Status: {'✅ ACHIEVED' if retail_auc >= 0.60 else '⚠️ Close but below target'}")
    
    print(f"\n   Coordinator R² Target: 0.20+")
    print(f"   Coordinator R² Actual: {results['test_r2']:.4f}")
    print(f"   Status: {'✅ ACHIEVED' if results['test_r2'] >= 0.20 else '⚠️ Below target'}")
    
    # Check if tech_x_calm is being used
    tech_x_calm_coef = results['coef_df'][results['coef_df']['feature'] == 'tech_x_calm']['coefficient'].values
    if len(tech_x_calm_coef) > 0:
        print(f"\n   tech_x_calm coefficient: {tech_x_calm_coef[0]:+.4f}")
        if abs(tech_x_calm_coef[0]) > 0.01:
            print(f"   Status: ✅ REGIME INTERACTION IS ACTIVE")
        else:
            print(f"   Status: ⚠️ Coefficient is small")
    
    # Timing
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("✅ TITAN V8 PHASE 13 COMPLETE")
    print(f"   Duration: {duration}")
    print(f"   Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
