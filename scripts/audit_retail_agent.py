"""
RetailRiskAgent Exhaustive Audit

Find ANY predictive signal from retail proxies (BTC/GME/IWM) for volatility prediction.
Tests every combination of targets, features, lags, models, and regimes.

Output: Table ranked by Test R² showing best configurations.
"""

import sys
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from lightgbm import LGBMRegressor, LGBMClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_all_data():
    """Load and merge all required data."""
    print("📂 Loading data...")
    
    # Load datasets
    retail = pd.read_parquet("data/processed/retail_signals.parquet")
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    targets = pd.read_parquet("data/processed/targets.parquet")
    
    # Normalize dates
    for df in [retail, residuals, targets]:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        if "ticker" in df.columns and df["ticker"].dtype.name == "category":
            df["ticker"] = df["ticker"].astype(str)
    
    # Merge targets with residuals
    merged = pd.merge(targets, residuals[["date", "ticker", "resid_tech"]], 
                      on=["date", "ticker"], how="inner")
    
    # Merge with retail signals (global - by date only)
    merged = pd.merge(merged, retail, on="date", how="left")
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # Fill missing retail signals
    retail_cols = [c for c in retail.columns if c != "date"]
    for col in retail_cols:
        merged[col] = merged[col].ffill().fillna(0)
    
    # Create VIX interactions
    merged["btc_vix_interaction"] = merged["btc_vol_5d"] * merged["VIX_close"]
    merged["gme_vix_interaction"] = merged["gme_vol_shock"] * merged["VIX_close"]
    
    # Create direction target
    merged["vol_direction"] = (merged["target_log_var"].diff() > 0).astype(int)
    
    print(f"   Loaded {len(merged):,} rows, {merged['ticker'].nunique()} tickers")
    print(f"   Date range: {merged['date'].min()} to {merged['date'].max()}")
    
    return merged


def create_lagged_features(df, cols, lags):
    """Create lagged versions of features."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            for lag in lags:
                df[f"{col}_lag{lag}"] = df.groupby("ticker")[col].shift(lag)
    return df


def get_feature_groups():
    """Define feature groups to test."""
    return {
        "bitcoin": ["btc_vol_5d", "btc_ret_5d", "btc_mom_20d"],
        "gamestop": ["gme_vol_shock", "gme_vol_5d", "gme_ret_5d"],
        "smallcap": ["small_cap_excess", "small_cap_mom"],
        "composite": ["retail_mania", "risk_on_signal"],
        "vix_interact": ["btc_vix_interaction", "gme_vix_interaction", "VIX_close"],
        "all_retail": ["btc_vol_5d", "btc_ret_5d", "gme_vol_shock", "small_cap_excess", 
                       "retail_mania", "risk_on_signal"],
    }


def run_experiment(df, features, target, model_type, train_mask, test_mask, is_classification=False):
    """Run a single experiment and return metrics."""
    # Filter available features
    available = [f for f in features if f in df.columns]
    if len(available) == 0:
        return None
    
    # Prepare data
    train = df[train_mask].dropna(subset=available + [target])
    test = df[test_mask].dropna(subset=available + [target])
    
    if len(train) < 50 or len(test) < 20:
        return None
    
    X_train = train[available].fillna(0)
    y_train = train[target]
    X_test = test[available].fillna(0)
    y_test = test[target]
    
    try:
        if is_classification:
            if model_type == "ridge":
                model = LogisticRegression(C=0.1, max_iter=1000)
            elif model_type == "lgbm_shallow":
                model = LGBMClassifier(n_estimators=100, max_depth=2, verbose=-1)
            else:
                model = LGBMClassifier(n_estimators=200, max_depth=4, verbose=-1)
            
            model.fit(X_train, y_train)
            train_score = accuracy_score(y_train, model.predict(X_train))
            test_score = accuracy_score(y_test, model.predict(X_test))
        else:
            if model_type == "ridge":
                model = Ridge(alpha=1.0)
            elif model_type == "lgbm_shallow":
                model = LGBMRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, verbose=-1)
            else:  # lgbm_deep
                model = LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.03, verbose=-1)
            
            model.fit(X_train, y_train)
            train_score = r2_score(y_train, model.predict(X_train))
            test_score = r2_score(y_test, model.predict(X_test))
        
        return {
            "train_score": train_score,
            "test_score": test_score,
            "n_train": len(train),
            "n_test": len(test),
            "n_features": len(available)
        }
    except Exception as e:
        return None


def main():
    print("\n" + "=" * 80)
    print("🔍 RETAILRISKAGENT EXHAUSTIVE AUDIT")
    print("   Testing every combination to find predictive signal")
    print("=" * 80)
    
    # Load data
    df = load_all_data()
    
    # Define experiment configurations
    targets = {
        "target_log_var": False,  # Raw volatility (regression)
        "resid_tech": False,       # Residuals (regression)
        "realized_vol": False,     # Daily RV (regression)
        "vol_direction": True      # Direction (classification)
    }
    
    lag_configs = {
        "no_lag": [],
        "short": [1, 3, 5],
        "medium": [5, 10, 15],
        "long": [10, 20]
    }
    
    model_types = ["ridge", "lgbm_shallow", "lgbm_deep"]
    
    split_configs = {
        "2023": ("2023-01-01", None),
        "2022": ("2022-01-01", None),
    }
    
    ticker_configs = {
        "all": None,
        "NVDA": "NVDA",
        "AAPL": "AAPL",
        "MSFT": "MSFT"
    }
    
    feature_groups = get_feature_groups()
    
    results = []
    total_experiments = 0
    
    print("\n🔬 Running experiments...")
    print("-" * 80)
    
    # Grid search
    for target_name, is_class in targets.items():
        if target_name not in df.columns:
            continue
            
        for fg_name, fg_features in feature_groups.items():
            for lag_name, lags in lag_configs.items():
                # Create lagged features
                df_lagged = create_lagged_features(df, fg_features, lags)
                
                # Get all features including lagged
                all_features = fg_features.copy()
                for f in fg_features:
                    for lag in lags:
                        lag_col = f"{f}_lag{lag}"
                        if lag_col in df_lagged.columns:
                            all_features.append(lag_col)
                
                for model_type in model_types:
                    for split_name, (cutoff, _) in split_configs.items():
                        cutoff_date = pd.to_datetime(cutoff)
                        
                        for ticker_name, ticker_filter in ticker_configs.items():
                            # Apply ticker filter
                            if ticker_filter:
                                df_exp = df_lagged[df_lagged["ticker"] == ticker_filter].copy()
                            else:
                                df_exp = df_lagged.copy()
                            
                            if len(df_exp) < 100:
                                continue
                            
                            # Create masks
                            train_mask = df_exp["date"] < cutoff_date
                            test_mask = df_exp["date"] >= cutoff_date
                            
                            # Run experiment
                            result = run_experiment(
                                df_exp, all_features, target_name, 
                                model_type, train_mask, test_mask, is_class
                            )
                            
                            total_experiments += 1
                            
                            if result:
                                results.append({
                                    "target": target_name,
                                    "features": fg_name,
                                    "lags": lag_name,
                                    "model": model_type,
                                    "split": split_name,
                                    "ticker": ticker_name,
                                    "train_score": result["train_score"],
                                    "test_score": result["test_score"],
                                    "n_train": result["n_train"],
                                    "n_test": result["n_test"],
                                    "n_features": result["n_features"],
                                    "is_classification": is_class
                                })
    
    print(f"\n   Completed {total_experiments} experiments")
    print(f"   Valid results: {len(results)}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("\n❌ No valid results!")
        return
    
    # =====================================================
    # ANALYSIS: REGRESSION RESULTS
    # =====================================================
    print("\n" + "=" * 80)
    print("📊 TOP 20 REGRESSION RESULTS (by Test R²)")
    print("=" * 80)
    
    reg_results = results_df[~results_df["is_classification"]].copy()
    reg_results = reg_results.sort_values("test_score", ascending=False)
    
    print(f"\n{'Rank':<5} {'Target':<15} {'Features':<12} {'Lags':<8} {'Model':<12} {'Ticker':<6} {'Train R²':>10} {'Test R²':>10}")
    print("-" * 90)
    
    for i, (_, row) in enumerate(reg_results.head(20).iterrows()):
        print(f"{i+1:<5} {row['target']:<15} {row['features']:<12} {row['lags']:<8} {row['model']:<12} {row['ticker']:<6} {row['train_score']:>10.4f} {row['test_score']:>10.4f}")
    
    # =====================================================
    # ANALYSIS: CLASSIFICATION RESULTS
    # =====================================================
    print("\n" + "=" * 80)
    print("📊 TOP 10 CLASSIFICATION RESULTS (by Test Accuracy)")
    print("=" * 80)
    
    class_results = results_df[results_df["is_classification"]].copy()
    class_results = class_results.sort_values("test_score", ascending=False)
    
    if len(class_results) > 0:
        print(f"\n{'Rank':<5} {'Target':<15} {'Features':<12} {'Lags':<8} {'Model':<12} {'Ticker':<6} {'Train Acc':>10} {'Test Acc':>10}")
        print("-" * 90)
        
        for i, (_, row) in enumerate(class_results.head(10).iterrows()):
            print(f"{i+1:<5} {row['target']:<15} {row['features']:<12} {row['lags']:<8} {row['model']:<12} {row['ticker']:<6} {row['train_score']*100:>9.1f}% {row['test_score']*100:>9.1f}%")
    
    # =====================================================
    # FEATURE GROUP ANALYSIS
    # =====================================================
    print("\n" + "=" * 80)
    print("📊 FEATURE GROUP PERFORMANCE (Mean Test R² for regression)")
    print("=" * 80)
    
    fg_analysis = reg_results.groupby("features")["test_score"].agg(["mean", "max", "count"])
    fg_analysis = fg_analysis.sort_values("max", ascending=False)
    
    print(f"\n{'Feature Group':<15} {'Mean R²':>10} {'Max R²':>10} {'Count':>8}")
    print("-" * 50)
    for fg, row in fg_analysis.iterrows():
        print(f"{fg:<15} {row['mean']:>10.4f} {row['max']:>10.4f} {int(row['count']):>8}")
    
    # =====================================================
    # TARGET ANALYSIS
    # =====================================================
    print("\n" + "=" * 80)
    print("📊 TARGET PERFORMANCE (Mean Test R² for regression)")
    print("=" * 80)
    
    target_analysis = reg_results.groupby("target")["test_score"].agg(["mean", "max", "count"])
    target_analysis = target_analysis.sort_values("max", ascending=False)
    
    print(f"\n{'Target':<20} {'Mean R²':>10} {'Max R²':>10} {'Count':>8}")
    print("-" * 55)
    for tgt, row in target_analysis.iterrows():
        print(f"{tgt:<20} {row['mean']:>10.4f} {row['max']:>10.4f} {int(row['count']):>8}")
    
    # =====================================================
    # LAG ANALYSIS
    # =====================================================
    print("\n" + "=" * 80)
    print("📊 LAG STRUCTURE PERFORMANCE")
    print("=" * 80)
    
    lag_analysis = reg_results.groupby("lags")["test_score"].agg(["mean", "max"])
    lag_analysis = lag_analysis.sort_values("max", ascending=False)
    
    print(f"\n{'Lag Config':<15} {'Mean R²':>10} {'Max R²':>10}")
    print("-" * 40)
    for lag, row in lag_analysis.iterrows():
        print(f"{lag:<15} {row['mean']:>10.4f} {row['max']:>10.4f}")
    
    # =====================================================
    # PER-TICKER ANALYSIS
    # =====================================================
    print("\n" + "=" * 80)
    print("📊 PER-TICKER PERFORMANCE")
    print("=" * 80)
    
    ticker_analysis = reg_results.groupby("ticker")["test_score"].agg(["mean", "max"])
    ticker_analysis = ticker_analysis.sort_values("max", ascending=False)
    
    print(f"\n{'Ticker':<10} {'Mean R²':>10} {'Max R²':>10}")
    print("-" * 35)
    for ticker, row in ticker_analysis.iterrows():
        print(f"{ticker:<10} {row['mean']:>10.4f} {row['max']:>10.4f}")
    
    # =====================================================
    # BEST CONFIGURATION
    # =====================================================
    print("\n" + "=" * 80)
    print("🏆 BEST CONFIGURATION")
    print("=" * 80)
    
    best = reg_results.iloc[0]
    print(f"""
    Target:       {best['target']}
    Features:     {best['features']}
    Lags:         {best['lags']}
    Model:        {best['model']}
    Ticker:       {best['ticker']}
    Split:        {best['split']}
    
    Train R²:     {best['train_score']:.4f} ({best['train_score']*100:.2f}%)
    Test R²:      {best['test_score']:.4f} ({best['test_score']*100:.2f}%)
    Gap:          {best['train_score'] - best['test_score']:.4f}
    """)
    
    # =====================================================
    # RECOMMENDATION
    # =====================================================
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATION")
    print("=" * 80)
    
    best_test_r2 = best['test_score']
    
    if best_test_r2 > 0.10:
        print(f"""
    ✅ STRONG SIGNAL FOUND! (Test R² = {best_test_r2*100:.2f}%)
    
    UPDATE RetailRiskAgent with:
    - Target: {best['target']}
    - Features: {best['features']}
    - Lags: {best['lags']}
    - Model: {best['model']}
    """)
    elif best_test_r2 > 0.05:
        print(f"""
    ⚠️ MODERATE SIGNAL FOUND (Test R² = {best_test_r2*100:.2f}%)
    
    Consider including in ensemble with lower weight.
    """)
    elif best_test_r2 > 0.02:
        print(f"""
    ⚠️ WEAK SIGNAL (Test R² = {best_test_r2*100:.2f}%)
    
    Marginal value. May not be worth the complexity.
    """)
    else:
        print(f"""
    ❌ NO USEFUL SIGNAL (Best Test R² = {best_test_r2*100:.2f}%)
    
    Retail proxies (BTC/GME/IWM) do not predict volatility for this dataset.
    Recommend: DROP RetailRiskAgent from the ensemble.
    """)
    
    # Check classification
    if len(class_results) > 0:
        best_class = class_results.iloc[0]
        if best_class['test_score'] > 0.55:
            print(f"""
    📈 DIRECTION PREDICTION:
    Best accuracy: {best_class['test_score']*100:.1f}% (baseline 50%)
    This could be useful for trading signals!
    """)
    
    print("=" * 80)
    
    # Save results
    results_df.to_csv("data/processed/retail_audit_results.csv", index=False)
    print(f"\n📁 Full results saved to: data/processed/retail_audit_results.csv")


if __name__ == "__main__":
    main()


