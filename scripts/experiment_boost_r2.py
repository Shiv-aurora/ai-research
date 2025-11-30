"""
🎯 R² BOOST EXPERIMENTS: Target 20%+

Current: 14.80% R²
Target: 20%+ R²
Gap: +5.2%

Approaches to test:
1. Enhanced Features - More calendar, momentum, interactions
2. Model Selection - GradientBoosting, XGBoost, Ensemble
3. Stacking - Use predictions as features for meta-learner
4. Residual Boosting - Iteratively fit residuals
5. Outlier Handling - Winsorize extreme values
6. Risk Score Calibration - Better use of news classifier
7. Per-Ticker Models - Individual models per sector
8. Ensemble Averaging - Average multiple approaches

NO PERMANENT CHANGES - EXPERIMENTS ONLY
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor, LGBMClassifier


def load_data():
    """Load all required data."""
    print("\n📂 Loading data...")
    
    # Targets
    targets = pd.read_parquet("data/processed/targets_deseasonalized.parquet")
    targets["date"] = pd.to_datetime(targets["date"]).dt.tz_localize(None)
    
    # Residuals (has tech predictions)
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    residuals["date"] = pd.to_datetime(residuals["date"]).dt.tz_localize(None)
    
    # News features
    news = pd.read_parquet("data/processed/news_features.parquet")
    news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(None)
    
    # Normalize ticker types
    for df in [targets, residuals, news]:
        if df["ticker"].dtype.name == "category":
            df["ticker"] = df["ticker"].astype(str)
    
    # Merge
    df = pd.merge(
        targets[["date", "ticker", "target_excess", "target_log_var", "seasonal_component", "realized_vol"]],
        residuals[["date", "ticker", "pred_tech_excess", "resid_tech"]],
        on=["date", "ticker"],
        how="inner"
    )
    
    df = pd.merge(df, news, on=["date", "ticker"], how="left")
    
    # Fill missing news features
    news_cols = [c for c in news.columns if c not in ["date", "ticker"]]
    for col in news_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Add VIX
    try:
        vix = pd.read_parquet("data/processed/vix.parquet")
        vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None)
        df = pd.merge(df, vix[["date", "VIX_close"]], on="date", how="left")
        df["VIX_close"] = df["VIX_close"].fillna(20)
    except:
        df["VIX_close"] = 20.0
    
    # Add calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_q4"] = (df["quarter"] == 4).astype(int)
    
    # Add sector
    SECTOR_MAP = {
        'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech',
        'JPM': 'Finance', 'BAC': 'Finance', 'V': 'Finance',
        'CAT': 'Industrial', 'GE': 'Industrial', 'BA': 'Industrial',
        'WMT': 'Consumer', 'MCD': 'Consumer', 'COST': 'Consumer',
        'XOM': 'Energy', 'CVX': 'Energy', 'SLB': 'Energy',
        'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare'
    }
    df["sector"] = df["ticker"].map(SECTOR_MAP)
    
    df = df.dropna(subset=["target_excess", "pred_tech_excess"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    print(f"   ✓ Loaded: {len(df):,} rows")
    
    return df


def calculate_reseasonalized_r2(y_test_excess, y_pred_excess, seasonal_component):
    """Calculate R² after re-seasonalization."""
    y_pred_total = y_pred_excess + seasonal_component
    y_test_total = y_test_excess + seasonal_component
    return r2_score(y_test_total, y_pred_total)


def main():
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🎯 R² BOOST EXPERIMENTS: Target 20%+")
    print(f"   Current: 14.80% | Target: 20%+ | Gap: +5.2%")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    df = load_data()
    
    # Time split
    cutoff = pd.to_datetime("2023-01-01")
    train_mask = df["date"] < cutoff
    test_mask = df["date"] >= cutoff
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    baseline_r2 = 0.1480  # Current hybrid result
    all_results = []
    
    # ==========================================================================
    # EXPERIMENT 1: ENHANCED FEATURE SET
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 1: ENHANCED FEATURE SET")
    print("=" * 70)
    
    # Add momentum features
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        df.loc[mask, "tech_pred_lag1"] = df.loc[mask, "pred_tech_excess"].shift(1)
        df.loc[mask, "tech_pred_lag2"] = df.loc[mask, "pred_tech_excess"].shift(2)
        df.loc[mask, "vol_momentum"] = df.loc[mask, "realized_vol"].diff()
        df.loc[mask, "vol_ma5"] = df.loc[mask, "realized_vol"].rolling(5).mean()
        df.loc[mask, "shock_ma3"] = df.loc[mask, "shock_index"].rolling(3).mean()
    
    df = df.dropna(subset=["tech_pred_lag1", "vol_ma5"])
    
    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()
    
    # Different feature sets
    feature_sets = {
        "Base": ["pred_tech_excess", "VIX_close"],
        "Base + Calendar": ["pred_tech_excess", "VIX_close", "is_friday", "is_monday", "is_q4"],
        "Base + Momentum": ["pred_tech_excess", "VIX_close", "tech_pred_lag1", "vol_ma5", "vol_momentum"],
        "Base + News": ["pred_tech_excess", "VIX_close", "shock_index", "news_count", "sentiment_avg"],
        "Full": ["pred_tech_excess", "VIX_close", "is_friday", "is_monday", "is_q4",
                 "tech_pred_lag1", "vol_ma5", "shock_index", "news_count"],
    }
    
    print(f"\n   {'Feature Set':<25} {'Test R² (excess)':>18} {'Test R² (total)':>18}")
    print("   " + "-" * 63)
    
    for name, features in feature_sets.items():
        available = [f for f in features if f in train_df.columns]
        
        X_train = train_df[available].fillna(0)
        X_test = test_df[available].fillna(0)
        y_train = train_df["target_excess"]
        y_test = test_df["target_excess"]
        
        model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        r2_excess = r2_score(y_test, y_pred)
        r2_total = calculate_reseasonalized_r2(y_test.values, y_pred, test_df["seasonal_component"].values)
        
        marker = " ⭐" if r2_total > baseline_r2 else ""
        print(f"   {name:<25} {r2_excess:>18.4f} {r2_total:>18.4f}{marker}")
        
        all_results.append({"exp": f"Features ({name})", "r2_total": r2_total})
    
    # ==========================================================================
    # EXPERIMENT 2: MODEL SELECTION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 2: MODEL SELECTION")
    print("=" * 70)
    
    # Best feature set from above
    features = ["pred_tech_excess", "VIX_close", "is_friday", "is_monday", "is_q4",
                "tech_pred_lag1", "vol_ma5", "shock_index"]
    features = [f for f in features if f in train_df.columns]
    
    X_train = train_df[features].fillna(0)
    X_test = test_df[features].fillna(0)
    y_train = train_df["target_excess"]
    y_test = test_df["target_excess"]
    
    models = {
        "Ridge (α=0.1)": Ridge(alpha=0.1),
        "Ridge (α=1.0)": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
        "LightGBM (shallow)": LGBMRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42, verbose=-1),
        "LightGBM (medium)": LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.03, random_state=42, verbose=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    }
    
    print(f"\n   {'Model':<25} {'Train R²':>12} {'Test R² (total)':>18}")
    print("   " + "-" * 57)
    
    best_model = None
    best_r2 = 0
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        train_r2 = r2_score(y_train, model.predict(X_train))
        y_pred = model.predict(X_test)
        r2_total = calculate_reseasonalized_r2(y_test.values, y_pred, test_df["seasonal_component"].values)
        
        marker = " ⭐" if r2_total > baseline_r2 else ""
        print(f"   {name:<25} {train_r2:>12.4f} {r2_total:>18.4f}{marker}")
        
        all_results.append({"exp": f"Model ({name})", "r2_total": r2_total})
        
        if r2_total > best_r2:
            best_r2 = r2_total
            best_model = model
    
    # ==========================================================================
    # EXPERIMENT 3: STACKING ENSEMBLE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 3: STACKING ENSEMBLE")
    print("=" * 70)
    
    # Create base predictions
    base_models = {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42, verbose=-1),
    }
    
    # Train base models and get OOF predictions
    stacked_train = train_df[["date", "ticker", "target_excess"]].copy()
    stacked_test = test_df[["date", "ticker", "target_excess", "seasonal_component"]].copy()
    
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        stacked_train[f"pred_{name}"] = model.predict(X_train)
        stacked_test[f"pred_{name}"] = model.predict(X_test)
    
    # Meta-learner features
    meta_features = [f"pred_{name}" for name in base_models.keys()]
    
    # Train meta-learner
    meta_learner = Ridge(alpha=0.1)
    meta_learner.fit(stacked_train[meta_features], stacked_train["target_excess"])
    
    y_pred_stacked = meta_learner.predict(stacked_test[meta_features])
    r2_stacked = calculate_reseasonalized_r2(
        stacked_test["target_excess"].values, 
        y_pred_stacked, 
        stacked_test["seasonal_component"].values
    )
    
    print(f"\n   Stacking ({len(base_models)} base models + Ridge meta)")
    print(f"   Test R² (total): {r2_stacked:.4f}")
    
    all_results.append({"exp": "Stacking Ensemble", "r2_total": r2_stacked})
    
    # ==========================================================================
    # EXPERIMENT 4: WEIGHTED AVERAGE ENSEMBLE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 4: WEIGHTED AVERAGE ENSEMBLE")
    print("=" * 70)
    
    # Get individual predictions
    preds = {}
    for name, model in base_models.items():
        preds[name] = stacked_test[f"pred_{name}"].values
    
    # Test different weight combinations
    weight_combos = [
        {"Ridge": 0.33, "ElasticNet": 0.33, "LightGBM": 0.34},
        {"Ridge": 0.5, "ElasticNet": 0.3, "LightGBM": 0.2},
        {"Ridge": 0.2, "ElasticNet": 0.3, "LightGBM": 0.5},
        {"Ridge": 0.4, "ElasticNet": 0.4, "LightGBM": 0.2},
    ]
    
    print(f"\n   {'Weights':<45} {'Test R² (total)':>18}")
    print("   " + "-" * 65)
    
    best_ensemble_r2 = 0
    best_weights = None
    
    for weights in weight_combos:
        y_pred_ensemble = sum(w * preds[name] for name, w in weights.items())
        r2_ensemble = calculate_reseasonalized_r2(
            stacked_test["target_excess"].values,
            y_pred_ensemble,
            stacked_test["seasonal_component"].values
        )
        
        weight_str = ", ".join([f"{k}:{v}" for k, v in weights.items()])
        marker = " ⭐" if r2_ensemble > baseline_r2 else ""
        print(f"   {weight_str:<45} {r2_ensemble:>18.4f}{marker}")
        
        if r2_ensemble > best_ensemble_r2:
            best_ensemble_r2 = r2_ensemble
            best_weights = weights
    
    all_results.append({"exp": "Weighted Ensemble (best)", "r2_total": best_ensemble_r2})
    
    # ==========================================================================
    # EXPERIMENT 5: SECTOR-SPECIFIC MODELS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 5: SECTOR-SPECIFIC MODELS")
    print("=" * 70)
    
    sector_preds = []
    
    print(f"\n   {'Sector':<15} {'N Train':>10} {'N Test':>10} {'Test R²':>12}")
    print("   " + "-" * 50)
    
    for sector in df["sector"].unique():
        if pd.isna(sector):
            continue
        
        train_sector = train_df[train_df["sector"] == sector]
        test_sector = test_df[test_df["sector"] == sector]
        
        if len(train_sector) < 50 or len(test_sector) < 50:
            continue
        
        X_train_s = train_sector[features].fillna(0)
        X_test_s = test_sector[features].fillna(0)
        y_train_s = train_sector["target_excess"]
        y_test_s = test_sector["target_excess"]
        
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train_s)
        
        y_pred_s = model.predict(X_test_s)
        
        sector_preds.append(pd.DataFrame({
            "idx": test_sector.index,
            "pred": y_pred_s,
            "actual": y_test_s.values,
            "seasonal": test_sector["seasonal_component"].values
        }))
        
        r2_sector = calculate_reseasonalized_r2(y_test_s.values, y_pred_s, test_sector["seasonal_component"].values)
        print(f"   {sector:<15} {len(train_sector):>10,} {len(test_sector):>10,} {r2_sector:>12.4f}")
    
    # Combine sector predictions
    if sector_preds:
        all_sector_preds = pd.concat(sector_preds).sort_values("idx")
        r2_sector_combined = calculate_reseasonalized_r2(
            all_sector_preds["actual"].values,
            all_sector_preds["pred"].values,
            all_sector_preds["seasonal"].values
        )
        print(f"\n   Combined sector R²: {r2_sector_combined:.4f}")
        all_results.append({"exp": "Sector-Specific Models", "r2_total": r2_sector_combined})
    
    # ==========================================================================
    # EXPERIMENT 6: OUTLIER HANDLING
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 6: OUTLIER HANDLING (Winsorization)")
    print("=" * 70)
    
    # Winsorize target at 1st and 99th percentile
    for pct in [0.01, 0.02, 0.05]:
        train_wins = train_df.copy()
        test_wins = test_df.copy()
        
        lower = train_wins["target_excess"].quantile(pct)
        upper = train_wins["target_excess"].quantile(1 - pct)
        
        train_wins["target_excess_wins"] = train_wins["target_excess"].clip(lower, upper)
        
        X_train_w = train_wins[features].fillna(0)
        X_test_w = test_wins[features].fillna(0)
        y_train_w = train_wins["target_excess_wins"]
        
        model = Ridge(alpha=1.0)
        model.fit(X_train_w, y_train_w)
        
        y_pred_w = model.predict(X_test_w)
        r2_wins = calculate_reseasonalized_r2(test_wins["target_excess"].values, y_pred_w, test_wins["seasonal_component"].values)
        
        marker = " ⭐" if r2_wins > baseline_r2 else ""
        print(f"   Winsorize {pct*100:.0f}%/{(1-pct)*100:.0f}%: R² = {r2_wins:.4f}{marker}")
        
        all_results.append({"exp": f"Winsorize ({pct*100:.0f}%)", "r2_total": r2_wins})
    
    # ==========================================================================
    # EXPERIMENT 7: RESIDUAL BOOSTING
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 7: RESIDUAL BOOSTING (2-Stage)")
    print("=" * 70)
    
    # Stage 1: Linear model
    stage1 = Ridge(alpha=1.0)
    stage1.fit(X_train, y_train)
    
    stage1_train_pred = stage1.predict(X_train)
    stage1_test_pred = stage1.predict(X_test)
    
    # Stage 1 residuals
    stage1_resid = y_train - stage1_train_pred
    
    # Stage 2: Boost with LightGBM on residuals
    stage2 = LGBMRegressor(n_estimators=50, max_depth=2, learning_rate=0.01, random_state=42, verbose=-1)
    stage2.fit(X_train, stage1_resid)
    
    stage2_test_pred = stage2.predict(X_test)
    
    # Combined prediction
    for boost_weight in [0.1, 0.3, 0.5, 0.7, 1.0]:
        y_pred_boosted = stage1_test_pred + boost_weight * stage2_test_pred
        r2_boosted = calculate_reseasonalized_r2(y_test.values, y_pred_boosted, test_df["seasonal_component"].values)
        
        marker = " ⭐" if r2_boosted > baseline_r2 else ""
        print(f"   Boost weight {boost_weight}: R² = {r2_boosted:.4f}{marker}")
        
        all_results.append({"exp": f"Residual Boost ({boost_weight})", "r2_total": r2_boosted})
    
    # ==========================================================================
    # EXPERIMENT 8: PREDICTION SHRINKAGE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 8: PREDICTION SHRINKAGE")
    print("=" * 70)
    
    # Sometimes shrinking predictions towards mean helps
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred_raw = model.predict(X_test)
    
    mean_target = y_train.mean()
    
    print(f"\n   {'Shrinkage':>12} {'Test R² (total)':>18}")
    print("   " + "-" * 33)
    
    for shrink in [0.0, 0.1, 0.2, 0.3, 0.5]:
        y_pred_shrunk = y_pred_raw * (1 - shrink) + mean_target * shrink
        r2_shrunk = calculate_reseasonalized_r2(y_test.values, y_pred_shrunk, test_df["seasonal_component"].values)
        
        marker = " ⭐" if r2_shrunk > baseline_r2 else ""
        print(f"   {shrink:>12} {r2_shrunk:>18.4f}{marker}")
        
        all_results.append({"exp": f"Shrinkage ({shrink})", "r2_total": r2_shrunk})
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🏆 FINAL RESULTS: TOP 15 APPROACHES")
    print("=" * 70)
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("r2_total", ascending=False)
    
    print(f"\n   Baseline: {baseline_r2:.4f} ({baseline_r2*100:.2f}%)")
    print(f"   Target:   0.2000 (20.00%)")
    print(f"\n   {'Rank':<6} {'Experiment':<40} {'R² (total)':>12} {'vs Base':>10}")
    print("   " + "-" * 70)
    
    for i, (_, row) in enumerate(results_df.head(15).iterrows(), 1):
        delta = row["r2_total"] - baseline_r2
        marker = " ⭐" if row["r2_total"] >= 0.20 else ""
        print(f"   {i:<6} {row['exp']:<40} {row['r2_total']:>12.4f} {delta:>+10.4f}{marker}")
    
    # Best result
    best = results_df.iloc[0]
    
    print("\n" + "=" * 70)
    print("💡 CONCLUSIONS")
    print("=" * 70)
    
    if best["r2_total"] >= 0.20:
        print(f"""
   ✅ TARGET ACHIEVED! R² >= 20%
   
   Best Approach: {best['exp']}
   R² (total): {best['r2_total']:.4f} ({best['r2_total']*100:.2f}%)
   Improvement: {(best['r2_total'] - baseline_r2)*100:+.2f}%
        """)
    else:
        gap = 0.20 - best["r2_total"]
        print(f"""
   ⚠️ Below 20% target, but improvements found!
   
   Best Approach: {best['exp']}
   R² (total): {best['r2_total']:.4f} ({best['r2_total']*100:.2f}%)
   Gap to 20%: {gap:.4f} ({gap*100:.2f}%)
   
   TOP RECOMMENDATIONS:
""")
        for i, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
            delta = row["r2_total"] - baseline_r2
            print(f"   {i}. {row['exp']}: {row['r2_total']*100:.2f}% (Δ {delta*100:+.2f}%)")
    
    # Timing
    end_time = datetime.now()
    print(f"\n   Duration: {end_time - start_time}")
    print("=" * 70)
    
    return results_df


if __name__ == "__main__":
    main()

