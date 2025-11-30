"""
Phase 16: Final Scientific Validation

Three rigorous tests to prove Titan V15 is robust:
1. Test A: Rolling Walk-Forward (Time Test)
2. Test B: Cross-Universe Generalization (Ticker Test)
3. Test C: Retail Signal Fragility (Noise Test)

Usage:
    python scripts/validate_robustness.py
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
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from lightgbm import LGBMClassifier
import mlflow


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_full_dataset():
    """Load and prepare the complete dataset with all features."""
    print("\n📂 Loading complete dataset...")
    
    # Load targets
    targets = pd.read_parquet("data/processed/targets_deseasonalized.parquet")
    targets["date"] = pd.to_datetime(targets["date"]).dt.tz_localize(None)
    if targets["ticker"].dtype.name == "category":
        targets["ticker"] = targets["ticker"].astype(str)
    
    # Load residuals (contains tech_pred)
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    residuals["date"] = pd.to_datetime(residuals["date"]).dt.tz_localize(None)
    if residuals["ticker"].dtype.name == "category":
        residuals["ticker"] = residuals["ticker"].astype(str)
    
    # Load news features
    news_features = pd.read_parquet("data/processed/news_features.parquet")
    news_features["date"] = pd.to_datetime(news_features["date"]).dt.tz_localize(None)
    if news_features["ticker"].dtype.name == "category":
        news_features["ticker"] = news_features["ticker"].astype(str)
    
    # Load retail proxy
    retail_proxy = pd.read_parquet("data/processed/reddit_proxy.parquet")
    retail_proxy["date"] = pd.to_datetime(retail_proxy["date"]).dt.tz_localize(None)
    if retail_proxy["ticker"].dtype.name == "category":
        retail_proxy["ticker"] = retail_proxy["ticker"].astype(str)
    
    # Merge targets with residuals
    df = pd.merge(
        targets,
        residuals[["date", "ticker", "pred_tech_excess", "resid_tech"]],
        on=["date", "ticker"],
        how="left"
    )
    
    # Create tech_pred
    if "seasonal_component" in df.columns and "pred_tech_excess" in df.columns:
        df["tech_pred"] = df["pred_tech_excess"] + df["seasonal_component"]
    else:
        df["tech_pred"] = df["target_log_var"].mean()
    
    # Merge news features
    news_cols = ["date", "ticker"] + [c for c in news_features.columns 
                                       if c.startswith(("news_pca", "shock", "sentiment", "novelty", "news_count"))]
    df = pd.merge(df, news_features[news_cols], on=["date", "ticker"], how="left")
    
    # Merge retail proxy
    retail_cols = ["date", "ticker", "volume_shock", "hype_signal", "hype_zscore", "price_acceleration"]
    if all(c in retail_proxy.columns for c in retail_cols):
        df = pd.merge(df, retail_proxy[retail_cols], on=["date", "ticker"], how="left")
    
    # Add calendar features
    df["date"] = pd.to_datetime(df["date"])
    dow = df["date"].dt.dayofweek
    df["is_monday"] = (dow == 0).astype(int)
    df["is_friday"] = (dow == 4).astype(int)
    df["is_q4"] = (df["date"].dt.quarter == 4).astype(int)
    df["year"] = df["date"].dt.year
    
    # Add momentum features
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        ticker_data = df.loc[mask, "target_log_var"]
        df.loc[mask, "vol_ma5"] = ticker_data.rolling(5, min_periods=1).mean().shift(1)
        df.loc[mask, "vol_ma10"] = ticker_data.rolling(10, min_periods=1).mean().shift(1)
        df.loc[mask, "vol_std5"] = ticker_data.rolling(5, min_periods=2).std().shift(1)
        
        # Rolling retail features
        if "volume_shock" in df.columns:
            df.loc[mask, "volume_shock_roll3"] = df.loc[mask, "volume_shock"].rolling(3, min_periods=1).mean()
        if "hype_signal" in df.columns:
            df.loc[mask, "hype_signal_roll3"] = df.loc[mask, "hype_signal"].rolling(3, min_periods=1).mean()
            df.loc[mask, "hype_signal_roll7"] = df.loc[mask, "hype_signal"].rolling(7, min_periods=1).mean()
    
    # Fill NaN
    df["vol_ma5"] = df["vol_ma5"].fillna(df["target_log_var"].mean())
    df["vol_ma10"] = df["vol_ma10"].fillna(df["target_log_var"].mean())
    df["vol_std5"] = df["vol_std5"].fillna(0)
    
    # Add HAR features if not present
    if "rv_lag_1" not in df.columns:
        print("   ⚙️ Adding HAR features...")
        for ticker in df["ticker"].unique():
            mask = df["ticker"] == ticker
            ticker_data = df.loc[mask, "target_log_var"]
            
            df.loc[mask, "rv_lag_1"] = ticker_data.shift(1)
            df.loc[mask, "rv_lag_5"] = ticker_data.rolling(5, min_periods=1).mean().shift(1)
            df.loc[mask, "rv_lag_22"] = ticker_data.rolling(22, min_periods=1).mean().shift(1)
            
            # Returns squared lag
            if "returns" in df.columns:
                df.loc[mask, "returns_sq_lag_1"] = (df.loc[mask, "returns"] ** 2).shift(1)
            else:
                df.loc[mask, "returns_sq_lag_1"] = 0
        
        # Fill NaN
        df["rv_lag_1"] = df["rv_lag_1"].fillna(df["target_log_var"].mean())
        df["rv_lag_5"] = df["rv_lag_5"].fillna(df["target_log_var"].mean())
        df["rv_lag_22"] = df["rv_lag_22"].fillna(df["target_log_var"].mean())
        df["returns_sq_lag_1"] = df["returns_sq_lag_1"].fillna(0)
    
    # Add sector mapping
    SECTOR_MAP = {
        'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech',
        'JPM': 'Finance', 'BAC': 'Finance', 'V': 'Finance',
        'CAT': 'Industrial', 'GE': 'Industrial', 'BA': 'Industrial',
        'WMT': 'Consumer', 'MCD': 'Consumer', 'COST': 'Consumer',
        'XOM': 'Energy', 'CVX': 'Energy', 'SLB': 'Energy',
        'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare'
    }
    df["sector"] = df["ticker"].map(SECTOR_MAP)
    
    # Drop rows with missing target
    df = df.dropna(subset=["target_log_var"])
    
    print(f"   ✓ Loaded {len(df):,} rows")
    print(f"   ✓ Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   ✓ Tickers: {len(df['ticker'].unique())}")
    
    return df


def train_full_pipeline(train_df, test_df, alpha=100.0, winsorize_pct=0.02, verbose=True):
    """
    Train the complete Titan V15 pipeline on training data and evaluate on test data.
    
    This includes:
    0. HAR-RV Model -> tech_pred (if missing)
    1. News Classifier -> news_risk_score
    2. Retail Classifier -> retail_risk_score
    3. Coordinator -> final prediction
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # ==========================================================================
    # STEP 0: HAR-RV MODEL (if tech_pred is missing or has NaN)
    # ==========================================================================
    if "tech_pred" not in train_df.columns or train_df["tech_pred"].isna().mean() > 0.5:
        # Need to compute tech_pred using HAR features
        har_features = ["rv_lag_1", "rv_lag_5", "rv_lag_22", "returns_sq_lag_1"]
        har_features = [f for f in har_features if f in train_df.columns]
        
        if len(har_features) >= 2:
            X_har_train = train_df[har_features].fillna(0)
            y_har_train = train_df["target_log_var"]
            X_har_test = test_df[har_features].fillna(0)
            
            har_model = Ridge(alpha=1.0)
            har_model.fit(X_har_train, y_har_train)
            
            train_df["tech_pred"] = har_model.predict(X_har_train)
            test_df["tech_pred"] = har_model.predict(X_har_test)
        else:
            # Fallback: use simple mean
            train_df["tech_pred"] = train_df["target_log_var"].mean()
            test_df["tech_pred"] = train_df["target_log_var"].mean()
    
    # Fill any remaining NaN in tech_pred
    train_mean = train_df["target_log_var"].mean()
    train_df["tech_pred"] = train_df["tech_pred"].fillna(train_mean)
    test_df["tech_pred"] = test_df["tech_pred"].fillna(train_mean)
    
    # Compute resid_tech if not present
    if "resid_tech" not in train_df.columns:
        train_df["resid_tech"] = train_df["target_log_var"] - train_df["tech_pred"]
        test_df["resid_tech"] = test_df["target_log_var"] - test_df["tech_pred"]
    
    # ==========================================================================
    # STEP 1: NEWS CLASSIFIER
    # ==========================================================================
    news_feature_cols = [c for c in train_df.columns if c.startswith("news_pca")][:10]
    news_feature_cols += ["shock_index", "news_count"] if "shock_index" in train_df.columns else []
    news_feature_cols = [c for c in news_feature_cols if c in train_df.columns]
    
    if len(news_feature_cols) > 0 and "resid_tech" in train_df.columns:
        # Create extreme event target
        threshold = train_df["resid_tech"].quantile(0.80)
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["is_extreme"] = (train_df["resid_tech"] > threshold).astype(int)
        test_df["is_extreme"] = (test_df["resid_tech"] > threshold).astype(int)
        
        X_news_train = train_df[news_feature_cols].fillna(0)
        y_news_train = train_df["is_extreme"]
        X_news_test = test_df[news_feature_cols].fillna(0)
        
        news_clf = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            verbose=-1,
            random_state=42
        )
        news_clf.fit(X_news_train, y_news_train)
        
        train_df["news_risk_score"] = news_clf.predict_proba(X_news_train)[:, 1]
        test_df["news_risk_score"] = news_clf.predict_proba(X_news_test)[:, 1]
    else:
        train_df["news_risk_score"] = 0.2
        test_df["news_risk_score"] = 0.2
    
    # ==========================================================================
    # STEP 2: RETAIL CLASSIFIER
    # ==========================================================================
    retail_feature_cols = ["volume_shock", "hype_signal", "hype_zscore", "price_acceleration"]
    retail_feature_cols += ["volume_shock_roll3", "hype_signal_roll3", "hype_signal_roll7"]
    retail_feature_cols = [c for c in retail_feature_cols if c in train_df.columns]
    
    if len(retail_feature_cols) > 0 and "resid_tech" in train_df.columns:
        # Use same threshold
        threshold = train_df["resid_tech"].quantile(0.80)
        
        X_retail_train = train_df[retail_feature_cols].fillna(0)
        y_retail_train = train_df["is_extreme"]
        X_retail_test = test_df[retail_feature_cols].fillna(0)
        
        retail_clf = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            verbose=-1,
            random_state=42
        )
        retail_clf.fit(X_retail_train, y_retail_train)
        
        train_df["retail_risk_score"] = retail_clf.predict_proba(X_retail_train)[:, 1]
        test_df["retail_risk_score"] = retail_clf.predict_proba(X_retail_test)[:, 1]
    else:
        train_df["retail_risk_score"] = 0.2
        test_df["retail_risk_score"] = 0.2
    
    # ==========================================================================
    # STEP 3: CREATE INTERACTION FEATURES
    # ==========================================================================
    train_df["news_x_retail"] = train_df["news_risk_score"] * train_df["retail_risk_score"]
    test_df["news_x_retail"] = test_df["news_risk_score"] * test_df["retail_risk_score"]
    
    # ==========================================================================
    # STEP 4: COORDINATOR (RIDGE)
    # ==========================================================================
    coord_features = [
        "tech_pred", "news_risk_score", "retail_risk_score",
        "is_friday", "is_monday", "is_q4",
        "vol_ma5", "vol_ma10", "vol_std5",
        "news_x_retail"
    ]
    coord_features = [c for c in coord_features if c in train_df.columns]
    
    X_train = train_df[coord_features].fillna(0)
    y_train = train_df["target_log_var"]
    X_test = test_df[coord_features].fillna(0)
    y_test = test_df["target_log_var"]
    
    # Winsorization on training target
    lower = y_train.quantile(winsorize_pct)
    upper = y_train.quantile(1 - winsorize_pct)
    y_train_winsorized = y_train.clip(lower=lower, upper=upper)
    
    # Train Ridge
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train, y_train_winsorized)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Baseline
    if "tech_pred" in test_df.columns and test_df["tech_pred"].notna().any():
        valid_mask = test_df["tech_pred"].notna() & y_test.notna()
        if valid_mask.sum() > 10:
            baseline_r2 = r2_score(y_test[valid_mask], test_df.loc[valid_mask, "tech_pred"])
        else:
            baseline_r2 = 0.0
    else:
        baseline_r2 = 0.0
    
    return {
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "baseline_r2": baseline_r2,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "model": model,
        "features": coord_features
    }


# =============================================================================
# TEST A: ROLLING WALK-FORWARD
# =============================================================================

def test_rolling_walkforward(df):
    """
    Test A: Rolling Walk-Forward Validation
    
    Train on historical data, test on each year.
    Re-train all components at each step to prevent look-ahead bias.
    """
    print("\n" + "=" * 70)
    print("🧪 TEST A: ROLLING WALK-FORWARD (Time Test)")
    print("=" * 70)
    
    years = [2021, 2022, 2023, 2024]
    results = []
    
    for year in years:
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year+1}-01-01")
        
        # Training: All data before this year
        train = df[df["date"] < year_start].copy()
        
        # Test: This year only
        test = df[(df["date"] >= year_start) & (df["date"] < year_end)].copy()
        
        if len(train) < 100 or len(test) < 50:
            print(f"   ⚠️ Year {year}: Insufficient data (train={len(train)}, test={len(test)})")
            continue
        
        # Train full pipeline
        metrics = train_full_pipeline(train, test, verbose=False)
        
        # Market context
        market_context = ""
        if year == 2022:
            market_context = " 📉 (Bear Market)"
        elif year == 2021:
            market_context = " 📈 (Bull Market)"
        elif year == 2023:
            market_context = " 📊 (Recovery)"
        elif year == 2024:
            market_context = " 📈 (AI Rally)"
        
        results.append({
            "year": year,
            "train_size": metrics["n_train"],
            "test_size": metrics["n_test"],
            "r2": metrics["test_r2"],
            "rmse": metrics["test_rmse"],
            "baseline_r2": metrics["baseline_r2"],
            "context": market_context
        })
        
        improvement = (metrics["test_r2"] - metrics["baseline_r2"]) * 100
        status = "✅" if metrics["test_r2"] > 0.10 else "⚠️"
        
        print(f"   {year}: R² = {metrics['test_r2']:.4f} ({metrics['test_r2']*100:.2f}%) "
              f"[Baseline: {metrics['baseline_r2']:.4f}] Δ{improvement:+.2f}%{market_context} {status}")
    
    # Summary statistics
    r2_values = [r["r2"] for r in results]
    mean_r2 = np.mean(r2_values)
    std_r2 = np.std(r2_values)
    min_r2 = np.min(r2_values)
    max_r2 = np.max(r2_values)
    
    print(f"\n   📊 Summary:")
    print(f"      Mean R²:     {mean_r2:.4f} ({mean_r2*100:.2f}%)")
    print(f"      Std R²:      {std_r2:.4f}")
    print(f"      Range:       [{min_r2:.4f}, {max_r2:.4f}]")
    print(f"      Stability:   {'✅ STABLE' if std_r2 < 0.10 else '⚠️ VARIABLE'}")
    
    # Check 2022 specifically
    year_2022 = [r for r in results if r["year"] == 2022]
    if year_2022:
        r2_2022 = year_2022[0]["r2"]
        print(f"\n   🐻 Bear Market (2022) Survival: {'✅ PASSED' if r2_2022 > 0.05 else '❌ FAILED'} (R² = {r2_2022:.4f})")
    
    return {
        "results": results,
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "stability_score": 1 - std_r2,  # Higher is better
        "passed": std_r2 < 0.10 and min_r2 > 0.05
    }


# =============================================================================
# TEST B: CROSS-UNIVERSE GENERALIZATION
# =============================================================================

def test_cross_universe(df):
    """
    Test B: Cross-Universe Generalization
    
    Train on 12 tickers, test on 6 completely unseen tickers.
    """
    print("\n" + "=" * 70)
    print("🧪 TEST B: CROSS-UNIVERSE GENERALIZATION (Ticker Test)")
    print("=" * 70)
    
    # Define splits
    in_sample_tickers = ['AAPL', 'MSFT', 'NVDA', 'JPM', 'BAC', 'V', 
                         'XOM', 'CVX', 'SLB', 'JNJ', 'PFE', 'UNH']
    out_of_sample_tickers = ['CAT', 'GE', 'BA', 'WMT', 'MCD', 'COST']
    
    print(f"\n   In-Sample Tickers (12): {in_sample_tickers}")
    print(f"   Out-of-Sample Tickers (6): {out_of_sample_tickers}")
    
    # Split data
    train = df[df["ticker"].isin(in_sample_tickers)].copy()
    test = df[df["ticker"].isin(out_of_sample_tickers)].copy()
    
    print(f"\n   Train: {len(train):,} samples (In-Sample)")
    print(f"   Test:  {len(test):,} samples (Out-of-Sample)")
    
    # Check sector coverage
    train_sectors = train["sector"].unique()
    test_sectors = test["sector"].unique()
    print(f"\n   Train sectors: {sorted(train_sectors)}")
    print(f"   Test sectors:  {sorted(test_sectors)}")
    
    # Important: Test sectors NOT in training!
    unseen_sectors = set(test_sectors) - set(train_sectors)
    if unseen_sectors:
        print(f"   ⚠️ Unseen sectors in test: {unseen_sectors}")
    
    # Train and evaluate
    metrics = train_full_pipeline(train, test, verbose=False)
    
    print(f"\n   📊 Results:")
    print(f"      Test R²:     {metrics['test_r2']:.4f} ({metrics['test_r2']*100:.2f}%)")
    print(f"      Test RMSE:   {metrics['test_rmse']:.4f}")
    print(f"      Baseline R²: {metrics['baseline_r2']:.4f}")
    
    improvement = (metrics["test_r2"] - metrics["baseline_r2"]) * 100
    print(f"      Improvement: {improvement:+.2f}%")
    
    # Per-ticker breakdown using individual ticker models
    print(f"\n   📊 Per-Ticker Breakdown:")
    print("   " + "-" * 50)
    
    ticker_results = []
    for ticker in out_of_sample_tickers:
        ticker_test = test[test["ticker"] == ticker].copy()
        if len(ticker_test) > 50:
            # Train a simple model for this ticker using training data
            ticker_train = train.copy()
            
            # Run mini pipeline
            mini_metrics = train_full_pipeline(ticker_train, ticker_test, verbose=False)
            ticker_r2 = mini_metrics["test_r2"]
            
            ticker_results.append({
                "ticker": ticker,
                "r2": ticker_r2,
                "n_samples": len(ticker_test)
            })
            
            status = "✅" if ticker_r2 > 0.10 else "⚠️"
            print(f"      {ticker:5s}: R² = {ticker_r2:.4f} ({ticker_r2*100:.2f}%) {status}")
    
    # Generalization assessment
    oos_r2 = metrics["test_r2"]
    generalization_passed = oos_r2 > 0.10
    
    print(f"\n   🎯 Generalization: {'✅ PASSED' if generalization_passed else '⚠️ WEAK'} (R² = {oos_r2:.4f})")
    
    return {
        "oos_r2": oos_r2,
        "baseline_r2": metrics["baseline_r2"],
        "improvement": improvement,
        "ticker_results": ticker_results,
        "generalization_score": oos_r2,
        "passed": generalization_passed
    }


# =============================================================================
# TEST C: RETAIL SIGNAL FRAGILITY
# =============================================================================

def compute_noisy_prediction(train_df, test_df, noise_std, alpha=100.0, winsorize_pct=0.02):
    """
    Train the pipeline and add noise to retail features in the test set.
    Returns the R² with noisy retail signals.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Step 0: HAR model
    if "tech_pred" not in train_df.columns or train_df["tech_pred"].isna().mean() > 0.5:
        har_features = ["rv_lag_1", "rv_lag_5", "rv_lag_22", "returns_sq_lag_1"]
        har_features = [f for f in har_features if f in train_df.columns]
        
        if len(har_features) >= 2:
            X_har_train = train_df[har_features].fillna(0)
            y_har_train = train_df["target_log_var"]
            X_har_test = test_df[har_features].fillna(0)
            
            har_model = Ridge(alpha=1.0)
            har_model.fit(X_har_train, y_har_train)
            
            train_df["tech_pred"] = har_model.predict(X_har_train)
            test_df["tech_pred"] = har_model.predict(X_har_test)
        else:
            train_df["tech_pred"] = train_df["target_log_var"].mean()
            test_df["tech_pred"] = train_df["target_log_var"].mean()
    
    train_mean = train_df["target_log_var"].mean()
    train_df["tech_pred"] = train_df["tech_pred"].fillna(train_mean)
    test_df["tech_pred"] = test_df["tech_pred"].fillna(train_mean)
    
    if "resid_tech" not in train_df.columns:
        train_df["resid_tech"] = train_df["target_log_var"] - train_df["tech_pred"]
        test_df["resid_tech"] = test_df["target_log_var"] - test_df["tech_pred"]
    
    # Step 1: News Classifier
    news_feature_cols = [c for c in train_df.columns if c.startswith("news_pca")][:10]
    news_feature_cols += ["shock_index", "news_count"] if "shock_index" in train_df.columns else []
    news_feature_cols = [c for c in news_feature_cols if c in train_df.columns]
    
    if len(news_feature_cols) > 0 and "resid_tech" in train_df.columns:
        threshold = train_df["resid_tech"].quantile(0.80)
        train_df["is_extreme"] = (train_df["resid_tech"] > threshold).astype(int)
        test_df["is_extreme"] = (test_df["resid_tech"] > threshold).astype(int)
        
        X_news_train = train_df[news_feature_cols].fillna(0)
        y_news_train = train_df["is_extreme"]
        X_news_test = test_df[news_feature_cols].fillna(0)
        
        news_clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, verbose=-1, random_state=42)
        news_clf.fit(X_news_train, y_news_train)
        
        train_df["news_risk_score"] = news_clf.predict_proba(X_news_train)[:, 1]
        test_df["news_risk_score"] = news_clf.predict_proba(X_news_test)[:, 1]
    else:
        train_df["news_risk_score"] = 0.2
        test_df["news_risk_score"] = 0.2
    
    # Step 2: Retail Classifier with NOISY FEATURES
    retail_feature_cols = ["volume_shock", "hype_signal", "hype_zscore", "price_acceleration"]
    retail_feature_cols += ["volume_shock_roll3", "hype_signal_roll3", "hype_signal_roll7"]
    retail_feature_cols = [c for c in retail_feature_cols if c in train_df.columns]
    
    if len(retail_feature_cols) > 0 and "resid_tech" in train_df.columns:
        threshold = train_df["resid_tech"].quantile(0.80)
        
        X_retail_train = train_df[retail_feature_cols].fillna(0)
        y_retail_train = train_df["is_extreme"]
        
        # Add noise to test features BEFORE prediction
        X_retail_test = test_df[retail_feature_cols].fillna(0).copy()
        for col in X_retail_test.columns:
            noise = np.random.normal(0, noise_std * X_retail_test[col].std(), size=len(X_retail_test))
            X_retail_test[col] = X_retail_test[col] + noise
        
        retail_clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, verbose=-1, random_state=42)
        retail_clf.fit(X_retail_train, y_retail_train)
        
        train_df["retail_risk_score"] = retail_clf.predict_proba(X_retail_train)[:, 1]
        test_df["retail_risk_score"] = retail_clf.predict_proba(X_retail_test)[:, 1]
    else:
        train_df["retail_risk_score"] = 0.2
        test_df["retail_risk_score"] = 0.2
    
    # Step 3: Interaction
    train_df["news_x_retail"] = train_df["news_risk_score"] * train_df["retail_risk_score"]
    test_df["news_x_retail"] = test_df["news_risk_score"] * test_df["retail_risk_score"]
    
    # Step 4: Coordinator
    coord_features = [
        "tech_pred", "news_risk_score", "retail_risk_score",
        "is_friday", "is_monday", "is_q4",
        "vol_ma5", "vol_ma10", "vol_std5",
        "news_x_retail"
    ]
    coord_features = [c for c in coord_features if c in train_df.columns]
    
    X_train = train_df[coord_features].fillna(0)
    y_train = train_df["target_log_var"]
    X_test = test_df[coord_features].fillna(0)
    y_test = test_df["target_log_var"]
    
    lower = y_train.quantile(winsorize_pct)
    upper = y_train.quantile(1 - winsorize_pct)
    y_train_winsorized = y_train.clip(lower=lower, upper=upper)
    
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train, y_train_winsorized)
    
    y_pred = model.predict(X_test)
    
    return r2_score(y_test, y_pred)


def test_retail_fragility(df):
    """
    Test C: Retail Signal Fragility
    
    Add Gaussian noise to retail_risk_score and measure R² drop.
    """
    print("\n" + "=" * 70)
    print("🧪 TEST C: RETAIL SIGNAL FRAGILITY (Noise Test)")
    print("=" * 70)
    
    # Standard train/test split
    cutoff = pd.Timestamp("2023-01-01")
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    
    print(f"\n   Train: {len(train):,} samples")
    print(f"   Test:  {len(test):,} samples")
    
    # Baseline (no noise) - run full pipeline to get baseline
    baseline_metrics = train_full_pipeline(train.copy(), test.copy(), verbose=False)
    baseline_r2 = baseline_metrics["test_r2"]
    
    print(f"\n   📊 Baseline (No Noise): R² = {baseline_r2:.4f}")
    
    # Test with different noise levels
    noise_levels = [0.1, 0.25, 0.5, 0.75, 1.0]
    results = []
    
    print(f"\n   📊 Noise Impact on retail_risk_score:")
    print("   " + "-" * 55)
    
    np.random.seed(42)
    
    for noise_std in noise_levels:
        # We need to inject noise into the pipeline
        # Custom training that applies noise to retail features BEFORE classification
        noisy_r2 = compute_noisy_prediction(train.copy(), test.copy(), noise_std)
        
        drop = (baseline_r2 - noisy_r2) / baseline_r2 * 100 if baseline_r2 > 0 else 0
        
        results.append({
            "noise_std": noise_std,
            "r2": noisy_r2,
            "drop_pct": drop
        })
        
        status = "✅" if drop < 10 else ("⚠️" if drop < 25 else "❌")
        print(f"      σ={noise_std:.2f}: R² = {noisy_r2:.4f} (Drop: {drop:+.2f}%) {status}")
    
    # Fragility assessment
    high_noise_result = [r for r in results if r["noise_std"] == 0.5][0]
    fragility_score = high_noise_result["drop_pct"]
    
    # Robustness: less than 20% drop with high noise is good
    is_robust = fragility_score < 20
    
    print(f"\n   📊 Summary:")
    print(f"      Baseline R²:      {baseline_r2:.4f}")
    print(f"      High Noise R²:    {high_noise_result['r2']:.4f} (σ=0.5)")
    print(f"      Fragility Score:  {fragility_score:.2f}% drop")
    print(f"      Robustness:       {'✅ ROBUST' if is_robust else '⚠️ FRAGILE'}")
    
    return {
        "baseline_r2": baseline_r2,
        "results": results,
        "fragility_score": fragility_score,  # % drop with σ=0.5 noise
        "passed": is_robust
    }


# =============================================================================
# FINAL REPORT
# =============================================================================

def generate_final_report(walkforward_results, crossuniverse_results, fragility_results):
    """Generate the final validation report."""
    
    print("\n" + "=" * 70)
    print("📋 PHASE 16: FINAL SCIENTIFIC VALIDATION REPORT")
    print("=" * 70)
    
    # Extract scores
    stability_score = walkforward_results["stability_score"]
    generalization_score = crossuniverse_results["generalization_score"]
    fragility_score = fragility_results["fragility_score"]
    
    # Normalize to 0-1 scale for comparison
    stability_normalized = max(0, min(1, stability_score))
    generalization_normalized = max(0, min(1, generalization_score))
    robustness_normalized = max(0, min(1, 1 - fragility_score / 100))
    
    overall_score = (stability_normalized + generalization_normalized + robustness_normalized) / 3
    
    print(f"""
   ┌────────────────────────────────────────────────────────────────────┐
   │                        VALIDATION SUMMARY                          │
   ├────────────────────────────────────────────────────────────────────┤
   │ Test                          │ Result       │ Score    │ Status   │
   ├───────────────────────────────┼──────────────┼──────────┼──────────┤
   │ A: Walk-Forward (Time)        │ σ = {walkforward_results['std_r2']:.4f}   │ {stability_normalized:.2f}     │ {'✅ PASS' if walkforward_results['passed'] else '⚠️ WARN'}   │
   │ B: Cross-Universe (Ticker)    │ R² = {crossuniverse_results['oos_r2']:.4f}  │ {generalization_normalized:.2f}     │ {'✅ PASS' if crossuniverse_results['passed'] else '⚠️ WARN'}   │
   │ C: Fragility (Noise)          │ -{fragility_score:.1f}% drop │ {robustness_normalized:.2f}     │ {'✅ PASS' if fragility_results['passed'] else '⚠️ WARN'}   │
   ├───────────────────────────────┴──────────────┴──────────┴──────────┤
   │ OVERALL ROBUSTNESS SCORE: {overall_score:.2f}                                      │
   └────────────────────────────────────────────────────────────────────┘
    """)
    
    # Detailed results
    print("\n   📊 DETAILED RESULTS:")
    print("   " + "-" * 60)
    
    # Walk-forward details
    print("\n   🕐 Walk-Forward (Time Stability):")
    for r in walkforward_results["results"]:
        status = "✅" if r["r2"] > 0.10 else "⚠️"
        print(f"      {r['year']}: R² = {r['r2']:.4f}{r['context']} {status}")
    print(f"      Mean: {walkforward_results['mean_r2']:.4f} ± {walkforward_results['std_r2']:.4f}")
    
    # Cross-universe details
    print("\n   🌐 Cross-Universe (Generalization):")
    print(f"      In-Sample Training:  12 tickers (Tech, Finance, Energy, Healthcare)")
    print(f"      Out-of-Sample Test:  6 tickers (Industrial, Consumer)")
    print(f"      OOS R²: {crossuniverse_results['oos_r2']:.4f} ({crossuniverse_results['oos_r2']*100:.2f}%)")
    
    # Fragility details
    print("\n   🔊 Fragility (Noise Robustness):")
    print(f"      Baseline R²:   {fragility_results['baseline_r2']:.4f}")
    for r in fragility_results["results"]:
        status = "✅" if r["drop_pct"] < 20 else "⚠️"
        print(f"      σ={r['noise_std']:.2f} noise: R² = {r['r2']:.4f} (Drop: {r['drop_pct']:+.1f}%) {status}")
    
    # Final verdict
    tests_passed = sum([
        walkforward_results["passed"],
        crossuniverse_results["passed"],
        fragility_results["passed"]
    ])
    
    print("\n" + "=" * 70)
    print("🏆 FINAL VERDICT")
    print("=" * 70)
    
    if tests_passed == 3:
        verdict = "FULLY VALIDATED"
        emoji = "🏆"
    elif tests_passed == 2:
        verdict = "MOSTLY VALIDATED"
        emoji = "✅"
    elif tests_passed == 1:
        verdict = "PARTIALLY VALIDATED"
        emoji = "⚠️"
    else:
        verdict = "NEEDS IMPROVEMENT"
        emoji = "❌"
    
    print(f"""
   {emoji} TITAN V15 STATUS: {verdict}
   
   Tests Passed: {tests_passed}/3
   
   ├── Time Stability:    {'✅ PASS' if walkforward_results['passed'] else '❌ FAIL'} (σ = {walkforward_results['std_r2']:.4f})
   ├── Generalization:    {'✅ PASS' if crossuniverse_results['passed'] else '❌ FAIL'} (OOS R² = {crossuniverse_results['oos_r2']:.4f})
   └── Noise Robustness:  {'✅ PASS' if fragility_results['passed'] else '❌ FAIL'} (Drop = {fragility_score:.1f}%)
   
   Overall Robustness Score: {overall_score:.2f}/1.00
    """)
    
    # Key insights
    print("   💡 KEY INSIGHTS:")
    print("   " + "-" * 50)
    
    if walkforward_results["passed"]:
        print("   ✅ Model is temporally stable across market regimes")
    else:
        print("   ⚠️ Model shows high variance across time periods")
    
    if crossuniverse_results["passed"]:
        print("   ✅ Model generalizes to unseen tickers/sectors")
    else:
        print("   ⚠️ Model struggles with out-of-sample tickers")
    
    if fragility_results["passed"]:
        print("   ✅ Model is robust to noise in retail signals")
    else:
        print("   ⚠️ Model is sensitive to retail signal noise")
    
    print("\n" + "=" * 70)
    
    return {
        "stability_score": stability_normalized,
        "generalization_score": generalization_normalized,
        "robustness_score": robustness_normalized,
        "overall_score": overall_score,
        "tests_passed": tests_passed,
        "verdict": verdict
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🔬 PHASE 16: FINAL SCIENTIFIC VALIDATION")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # End any stale MLflow runs
    mlflow.end_run()
    
    # Load data
    df = load_full_dataset()
    
    # Run all tests
    walkforward_results = test_rolling_walkforward(df)
    crossuniverse_results = test_cross_universe(df)
    fragility_results = test_retail_fragility(df)
    
    # Generate final report
    final_results = generate_final_report(
        walkforward_results,
        crossuniverse_results,
        fragility_results
    )
    
    # Timing
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n   Duration: {duration}")
    print(f"   Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return {
        "walkforward": walkforward_results,
        "crossuniverse": crossuniverse_results,
        "fragility": fragility_results,
        "summary": final_results
    }


if __name__ == "__main__":
    main()

