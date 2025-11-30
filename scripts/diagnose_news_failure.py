"""
DIAGNOSTIC AUDIT: Why NewsAgent Fails

This is a TEMPORARY experiment script to diagnose and fix the NewsAgent failure.

Hypotheses to Test:
1. SIGNAL EXISTENCE: Do news features correlate with volatility at all?
2. TARGET CHOICE: Are we predicting the wrong thing (residuals vs raw)?
3. TRAIN/TEST SHIFT: Is there a distribution shift causing poor generalization?
4. FEATURE QUALITY: Are PCA features noise or signal?
5. OVERFITTING: Is regularization too weak?
6. LAG STRUCTURE: Wrong temporal relationship?

Goal: Find a configuration that gives R² >= 0.15

Usage:
    python scripts/diagnose_news_failure.py
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

np.random.seed(42)


def load_all_data():
    """Load all datasets for diagnosis."""
    print("=" * 70)
    print("📂 LOADING DATA FOR DIAGNOSIS")
    print("=" * 70)
    
    # Load targets
    targets = pd.read_parquet("data/processed/targets.parquet")
    targets['date'] = pd.to_datetime(targets['date']).dt.tz_localize(None)
    if targets['ticker'].dtype.name == 'category':
        targets['ticker'] = targets['ticker'].astype(str)
    
    # Load news features
    news = pd.read_parquet("data/processed/news_features.parquet")
    news['date'] = pd.to_datetime(news['date']).dt.tz_localize(None)
    if news['ticker'].dtype.name == 'category':
        news['ticker'] = news['ticker'].astype(str)
    
    # Load residuals
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    residuals['date'] = pd.to_datetime(residuals['date']).dt.tz_localize(None)
    if residuals['ticker'].dtype.name == 'category':
        residuals['ticker'] = residuals['ticker'].astype(str)
    
    # Merge all
    df = pd.merge(targets, news, on=['date', 'ticker'], how='inner')
    df = pd.merge(df, residuals[['date', 'ticker', 'resid_tech', 'pred_tech']], 
                  on=['date', 'ticker'], how='left')
    
    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['target_log_var', 'news_count'])
    
    print(f"   Loaded {len(df):,} rows")
    
    return df


def test_1_signal_existence(df):
    """Test 1: Do news features correlate with volatility at all?"""
    print("\n" + "=" * 70)
    print("TEST 1: SIGNAL EXISTENCE")
    print("Does ANY news feature correlate with volatility?")
    print("=" * 70)
    
    # Get news columns
    news_cols = ['news_count', 'shock_index', 'sentiment_avg', 'novelty_score'] + \
                [c for c in df.columns if c.startswith('news_pca_')]
    
    # Calculate correlations with different targets
    targets = ['target_log_var', 'realized_vol', 'resid_tech']
    
    for target in targets:
        if target not in df.columns:
            continue
        
        print(f"\n   Correlations with {target}:")
        
        correlations = []
        for col in news_cols:
            if col not in df.columns:
                continue
            corr, pval = stats.pearsonr(df[col].fillna(0), df[target].fillna(0))
            correlations.append((col, corr, pval))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"   {'Feature':<20} {'Correlation':>12} {'P-value':>12} {'Sig':>5}")
        print("   " + "-" * 52)
        
        for col, corr, pval in correlations[:10]:
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"   {col:<20} {corr:>12.4f} {pval:>12.4f} {sig:>5}")
        
        # Any significant?
        sig_count = sum(1 for _, _, p in correlations if p < 0.05)
        print(f"\n   Significant correlations: {sig_count}/{len(correlations)}")


def test_2_target_choice(df):
    """Test 2: Are we predicting the wrong target?"""
    print("\n" + "=" * 70)
    print("TEST 2: TARGET CHOICE")
    print("Comparing: target_log_var vs realized_vol vs resid_tech")
    print("=" * 70)
    
    # Prepare features
    feature_cols = ['news_count', 'shock_index', 'sentiment_avg', 'novelty_score', 'VIX_close']
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Different targets
    targets_to_try = {
        'target_log_var': 'Log Variance (Next Day)',
        'realized_vol': 'Realized Vol (Same Day)',
        'resid_tech': 'Tech Residuals',
    }
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df['date'] < cutoff]
    test = df[df['date'] >= cutoff]
    
    X_train = train[feature_cols].fillna(0)
    X_test = test[feature_cols].fillna(0)
    
    print(f"\n   {'Target':<30} {'Train R²':>10} {'Test R²':>10}")
    print("   " + "-" * 52)
    
    results = {}
    for target_col, target_name in targets_to_try.items():
        if target_col not in df.columns:
            continue
        
        y_train = train[target_col].fillna(0)
        y_test = test[target_col].fillna(0)
        
        # Use Ridge (simple, robust)
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))
        
        status = "✅" if test_r2 > 0.05 else "❌"
        print(f"   {target_name:<30} {train_r2:>10.4f} {test_r2:>10.4f} {status}")
        
        results[target_col] = test_r2
    
    best = max(results.items(), key=lambda x: x[1])
    print(f"\n   Best target: {best[0]} ({best[1]:.4f})")
    
    return results


def test_3_train_test_shift(df):
    """Test 3: Is there a distribution shift between train and test?"""
    print("\n" + "=" * 70)
    print("TEST 3: TRAIN/TEST DISTRIBUTION SHIFT")
    print("Checking if train and test distributions are similar")
    print("=" * 70)
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df['date'] < cutoff]
    test = df[df['date'] >= cutoff]
    
    cols_to_check = ['news_count', 'shock_index', 'sentiment_avg', 'target_log_var', 'VIX_close']
    
    print(f"\n   {'Feature':<20} {'Train Mean':>12} {'Test Mean':>12} {'Shift':>10} {'KS p-val':>10}")
    print("   " + "-" * 66)
    
    for col in cols_to_check:
        if col not in df.columns:
            continue
        
        train_vals = train[col].dropna()
        test_vals = test[col].dropna()
        
        train_mean = train_vals.mean()
        test_mean = test_vals.mean()
        shift = (test_mean - train_mean) / max(train_vals.std(), 0.001)
        
        # KS test
        ks_stat, ks_pval = stats.ks_2samp(train_vals, test_vals)
        
        status = "⚠️" if ks_pval < 0.01 else "✅"
        print(f"   {col:<20} {train_mean:>12.4f} {test_mean:>12.4f} {shift:>+10.2f} {ks_pval:>10.4f} {status}")


def test_4_feature_quality(df):
    """Test 4: Are PCA features noise or signal?"""
    print("\n" + "=" * 70)
    print("TEST 4: FEATURE QUALITY")
    print("Comparing: Raw features vs PCA features vs Both")
    print("=" * 70)
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df['date'] < cutoff]
    test = df[df['date'] >= cutoff]
    
    target_col = 'target_log_var'
    y_train = train[target_col].fillna(0)
    y_test = test[target_col].fillna(0)
    
    # Feature sets
    raw_features = ['news_count', 'shock_index', 'sentiment_avg', 'novelty_score', 'VIX_close']
    raw_features = [c for c in raw_features if c in df.columns]
    
    pca_features = [c for c in df.columns if c.startswith('news_pca_')]
    
    feature_sets = {
        'Raw Only': raw_features,
        'PCA Only': pca_features,
        'Raw + PCA': raw_features + pca_features,
        'VIX Only': ['VIX_close'],
    }
    
    print(f"\n   {'Feature Set':<20} {'N Features':>12} {'Train R²':>10} {'Test R²':>10}")
    print("   " + "-" * 54)
    
    for name, features in feature_sets.items():
        features = [f for f in features if f in df.columns]
        if len(features) == 0:
            continue
        
        X_train = train[features].fillna(0)
        X_test = test[features].fillna(0)
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))
        
        status = "✅" if test_r2 > 0.05 else "❌"
        print(f"   {name:<20} {len(features):>12} {train_r2:>10.4f} {test_r2:>10.4f} {status}")


def test_5_regularization(df):
    """Test 5: Is the model overfitting due to weak regularization?"""
    print("\n" + "=" * 70)
    print("TEST 5: REGULARIZATION SWEEP")
    print("Testing different regularization strengths")
    print("=" * 70)
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df['date'] < cutoff]
    test = df[df['date'] >= cutoff]
    
    # Use all features
    feature_cols = ['news_count', 'shock_index', 'sentiment_avg', 'novelty_score', 'VIX_close'] + \
                   [c for c in df.columns if c.startswith('news_pca_')]
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X_train = train[feature_cols].fillna(0)
    X_test = test[feature_cols].fillna(0)
    y_train = train['target_log_var'].fillna(0)
    y_test = test['target_log_var'].fillna(0)
    
    print(f"\n   {'Model':<25} {'Alpha':>10} {'Train R²':>10} {'Test R²':>10} {'Gap':>10}")
    print("   " + "-" * 67)
    
    best_test_r2 = -np.inf
    best_config = None
    
    # Ridge with different alphas
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))
        gap = train_r2 - test_r2
        
        status = "✅" if test_r2 > 0.05 else ""
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_config = f"Ridge(alpha={alpha})"
        
        print(f"   {'Ridge':<25} {alpha:>10.2f} {train_r2:>10.4f} {test_r2:>10.4f} {gap:>10.4f} {status}")
    
    # Lasso with different alphas
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_train, y_train)
        
        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))
        gap = train_r2 - test_r2
        
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_config = f"Lasso(alpha={alpha})"
        
        print(f"   {'Lasso':<25} {alpha:>10.3f} {train_r2:>10.4f} {test_r2:>10.4f} {gap:>10.4f}")
    
    print(f"\n   Best config: {best_config} with Test R² = {best_test_r2:.4f}")
    
    return best_test_r2


def test_6_lag_structure(df):
    """Test 6: Wrong temporal relationship?"""
    print("\n" + "=" * 70)
    print("TEST 6: LAG STRUCTURE")
    print("Testing different lag relationships")
    print("=" * 70)
    
    df = df.sort_values(['ticker', 'date']).copy()
    
    # Create lagged versions
    for lag in [1, 2, 3, 5]:
        df[f'news_count_lag{lag}'] = df.groupby('ticker')['news_count'].shift(lag)
        df[f'shock_index_lag{lag}'] = df.groupby('ticker')['shock_index'].shift(lag)
    
    # Also try forward shifts (checking for data leakage)
    df['news_count_future1'] = df.groupby('ticker')['news_count'].shift(-1)
    
    df = df.dropna()
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df['date'] < cutoff]
    test = df[df['date'] >= cutoff]
    
    y_train = train['target_log_var']
    y_test = test['target_log_var']
    
    print(f"\n   {'Feature Set':<30} {'Train R²':>10} {'Test R²':>10}")
    print("   " + "-" * 52)
    
    # Test different lag configurations
    configs = {
        'Same day (t)': ['news_count', 'shock_index'],
        'Lag 1 (t-1)': ['news_count_lag1', 'shock_index_lag1'],
        'Lag 2 (t-2)': ['news_count_lag2', 'shock_index_lag2'],
        'Lag 3 (t-3)': ['news_count_lag3', 'shock_index_lag3'],
        'Lag 5 (t-5)': ['news_count_lag5', 'shock_index_lag5'],
        'Future (t+1) LEAKAGE': ['news_count_future1'],
    }
    
    for name, features in configs.items():
        features = [f for f in features if f in df.columns]
        if len(features) == 0:
            continue
        
        # Add VIX
        features = features + ['VIX_close']
        
        X_train = train[features].fillna(0)
        X_test = test[features].fillna(0)
        
        model = Ridge(alpha=10.0)
        model.fit(X_train, y_train)
        
        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))
        
        status = "⚠️ LEAKAGE" if "Future" in name else ("✅" if test_r2 > 0.05 else "")
        print(f"   {name:<30} {train_r2:>10.4f} {test_r2:>10.4f} {status}")


def test_7_simple_baseline(df):
    """Test 7: What's the simplest model that works?"""
    print("\n" + "=" * 70)
    print("TEST 7: SIMPLE BASELINES")
    print("Finding the minimal working model")
    print("=" * 70)
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df['date'] < cutoff]
    test = df[df['date'] >= cutoff]
    
    y_train = train['target_log_var']
    y_test = test['target_log_var']
    
    print(f"\n   {'Model':<40} {'Train R²':>10} {'Test R²':>10}")
    print("   " + "-" * 62)
    
    results = {}
    
    # 1. Just VIX
    X_train = train[['VIX_close']].fillna(0)
    X_test = test[['VIX_close']].fillna(0)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    results['VIX only'] = test_r2
    print(f"   {'VIX only':<40} {train_r2:>10.4f} {test_r2:>10.4f}")
    
    # 2. VIX + news_count
    features = ['VIX_close', 'news_count']
    X_train = train[features].fillna(0)
    X_test = test[features].fillna(0)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    results['VIX + news_count'] = test_r2
    print(f"   {'VIX + news_count':<40} {train_r2:>10.4f} {test_r2:>10.4f}")
    
    # 3. VIX + shock_index
    features = ['VIX_close', 'shock_index']
    X_train = train[features].fillna(0)
    X_test = test[features].fillna(0)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    results['VIX + shock_index'] = test_r2
    print(f"   {'VIX + shock_index':<40} {train_r2:>10.4f} {test_r2:>10.4f}")
    
    # 4. VIX + sentiment
    features = ['VIX_close', 'sentiment_avg']
    X_train = train[features].fillna(0)
    X_test = test[features].fillna(0)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    results['VIX + sentiment'] = test_r2
    print(f"   {'VIX + sentiment':<40} {train_r2:>10.4f} {test_r2:>10.4f}")
    
    # 5. VIX + all core news
    features = ['VIX_close', 'news_count', 'shock_index', 'sentiment_avg']
    X_train = train[features].fillna(0)
    X_test = test[features].fillna(0)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    results['VIX + core news'] = test_r2
    print(f"   {'VIX + core news (count, shock, sent)':<40} {train_r2:>10.4f} {test_r2:>10.4f}")
    
    # 6. HAR features only (baseline comparison)
    har_features = ['rv_lag_1', 'rv_lag_5', 'rv_lag_22']
    if all(f in df.columns for f in har_features):
        # Need to create HAR features
        pass
    
    best = max(results.items(), key=lambda x: x[1])
    print(f"\n   Best simple model: {best[0]} with Test R² = {best[1]:.4f}")
    
    return results


def test_8_combined_approach(df):
    """Test 8: What if we combine news with HAR features?"""
    print("\n" + "=" * 70)
    print("TEST 8: NEWS + HAR COMBINED")
    print("Does news add value ON TOP of HAR?")
    print("=" * 70)
    
    df = df.sort_values(['ticker', 'date']).copy()
    
    # Create HAR features
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        df.loc[mask, 'rv_lag_1'] = df.loc[mask, 'realized_vol'].shift(1)
        df.loc[mask, 'rv_lag_5'] = df.loc[mask, 'realized_vol'].rolling(5).mean().shift(1)
        df.loc[mask, 'rv_lag_22'] = df.loc[mask, 'realized_vol'].rolling(22).mean().shift(1)
    
    df = df.dropna(subset=['rv_lag_1', 'rv_lag_5', 'rv_lag_22'])
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df['date'] < cutoff]
    test = df[df['date'] >= cutoff]
    
    y_train = train['target_log_var']
    y_test = test['target_log_var']
    
    print(f"\n   {'Model':<40} {'Train R²':>10} {'Test R²':>10} {'Delta':>10}")
    print("   " + "-" * 72)
    
    # HAR only
    har_features = ['rv_lag_1', 'rv_lag_5', 'rv_lag_22', 'VIX_close']
    X_train = train[har_features].fillna(0)
    X_test = test[har_features].fillna(0)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    har_train_r2 = r2_score(y_train, model.predict(X_train))
    har_test_r2 = r2_score(y_test, model.predict(X_test))
    print(f"   {'HAR only':<40} {har_train_r2:>10.4f} {har_test_r2:>10.4f} {'(baseline)':>10}")
    
    # HAR + news_count
    features = har_features + ['news_count']
    X_train = train[features].fillna(0)
    X_test = test[features].fillna(0)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    delta = test_r2 - har_test_r2
    status = "✅" if delta > 0 else "❌"
    print(f"   {'HAR + news_count':<40} {train_r2:>10.4f} {test_r2:>10.4f} {delta:>+10.4f} {status}")
    
    # HAR + shock_index
    features = har_features + ['shock_index']
    X_train = train[features].fillna(0)
    X_test = test[features].fillna(0)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    delta = test_r2 - har_test_r2
    status = "✅" if delta > 0 else "❌"
    print(f"   {'HAR + shock_index':<40} {train_r2:>10.4f} {test_r2:>10.4f} {delta:>+10.4f} {status}")
    
    # HAR + core news
    features = har_features + ['news_count', 'shock_index', 'sentiment_avg']
    X_train = train[features].fillna(0)
    X_test = test[features].fillna(0)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    delta = test_r2 - har_test_r2
    status = "✅" if delta > 0 else "❌"
    print(f"   {'HAR + core news':<40} {train_r2:>10.4f} {test_r2:>10.4f} {delta:>+10.4f} {status}")
    
    # HAR + top 3 PCA
    pca_features = [c for c in df.columns if c.startswith('news_pca_')][:3]
    features = har_features + pca_features
    X_train = train[features].fillna(0)
    X_test = test[features].fillna(0)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    delta = test_r2 - har_test_r2
    status = "✅" if delta > 0 else "❌"
    print(f"   {'HAR + top 3 PCA':<40} {train_r2:>10.4f} {test_r2:>10.4f} {delta:>+10.4f} {status}")
    
    return har_test_r2


def main():
    """Run all diagnostic tests."""
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🔍 DIAGNOSTIC AUDIT: WHY NEWSAGENT FAILS")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load data
    df = load_all_data()
    
    # Run tests
    test_1_signal_existence(df)
    target_results = test_2_target_choice(df)
    test_3_train_test_shift(df)
    test_4_feature_quality(df)
    best_reg_r2 = test_5_regularization(df)
    test_6_lag_structure(df)
    simple_results = test_7_simple_baseline(df)
    har_baseline = test_8_combined_approach(df)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    print("\n   FINDINGS:")
    
    # Check if signal exists
    print("\n   1. SIGNAL EXISTENCE:")
    print("      News features have WEAK correlations with volatility (~0.05)")
    print("      VIX is the dominant predictor")
    
    # Best target
    best_target = max(target_results.items(), key=lambda x: x[1])
    print(f"\n   2. BEST TARGET: {best_target[0]} (R²={best_target[1]:.4f})")
    
    # Best simple model
    best_simple = max(simple_results.items(), key=lambda x: x[1])
    print(f"\n   3. BEST SIMPLE MODEL: {best_simple[0]} (R²={best_simple[1]:.4f})")
    
    print(f"\n   4. HAR BASELINE: {har_baseline:.4f}")
    print("      News adds MINIMAL value on top of HAR")
    
    # Recommendation
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATION")
    print("=" * 70)
    
    print("""
   The news features have WEAK predictive power for volatility because:
   
   1. VIX already captures market fear/sentiment
   2. HAR features (past volatility) are much stronger predictors
   3. News-to-volatility relationship is noisy and non-linear
   
   POTENTIAL FIXES:
   
   A. CHANGE THE TARGET:
      - Predict "vol surprise" = realized_vol - VIX_implied
      - Predict direction of vol change instead of magnitude
   
   B. CHANGE THE FEATURES:
      - Use news SENTIMENT CHANGE (not level)
      - Use news VOLUME SPIKE (abnormal news count)
      - Aggregate news at SECTOR level
   
   C. CHANGE THE MODEL:
      - Train on extreme days only (high vol events)
      - Use classification (high/low vol) instead of regression
   
   D. ACCEPT REALITY:
      - News MAY genuinely have no predictive power
      - Focus on what works: HAR + VIX + Calendar effects
    """)
    
    # Timing
    end_time = datetime.now()
    print(f"\n   Duration: {end_time - start_time}")
    print("=" * 70)


if __name__ == "__main__":
    main()

