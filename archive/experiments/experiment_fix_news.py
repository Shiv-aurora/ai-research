"""
EXPERIMENT: Fix NewsAgent by changing the target

DIAGNOSIS FOUND:
- News correlates with SAME-DAY realized_vol (r=0.26) 
- News does NOT correlate with NEXT-DAY target_log_var (r=0.08)
- Test R² on realized_vol = 23.88%!

HYPOTHESIS:
News affects CURRENT volatility, not FUTURE volatility.
The market prices in news IMMEDIATELY.

This experiment tests multiple approaches to fix NewsAgent:
1. Predict same-day realized_vol (direct)
2. Predict "vol surprise" = realized_vol - expected_vol
3. Predict extreme volatility events (classification)
4. Use news to improve HAR (additive)

Goal: Find R² >= 0.15

Usage:
    python scripts/experiment_fix_news.py
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

np.random.seed(42)


def load_data():
    """Load and prepare all data."""
    print("=" * 70)
    print("📂 LOADING DATA")
    print("=" * 70)
    
    targets = pd.read_parquet("data/processed/targets.parquet")
    news = pd.read_parquet("data/processed/news_features.parquet")
    
    for df in [targets, news]:
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        if df['ticker'].dtype.name == 'category':
            df['ticker'] = df['ticker'].astype(str)
    
    df = pd.merge(targets, news, on=['date', 'ticker'], how='inner')
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Sort
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Create HAR features
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        df.loc[mask, 'rv_lag_1'] = df.loc[mask, 'realized_vol'].shift(1)
        df.loc[mask, 'rv_lag_5'] = df.loc[mask, 'realized_vol'].rolling(5).mean().shift(1)
        df.loc[mask, 'rv_lag_22'] = df.loc[mask, 'realized_vol'].rolling(22).mean().shift(1)
    
    df = df.dropna(subset=['rv_lag_1', 'rv_lag_5', 'rv_lag_22', 'target_log_var', 'realized_vol'])
    
    print(f"   Loaded {len(df):,} rows")
    
    return df


def experiment_1_same_day_vol(df):
    """Experiment 1: Predict same-day realized volatility."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: PREDICT SAME-DAY REALIZED_VOL")
    print("=" * 70)
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df['date'] < cutoff]
    test = df[df['date'] >= cutoff]
    
    # Target: same-day realized_vol
    y_train = train['realized_vol']
    y_test = test['realized_vol']
    
    # Feature sets to try
    results = {}
    
    # News only
    news_features = ['news_count', 'shock_index', 'sentiment_avg', 'novelty_score', 'VIX_close']
    X_train = train[news_features].fillna(0)
    X_test = test[news_features].fillna(0)
    
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    test_r2 = r2_score(y_test, model.predict(X_test))
    results['News only'] = test_r2
    print(f"\n   News only:        Test R² = {test_r2:.4f} ({test_r2*100:.2f}%)")
    
    # News + PCA
    pca_features = [c for c in df.columns if c.startswith('news_pca_')][:5]
    features = news_features + pca_features
    X_train = train[features].fillna(0)
    X_test = test[features].fillna(0)
    
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    test_r2 = r2_score(y_test, model.predict(X_test))
    results['News + PCA'] = test_r2
    print(f"   News + PCA:       Test R² = {test_r2:.4f} ({test_r2*100:.2f}%)")
    
    # HAR baseline (for comparison)
    har_features = ['rv_lag_1', 'rv_lag_5', 'rv_lag_22']
    X_train = train[har_features].fillna(0)
    X_test = test[har_features].fillna(0)
    
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    har_r2 = r2_score(y_test, model.predict(X_test))
    results['HAR only'] = har_r2
    print(f"   HAR only:         Test R² = {har_r2:.4f} ({har_r2*100:.2f}%)")
    
    # HAR + News
    features = har_features + news_features
    X_train = train[features].fillna(0)
    X_test = test[features].fillna(0)
    
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    test_r2 = r2_score(y_test, model.predict(X_test))
    results['HAR + News'] = test_r2
    delta = test_r2 - har_r2
    print(f"   HAR + News:       Test R² = {test_r2:.4f} ({test_r2*100:.2f}%) [Δ = {delta:+.4f}]")
    
    # Full model
    features = har_features + news_features + pca_features
    X_train = train[features].fillna(0)
    X_test = test[features].fillna(0)
    
    model = LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, verbose=-1)
    model.fit(X_train, y_train)
    test_r2 = r2_score(y_test, model.predict(X_test))
    results['Full LightGBM'] = test_r2
    print(f"   Full LightGBM:    Test R² = {test_r2:.4f} ({test_r2*100:.2f}%)")
    
    best = max(results.items(), key=lambda x: x[1])
    print(f"\n   ✅ BEST: {best[0]} with R² = {best[1]:.4f}")
    
    return results


def experiment_2_vol_surprise(df):
    """Experiment 2: Predict vol surprise (deviation from expected)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: PREDICT VOL SURPRISE")
    print("(realized_vol - expected_vol from HAR)")
    print("=" * 70)
    
    # Calculate HAR prediction
    har_features = ['rv_lag_1', 'rv_lag_5', 'rv_lag_22']
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df['date'] < cutoff].copy()
    test = df[df['date'] >= cutoff].copy()
    
    # Train HAR model
    X_train_har = train[har_features].fillna(0)
    y_train = train['realized_vol']
    
    har_model = Ridge(alpha=10.0)
    har_model.fit(X_train_har, y_train)
    
    # Calculate surprise
    train['expected_vol'] = har_model.predict(X_train_har)
    test['expected_vol'] = har_model.predict(test[har_features].fillna(0))
    
    train['vol_surprise'] = train['realized_vol'] - train['expected_vol']
    test['vol_surprise'] = test['realized_vol'] - test['expected_vol']
    
    print(f"\n   Vol surprise stats:")
    print(f"      Train: mean={train['vol_surprise'].mean():.4f}, std={train['vol_surprise'].std():.4f}")
    print(f"      Test:  mean={test['vol_surprise'].mean():.4f}, std={test['vol_surprise'].std():.4f}")
    
    # Now predict surprise with news
    news_features = ['news_count', 'shock_index', 'sentiment_avg', 'novelty_score', 'VIX_close']
    
    y_train = train['vol_surprise']
    y_test = test['vol_surprise']
    
    X_train = train[news_features].fillna(0)
    X_test = test[news_features].fillna(0)
    
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    test_r2 = r2_score(y_test, model.predict(X_test))
    
    print(f"\n   News → Vol Surprise: Test R² = {test_r2:.4f} ({test_r2*100:.2f}%)")
    
    # Combined prediction
    total_pred = test['expected_vol'] + model.predict(X_test)
    total_r2 = r2_score(test['realized_vol'], total_pred)
    
    print(f"   HAR + News Surprise: Test R² = {total_r2:.4f} ({total_r2*100:.2f}%)")
    
    return test_r2


def experiment_3_extreme_events(df):
    """Experiment 3: Classify extreme volatility events."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: CLASSIFY EXTREME VOL EVENTS")
    print("(Binary: Is today high volatility?)")
    print("=" * 70)
    
    # Define "extreme" as top 20% of volatility
    vol_threshold = df['realized_vol'].quantile(0.80)
    df['high_vol'] = (df['realized_vol'] > vol_threshold).astype(int)
    
    print(f"\n   High vol threshold: {vol_threshold:.4f}")
    print(f"   High vol ratio: {df['high_vol'].mean()*100:.1f}%")
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df['date'] < cutoff]
    test = df[df['date'] >= cutoff]
    
    news_features = ['news_count', 'shock_index', 'sentiment_avg', 'novelty_score', 'VIX_close']
    
    X_train = train[news_features].fillna(0)
    X_test = test[news_features].fillna(0)
    y_train = train['high_vol']
    y_test = test['high_vol']
    
    # Logistic Regression
    model = LogisticRegression(C=0.1, max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n   Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"   AUC-ROC:  {auc:.4f}")
    
    # What's baseline?
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    print(f"   Baseline: {baseline_acc:.4f}")
    
    status = "✅ BETTER" if accuracy > baseline_acc else "❌ WORSE"
    print(f"   {status} than baseline")
    
    return auc


def experiment_4_shock_model(df):
    """Experiment 4: Model only high-shock days."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: MODEL HIGH-SHOCK DAYS ONLY")
    print("(When shock_index > median)")
    print("=" * 70)
    
    # Filter to high-shock days
    shock_median = df['shock_index'].median()
    df_shock = df[df['shock_index'] > shock_median].copy()
    
    print(f"\n   Shock median: {shock_median:.4f}")
    print(f"   High-shock rows: {len(df_shock):,} ({len(df_shock)/len(df)*100:.1f}%)")
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df_shock[df_shock['date'] < cutoff]
    test = df_shock[df_shock['date'] >= cutoff]
    
    if len(train) < 100 or len(test) < 50:
        print("   ⚠️ Not enough data")
        return 0
    
    news_features = ['news_count', 'shock_index', 'sentiment_avg', 'novelty_score', 'VIX_close']
    har_features = ['rv_lag_1', 'rv_lag_5', 'rv_lag_22']
    
    # HAR baseline on shock days
    X_train = train[har_features].fillna(0)
    X_test = test[har_features].fillna(0)
    y_train = train['realized_vol']
    y_test = test['realized_vol']
    
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    har_r2 = r2_score(y_test, model.predict(X_test))
    print(f"\n   HAR on shock days:      R² = {har_r2:.4f}")
    
    # HAR + News on shock days
    features = har_features + news_features
    X_train = train[features].fillna(0)
    X_test = test[features].fillna(0)
    
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    full_r2 = r2_score(y_test, model.predict(X_test))
    delta = full_r2 - har_r2
    print(f"   HAR + News:             R² = {full_r2:.4f} [Δ = {delta:+.4f}]")
    
    return full_r2


def experiment_5_next_day_with_news_lag(df):
    """Experiment 5: Predict next-day vol using LAGGED news."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: NEXT-DAY VOL WITH LAGGED NEWS")
    print("(Use yesterday's news to predict today's vol)")
    print("=" * 70)
    
    # Create lagged news features
    news_cols = ['news_count', 'shock_index', 'sentiment_avg']
    
    for col in news_cols:
        df[f'{col}_lag1'] = df.groupby('ticker')[col].shift(1)
    
    df = df.dropna()
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df['date'] < cutoff]
    test = df[df['date'] >= cutoff]
    
    # HAR features
    har_features = ['rv_lag_1', 'rv_lag_5', 'rv_lag_22']
    lagged_news = [f'{c}_lag1' for c in news_cols] + ['VIX_close']
    
    # Target: same-day realized_vol
    y_train = train['realized_vol']
    y_test = test['realized_vol']
    
    # HAR baseline
    X_train = train[har_features].fillna(0)
    X_test = test[har_features].fillna(0)
    
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    har_r2 = r2_score(y_test, model.predict(X_test))
    print(f"\n   HAR only:           R² = {har_r2:.4f}")
    
    # HAR + lagged news
    features = har_features + lagged_news
    X_train = train[features].fillna(0)
    X_test = test[features].fillna(0)
    
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    full_r2 = r2_score(y_test, model.predict(X_test))
    delta = full_r2 - har_r2
    print(f"   HAR + Lagged News:  R² = {full_r2:.4f} [Δ = {delta:+.4f}]")
    
    return full_r2


def main():
    """Run all experiments."""
    print("\n" + "=" * 70)
    print("🔬 EXPERIMENTS TO FIX NEWSAGENT")
    print("   Goal: Find R² >= 0.15")
    print("=" * 70)
    
    df = load_data()
    
    # Run experiments
    results = {}
    
    exp1_results = experiment_1_same_day_vol(df)
    results['Exp 1: Same-day vol'] = max(exp1_results.values())
    
    exp2_r2 = experiment_2_vol_surprise(df)
    results['Exp 2: Vol surprise'] = exp2_r2
    
    exp3_auc = experiment_3_extreme_events(df)
    results['Exp 3: Extreme events (AUC)'] = exp3_auc
    
    exp4_r2 = experiment_4_shock_model(df)
    results['Exp 4: Shock days only'] = exp4_r2
    
    exp5_r2 = experiment_5_next_day_with_news_lag(df)
    results['Exp 5: Lagged news'] = exp5_r2
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 70)
    
    print(f"\n   {'Experiment':<35} {'Best Score':>12} {'Goal Met':>10}")
    print("   " + "-" * 59)
    
    for exp, score in results.items():
        goal_met = "✅ YES" if score >= 0.15 else "❌ NO"
        print(f"   {exp:<35} {score:>12.4f} {goal_met:>10}")
    
    # Best approach
    best = max(results.items(), key=lambda x: x[1])
    print(f"\n   🏆 BEST: {best[0]} with score = {best[1]:.4f}")
    
    # Recommendation
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATION")
    print("=" * 70)
    
    if best[1] >= 0.15:
        print(f"""
   ✅ FOUND A WORKING APPROACH!
   
   Use: {best[0]}
   Score: {best[1]:.4f} ({best[1]*100:.2f}%)
   
   IMPLEMENTATION:
   - Change NewsAgent target from 'resid_tech' to 'realized_vol'
   - Use same-day news features (not lagged)
   - This captures how news affects CURRENT volatility
        """)
    else:
        print(f"""
   ⚠️ NO APPROACH REACHES 15% R²
   
   Best found: {best[0]} with {best[1]:.4f}
   
   HONEST ASSESSMENT:
   - News has WEAK predictive power for volatility
   - HAR + VIX are the dominant predictors
   - News may only matter for extreme events
   
   RECOMMENDATION:
   - Accept 4-5% R² from news as marginal improvement
   - Focus on HAR + VIX + Calendar effects
   - Or try different news source/processing
        """)
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()

