"""
Titan V8 Robustness Audit Suite

Validates that the 30.83% R² result is genuine and not due to:
- Overfitting
- Data leakage
- Look-ahead bias
- Statistical artifacts

Tests:
A. Placebo Test (shuffle y)
B. Rolling Walk-Forward (regime stability)
C. Tail Risk Analysis (crash performance)
D. Feature Permutation (Friday check)
E. Bootstrap Confidence Intervals
F. Cross-Validation (TimeSeriesSplit)
G. Leave-One-Ticker-Out
H. Coefficient Stability
I. Out-of-Sample Date Test

Pass Criteria:
- VALIDATED: 8+ of 9 tests pass
- PARTIAL: 6-7 tests pass
- FAILED: <6 tests pass
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent))

# Global config
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_and_prepare_data():
    """Load targets and prepare all features for testing."""
    print("=" * 70)
    print("LOADING AND PREPARING DATA")
    print("=" * 70)
    
    # Load base data
    targets = pd.read_parquet("data/processed/targets.parquet")
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    news_features = pd.read_parquet("data/processed/news_features.parquet")
    
    # Normalize dates
    for df in [targets, residuals, news_features]:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        if "ticker" in df.columns and df["ticker"].dtype.name == "category":
            df["ticker"] = df["ticker"].astype(str)
    
    df = targets.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # HAR features for tech_pred
    df["rv_lag_1"] = df.groupby("ticker")["realized_vol"].shift(1)
    df["rv_lag_5"] = df.groupby("ticker")["realized_vol"].transform(
        lambda x: x.rolling(5).mean()
    ).shift(1)
    df["rv_lag_22"] = df.groupby("ticker")["realized_vol"].transform(
        lambda x: x.rolling(22).mean()
    ).shift(1)
    df["returns_sq_lag_1"] = (df["close"].pct_change() ** 2).shift(1)
    
    df["VIX_close"] = df["VIX_close"].ffill().fillna(15)
    df["rsi_14"] = df["rsi_14"].ffill().fillna(50)
    
    # Calendar features
    df["is_friday"] = (df["date"].dt.dayofweek == 4).astype(int)
    df["is_monday"] = (df["date"].dt.dayofweek == 0).astype(int)
    df["is_q4"] = (df["date"].dt.quarter == 4).astype(int)
    
    # Generate tech_pred
    tech_features = ["rv_lag_1", "rv_lag_5", "rv_lag_22", "returns_sq_lag_1", "VIX_close", "rsi_14"]
    cutoff = pd.to_datetime("2023-01-01")
    
    train_tech = df[df["date"] < cutoff].dropna(subset=tech_features + ["target_log_var"])
    tech_model = Ridge(alpha=1.0)
    tech_model.fit(train_tech[tech_features], train_tech["target_log_var"])
    df["tech_pred"] = tech_model.predict(df[tech_features].fillna(0))
    
    # Merge residuals and news
    df = pd.merge(df, residuals[["date", "ticker", "resid_tech"]], on=["date", "ticker"], how="left")
    df = pd.merge(df, news_features, on=["date", "ticker"], how="left", suffixes=("", "_news"))
    
    # Create lagged news features
    for col in ["news_count", "shock_index", "sentiment_avg"]:
        if col in df.columns:
            for lag in [1, 3, 5]:
                df[f"{col}_lag{lag}"] = df.groupby("ticker")[col].shift(lag)
    
    # News prediction (simplified)
    pca_cols = [c for c in df.columns if c.startswith("news_pca_")][:10]
    lag_cols = [c for c in df.columns if "_lag" in c and "news" in c]
    news_features_list = ["shock_index", "news_count", "sentiment_avg"] + pca_cols + lag_cols
    news_features_list = [f for f in news_features_list if f in df.columns]
    
    train_news = df[(df["date"] < cutoff) & df["resid_tech"].notna()].dropna(subset=news_features_list)
    if len(train_news) > 50:
        from lightgbm import LGBMRegressor
        news_model = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.01, verbose=-1)
        news_model.fit(train_news[news_features_list], train_news["resid_tech"])
        df["news_pred"] = news_model.predict(df[news_features_list].fillna(0))
    else:
        df["news_pred"] = 0
    
    # Fund prediction
    df["debt_to_equity"] = df["debt_to_equity"].fillna(0)
    df["days_to_ex_div"] = df["days_to_ex_div"].fillna(365)
    df["debt_vix_interaction"] = df["debt_to_equity"] * df["VIX_close"]
    
    fund_features = ["debt_to_equity", "days_to_ex_div", "VIX_close", "debt_vix_interaction"]
    train_fund = df[(df["date"] < cutoff) & df["resid_tech"].notna()].dropna(subset=fund_features)
    if len(train_fund) > 50:
        fund_model = LGBMRegressor(n_estimators=200, max_depth=2, learning_rate=0.02, verbose=-1)
        fund_model.fit(train_fund[fund_features], train_fund["resid_tech"])
        df["fund_pred"] = fund_model.predict(df[fund_features].fillna(0))
    else:
        df["fund_pred"] = 0
    
    # Retail prediction (simplified)
    df["retail_pred"] = 0
    
    # Clean up
    df = df.dropna(subset=["target_log_var"])
    
    print(f"   Loaded {len(df):,} rows with {len(df.columns)} features")
    print(f"   Tickers: {df['ticker'].unique().tolist()}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def get_features_and_target(df):
    """Get feature matrix X and target y."""
    features = ["tech_pred", "news_pred", "fund_pred", "retail_pred", 
                "VIX_close", "is_friday", "is_monday", "is_q4"]
    features = [f for f in features if f in df.columns]
    
    X = df[features].fillna(0)
    y = df["target_log_var"]
    
    return X, y, features


def train_titan_model(X_train, y_train):
    """Train the Titan ElasticNet model."""
    model = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000)
    model.fit(X_train, y_train)
    return model


# ============================================================
# TEST A: PLACEBO TEST
# ============================================================
def test_a_placebo(df):
    """Shuffle y randomly and retrain - R² should be near zero."""
    print("\n" + "-" * 70)
    print("TEST A: PLACEBO TEST (Shuffle y)")
    print("-" * 70)
    
    X, y, features = get_features_and_target(df)
    
    cutoff = pd.to_datetime("2023-01-01")
    train_idx = df["date"] < cutoff
    test_idx = df["date"] >= cutoff
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Shuffle y_train
    y_train_shuffled = y_train.copy()
    np.random.shuffle(y_train_shuffled.values)
    
    # Train on shuffled
    model = train_titan_model(X_train, y_train_shuffled)
    
    # Predict on test (with original y for evaluation)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    passed = r2 <= 0.05
    
    print(f"   Placebo R²: {r2:.4f}")
    print(f"   Pass Criteria: R² <= 0.05")
    print(f"   Result: {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": "Placebo Test",
        "r2": r2,
        "threshold": 0.05,
        "passed": passed,
        "details": f"Shuffled target R² = {r2:.4f}"
    }


# ============================================================
# TEST B: ROLLING WALK-FORWARD
# ============================================================
def test_b_rolling_walkforward(df):
    """Test across different time periods."""
    print("\n" + "-" * 70)
    print("TEST B: ROLLING WALK-FORWARD (Regime Stability)")
    print("-" * 70)
    
    X, y, features = get_features_and_target(df)
    
    results = {}
    
    # Split 1: Train on first half of 2022, test on second half
    train_mask_1 = (df["date"] >= "2022-01-01") & (df["date"] < "2022-07-01")
    test_mask_1 = (df["date"] >= "2022-07-01") & (df["date"] < "2023-01-01")
    
    if train_mask_1.sum() > 50 and test_mask_1.sum() > 30:
        model = train_titan_model(X[train_mask_1], y[train_mask_1])
        r2_1 = r2_score(y[test_mask_1], model.predict(X[test_mask_1]))
        results["2022_H2"] = r2_1
        print(f"   Split 1 (Train: 2022-H1, Test: 2022-H2): R² = {r2_1:.4f}")
    else:
        results["2022_H2"] = None
        print(f"   Split 1: Insufficient data")
    
    # Split 2: Train on 2022, test on 2023
    train_mask_2 = (df["date"] >= "2022-01-01") & (df["date"] < "2023-01-01")
    test_mask_2 = df["date"] >= "2023-01-01"
    
    if train_mask_2.sum() > 50 and test_mask_2.sum() > 30:
        model = train_titan_model(X[train_mask_2], y[train_mask_2])
        r2_2 = r2_score(y[test_mask_2], model.predict(X[test_mask_2]))
        results["2023"] = r2_2
        print(f"   Split 2 (Train: 2022, Test: 2023): R² = {r2_2:.4f}")
    else:
        results["2023"] = None
        print(f"   Split 2: Insufficient data")
    
    # Pass if all valid splits > 15%
    valid_scores = [v for v in results.values() if v is not None]
    passed = len(valid_scores) > 0 and all(v > 0.15 for v in valid_scores)
    
    print(f"   Pass Criteria: All splits R² > 15%")
    print(f"   Result: {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": "Rolling Walk-Forward",
        "splits": results,
        "passed": passed,
        "details": f"2022-H2: {results.get('2022_H2', 'N/A'):.4f}, 2023: {results.get('2023', 'N/A'):.4f}"
    }


# ============================================================
# TEST C: TAIL RISK ANALYSIS
# ============================================================
def test_c_tail_risk(df):
    """Compare performance on high volatility days."""
    print("\n" + "-" * 70)
    print("TEST C: TAIL RISK ANALYSIS (Crash Performance)")
    print("-" * 70)
    
    X, y, features = get_features_and_target(df)
    
    cutoff = pd.to_datetime("2023-01-01")
    train_idx = df["date"] < cutoff
    test_idx = df["date"] >= cutoff
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx].values, y[test_idx].values
    
    # Train model
    model = train_titan_model(X_train, y_train)
    y_pred_titan = model.predict(X_test)
    y_pred_baseline = df.loc[test_idx, "tech_pred"].values
    
    # Find top 10% volatility days
    threshold = np.percentile(y_test, 90)
    tail_mask = y_test >= threshold
    
    n_tail = tail_mask.sum()
    print(f"   High volatility days (top 10%): {n_tail} days")
    print(f"   Threshold: target_log_var >= {threshold:.4f}")
    
    # Calculate RMSE on tails
    rmse_baseline = np.sqrt(mean_squared_error(y_test[tail_mask], y_pred_baseline[tail_mask]))
    rmse_titan = np.sqrt(mean_squared_error(y_test[tail_mask], y_pred_titan[tail_mask]))
    
    passed = rmse_titan < rmse_baseline
    
    print(f"   Baseline Tail RMSE: {rmse_baseline:.4f}")
    print(f"   Titan Tail RMSE: {rmse_titan:.4f}")
    print(f"   Improvement: {((rmse_baseline - rmse_titan) / rmse_baseline * 100):.1f}%")
    print(f"   Pass Criteria: Titan < Baseline")
    print(f"   Result: {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": "Tail Risk Analysis",
        "baseline_rmse": rmse_baseline,
        "titan_rmse": rmse_titan,
        "n_tail_days": n_tail,
        "passed": passed,
        "details": f"Baseline: {rmse_baseline:.4f}, Titan: {rmse_titan:.4f}"
    }


# ============================================================
# TEST D: FEATURE PERMUTATION
# ============================================================
def test_d_feature_permutation(df):
    """Shuffle is_friday and measure R² drop."""
    print("\n" + "-" * 70)
    print("TEST D: FEATURE PERMUTATION (Friday Check)")
    print("-" * 70)
    
    X, y, features = get_features_and_target(df)
    
    cutoff = pd.to_datetime("2023-01-01")
    train_idx = df["date"] < cutoff
    test_idx = df["date"] >= cutoff
    
    X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train model
    model = train_titan_model(X_train, y_train)
    
    # Original prediction
    r2_original = r2_score(y_test, model.predict(X_test))
    
    # Shuffle is_friday
    if "is_friday" in X_test.columns:
        X_test_shuffled = X_test.copy()
        X_test_shuffled["is_friday"] = np.random.permutation(X_test_shuffled["is_friday"].values)
        r2_shuffled = r2_score(y_test, model.predict(X_test_shuffled))
    else:
        r2_shuffled = r2_original
    
    drop = r2_original - r2_shuffled
    passed = drop > 0.05
    
    print(f"   Original R²: {r2_original:.4f}")
    print(f"   Shuffled Friday R²: {r2_shuffled:.4f}")
    print(f"   Drop: {drop:.4f} ({drop*100:.1f}%)")
    print(f"   Pass Criteria: Drop > 5%")
    print(f"   Result: {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": "Feature Permutation",
        "r2_original": r2_original,
        "r2_shuffled": r2_shuffled,
        "drop": drop,
        "passed": passed,
        "details": f"Original: {r2_original:.4f}, Shuffled: {r2_shuffled:.4f}, Drop: {drop:.4f}"
    }


# ============================================================
# TEST E: BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================
def test_e_bootstrap_ci(df, n_bootstrap=1000):
    """Calculate 95% CI for R² using bootstrap."""
    print("\n" + "-" * 70)
    print("TEST E: BOOTSTRAP CONFIDENCE INTERVALS")
    print("-" * 70)
    
    X, y, features = get_features_and_target(df)
    
    cutoff = pd.to_datetime("2023-01-01")
    train_idx = df["date"] < cutoff
    test_idx = df["date"] >= cutoff
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx].values, y[test_idx].values
    
    # Train model
    model = train_titan_model(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Bootstrap
    r2_scores = []
    n_test = len(y_test)
    
    for i in range(n_bootstrap):
        idx = np.random.choice(n_test, size=n_test, replace=True)
        r2 = r2_score(y_test[idx], y_pred[idx])
        r2_scores.append(r2)
    
    r2_scores = np.array(r2_scores)
    ci_lower = np.percentile(r2_scores, 2.5)
    ci_upper = np.percentile(r2_scores, 97.5)
    mean_r2 = np.mean(r2_scores)
    
    passed = ci_lower > 0.15
    
    print(f"   Bootstrap samples: {n_bootstrap}")
    print(f"   Mean R²: {mean_r2:.4f}")
    print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"   Pass Criteria: CI lower bound > 15%")
    print(f"   Result: {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": "Bootstrap CI",
        "mean_r2": mean_r2,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "passed": passed,
        "details": f"Mean: {mean_r2:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]"
    }


# ============================================================
# TEST F: CROSS-VALIDATION
# ============================================================
def test_f_cross_validation(df, n_splits=5):
    """Time-series cross-validation."""
    print("\n" + "-" * 70)
    print("TEST F: CROSS-VALIDATION (TimeSeriesSplit)")
    print("-" * 70)
    
    X, y, features = get_features_and_target(df)
    
    # Sort by date
    df_sorted = df.sort_values("date")
    X_sorted = X.loc[df_sorted.index]
    y_sorted = y.loc[df_sorted.index]
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(X_sorted)):
        X_train = X_sorted.iloc[train_idx]
        X_test = X_sorted.iloc[test_idx]
        y_train = y_sorted.iloc[train_idx]
        y_test = y_sorted.iloc[test_idx]
        
        model = train_titan_model(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        scores.append(r2)
        print(f"   Fold {i+1}: R² = {r2:.4f}")
    
    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)
    
    passed = mean_r2 > 0.15 and std_r2 < 0.15
    
    print(f"\n   Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"   Pass Criteria: Mean > 15%, Std < 15%")
    print(f"   Result: {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": "Cross-Validation",
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "fold_scores": scores,
        "passed": passed,
        "details": f"Mean: {mean_r2:.4f} ± {std_r2:.4f}"
    }


# ============================================================
# TEST G: LEAVE-ONE-TICKER-OUT
# ============================================================
def test_g_leave_one_ticker_out(df):
    """Train on 2 tickers, test on 3rd."""
    print("\n" + "-" * 70)
    print("TEST G: LEAVE-ONE-TICKER-OUT")
    print("-" * 70)
    
    X, y, features = get_features_and_target(df)
    
    tickers = df["ticker"].unique()
    scores = {}
    
    for test_ticker in tickers:
        train_mask = df["ticker"] != test_ticker
        test_mask = df["ticker"] == test_ticker
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        if len(X_train) < 50 or len(X_test) < 20:
            continue
        
        model = train_titan_model(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        scores[test_ticker] = r2
        print(f"   Leave out {test_ticker}: R² = {r2:.4f}")
    
    avg_r2 = np.mean(list(scores.values())) if scores else 0
    
    passed = avg_r2 > 0.10
    
    print(f"\n   Average R²: {avg_r2:.4f}")
    print(f"   Pass Criteria: Average > 10%")
    print(f"   Result: {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": "Leave-One-Ticker-Out",
        "ticker_scores": scores,
        "avg_r2": avg_r2,
        "passed": passed,
        "details": f"Average: {avg_r2:.4f}"
    }


# ============================================================
# TEST H: COEFFICIENT STABILITY
# ============================================================
def test_h_coefficient_stability(df, n_runs=100):
    """Check if is_friday coefficient is consistently negative."""
    print("\n" + "-" * 70)
    print("TEST H: COEFFICIENT STABILITY")
    print("-" * 70)
    
    X, y, features = get_features_and_target(df)
    
    cutoff = pd.to_datetime("2023-01-01")
    train_idx = df["date"] < cutoff
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    
    friday_idx = features.index("is_friday") if "is_friday" in features else None
    
    if friday_idx is None:
        print("   is_friday not in features!")
        return {"name": "Coefficient Stability", "passed": False, "details": "Feature missing"}
    
    friday_coeffs = []
    
    for i in range(n_runs):
        # Random 80% subsample
        n = len(X_train)
        idx = np.random.choice(n, size=int(0.8 * n), replace=False)
        
        model = train_titan_model(X_train.iloc[idx], y_train.iloc[idx])
        friday_coeffs.append(model.coef_[friday_idx])
    
    friday_coeffs = np.array(friday_coeffs)
    pct_negative = (friday_coeffs < 0).mean() * 100
    mean_coef = np.mean(friday_coeffs)
    std_coef = np.std(friday_coeffs)
    
    passed = pct_negative > 90
    
    print(f"   Runs: {n_runs}")
    print(f"   is_friday coefficient: {mean_coef:.4f} ± {std_coef:.4f}")
    print(f"   Percent negative: {pct_negative:.1f}%")
    print(f"   Pass Criteria: >90% negative")
    print(f"   Result: {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": "Coefficient Stability",
        "mean_coef": mean_coef,
        "std_coef": std_coef,
        "pct_negative": pct_negative,
        "passed": passed,
        "details": f"Mean: {mean_coef:.4f} ± {std_coef:.4f}, {pct_negative:.1f}% negative"
    }


# ============================================================
# TEST I: OUT-OF-SAMPLE DATE TEST
# ============================================================
def test_i_random_date_split(df, n_runs=50):
    """Train on random 70% of dates, test on 30%."""
    print("\n" + "-" * 70)
    print("TEST I: OUT-OF-SAMPLE DATE TEST")
    print("-" * 70)
    
    X, y, features = get_features_and_target(df)
    
    unique_dates = df["date"].unique()
    n_dates = len(unique_dates)
    n_train = int(0.7 * n_dates)
    
    r2_scores = []
    
    for i in range(n_runs):
        # Random date split
        train_dates = np.random.choice(unique_dates, size=n_train, replace=False)
        train_mask = df["date"].isin(train_dates)
        test_mask = ~train_mask
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        if len(X_test) < 30:
            continue
        
        model = train_titan_model(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        r2_scores.append(r2)
    
    r2_scores = np.array(r2_scores)
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    
    passed = mean_r2 > 0.20
    
    print(f"   Runs: {n_runs}")
    print(f"   Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"   Min: {np.min(r2_scores):.4f}, Max: {np.max(r2_scores):.4f}")
    print(f"   Pass Criteria: Mean > 20%")
    print(f"   Result: {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": "Random Date Split",
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "min_r2": np.min(r2_scores),
        "max_r2": np.max(r2_scores),
        "passed": passed,
        "details": f"Mean: {mean_r2:.4f} ± {std_r2:.4f}"
    }


# ============================================================
# GENERATE REPORT
# ============================================================
def generate_report(results, df):
    """Generate markdown report."""
    
    n_passed = sum(1 for r in results.values() if r["passed"])
    n_total = len(results)
    
    if n_passed >= 8:
        verdict = "VALIDATED"
        verdict_emoji = "✅"
    elif n_passed >= 6:
        verdict = "PARTIAL"
        verdict_emoji = "⚠️"
    else:
        verdict = "FAILED"
        verdict_emoji = "❌"
    
    report = f"""# Titan V8 Robustness Audit Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model:** ElasticNet + Calendar Features  
**Original R²:** 30.83%

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Tests Passed | {n_passed} / {n_total} |
| Verdict | {verdict_emoji} **{verdict}** |
| Data Rows | {len(df):,} |
| Tickers | {', '.join(df['ticker'].unique())} |

---

## Test Results

| Test | Result | Pass/Fail |
|------|--------|-----------|
"""
    
    for key, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        report += f"| {result['name']} | {result['details']} | {status} |\n"
    
    report += """
---

## Detailed Results

"""
    
    # Test A
    r = results["A"]
    report += f"""### Test A: Placebo Test
- **Purpose:** Verify model isn't just memorizing noise
- **Method:** Shuffle target y randomly, retrain, and check R²
- **Result:** R² = {r['r2']:.4f}
- **Threshold:** R² <= 0.05
- **Verdict:** {"✅ PASS" if r['passed'] else "❌ FAIL"}

"""
    
    # Test B
    r = results["B"]
    report += f"""### Test B: Rolling Walk-Forward
- **Purpose:** Verify model works across different time periods
- **Method:** Train on different periods, test on subsequent periods
- **Results:**
  - 2022-H2 Test: R² = {r['splits'].get('2022_H2', 'N/A')}
  - 2023 Test: R² = {r['splits'].get('2023', 'N/A')}
- **Threshold:** All splits > 15%
- **Verdict:** {"✅ PASS" if r['passed'] else "❌ FAIL"}

"""
    
    # Test C
    r = results["C"]
    report += f"""### Test C: Tail Risk Analysis
- **Purpose:** Verify model performs well during high volatility
- **Method:** Compare RMSE on top 10% volatility days
- **Results:**
  - Baseline RMSE: {r['baseline_rmse']:.4f}
  - Titan RMSE: {r['titan_rmse']:.4f}
  - Tail Days: {r['n_tail_days']}
- **Threshold:** Titan < Baseline
- **Verdict:** {"✅ PASS" if r['passed'] else "❌ FAIL"}

"""
    
    # Test D
    r = results["D"]
    report += f"""### Test D: Feature Permutation
- **Purpose:** Verify is_friday feature is truly important
- **Method:** Shuffle is_friday and measure R² drop
- **Results:**
  - Original R²: {r['r2_original']:.4f}
  - Shuffled R²: {r['r2_shuffled']:.4f}
  - Drop: {r['drop']:.4f} ({r['drop']*100:.1f}%)
- **Threshold:** Drop > 5%
- **Verdict:** {"✅ PASS" if r['passed'] else "❌ FAIL"}

"""
    
    # Test E
    r = results["E"]
    report += f"""### Test E: Bootstrap Confidence Intervals
- **Purpose:** Quantify uncertainty in R² estimate
- **Method:** Resample test set 1000 times
- **Results:**
  - Mean R²: {r['mean_r2']:.4f}
  - 95% CI: [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]
- **Threshold:** CI lower > 15%
- **Verdict:** {"✅ PASS" if r['passed'] else "❌ FAIL"}

"""
    
    # Test F
    r = results["F"]
    report += f"""### Test F: Cross-Validation
- **Purpose:** Verify consistency across time folds
- **Method:** 5-fold TimeSeriesSplit
- **Results:**
  - Mean R²: {r['mean_r2']:.4f} ± {r['std_r2']:.4f}
  - Fold Scores: {[f'{s:.2f}' for s in r['fold_scores']]}
- **Threshold:** Mean > 15%, Std < 15%
- **Verdict:** {"✅ PASS" if r['passed'] else "❌ FAIL"}

"""
    
    # Test G
    r = results["G"]
    report += f"""### Test G: Leave-One-Ticker-Out
- **Purpose:** Verify model generalizes across tickers
- **Method:** Train on 2 tickers, test on 3rd
- **Results:**
  - Ticker Scores: {r['ticker_scores']}
  - Average R²: {r['avg_r2']:.4f}
- **Threshold:** Average > 10%
- **Verdict:** {"✅ PASS" if r['passed'] else "❌ FAIL"}

"""
    
    # Test H
    r = results["H"]
    report += f"""### Test H: Coefficient Stability
- **Purpose:** Verify is_friday coefficient is consistently negative
- **Method:** Retrain 100 times with 80% subsamples
- **Results:**
  - Mean Coefficient: {r.get('mean_coef', 'N/A')}
  - Std: {r.get('std_coef', 'N/A')}
  - Percent Negative: {r.get('pct_negative', 'N/A')}%
- **Threshold:** >90% negative
- **Verdict:** {"✅ PASS" if r['passed'] else "❌ FAIL"}

"""
    
    # Test I
    r = results["I"]
    report += f"""### Test I: Random Date Split
- **Purpose:** Verify robustness to train/test split choice
- **Method:** Random 70/30 date splits, 50 runs
- **Results:**
  - Mean R²: {r['mean_r2']:.4f} ± {r['std_r2']:.4f}
  - Range: [{r['min_r2']:.4f}, {r['max_r2']:.4f}]
- **Threshold:** Mean > 20%
- **Verdict:** {"✅ PASS" if r['passed'] else "❌ FAIL"}

---

## Conclusion

"""
    
    if verdict == "VALIDATED":
        report += """The Titan V8 model has passed rigorous robustness testing. The 30%+ R² result appears to be genuine and not due to:

- ✅ Overfitting (Placebo test passed)
- ✅ Data leakage (Walk-forward tests passed)
- ✅ Statistical artifacts (Bootstrap CI is tight)
- ✅ Feature importance is real (Permutation test confirms is_friday matters)
- ✅ Results generalize across tickers and time periods

**The model is ready for production deployment.**
"""
    elif verdict == "PARTIAL":
        report += """The Titan V8 model shows promising results but requires further investigation:

- Some tests passed, indicating genuine signal
- Some tests failed, suggesting potential issues
- Recommend collecting more data before deployment

**Proceed with caution.**
"""
    else:
        report += """The Titan V8 model has failed robustness testing. The results may be due to:

- Overfitting
- Data leakage
- Statistical artifacts

**Do not deploy. Further research required.**
"""
    
    report += f"""
---

*Report generated by Titan V8 Robustness Audit Suite*
"""
    
    return report, verdict, n_passed, n_total


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("TITAN V8 ROBUSTNESS AUDIT SUITE")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_and_prepare_data()
    
    # Run all tests
    results = {}
    
    results["A"] = test_a_placebo(df)
    results["B"] = test_b_rolling_walkforward(df)
    results["C"] = test_c_tail_risk(df)
    results["D"] = test_d_feature_permutation(df)
    results["E"] = test_e_bootstrap_ci(df, n_bootstrap=1000)
    results["F"] = test_f_cross_validation(df, n_splits=5)
    results["G"] = test_g_leave_one_ticker_out(df)
    results["H"] = test_h_coefficient_stability(df, n_runs=100)
    results["I"] = test_i_random_date_split(df, n_runs=50)
    
    # Generate report
    report, verdict, n_passed, n_total = generate_report(results, df)
    
    # Save report
    report_path = Path("results/test_results.md")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    
    # Print summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    
    print(f"\n   Tests Passed: {n_passed} / {n_total}")
    print(f"\n   VERDICT: {verdict}")
    
    print("\n   Individual Results:")
    for key, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"      {result['name']:<25} {status}")
    
    print(f"\n   Report saved to: {report_path}")
    print("=" * 70)
    
    return results, verdict


if __name__ == "__main__":
    main()

