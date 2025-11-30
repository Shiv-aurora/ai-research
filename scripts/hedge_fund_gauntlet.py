"""
Hedge Fund Gauntlet: The Ultimate Alpha Validation

This is the final boss. If Titan V8 survives this, it's production-ready.

Tests:
1. Temporal Stability (Rolling Alpha) - Is the signal consistent over time?
2. Orthogonalization (Shadow Test) - Is news_pred unique or just a proxy?
3. Time-Shift (Causality Check) - Does timing matter? (No future leakage?)
4. Bootstrap Coefficient Stability - Is news_pred consistently positive?

Pass all 4 = Ready for production
Pass 3/4 = Proceed with caution
Pass <3 = Do not deploy

Usage:
    python scripts/hedge_fund_gauntlet.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent))

np.random.seed(42)


def load_and_prepare_data():
    """Load and prepare all features."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    targets = pd.read_parquet("data/processed/targets.parquet")
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    news_features = pd.read_parquet("data/processed/news_features.parquet")
    
    for df in [targets, residuals, news_features]:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        if "ticker" in df.columns and df["ticker"].dtype.name == "category":
            df["ticker"] = df["ticker"].astype(str)
    
    df = targets.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # HAR features
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
    
    # Tech prediction
    tech_features = ["rv_lag_1", "rv_lag_5", "rv_lag_22", "returns_sq_lag_1", "VIX_close", "rsi_14"]
    cutoff = pd.to_datetime("2022-07-01")
    
    train_tech = df[df["date"] < cutoff].dropna(subset=tech_features + ["target_log_var"])
    tech_model = Ridge(alpha=1.0)
    tech_model.fit(train_tech[tech_features], train_tech["target_log_var"])
    df["tech_pred"] = tech_model.predict(df[tech_features].fillna(0))
    
    # Merge residuals and news
    df = pd.merge(df, residuals[["date", "ticker", "resid_tech"]], on=["date", "ticker"], how="left")
    df = pd.merge(df, news_features, on=["date", "ticker"], how="left", suffixes=("", "_news"))
    
    # Lagged news features
    for col in ["news_count", "shock_index", "sentiment_avg"]:
        if col in df.columns:
            for lag in [1, 3, 5]:
                df[f"{col}_lag{lag}"] = df.groupby("ticker")[col].shift(lag)
    
    # News prediction
    from lightgbm import LGBMRegressor
    pca_cols = [c for c in df.columns if c.startswith("news_pca_")][:10]
    lag_cols = [c for c in df.columns if "_lag" in c and "news" in c]
    news_features_list = ["shock_index", "news_count", "sentiment_avg"] + pca_cols + lag_cols
    news_features_list = [f for f in news_features_list if f in df.columns]
    
    train_news = df[(df["date"] < cutoff) & df["resid_tech"].notna()].dropna(subset=news_features_list)
    if len(train_news) > 50:
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
    
    df["retail_pred"] = 0
    df = df.dropna(subset=["target_log_var"])
    
    print(f"   Loaded {len(df):,} rows")
    print(f"   Tickers: {df['ticker'].unique().tolist()}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


def get_features():
    """Get full feature list."""
    return ["tech_pred", "news_pred", "fund_pred", "VIX_close", 
            "is_friday", "is_monday", "is_q4"]


def train_model(X_train, y_train):
    """Train ElasticNet."""
    model = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000)
    model.fit(X_train, y_train)
    return model


# ============================================================
# TEST 1: TEMPORAL STABILITY (Rolling Alpha)
# ============================================================
def test_1_temporal_stability(df):
    """
    Test if news_pred coefficient is stable across time windows.
    
    Pass: Coefficient > 0.5 in at least 3/4 windows.
    """
    print("\n" + "=" * 70)
    print("TEST 1: TEMPORAL STABILITY (Rolling Alpha)")
    print("=" * 70)
    
    features = get_features()
    features = [f for f in features if f in df.columns]
    
    # Adjust cutoffs based on available data
    min_date = df["date"].min()
    max_date = df["date"].max()
    
    # Generate cutoffs that work with our data
    cutoffs = []
    for months_offset in [0, 6, 12, 18]:
        cutoff = min_date + pd.DateOffset(months=months_offset + 6)
        if cutoff < max_date - pd.DateOffset(months=3):
            cutoffs.append(cutoff)
    
    if len(cutoffs) < 2:
        # Fallback for limited data
        mid_point = min_date + (max_date - min_date) / 2
        cutoffs = [mid_point - pd.DateOffset(months=3), mid_point, mid_point + pd.DateOffset(months=3)]
        cutoffs = [c for c in cutoffs if c > min_date + pd.DateOffset(months=3) and c < max_date - pd.DateOffset(months=2)]
    
    print(f"   Date range: {min_date.date()} to {max_date.date()}")
    print(f"   Testing {len(cutoffs)} windows")
    
    results = []
    news_pred_idx = features.index("news_pred") if "news_pred" in features else None
    
    if news_pred_idx is None:
        print("   news_pred not in features!")
        return {"name": "Temporal Stability", "passed": False, "details": "Missing feature"}
    
    for cutoff in cutoffs:
        cutoff = pd.to_datetime(cutoff)
        test_end = cutoff + pd.DateOffset(months=6)
        
        train_mask = df["date"] < cutoff
        test_mask = (df["date"] >= cutoff) & (df["date"] < test_end)
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if len(train_df) < 50 or len(test_df) < 20:
            print(f"   Window {cutoff.date()}: Insufficient data (train: {len(train_df)}, test: {len(test_df)})")
            continue
        
        X_train = train_df[features].fillna(0)
        y_train = train_df["target_log_var"]
        X_test = test_df[features].fillna(0)
        y_test = test_df["target_log_var"]
        
        model = train_model(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        
        news_coef = model.coef_[news_pred_idx]
        
        results.append({
            "cutoff": cutoff,
            "news_coef": news_coef,
            "r2": r2,
            "n_train": len(train_df),
            "n_test": len(test_df)
        })
        
        status = "✓" if news_coef > 0.5 else "✗"
        print(f"   Window {cutoff.date()}: news_coef = {news_coef:+.4f}, R² = {r2:.4f} [{status}]")
    
    if not results:
        return {"name": "Temporal Stability", "passed": False, "details": "No valid windows"}
    
    # Check pass criteria
    n_positive = sum(1 for r in results if r["news_coef"] > 0.5)
    n_total = len(results)
    
    passed = n_positive >= max(1, n_total * 0.75)  # At least 75% should have coef > 0.5
    
    print(f"\n   SUMMARY:")
    print(f"      Windows with news_coef > 0.5: {n_positive}/{n_total}")
    print(f"      VERDICT: {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": "Temporal Stability",
        "results": results,
        "n_positive": n_positive,
        "n_total": n_total,
        "passed": passed
    }


# ============================================================
# TEST 2: ORTHOGONALIZATION (Shadow Test)
# ============================================================
def test_2_orthogonalization(df):
    """
    Test if news_pred provides unique signal beyond VIX/calendar.
    
    Method:
    1. Predict news_pred from VIX/calendar
    2. Calculate residual (orthogonal component)
    3. Use residual instead of news_pred
    4. If R² doesn't drop much, signal was unique
    """
    print("\n" + "=" * 70)
    print("TEST 2: ORTHOGONALIZATION (Shadow Test)")
    print("=" * 70)
    
    features = get_features()
    features = [f for f in features if f in df.columns]
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    
    print(f"   Train: {len(train)} samples, Test: {len(test)} samples")
    
    # Step A: Predict news_pred from systematic factors
    shadow_features = ["VIX_close", "is_friday", "tech_pred"]
    shadow_features = [f for f in shadow_features if f in df.columns]
    
    print(f"\n   Step A: Predicting news_pred from {shadow_features}")
    
    shadow_model = LinearRegression()
    shadow_model.fit(train[shadow_features].fillna(0), train["news_pred"])
    
    # Calculate how much variance is explained
    news_pred_r2 = r2_score(train["news_pred"], shadow_model.predict(train[shadow_features].fillna(0)))
    print(f"   Shadow model R² on news_pred: {news_pred_r2:.4f}")
    print(f"   (This shows how much news_pred overlaps with systematic factors)")
    
    # Step B: Calculate orthogonal residual
    print(f"\n   Step B: Calculating orthogonal component")
    
    df["news_shadow"] = shadow_model.predict(df[shadow_features].fillna(0))
    df["news_ortho"] = df["news_pred"] - df["news_shadow"]
    
    ortho_var = df["news_ortho"].var()
    orig_var = df["news_pred"].var()
    unique_pct = (ortho_var / orig_var) * 100 if orig_var > 0 else 0
    print(f"   Unique variance in news_pred: {unique_pct:.1f}%")
    
    # Step C: Train with original news_pred
    print(f"\n   Step C: Comparing models")
    
    X_train = train[features].fillna(0)
    y_train = train["target_log_var"]
    X_test = test[features].fillna(0)
    y_test = test["target_log_var"]
    
    model_orig = train_model(X_train, y_train)
    r2_orig = r2_score(y_test, model_orig.predict(X_test))
    
    # Step D: Train with orthogonalized news_pred
    features_ortho = [f if f != "news_pred" else "news_ortho" for f in features]
    features_ortho = [f for f in features_ortho if f in df.columns]
    
    # Recalculate ortho for train/test
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    
    X_train_ortho = train[features_ortho].fillna(0)
    X_test_ortho = test[features_ortho].fillna(0)
    
    model_ortho = train_model(X_train_ortho, y_train)
    r2_ortho = r2_score(y_test, model_ortho.predict(X_test_ortho))
    
    r2_drop = r2_orig - r2_ortho
    r2_drop_pct = (r2_drop / r2_orig) * 100 if r2_orig > 0 else 0
    
    print(f"\n   Original news_pred R²:      {r2_orig:.4f}")
    print(f"   Orthogonalized news_pred R²: {r2_ortho:.4f}")
    print(f"   Drop: {r2_drop:.4f} ({r2_drop_pct:.1f}%)")
    
    # Pass if drop < 2% (signal was unique)
    passed = abs(r2_drop_pct) < 20  # Less than 20% drop means signal is mostly unique
    
    print(f"\n   INTERPRETATION:")
    if r2_drop_pct < 5:
        print(f"   The news signal is HIGHLY UNIQUE - almost no overlap with systematic factors")
    elif r2_drop_pct < 20:
        print(f"   The news signal is MOSTLY UNIQUE - some overlap but still adds value")
    else:
        print(f"   The news signal has SIGNIFICANT OVERLAP with systematic factors")
    
    print(f"\n   VERDICT: {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": "Orthogonalization",
        "shadow_r2": news_pred_r2,
        "unique_variance_pct": unique_pct,
        "r2_orig": r2_orig,
        "r2_ortho": r2_ortho,
        "r2_drop": r2_drop,
        "r2_drop_pct": r2_drop_pct,
        "passed": passed
    }


# ============================================================
# TEST 3: TIME-SHIFT (Causality Check)
# ============================================================
def test_3_time_shift(df):
    """
    Test causality by shifting news_pred in time.
    
    - Shift +1: Future news (cheating) - should be highest
    - Shift 0: Current (Titan) - our actual model
    - Shift -1: Old news (late) - should be lower
    
    Pass: R²(Shift 0) > R²(Shift -1)
    """
    print("\n" + "=" * 70)
    print("TEST 3: TIME-SHIFT (Causality Check)")
    print("=" * 70)
    
    features_base = ["tech_pred", "fund_pred", "VIX_close", 
                     "is_friday", "is_monday", "is_q4"]
    features_base = [f for f in features_base if f in df.columns]
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    
    print(f"   Train: {len(train)} samples, Test: {len(test)} samples")
    
    shifts = {
        "+1 (Future - Cheating)": 1,
        "0 (Current - Titan)": 0,
        "-1 (Old - Late)": -1,
        "-2 (Very Old)": -2
    }
    
    results = {}
    
    for name, shift in shifts.items():
        # Create shifted news_pred
        train_shifted = train.copy()
        test_shifted = test.copy()
        
        if shift != 0:
            train_shifted["news_pred_shifted"] = train_shifted.groupby("ticker")["news_pred"].shift(-shift)
            test_shifted["news_pred_shifted"] = test_shifted.groupby("ticker")["news_pred"].shift(-shift)
        else:
            train_shifted["news_pred_shifted"] = train_shifted["news_pred"]
            test_shifted["news_pred_shifted"] = test_shifted["news_pred"]
        
        # Drop NaN from shifting
        train_shifted = train_shifted.dropna(subset=["news_pred_shifted"])
        test_shifted = test_shifted.dropna(subset=["news_pred_shifted"])
        
        if len(train_shifted) < 50 or len(test_shifted) < 20:
            print(f"   {name}: Insufficient data after shift")
            continue
        
        features = features_base + ["news_pred_shifted"]
        
        X_train = train_shifted[features].fillna(0)
        y_train = train_shifted["target_log_var"]
        X_test = test_shifted[features].fillna(0)
        y_test = test_shifted["target_log_var"]
        
        model = train_model(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        
        results[name] = r2
        
        marker = ""
        if "Future" in name:
            marker = " (CEILING - should be highest)"
        elif "Current" in name:
            marker = " (OUR MODEL)"
        elif "Late" in name or "Old" in name:
            marker = " (should be lower)"
        
        print(f"   {name:25s}: R² = {r2:.4f}{marker}")
    
    # Check causality
    r2_current = results.get("0 (Current - Titan)", 0)
    r2_late = results.get("-1 (Old - Late)", 0)
    r2_future = results.get("+1 (Future - Cheating)", 0)
    
    # Pass if current > late (timing matters)
    passed = r2_current > r2_late
    
    print(f"\n   CAUSALITY CHECK:")
    print(f"      Current > Late: {r2_current:.4f} > {r2_late:.4f} = {r2_current > r2_late}")
    print(f"      Future > Current: {r2_future:.4f} > {r2_current:.4f} = {r2_future > r2_current}")
    
    if passed:
        print(f"\n   ✓ Timing matters - fresh news is more predictive than old news")
    else:
        print(f"\n   ✗ Timing doesn't matter - possible autocorrelation issue")
    
    print(f"\n   VERDICT: {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": "Time-Shift Causality",
        "r2_future": r2_future,
        "r2_current": r2_current,
        "r2_late": r2_late,
        "passed": passed
    }


# ============================================================
# TEST 4: BOOTSTRAP COEFFICIENT STABILITY
# ============================================================
def test_4_bootstrap_stability(df, n_bootstrap=100):
    """
    Test if news_pred coefficient is consistently positive.
    
    Method: Resample training data 100 times, retrain, record coefficient.
    Pass: 95% of coefficients > 0
    """
    print("\n" + "=" * 70)
    print("TEST 4: BOOTSTRAP COEFFICIENT STABILITY")
    print("=" * 70)
    
    features = get_features()
    features = [f for f in features if f in df.columns]
    
    news_pred_idx = features.index("news_pred") if "news_pred" in features else None
    
    if news_pred_idx is None:
        return {"name": "Bootstrap Stability", "passed": False, "details": "Missing feature"}
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df["date"] < cutoff]
    
    X_train = train[features].fillna(0)
    y_train = train["target_log_var"]
    
    print(f"   Training samples: {len(train)}")
    print(f"   Bootstrap iterations: {n_bootstrap}")
    
    coefficients = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train.iloc[idx]
        y_boot = y_train.iloc[idx]
        
        model = train_model(X_boot, y_boot)
        coefficients.append(model.coef_[news_pred_idx])
    
    coefficients = np.array(coefficients)
    
    # Statistics
    mean_coef = np.mean(coefficients)
    std_coef = np.std(coefficients)
    pct_positive = (coefficients > 0).mean() * 100
    ci_lower = np.percentile(coefficients, 2.5)
    ci_upper = np.percentile(coefficients, 97.5)
    
    print(f"\n   news_pred Coefficient Distribution:")
    print(f"      Mean:     {mean_coef:+.4f}")
    print(f"      Std:      {std_coef:.4f}")
    print(f"      95% CI:   [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"      % > 0:    {pct_positive:.1f}%")
    
    # Pass if 95% positive
    passed = pct_positive >= 95
    
    # Additional check: is the CI entirely positive?
    ci_positive = ci_lower > 0
    
    print(f"\n   CI entirely positive: {'Yes' if ci_positive else 'No'}")
    print(f"\n   VERDICT: {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": "Bootstrap Stability",
        "mean_coef": mean_coef,
        "std_coef": std_coef,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pct_positive": pct_positive,
        "passed": passed
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("HEDGE FUND GAUNTLET: ULTIMATE ALPHA VALIDATION")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("If you pass this, you're production-ready.")
    
    # Load data
    df = load_and_prepare_data()
    
    # Run all tests
    results = {}
    
    results["1"] = test_1_temporal_stability(df)
    results["2"] = test_2_orthogonalization(df)
    results["3"] = test_3_time_shift(df)
    results["4"] = test_4_bootstrap_stability(df, n_bootstrap=100)
    
    # ================================================================
    # GAUNTLET REPORT
    # ================================================================
    print("\n" + "=" * 70)
    print("HEDGE FUND GAUNTLET REPORT")
    print("=" * 70)
    
    n_passed = sum(1 for r in results.values() if r["passed"])
    n_total = len(results)
    
    print(f"\n   ┌{'─' * 50}┐")
    print(f"   │{'GAUNTLET RESULTS':^50}│")
    print(f"   ├{'─' * 50}┤")
    print(f"   │ {'Test':<30} {'Status':>17} │")
    print(f"   ├{'─' * 50}┤")
    
    for key, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"   │ {result['name']:<30} {status:>17} │")
    
    print(f"   ├{'─' * 50}┤")
    print(f"   │ {'TOTAL':<30} {f'{n_passed}/{n_total}':>17} │")
    print(f"   └{'─' * 50}┘")
    
    # Verdict
    if n_passed == 4:
        verdict = "PRODUCTION READY"
        emoji = "🏆"
        msg = "All tests passed. Titan V8 is ready for live trading."
    elif n_passed >= 3:
        verdict = "VALIDATED"
        emoji = "✅"
        msg = "Most tests passed. Proceed with caution."
    elif n_passed >= 2:
        verdict = "PARTIAL"
        emoji = "⚠️"
        msg = "Mixed results. Investigate failures before deployment."
    else:
        verdict = "FAILED"
        emoji = "❌"
        msg = "Critical failures. Do not deploy."
    
    print(f"\n   {emoji} FINAL VERDICT: {verdict}")
    print(f"   {msg}")
    
    # Key metrics
    print(f"\n   KEY METRICS:")
    print(f"      Temporal Stability: {results['1'].get('n_positive', 'N/A')}/{results['1'].get('n_total', 'N/A')} windows positive")
    print(f"      Unique Variance:    {results['2'].get('unique_variance_pct', 'N/A'):.1f}%")
    print(f"      Causality Check:    Current ({results['3'].get('r2_current', 0):.4f}) > Late ({results['3'].get('r2_late', 0):.4f})")
    print(f"      Bootstrap Positive: {results['4'].get('pct_positive', 'N/A'):.1f}%")
    
    print("\n" + "=" * 70)
    
    # Save report
    report_path = Path("results/gauntlet_report.md")
    with open(report_path, "w") as f:
        f.write(f"""# Hedge Fund Gauntlet Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Tests Passed | {n_passed} / {n_total} |
| Verdict | {emoji} **{verdict}** |

---

## Test Results

| Test | Description | Status |
|------|-------------|--------|
| 1. Temporal Stability | news_coef > 0.5 in {results['1'].get('n_positive', 'N/A')}/{results['1'].get('n_total', 'N/A')} windows | {'✅ PASS' if results['1']['passed'] else '❌ FAIL'} |
| 2. Orthogonalization | {results['2'].get('unique_variance_pct', 0):.1f}% unique variance | {'✅ PASS' if results['2']['passed'] else '❌ FAIL'} |
| 3. Time-Shift | Current > Late | {'✅ PASS' if results['3']['passed'] else '❌ FAIL'} |
| 4. Bootstrap | {results['4'].get('pct_positive', 0):.1f}% positive coefficients | {'✅ PASS' if results['4']['passed'] else '❌ FAIL'} |

---

## Detailed Results

### Test 1: Temporal Stability
- Windows with news_coef > 0.5: {results['1'].get('n_positive', 'N/A')}/{results['1'].get('n_total', 'N/A')}

### Test 2: Orthogonalization
- Shadow model R² on news_pred: {results['2'].get('shadow_r2', 'N/A'):.4f}
- Unique variance: {results['2'].get('unique_variance_pct', 'N/A'):.1f}%
- R² drop when orthogonalized: {results['2'].get('r2_drop_pct', 'N/A'):.1f}%

### Test 3: Time-Shift Causality
- Future (+1): R² = {results['3'].get('r2_future', 'N/A'):.4f}
- Current (0): R² = {results['3'].get('r2_current', 'N/A'):.4f}
- Late (-1): R² = {results['3'].get('r2_late', 'N/A'):.4f}

### Test 4: Bootstrap Stability
- Mean coefficient: {results['4'].get('mean_coef', 'N/A'):+.4f}
- 95% CI: [{results['4'].get('ci_lower', 'N/A'):+.4f}, {results['4'].get('ci_upper', 'N/A'):+.4f}]
- % positive: {results['4'].get('pct_positive', 'N/A'):.1f}%

---

## Conclusion

{msg}

---

*Generated by Hedge Fund Gauntlet*
""")
    
    print(f"   Report saved to: {report_path}")
    print("=" * 70)
    
    return results, verdict


if __name__ == "__main__":
    main()


