"""
Verify News Agent Fix (Phase 6)

Comprehensive test suite to verify the decay kernel fix works:

Test 1: Noise Injection (Fragility) - Does noise kill the signal?
Test 2: Alpha Decay (Freshness) - Is fresh news better than stale?
Test 3: Feature Ablation (Dominance) - Does news_memory matter?
Test 4: Orthogonalization (Independence) - Is signal unique from VIX/Friday?
Test 5: Residual Autocorrelation (Root Cause) - No spike at lag 5?

Pass all 5 = Fix is verified
Pass <5 = Investigate further

Usage:
    python scripts/verify_news_fix.py
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

np.random.seed(42)


def load_and_prepare_data():
    """Load data with decay kernel features."""
    print("=" * 70)
    print("LOADING DATA WITH DECAY KERNEL FEATURES")
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
    
    # Merge news
    df = pd.merge(df, residuals[["date", "ticker", "resid_tech"]], on=["date", "ticker"], how="left")
    df = pd.merge(df, news_features, on=["date", "ticker"], how="left", suffixes=("", "_news"))
    
    # =============================================
    # DECAY KERNEL FEATURES (Phase 6 Fix)
    # =============================================
    print("\n   Applying Decay Kernel (0.50, 0.25, 0.15, 0.10)...")
    
    weights = [0.50, 0.25, 0.15, 0.10]  # Stops at lag 4
    
    for col in ["news_count", "shock_index", "sentiment_avg"]:
        if col in df.columns:
            memory_col = f"{col.replace('_count', '').replace('_index', '').replace('_avg', '')}_memory"
            df[memory_col] = 0.0
            for i, w in enumerate(weights, 1):
                df[memory_col] += w * df.groupby("ticker")[col].shift(i).fillna(0)
    
    # Create shock_vix_memory
    df["shock_vix_memory"] = df["shock_memory"] * df["VIX_close"]
    
    print(f"   Created: news_memory, shock_memory, sentiment_memory, shock_vix_memory")
    
    # Tech prediction
    tech_features = ["rv_lag_1", "rv_lag_5", "rv_lag_22", "returns_sq_lag_1", "VIX_close", "rsi_14"]
    cutoff = pd.to_datetime("2022-07-01")
    
    train_tech = df[df["date"] < cutoff].dropna(subset=tech_features + ["target_log_var"])
    if len(train_tech) > 50:
        tech_model = Ridge(alpha=1.0)
        tech_model.fit(train_tech[tech_features], train_tech["target_log_var"])
        df["tech_pred"] = tech_model.predict(df[tech_features].fillna(0))
    else:
        df["tech_pred"] = 0
    
    df = df.dropna(subset=["target_log_var", "news_memory"])
    
    print(f"\n   Loaded {len(df):,} rows")
    
    return df


def get_features():
    """Get feature list including decay kernel features."""
    return ["tech_pred", "news_memory", "shock_memory", "sentiment_memory",
            "shock_vix_memory", "VIX_close", "is_friday", "is_monday", "is_q4"]


def train_model(X_train, y_train):
    """Train ElasticNet model."""
    model = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000)
    model.fit(X_train, y_train)
    return model


# ============================================================
# TEST 1: NOISE INJECTION (FRAGILITY CHECK)
# ============================================================
def test_1_noise_injection(df):
    """
    Add noise to news_memory and check if R² drops monotonically.
    
    Pass: R² should drop significantly with increasing noise.
    If 100% noise only drops R² by ~10%, the model ignores news.
    """
    print("\n" + "=" * 70)
    print("TEST 1: NOISE INJECTION (Fragility Check)")
    print("=" * 70)
    
    features = get_features()
    features = [f for f in features if f in df.columns]
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    
    X_train = train[features].fillna(0)
    y_train = train["target_log_var"]
    X_test_orig = test[features].fillna(0).copy()
    y_test = test["target_log_var"]
    
    # Train model
    model = train_model(X_train, y_train)
    baseline_r2 = r2_score(y_test, model.predict(X_test_orig))
    
    print(f"\n   Baseline R²: {baseline_r2:.4f}")
    
    noise_levels = [0.05, 0.10, 0.25, 0.50, 1.00]
    results = {}
    
    print(f"\n   {'Noise':<10} {'R²':>10} {'Drop':>12} {'% Drop':>10}")
    print("   " + "-" * 45)
    
    for noise_level in noise_levels:
        X_test_noisy = X_test_orig.copy()
        noise = np.random.normal(0, noise_level, size=len(X_test_noisy))
        X_test_noisy["news_memory"] = X_test_noisy["news_memory"] * (1 + noise)
        
        r2_noisy = r2_score(y_test, model.predict(X_test_noisy))
        drop = baseline_r2 - r2_noisy
        pct_drop = (drop / baseline_r2) * 100 if baseline_r2 > 0 else 0
        
        results[noise_level] = {"r2": r2_noisy, "drop": drop, "pct_drop": pct_drop}
        print(f"   {noise_level*100:>5.0f}%     {r2_noisy:>10.4f} {drop:>+12.4f} {pct_drop:>+9.1f}%")
    
    # Check monotonic decrease
    r2_values = [results[n]["r2"] for n in noise_levels]
    monotonic = all(r2_values[i] >= r2_values[i+1] for i in range(len(r2_values)-1))
    
    # Check significant drop at 100% noise
    drop_100 = results[1.00]["pct_drop"]
    significant_drop = drop_100 > 15
    
    passed = monotonic and significant_drop
    
    print(f"\n   Monotonic decrease: {'Yes' if monotonic else 'No'}")
    print(f"   100% noise drop:    {drop_100:.1f}% {'(> 15% = Good)' if significant_drop else '(< 15% = Model ignores news)'}")
    print(f"\n   VERDICT: {'PASS' if passed else 'FAIL'}")
    
    return {"name": "Noise Injection", "results": results, "passed": passed}


# ============================================================
# TEST 2: ALPHA DECAY (FRESHNESS CHECK)
# ============================================================
def test_2_alpha_decay(df):
    """
    Shift news_memory by different lags to test freshness.
    
    Pass: Shift 0 should perform best, negative shifts should be worse.
    """
    print("\n" + "=" * 70)
    print("TEST 2: ALPHA DECAY (Freshness Check)")
    print("=" * 70)
    
    features_base = ["tech_pred", "shock_memory", "VIX_close", "is_friday", "is_monday", "is_q4"]
    features_base = [f for f in features_base if f in df.columns]
    
    cutoff = pd.to_datetime("2023-01-01")
    
    shifts = [
        ("+2 (Future)", 2),
        ("+1 (Tomorrow)", 1),
        ("0 (Current)", 0),
        ("-1 (Yesterday)", -1),
        ("-2 (Old)", -2),
        ("-3 (Older)", -3),
        ("-5 (Weekly)", -5),
    ]
    
    print(f"\n   {'Shift':<20} {'R²':>10} {'vs Current':>12}")
    print("   " + "-" * 45)
    
    results = {}
    baseline_r2 = None
    
    for name, shift in shifts:
        df_shifted = df.copy()
        
        if shift != 0:
            df_shifted["news_memory_shifted"] = df_shifted.groupby("ticker")["news_memory"].shift(-shift)
        else:
            df_shifted["news_memory_shifted"] = df_shifted["news_memory"]
        
        df_shifted = df_shifted.dropna(subset=["news_memory_shifted", "target_log_var"])
        
        train = df_shifted[df_shifted["date"] < cutoff]
        test = df_shifted[df_shifted["date"] >= cutoff]
        
        if len(train) < 50 or len(test) < 20:
            continue
        
        features = features_base + ["news_memory_shifted"]
        
        X_train = train[features].fillna(0)
        y_train = train["target_log_var"]
        X_test = test[features].fillna(0)
        y_test = test["target_log_var"]
        
        model = train_model(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        
        if shift == 0:
            baseline_r2 = r2
        
        vs_baseline = r2 - baseline_r2 if baseline_r2 is not None else 0
        
        results[shift] = {"name": name, "r2": r2, "vs_baseline": vs_baseline}
        print(f"   {name:<20} {r2:>10.4f} {vs_baseline:>+12.4f}")
    
    # Pass criteria: Current > Old (shift -2)
    r2_current = results.get(0, {}).get("r2", 0)
    r2_old = results.get(-2, {}).get("r2", 0)
    r2_weekly = results.get(-5, {}).get("r2", 0)
    
    freshness_valid = r2_current >= r2_old
    no_weekly_echo = r2_weekly < r2_current
    
    passed = freshness_valid and no_weekly_echo
    
    print(f"\n   Current >= Old (-2): {r2_current:.4f} >= {r2_old:.4f} = {freshness_valid}")
    print(f"   No weekly echo (-5): {r2_weekly:.4f} < {r2_current:.4f} = {no_weekly_echo}")
    print(f"\n   VERDICT: {'PASS' if passed else 'FAIL'}")
    
    return {"name": "Alpha Decay", "results": results, "passed": passed}


# ============================================================
# TEST 3: FEATURE ABLATION (DOMINANCE CHECK)
# ============================================================
def test_3_feature_ablation(df):
    """
    Test contribution of news_memory vs other features.
    
    Pass: Removing news_memory should cause measurable R² drop.
    """
    print("\n" + "=" * 70)
    print("TEST 3: FEATURE ABLATION (Dominance Check)")
    print("=" * 70)
    
    features = get_features()
    features = [f for f in features if f in df.columns]
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    
    X_train = train[features].fillna(0)
    y_train = train["target_log_var"]
    X_test = test[features].fillna(0)
    y_test = test["target_log_var"]
    
    # Baseline
    model = train_model(X_train, y_train)
    baseline_r2 = r2_score(y_test, model.predict(X_test))
    
    print(f"\n   Base R²: {baseline_r2:.4f}")
    print(f"\n   {'Feature':<20} {'R² Without':>12} {'Drop':>10} {'% Drop':>10}")
    print("   " + "-" * 55)
    
    results = {}
    ablate_features = ["news_memory", "shock_memory", "is_friday", "VIX_close"]
    
    for feat in ablate_features:
        if feat not in features:
            continue
        
        X_test_ablated = X_test.copy()
        X_test_ablated[feat] = 0
        
        r2_ablated = r2_score(y_test, model.predict(X_test_ablated))
        drop = baseline_r2 - r2_ablated
        pct_drop = (drop / baseline_r2) * 100 if baseline_r2 > 0 else 0
        
        results[feat] = {"r2": r2_ablated, "drop": drop, "pct_drop": pct_drop}
        print(f"   {feat:<20} {r2_ablated:>12.4f} {drop:>+10.4f} {pct_drop:>+9.1f}%")
    
    # Pass: news_memory should have measurable contribution
    news_drop = results.get("news_memory", {}).get("pct_drop", 0)
    passed = news_drop > 2  # At least 2% contribution
    
    print(f"\n   news_memory contribution: {news_drop:.1f}%")
    print(f"\n   VERDICT: {'PASS' if passed else 'FAIL'}")
    
    return {"name": "Feature Ablation", "results": results, "passed": passed}


# ============================================================
# TEST 4: ORTHOGONALIZATION (INDEPENDENCE CHECK)
# ============================================================
def test_4_orthogonalization(df):
    """
    Test if news_memory is independent from VIX/Friday.
    
    Pass: Performance should remain >80% when using orthogonalized signal.
    """
    print("\n" + "=" * 70)
    print("TEST 4: ORTHOGONALIZATION (Independence Check)")
    print("=" * 70)
    
    features = get_features()
    features = [f for f in features if f in df.columns]
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    
    # Step 1: Regress news_memory against VIX/Friday
    shadow_features = ["VIX_close", "is_friday"]
    shadow_features = [f for f in shadow_features if f in df.columns]
    
    shadow_model = LinearRegression()
    shadow_model.fit(train[shadow_features].fillna(0), train["news_memory"])
    
    shadow_r2 = r2_score(train["news_memory"], shadow_model.predict(train[shadow_features].fillna(0)))
    print(f"\n   Shadow model R² (how much VIX/Friday explains news_memory): {shadow_r2:.4f}")
    
    # Step 2: Calculate orthogonal residual
    df["news_ortho"] = df["news_memory"] - shadow_model.predict(df[shadow_features].fillna(0))
    
    unique_variance = 1 - shadow_r2
    print(f"   Unique variance in news_memory: {unique_variance*100:.1f}%")
    
    # Step 3: Train with original
    X_train = train[features].fillna(0)
    y_train = train["target_log_var"]
    X_test = test[features].fillna(0)
    y_test = test["target_log_var"]
    
    model_orig = train_model(X_train, y_train)
    r2_orig = r2_score(y_test, model_orig.predict(X_test))
    
    # Step 4: Train with orthogonalized
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    
    features_ortho = [f if f != "news_memory" else "news_ortho" for f in features]
    features_ortho = [f for f in features_ortho if f in df.columns]
    
    X_train_ortho = train[features_ortho].fillna(0)
    X_test_ortho = test[features_ortho].fillna(0)
    
    model_ortho = train_model(X_train_ortho, y_train)
    r2_ortho = r2_score(y_test, model_ortho.predict(X_test_ortho))
    
    retention = (r2_ortho / r2_orig) * 100 if r2_orig > 0 else 0
    
    print(f"\n   Original R²:      {r2_orig:.4f}")
    print(f"   Orthogonalized R²: {r2_ortho:.4f}")
    print(f"   Retention:        {retention:.1f}%")
    
    passed = retention >= 80
    
    print(f"\n   VERDICT: {'PASS' if passed else 'FAIL'} (need >= 80% retention)")
    
    return {"name": "Orthogonalization", "unique_variance": unique_variance, 
            "retention": retention, "passed": passed}


# ============================================================
# TEST 5: RESIDUAL AUTOCORRELATION (ROOT CAUSE CHECK)
# ============================================================
def test_5_residual_autocorrelation(df):
    """
    Check if residuals have autocorrelation spike at lag 5.
    
    Pass: No significant spike at lag 5 (the weekly echo should be gone).
    """
    print("\n" + "=" * 70)
    print("TEST 5: RESIDUAL AUTOCORRELATION (Root Cause Check)")
    print("=" * 70)
    
    features = get_features()
    features = [f for f in features if f in df.columns]
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    
    X_train = train[features].fillna(0)
    y_train = train["target_log_var"]
    X_test = test[features].fillna(0)
    y_test = test["target_log_var"]
    
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate residuals
    residuals = y_test.values - y_pred
    
    # Calculate ACF
    try:
        from statsmodels.tsa.stattools import acf
        acf_values = acf(residuals, nlags=10, fft=True)
    except:
        # Fallback manual ACF
        n = len(residuals)
        mean = np.mean(residuals)
        var = np.var(residuals)
        acf_values = []
        for lag in range(11):
            if lag == 0:
                acf_values.append(1.0)
            else:
                cov = np.sum((residuals[lag:] - mean) * (residuals[:-lag] - mean)) / n
                acf_values.append(cov / var)
        acf_values = np.array(acf_values)
    
    print(f"\n   Autocorrelation of Model Residuals:")
    print(f"\n   {'Lag':<6} {'ACF':>10} {'Status':>15}")
    print("   " + "-" * 35)
    
    for lag in range(1, 11):
        acf_val = acf_values[lag]
        # Significance threshold (approximate)
        threshold = 1.96 / np.sqrt(len(residuals))
        significant = abs(acf_val) > threshold
        
        status = "⚠️ SIGNIFICANT" if significant else "✓ OK"
        if lag == 5:
            status += " <- WEEKLY LAG"
        
        print(f"   {lag:<6} {acf_val:>+10.4f} {status:>15}")
    
    # Check lag 5 specifically
    acf_lag5 = acf_values[5]
    threshold = 1.96 / np.sqrt(len(residuals))
    lag5_significant = abs(acf_lag5) > threshold
    
    passed = not lag5_significant
    
    print(f"\n   Lag 5 ACF: {acf_lag5:.4f}")
    print(f"   Significance threshold: ±{threshold:.4f}")
    print(f"   Lag 5 significant: {'Yes (BAD - weekly echo still present)' if lag5_significant else 'No (GOOD - weekly echo eliminated)'}")
    print(f"\n   VERDICT: {'PASS' if passed else 'FAIL'}")
    
    return {"name": "Residual ACF", "acf_lag5": acf_lag5, "threshold": threshold, "passed": passed}


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("VERIFY NEWS AGENT FIX (Phase 6)")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Testing decay kernel implementation")
    
    # Load data
    df = load_and_prepare_data()
    
    # Run all tests
    results = {}
    
    results["1"] = test_1_noise_injection(df)
    results["2"] = test_2_alpha_decay(df)
    results["3"] = test_3_feature_ablation(df)
    results["4"] = test_4_orthogonalization(df)
    results["5"] = test_5_residual_autocorrelation(df)
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION REPORT")
    print("=" * 70)
    
    n_passed = sum(1 for r in results.values() if r["passed"])
    n_total = len(results)
    
    print(f"\n   {'Test':<35} {'Status':>10}")
    print("   " + "-" * 47)
    
    for key, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"   {result['name']:<35} {status:>10}")
    
    print("   " + "-" * 47)
    print(f"   {'TOTAL':<35} {n_passed}/{n_total}")
    
    # Verdict
    if n_passed == 5:
        verdict = "FIX VERIFIED"
        emoji = "🏆"
    elif n_passed >= 4:
        verdict = "MOSTLY VERIFIED"
        emoji = "✅"
    elif n_passed >= 3:
        verdict = "PARTIAL"
        emoji = "⚠️"
    else:
        verdict = "FIX FAILED"
        emoji = "❌"
    
    print(f"\n   {emoji} VERDICT: {verdict}")
    
    print("\n" + "=" * 70)
    
    # Save report
    report_path = Path("results/news_fix_verification.md")
    with open(report_path, "w") as f:
        f.write(f"""# News Agent Fix Verification Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Summary

| Metric | Value |
|--------|-------|
| Tests Passed | {n_passed} / {n_total} |
| Verdict | {emoji} **{verdict}** |

---

## Test Results

| Test | Status |
|------|--------|
| 1. Noise Injection | {'✅ PASS' if results['1']['passed'] else '❌ FAIL'} |
| 2. Alpha Decay | {'✅ PASS' if results['2']['passed'] else '❌ FAIL'} |
| 3. Feature Ablation | {'✅ PASS' if results['3']['passed'] else '❌ FAIL'} |
| 4. Orthogonalization | {'✅ PASS' if results['4']['passed'] else '❌ FAIL'} |
| 5. Residual ACF | {'✅ PASS' if results['5']['passed'] else '❌ FAIL'} |

---

## Key Metrics

- Unique variance in news_memory: {results['4'].get('unique_variance', 0)*100:.1f}%
- Orthogonalization retention: {results['4'].get('retention', 0):.1f}%
- Lag 5 ACF: {results['5'].get('acf_lag5', 0):.4f}

---

*Generated by News Fix Verification*
""")
    
    print(f"   Report saved to: {report_path}")
    print("=" * 70)
    
    return results, verdict


if __name__ == "__main__":
    main()


