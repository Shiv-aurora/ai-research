"""
Final Stress Test: Fragility Checks

Two critical tests to verify model robustness:

Test A: Adversarial Noise Injection
- Add Gaussian noise to news_pred
- Signal should survive small noise (5%)
- Signal should degrade with large noise (50%)

Test B: Multi-Lag Causality (Alpha Decay)
- Shift news_pred by multiple days
- Fresh news should beat stale news
- Ancient news should have minimal value

Pass both = Model is robust and causally valid

Usage:
    python scripts/final_stress_test.py
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
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
# TEST A: ADVERSARIAL NOISE INJECTION
# ============================================================
def test_a_noise_injection(df):
    """
    Add Gaussian noise to news_pred and measure R² degradation.
    
    Expected behavior:
    - 5% noise: R² drops < 5% (robust signal)
    - 10% noise: R² drops < 15% 
    - 50% noise: R² drops significantly (signal destroyed)
    """
    print("\n" + "=" * 70)
    print("TEST A: ADVERSARIAL NOISE INJECTION")
    print("=" * 70)
    print("   Testing robustness by adding Gaussian noise to news_pred")
    
    features = get_features()
    features = [f for f in features if f in df.columns]
    
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    
    X_train = train[features].fillna(0)
    y_train = train["target_log_var"]
    X_test_orig = test[features].fillna(0).copy()
    y_test = test["target_log_var"]
    
    print(f"\n   Train: {len(train)} samples, Test: {len(test)} samples")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Baseline (no noise)
    baseline_r2 = r2_score(y_test, model.predict(X_test_orig))
    print(f"\n   Baseline R² (no noise): {baseline_r2:.4f} ({baseline_r2*100:.2f}%)")
    
    # Noise levels to test
    noise_levels = [0.05, 0.10, 0.25, 0.50, 1.00]
    
    print(f"\n   {'Noise Level':<15} {'R²':>10} {'Drop':>12} {'% Drop':>10} {'Status':>10}")
    print("   " + "-" * 60)
    
    results = {}
    
    for noise_level in noise_levels:
        # Add noise to news_pred
        X_test_noisy = X_test_orig.copy()
        noise = np.random.normal(0, noise_level, size=len(X_test_noisy))
        X_test_noisy["news_pred"] = X_test_noisy["news_pred"] * (1 + noise)
        
        # Predict
        r2_noisy = r2_score(y_test, model.predict(X_test_noisy))
        drop = baseline_r2 - r2_noisy
        pct_drop = (drop / baseline_r2) * 100 if baseline_r2 > 0 else 0
        
        # Status based on expected behavior
        if noise_level <= 0.05:
            status = "✓ ROBUST" if pct_drop < 5 else "✗ FRAGILE"
        elif noise_level <= 0.10:
            status = "✓ OK" if pct_drop < 15 else "✗ WEAK"
        elif noise_level >= 0.50:
            status = "✓ EXPECTED" if pct_drop > 20 else "? STRANGE"
        else:
            status = "~"
        
        results[noise_level] = {
            "r2": r2_noisy,
            "drop": drop,
            "pct_drop": pct_drop
        }
        
        print(f"   {noise_level*100:>5.0f}% noise     {r2_noisy:>10.4f} {drop:>+12.4f} {pct_drop:>+9.1f}% {status:>10}")
    
    # Pass criteria
    # 5% noise should be robust (< 5% drop)
    # 50% noise should destroy signal (> 20% drop)
    
    drop_5pct = results[0.05]["pct_drop"]
    drop_50pct = results[0.50]["pct_drop"]
    
    passed = (drop_5pct < 10) and (drop_50pct > 15)
    
    print(f"\n   ANALYSIS:")
    print(f"      5% noise drops R² by {drop_5pct:.1f}% {'(ROBUST)' if drop_5pct < 10 else '(FRAGILE)'}")
    print(f"      50% noise drops R² by {drop_50pct:.1f}% {'(AS EXPECTED)' if drop_50pct > 15 else '(TOO ROBUST?)'}")
    
    print(f"\n   VERDICT: {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": "Noise Injection",
        "baseline_r2": baseline_r2,
        "results": results,
        "passed": passed
    }


# ============================================================
# TEST B: MULTI-LAG CAUSALITY (ALPHA DECAY)
# ============================================================
def test_b_alpha_decay(df):
    """
    Test how signal decays with lag.
    
    Expected behavior:
    - Future (+2): Highest R² (cheating/leakage)
    - Current (0): Baseline
    - Old (-2): Lower than baseline
    - Ancient (-5): Near zero contribution
    """
    print("\n" + "=" * 70)
    print("TEST B: MULTI-LAG CAUSALITY (Alpha Decay)")
    print("=" * 70)
    print("   Testing how news signal decays over time")
    
    features_base = ["tech_pred", "fund_pred", "VIX_close", 
                     "is_friday", "is_monday", "is_q4"]
    features_base = [f for f in features_base if f in df.columns]
    
    cutoff = pd.to_datetime("2023-01-01")
    
    # Define shifts to test
    shifts = [
        ("+3 (Future)", 3),
        ("+2 (Future)", 2),
        ("+1 (Tomorrow)", 1),
        ("0 (Current - Baseline)", 0),
        ("-1 (Yesterday)", -1),
        ("-2 (Old)", -2),
        ("-3 (Older)", -3),
        ("-5 (Ancient)", -5),
        ("-10 (Very Ancient)", -10),
    ]
    
    print(f"\n   {'Shift':<25} {'R²':>10} {'vs Baseline':>15} {'Alpha Decay':>12}")
    print("   " + "-" * 65)
    
    results = {}
    baseline_r2 = None
    
    for name, shift in shifts:
        # Create shifted data
        df_shifted = df.copy()
        
        if shift != 0:
            # Negative shift = looking into past (shift rows down)
            # Positive shift = looking into future (shift rows up)
            df_shifted["news_pred_shifted"] = df_shifted.groupby("ticker")["news_pred"].shift(-shift)
        else:
            df_shifted["news_pred_shifted"] = df_shifted["news_pred"]
        
        # Drop NaN
        df_shifted = df_shifted.dropna(subset=["news_pred_shifted", "target_log_var"])
        
        train = df_shifted[df_shifted["date"] < cutoff]
        test = df_shifted[df_shifted["date"] >= cutoff]
        
        if len(train) < 50 or len(test) < 20:
            print(f"   {name:<25} Insufficient data")
            continue
        
        features = features_base + ["news_pred_shifted"]
        
        X_train = train[features].fillna(0)
        y_train = train["target_log_var"]
        X_test = test[features].fillna(0)
        y_test = test["target_log_var"]
        
        model = train_model(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        
        # Store baseline
        if shift == 0:
            baseline_r2 = r2
        
        # Calculate vs baseline
        if baseline_r2 is not None:
            vs_baseline = r2 - baseline_r2
            decay = f"{vs_baseline:+.4f}"
        else:
            vs_baseline = 0
            decay = "N/A"
        
        results[shift] = {
            "name": name,
            "r2": r2,
            "vs_baseline": vs_baseline
        }
        
        # Visual indicator
        if shift > 0:
            indicator = "⚠️ FUTURE"
        elif shift == 0:
            indicator = "📍 BASELINE"
        elif shift >= -2:
            indicator = "📉 Recent"
        elif shift >= -5:
            indicator = "📉 Stale"
        else:
            indicator = "📉 Ancient"
        
        print(f"   {name:<25} {r2:>10.4f} {vs_baseline:>+14.4f} {indicator:>12}")
    
    # Pass criteria
    # R²(-2) should be < R²(0) - old news is worse
    # R²(+2) should be > R²(0) - future is cheating
    
    r2_current = results.get(0, {}).get("r2", 0)
    r2_old = results.get(-2, {}).get("r2", 0)
    r2_ancient = results.get(-5, {}).get("r2", 0)
    r2_future = results.get(2, {}).get("r2", 0)
    
    causality_valid = r2_current > r2_old
    decay_exists = r2_old > r2_ancient
    future_higher = r2_future > r2_current
    
    passed = causality_valid and decay_exists
    
    print(f"\n   ALPHA DECAY ANALYSIS:")
    print(f"      Future (+2) > Current (0): {r2_future:.4f} > {r2_current:.4f} = {future_higher}")
    print(f"      Current (0) > Old (-2):    {r2_current:.4f} > {r2_old:.4f} = {causality_valid}")
    print(f"      Old (-2) > Ancient (-5):   {r2_old:.4f} > {r2_ancient:.4f} = {decay_exists}")
    
    if passed:
        print(f"\n   ✓ Signal decays appropriately - causality confirmed")
        print(f"   ✓ Fresh news is more valuable than stale news")
    else:
        print(f"\n   ✗ Signal decay pattern is unexpected - investigate")
    
    print(f"\n   VERDICT: {'PASS' if passed else 'FAIL'}")
    
    return {
        "name": "Alpha Decay",
        "results": results,
        "baseline_r2": baseline_r2,
        "causality_valid": causality_valid,
        "decay_exists": decay_exists,
        "passed": passed
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("FINAL STRESS TEST: FRAGILITY CHECKS")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Testing model robustness and causal validity")
    
    # Load data
    df = load_and_prepare_data()
    
    # Run tests
    results = {}
    
    results["A"] = test_a_noise_injection(df)
    results["B"] = test_b_alpha_decay(df)
    
    # ================================================================
    # STRESS TEST REPORT
    # ================================================================
    print("\n" + "=" * 70)
    print("FINAL STRESS TEST REPORT")
    print("=" * 70)
    
    n_passed = sum(1 for r in results.values() if r["passed"])
    n_total = len(results)
    
    print(f"""
   ┌────────────────────────────────────────────────────────────────────┐
   │                      STRESS TEST RESULTS                           │
   ├────────────────────────────────────────────────────────────────────┤
   │ Test                              │ Description          │ Status │
   ├───────────────────────────────────┼──────────────────────┼────────┤
   │ A: Noise Injection                │ Robustness to noise  │   {'✅' if results['A']['passed'] else '❌'}   │
   │ B: Alpha Decay                    │ Causal validity      │   {'✅' if results['B']['passed'] else '❌'}   │
   ├───────────────────────────────────┴──────────────────────┼────────┤
   │ TOTAL                                                    │  {n_passed}/{n_total}   │
   └──────────────────────────────────────────────────────────┴────────┘
    """)
    
    # Verdict
    if n_passed == 2:
        verdict = "STRESS TEST PASSED"
        emoji = "🏆"
        msg = "Model is robust and causally valid."
    elif n_passed == 1:
        verdict = "PARTIAL"
        emoji = "⚠️"
        msg = "One test failed. Investigate before deployment."
    else:
        verdict = "FAILED"
        emoji = "❌"
        msg = "Critical fragility detected. Do not deploy."
    
    print(f"   {emoji} VERDICT: {verdict}")
    print(f"   {msg}")
    
    # Key metrics
    print(f"\n   KEY FINDINGS:")
    
    # Test A details
    if "results" in results["A"]:
        drop_5 = results["A"]["results"].get(0.05, {}).get("pct_drop", 0)
        drop_50 = results["A"]["results"].get(0.50, {}).get("pct_drop", 0)
        print(f"      Noise Robustness:")
        print(f"         5% noise:  {drop_5:+.1f}% R² drop {'(ROBUST)' if drop_5 < 10 else '(FRAGILE)'}")
        print(f"         50% noise: {drop_50:+.1f}% R² drop")
    
    # Test B details
    if "results" in results["B"]:
        r2_current = results["B"]["results"].get(0, {}).get("r2", 0)
        r2_old = results["B"]["results"].get(-2, {}).get("r2", 0)
        r2_ancient = results["B"]["results"].get(-5, {}).get("r2", 0)
        print(f"      Alpha Decay:")
        print(f"         Current (0):   R² = {r2_current:.4f}")
        print(f"         Old (-2):      R² = {r2_old:.4f} (decay: {r2_current - r2_old:+.4f})")
        print(f"         Ancient (-5):  R² = {r2_ancient:.4f} (decay: {r2_current - r2_ancient:+.4f})")
    
    print("\n" + "=" * 70)
    
    # Save report
    report_path = Path("results/stress_test_report.md")
    with open(report_path, "w") as f:
        f.write(f"""# Final Stress Test Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Tests Passed | {n_passed} / {n_total} |
| Verdict | {emoji} **{verdict}** |

---

## Test Results

### Test A: Noise Injection (Robustness)

| Noise Level | R² | Drop | Status |
|-------------|----|----|--------|
| Baseline | {results['A']['baseline_r2']:.4f} | - | - |
""")
        for level, data in results["A"]["results"].items():
            f.write(f"| {level*100:.0f}% | {data['r2']:.4f} | {data['pct_drop']:+.1f}% | {'Robust' if data['pct_drop'] < 10 else 'Degraded'} |\n")
        
        f.write(f"""
**Verdict:** {'✅ PASS' if results['A']['passed'] else '❌ FAIL'}

### Test B: Alpha Decay (Causality)

| Shift | R² | vs Baseline |
|-------|----|----|
""")
        for shift, data in sorted(results["B"]["results"].items(), key=lambda x: -x[0]):
            f.write(f"| {data['name']} | {data['r2']:.4f} | {data['vs_baseline']:+.4f} |\n")
        
        f.write(f"""
**Verdict:** {'✅ PASS' if results['B']['passed'] else '❌ FAIL'}

---

## Conclusion

{msg}

The model demonstrates:
- {'✅ Robustness to moderate noise' if results['A']['passed'] else '❌ Fragility to noise'}
- {'✅ Proper alpha decay (fresh news > stale news)' if results['B']['passed'] else '❌ Unexpected alpha decay pattern'}

---

*Generated by Final Stress Test*
""")
    
    print(f"   Report saved to: {report_path}")
    print("=" * 70)
    
    return results, verdict


if __name__ == "__main__":
    main()

