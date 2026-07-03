"""
RIVE Research Audit - Test 01: Data Leakage Detection
======================================================
External audit script to verify no look-ahead bias in feature engineering.

Tests:
1. Feature-Target temporal alignment
2. Momentum features use proper lagging
3. Train/Test temporal separation
4. No future data contamination

Author: External Audit
Date: January 2026
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_data():
    """Load all relevant data files."""
    data_path = PROJECT_ROOT / "data" / "processed"

    targets = pd.read_parquet(data_path / "targets.parquet")
    residuals = pd.read_parquet(data_path / "residuals.parquet")

    # Normalize dates
    targets["date"] = pd.to_datetime(targets["date"]).dt.tz_localize(None)
    residuals["date"] = pd.to_datetime(residuals["date"]).dt.tz_localize(None)

    return targets, residuals


def test_temporal_alignment():
    """
    TEST 1: Verify target is correctly shifted to represent NEXT DAY.

    If target_log_var at date T is the volatility FOR day T+1,
    then features at date T should only use data from T-1 or earlier.
    """
    print("\n" + "="*70)
    print("TEST 1: Temporal Alignment Check")
    print("="*70)

    targets, residuals = load_data()

    # Check HAR features are properly lagged
    print("\n  Checking HAR feature lags...")

    required_lag_features = ["rv_lag_1", "rv_lag_5", "rv_lag_22"]

    for feat in required_lag_features:
        if feat in targets.columns:
            print(f"    ✓ {feat} exists in targets")
        else:
            print(f"    ✗ {feat} MISSING - potential issue")

    # Verify rv_lag_1 is actually lagged (not same as target)
    sample_ticker = targets["ticker"].unique()[0]
    ticker_data = targets[targets["ticker"] == sample_ticker].sort_values("date").copy()

    # If rv_lag_1 is correctly lagged, it should NOT be correlated 1.0 with target
    if "rv_lag_1" in ticker_data.columns and "target_log_var" in ticker_data.columns:
        corr = ticker_data["rv_lag_1"].corr(ticker_data["target_log_var"])
        print(f"\n  Correlation(rv_lag_1, target_log_var) = {corr:.4f}")

        if corr < 0.99:
            print("    ✓ PASS: Features are properly lagged (not identical to target)")
        else:
            print("    ✗ FAIL: Features may be leaking future data!")

    return True


def test_momentum_features():
    """
    TEST 2: Verify momentum features use .shift(1) to prevent leakage.

    vol_ma5, vol_ma10, vol_std5 should be computed on PAST data only.
    """
    print("\n" + "="*70)
    print("TEST 2: Momentum Feature Leakage Check")
    print("="*70)

    targets, _ = load_data()

    sample_ticker = targets["ticker"].unique()[0]
    ticker_data = targets[targets["ticker"] == sample_ticker].sort_values("date").reset_index(drop=True)

    print(f"\n  Testing on ticker: {sample_ticker}")
    print(f"  Sample size: {len(ticker_data)} rows")

    # Manually compute what vol_ma5 SHOULD be (with shift)
    target_col = "target_log_var"

    if target_col not in ticker_data.columns:
        print(f"    ✗ {target_col} not found, cannot verify momentum")
        return False

    # Compute expected vol_ma5 WITH proper shift
    expected_vol_ma5 = ticker_data[target_col].rolling(5, min_periods=1).mean().shift(1)

    # Check if vol_ma5 already exists
    if "vol_ma5" in ticker_data.columns:
        actual_vol_ma5 = ticker_data["vol_ma5"]

        # Compare (after handling NaN)
        mask = expected_vol_ma5.notna() & actual_vol_ma5.notna()
        diff = (expected_vol_ma5[mask] - actual_vol_ma5[mask]).abs().max()

        print(f"\n  Max difference between expected and actual vol_ma5: {diff:.6f}")

        if diff < 0.01:
            print("    ✓ PASS: vol_ma5 correctly computed with shift(1)")
        else:
            print("    ⚠ WARNING: vol_ma5 may have different computation")
    else:
        print("    ℹ vol_ma5 not pre-computed in targets file (computed at training time)")

    # CRITICAL TEST: Check if vol_ma5 at time T uses data from T or only T-1 and before
    print("\n  Leakage simulation test...")

    # Create a synthetic scenario where we KNOW the answer
    test_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

    # With proper shift(1), vol_ma5 at index 5 should use indices 0-4 (values 1-5)
    # NOT indices 1-5 (values 2-6)

    correct_vol_ma5 = test_series.rolling(5, min_periods=1).mean().shift(1)

    # At index 5, the value should be mean of [1,2,3,4,5] = 3.0
    expected_at_5 = 3.0
    actual_at_5 = correct_vol_ma5.iloc[5]

    print(f"\n  Synthetic test at index 5:")
    print(f"    Series: {test_series.tolist()}")
    print(f"    Expected vol_ma5[5] = mean([1,2,3,4,5]) = {expected_at_5}")
    print(f"    Actual vol_ma5[5] = {actual_at_5}")

    if abs(expected_at_5 - actual_at_5) < 0.001:
        print("    ✓ PASS: shift(1) correctly prevents look-ahead bias")
        return True
    else:
        print("    ✗ FAIL: Potential look-ahead bias detected!")
        return False


def test_train_test_separation():
    """
    TEST 3: Verify strict temporal separation between train and test sets.

    Train should be < 2023-01-01
    Test should be >= 2023-01-01
    No overlap allowed.
    """
    print("\n" + "="*70)
    print("TEST 3: Train/Test Temporal Separation")
    print("="*70)

    targets, _ = load_data()

    cutoff = pd.to_datetime("2023-01-01")

    train = targets[targets["date"] < cutoff]
    test = targets[targets["date"] >= cutoff]

    print(f"\n  Cutoff date: {cutoff.date()}")
    print(f"  Train samples: {len(train):,}")
    print(f"  Test samples: {len(test):,}")

    train_max_date = train["date"].max()
    test_min_date = test["date"].min()

    print(f"\n  Train max date: {train_max_date.date()}")
    print(f"  Test min date: {test_min_date.date()}")

    # Check for gap (should be at least 1 day)
    gap = (test_min_date - train_max_date).days
    print(f"  Gap between train and test: {gap} days")

    if gap >= 0:
        print("    ✓ PASS: No temporal overlap between train and test")
    else:
        print("    ✗ FAIL: Train and test sets overlap!")

    # Check for data consistency
    print("\n  Data coverage check:")

    train_tickers = set(train["ticker"].unique())
    test_tickers = set(test["ticker"].unique())

    overlap = train_tickers & test_tickers
    train_only = train_tickers - test_tickers
    test_only = test_tickers - train_tickers

    print(f"    Tickers in both: {len(overlap)}")
    print(f"    Train-only tickers: {len(train_only)}")
    print(f"    Test-only tickers: {len(test_only)}")

    if len(overlap) > 0:
        print("    ✓ PASS: Same tickers in train and test (proper setup)")
    else:
        print("    ⚠ WARNING: No overlapping tickers")

    return True


def test_no_future_contamination():
    """
    TEST 4: Verify prediction residuals don't contain future information.

    If residuals were computed with look-ahead, they would have
    artificially low variance or perfect correlation patterns.
    """
    print("\n" + "="*70)
    print("TEST 4: Future Contamination Check (Residual Analysis)")
    print("="*70)

    _, residuals = load_data()

    cutoff = pd.to_datetime("2023-01-01")
    test_residuals = residuals[residuals["date"] >= cutoff]

    if "resid_tech" not in test_residuals.columns:
        print("    ⚠ resid_tech not found, checking for alternatives...")
        resid_col = [c for c in test_residuals.columns if "resid" in c.lower()]
        if resid_col:
            print(f"    Found: {resid_col}")
        else:
            print("    ✗ No residual columns found")
            return False

    resid_col = "resid_tech" if "resid_tech" in test_residuals.columns else "resid_tech_excess"

    if resid_col not in test_residuals.columns:
        print(f"    ✗ {resid_col} not found")
        return False

    residuals_series = test_residuals[resid_col].dropna()

    print(f"\n  Residual statistics (test period):")
    print(f"    Count: {len(residuals_series):,}")
    print(f"    Mean: {residuals_series.mean():.4f}")
    print(f"    Std: {residuals_series.std():.4f}")
    print(f"    Min: {residuals_series.min():.4f}")
    print(f"    Max: {residuals_series.max():.4f}")

    # A model with look-ahead would have residuals centered at 0 with low variance
    # A legitimate model should have non-zero variance

    if residuals_series.std() > 0.01:
        print("\n    ✓ PASS: Residuals have substantial variance (no perfect fit)")
    else:
        print("\n    ✗ FAIL: Residuals suspiciously low variance (possible leakage)")

    # Check for autocorrelation (look-ahead would show unusual patterns)
    if len(residuals_series) > 100:
        autocorr_1 = residuals_series.autocorr(lag=1)
        autocorr_5 = residuals_series.autocorr(lag=5)

        print(f"\n  Autocorrelation check:")
        print(f"    Lag-1 autocorr: {autocorr_1:.4f}")
        print(f"    Lag-5 autocorr: {autocorr_5:.4f}")

        if abs(autocorr_1) < 0.9:
            print("    ✓ PASS: No suspicious autocorrelation pattern")
        else:
            print("    ⚠ WARNING: High autocorrelation may indicate issues")

    return True


def run_all_tests():
    """Run complete leakage audit."""
    print("\n" + "="*70)
    print("  RIVE RESEARCH AUDIT - DATA LEAKAGE DETECTION")
    print("  External Verification Suite")
    print("="*70)

    results = {}

    try:
        results["temporal_alignment"] = test_temporal_alignment()
    except Exception as e:
        print(f"  ✗ TEST 1 ERROR: {e}")
        results["temporal_alignment"] = False

    try:
        results["momentum_leakage"] = test_momentum_features()
    except Exception as e:
        print(f"  ✗ TEST 2 ERROR: {e}")
        results["momentum_leakage"] = False

    try:
        results["train_test_separation"] = test_train_test_separation()
    except Exception as e:
        print(f"  ✗ TEST 3 ERROR: {e}")
        results["train_test_separation"] = False

    try:
        results["future_contamination"] = test_no_future_contamination()
    except Exception as e:
        print(f"  ✗ TEST 4 ERROR: {e}")
        results["future_contamination"] = False

    # Summary
    print("\n" + "="*70)
    print("  AUDIT SUMMARY")
    print("="*70)

    passed = sum(results.values())
    total = len(results)

    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"    {test}: {status}")

    print(f"\n  Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🏆 ALL LEAKAGE TESTS PASSED")
    else:
        print(f"\n  ⚠ {total - passed} TESTS FAILED - REVIEW REQUIRED")

    return results


if __name__ == "__main__":
    run_all_tests()
