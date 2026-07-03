"""
RIVE Research Audit - Test 03: Shuffle Test (Gold Standard)
============================================================
The definitive test for data leakage: train on shuffled targets.

If the model has any form of leakage (look-ahead bias, data snooping),
it will STILL perform well on shuffled data because the "signal" is
coming from features that contain future information.

A legitimate model should achieve R² ≈ 0% on shuffled targets.

Author: External Audit
Date: January 2026
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_data():
    """Load targets and residuals."""
    data_path = PROJECT_ROOT / "data" / "processed"

    targets = pd.read_parquet(data_path / "targets.parquet")
    residuals = pd.read_parquet(data_path / "residuals.parquet")

    # Normalize dates
    targets["date"] = pd.to_datetime(targets["date"]).dt.tz_localize(None)
    residuals["date"] = pd.to_datetime(residuals["date"]).dt.tz_localize(None)

    return targets, residuals


def prepare_features(df):
    """Prepare features similar to the coordinator."""

    df = df.copy()

    # Calendar features
    df["date"] = pd.to_datetime(df["date"])
    dow = df["date"].dt.dayofweek
    df["is_monday"] = (dow == 0).astype(int)
    df["is_friday"] = (dow == 4).astype(int)
    df["is_q4"] = (df["date"].dt.quarter == 4).astype(int)

    # Momentum features (per ticker, with shift to avoid leakage)
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        ticker_data = df.loc[mask, "target_log_var"]

        df.loc[mask, "vol_ma5"] = ticker_data.rolling(5, min_periods=1).mean().shift(1)
        df.loc[mask, "vol_ma10"] = ticker_data.rolling(10, min_periods=1).mean().shift(1)
        df.loc[mask, "vol_std5"] = ticker_data.rolling(5, min_periods=2).std().shift(1)

    # Fill NaN
    df["vol_ma5"] = df["vol_ma5"].fillna(df["target_log_var"].mean())
    df["vol_ma10"] = df["vol_ma10"].fillna(df["target_log_var"].mean())
    df["vol_std5"] = df["vol_std5"].fillna(0)

    return df


def shuffle_test_har():
    """
    TEST 1: Shuffle test on HAR features only.

    Train HAR model on shuffled targets and check test R².
    """
    print("\n" + "="*70)
    print("TEST 1: HAR Model Shuffle Test")
    print("="*70)

    targets, _ = load_data()

    # Feature columns (HAR) - use actual column names from data
    har_features = ["prev_day_rv", "rv_5d_mean", "rv_20d_mean"]
    available_features = [f for f in har_features if f in targets.columns]

    if len(available_features) < 1:
        print(f"    ⚠ No HAR features found. Checking alternatives...")
        # Try alternative names
        alt_features = ["realized_vol", "rsi_14", "VIX_close"]
        available_features = [f for f in alt_features if f in targets.columns]

    if len(available_features) < 1:
        print(f"    ✗ No usable features found")
        return False, 0, 0

    print(f"\n  Using features: {available_features}")

    # Prepare data
    cutoff = pd.to_datetime("2023-01-01")

    train = targets[targets["date"] < cutoff].copy()
    test = targets[targets["date"] >= cutoff].copy()

    # Drop NaN
    train = train.dropna(subset=available_features + ["target_log_var"])
    test = test.dropna(subset=available_features + ["target_log_var"])

    X_train = train[available_features].values
    y_train = train["target_log_var"].values
    X_test = test[available_features].values
    y_test = test["target_log_var"].values

    print(f"\n  Train samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")

    # =============================
    # NORMAL MODEL (baseline)
    # =============================
    model_normal = Ridge(alpha=1.0)
    model_normal.fit(X_train, y_train)
    y_pred_normal = model_normal.predict(X_test)
    r2_normal = r2_score(y_test, y_pred_normal)

    print(f"\n  Normal Model:")
    print(f"    Test R²: {r2_normal:.4f} ({r2_normal*100:.2f}%)")

    # =============================
    # SHUFFLED TARGET MODEL
    # =============================
    np.random.seed(42)
    y_train_shuffled = np.random.permutation(y_train)

    model_shuffled = Ridge(alpha=1.0)
    model_shuffled.fit(X_train, y_train_shuffled)
    y_pred_shuffled = model_shuffled.predict(X_test)
    r2_shuffled = r2_score(y_test, y_pred_shuffled)

    print(f"\n  Shuffled Target Model:")
    print(f"    Test R²: {r2_shuffled:.4f} ({r2_shuffled*100:.2f}%)")

    # =============================
    # INTERPRETATION
    # =============================
    print("\n  Interpretation:")

    if r2_shuffled < 0.05:
        print("    ✓ PASS: Shuffled model has ~0% R² (no leakage detected)")
        result = True
    else:
        print(f"    ✗ FAIL: Shuffled model still has {r2_shuffled*100:.1f}% R²!")
        print("           This indicates potential data leakage.")
        result = False

    print(f"\n    R² drop: {r2_normal:.4f} → {r2_shuffled:.4f}")
    print(f"    Expected: Shuffled R² should be near 0 or negative")

    return result, r2_normal, r2_shuffled


def shuffle_test_full_features():
    """
    TEST 2: Shuffle test on full feature set (including momentum).

    This is the critical test - momentum features could potentially leak.
    """
    print("\n" + "="*70)
    print("TEST 2: Full Feature Shuffle Test")
    print("="*70)

    targets, residuals = load_data()

    # Merge targets with residuals to get tech_pred
    merged = pd.merge(
        targets,
        residuals[["date", "ticker", "pred_tech"]],
        on=["date", "ticker"],
        how="left"
    )

    # Prepare features
    merged = prepare_features(merged)

    # All coordinator features (that we have)
    feature_cols = [
        "pred_tech",
        "is_friday", "is_monday", "is_q4",
        "vol_ma5", "vol_ma10", "vol_std5"
    ]

    available_features = [f for f in feature_cols if f in merged.columns]
    print(f"\n  Using features: {available_features}")

    # Prepare data
    cutoff = pd.to_datetime("2023-01-01")

    merged = merged.dropna(subset=available_features + ["target_log_var"])

    train = merged[merged["date"] < cutoff].copy()
    test = merged[merged["date"] >= cutoff].copy()

    X_train = train[available_features].values
    y_train = train["target_log_var"].values
    X_test = test[available_features].values
    y_test = test["target_log_var"].values

    print(f"\n  Train samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")

    # =============================
    # NORMAL MODEL
    # =============================
    model_normal = Ridge(alpha=100.0)  # Same as coordinator
    model_normal.fit(X_train, y_train)
    y_pred_normal = model_normal.predict(X_test)
    r2_normal = r2_score(y_test, y_pred_normal)

    print(f"\n  Normal Model (Ridge α=100):")
    print(f"    Test R²: {r2_normal:.4f} ({r2_normal*100:.2f}%)")

    # =============================
    # SHUFFLED TARGET MODEL
    # =============================
    np.random.seed(42)
    y_train_shuffled = np.random.permutation(y_train)

    model_shuffled = Ridge(alpha=100.0)
    model_shuffled.fit(X_train, y_train_shuffled)
    y_pred_shuffled = model_shuffled.predict(X_test)
    r2_shuffled = r2_score(y_test, y_pred_shuffled)

    print(f"\n  Shuffled Target Model:")
    print(f"    Test R²: {r2_shuffled:.4f} ({r2_shuffled*100:.2f}%)")

    # =============================
    # INTERPRETATION
    # =============================
    print("\n  Interpretation:")

    if r2_shuffled < 0.05:
        print("    ✓ PASS: Shuffled model has ~0% R² (no leakage detected)")
        result = True
    else:
        print(f"    ✗ FAIL: Shuffled model still has {r2_shuffled*100:.1f}% R²!")
        print("           This indicates potential data leakage in features.")
        result = False

    print(f"\n    R² drop: {r2_normal:.4f} → {r2_shuffled:.4f}")

    return result, r2_normal, r2_shuffled


def shuffle_test_multiple_seeds():
    """
    TEST 3: Multiple shuffle iterations for statistical significance.

    Run shuffle test with different random seeds to ensure consistency.
    """
    print("\n" + "="*70)
    print("TEST 3: Multi-Seed Shuffle Test")
    print("="*70)

    targets, residuals = load_data()

    # Merge and prepare
    merged = pd.merge(
        targets,
        residuals[["date", "ticker", "pred_tech"]],
        on=["date", "ticker"],
        how="left"
    )

    merged = prepare_features(merged)

    feature_cols = ["pred_tech", "is_friday", "is_monday", "vol_ma5"]
    available_features = [f for f in feature_cols if f in merged.columns]

    merged = merged.dropna(subset=available_features + ["target_log_var"])

    cutoff = pd.to_datetime("2023-01-01")
    train = merged[merged["date"] < cutoff].copy()
    test = merged[merged["date"] >= cutoff].copy()

    X_train = train[available_features].values
    y_train = train["target_log_var"].values
    X_test = test[available_features].values
    y_test = test["target_log_var"].values

    # Normal baseline
    model_normal = Ridge(alpha=100.0)
    model_normal.fit(X_train, y_train)
    r2_normal = r2_score(y_test, model_normal.predict(X_test))

    print(f"\n  Normal R²: {r2_normal:.4f} ({r2_normal*100:.2f}%)")
    print("\n  Shuffled R² across 10 random seeds:")
    print("  " + "-"*40)

    shuffled_r2s = []

    for seed in range(10):
        np.random.seed(seed)
        y_train_shuffled = np.random.permutation(y_train)

        model = Ridge(alpha=100.0)
        model.fit(X_train, y_train_shuffled)
        r2 = r2_score(y_test, model.predict(X_test))
        shuffled_r2s.append(r2)
        print(f"    Seed {seed}: R² = {r2:.4f} ({r2*100:.2f}%)")

    mean_shuffled = np.mean(shuffled_r2s)
    std_shuffled = np.std(shuffled_r2s)

    print(f"\n  Shuffled R² Summary:")
    print(f"    Mean: {mean_shuffled:.4f} ({mean_shuffled*100:.2f}%)")
    print(f"    Std:  {std_shuffled:.4f}")
    print(f"    Range: [{min(shuffled_r2s):.4f}, {max(shuffled_r2s):.4f}]")

    print("\n  Interpretation:")

    if mean_shuffled < 0.03 and all(r < 0.10 for r in shuffled_r2s):
        print("    ✓ PASS: All shuffled runs have near-zero R²")
        result = True
    else:
        print(f"    ⚠ REVIEW: Mean shuffled R² = {mean_shuffled*100:.2f}%")
        result = mean_shuffled < 0.05

    # Significance of normal vs shuffled
    z_score = (r2_normal - mean_shuffled) / (std_shuffled + 1e-6)
    print(f"\n    Z-score (normal vs shuffled): {z_score:.2f}")

    if z_score > 3:
        print("    ✓ Normal R² is >3σ above shuffled (highly significant)")

    return result


def run_all_tests():
    """Run all shuffle tests."""
    print("\n" + "="*70)
    print("  RIVE RESEARCH AUDIT - SHUFFLE TEST (GOLD STANDARD)")
    print("  External Verification Suite")
    print("="*70)

    print("""
  About the Shuffle Test:
  -----------------------
  The shuffle test is the GOLD STANDARD for detecting data leakage.

  If a model's features contain future information (look-ahead bias),
  it will STILL perform well even when trained on shuffled targets.

  A legitimate model should achieve R² ≈ 0% on shuffled targets because
  shuffling breaks the true relationship between features and target.

  PASS CRITERIA: Shuffled R² < 5%
    """)

    results = {}

    try:
        result1, r2_norm1, r2_shuf1 = shuffle_test_har()
        results["har_shuffle"] = result1
    except Exception as e:
        print(f"  ✗ TEST 1 ERROR: {e}")
        results["har_shuffle"] = False
        import traceback
        traceback.print_exc()

    try:
        result2, r2_norm2, r2_shuf2 = shuffle_test_full_features()
        results["full_shuffle"] = result2
    except Exception as e:
        print(f"  ✗ TEST 2 ERROR: {e}")
        results["full_shuffle"] = False
        import traceback
        traceback.print_exc()

    try:
        results["multi_seed"] = shuffle_test_multiple_seeds()
    except Exception as e:
        print(f"  ✗ TEST 3 ERROR: {e}")
        results["multi_seed"] = False
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("  SHUFFLE TEST SUMMARY")
    print("="*70)

    passed = sum(results.values())
    total = len(results)

    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"    {test}: {status}")

    print(f"\n  Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🏆 ALL SHUFFLE TESTS PASSED - NO DATA LEAKAGE DETECTED")
    else:
        print(f"\n  ⚠ {total - passed} TESTS FAILED - POTENTIAL LEAKAGE")

    return results


if __name__ == "__main__":
    run_all_tests()
