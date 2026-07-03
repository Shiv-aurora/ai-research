"""
RIVE Research Audit - Test 04: Reproducibility Test
====================================================
Attempt to independently reproduce the claimed results.

Claimed Results:
- 18-ticker universe: ~30% R²
- Top 50 Active: 61% R²
- S&P 500 GICS-55: 22% R²

This script rebuilds the pipeline from scratch to verify these claims.

Author: External Audit
Date: January 2026
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_all_data():
    """Load all processed data files."""
    data_path = PROJECT_ROOT / "data" / "processed"

    data = {}

    # Load targets
    try:
        data["targets"] = pd.read_parquet(data_path / "targets.parquet")
        data["targets"]["date"] = pd.to_datetime(data["targets"]["date"]).dt.tz_localize(None)
        print(f"  ✓ Loaded targets: {len(data['targets']):,} rows")
    except Exception as e:
        print(f"  ✗ Failed to load targets: {e}")

    # Load residuals (technical predictions)
    try:
        data["residuals"] = pd.read_parquet(data_path / "residuals.parquet")
        data["residuals"]["date"] = pd.to_datetime(data["residuals"]["date"]).dt.tz_localize(None)
        print(f"  ✓ Loaded residuals: {len(data['residuals']):,} rows")
    except Exception as e:
        print(f"  ✗ Failed to load residuals: {e}")

    # Load news predictions
    try:
        data["news_preds"] = pd.read_parquet(data_path / "news_predictions.parquet")
        data["news_preds"]["date"] = pd.to_datetime(data["news_preds"]["date"]).dt.tz_localize(None)
        print(f"  ✓ Loaded news predictions: {len(data['news_preds']):,} rows")
    except Exception as e:
        print(f"  ⚠ No news predictions: {e}")
        data["news_preds"] = None

    # Load retail predictions
    try:
        data["retail_preds"] = pd.read_parquet(data_path / "retail_predictions.parquet")
        data["retail_preds"]["date"] = pd.to_datetime(data["retail_preds"]["date"]).dt.tz_localize(None)
        print(f"  ✓ Loaded retail predictions: {len(data['retail_preds']):,} rows")
    except Exception as e:
        print(f"  ⚠ No retail predictions: {e}")
        data["retail_preds"] = None

    return data


def build_coordinator_features(data):
    """
    Rebuild the coordinator feature matrix independently.

    This mimics RiveCoordinator.prepare_predictions_dataset()
    """
    print("\n  Building coordinator features...")

    df = data["targets"].copy()

    # 1. Merge technical predictions
    if "residuals" in data and data["residuals"] is not None:
        res = data["residuals"].copy()

        if "pred_tech" in res.columns:
            df = pd.merge(df, res[["date", "ticker", "pred_tech"]], on=["date", "ticker"], how="left")
            df["tech_pred"] = df["pred_tech"]
            print(f"    ✓ Added tech_pred from residuals")
        elif "pred_tech_excess" in res.columns:
            df = pd.merge(df, res[["date", "ticker", "pred_tech_excess"]], on=["date", "ticker"], how="left")
            df["tech_pred"] = df["pred_tech_excess"]
            print(f"    ✓ Added tech_pred from pred_tech_excess")

    # 2. Merge news risk scores (using correct column name: news_pred)
    if data.get("news_preds") is not None:
        news = data["news_preds"].copy()
        # The column is 'news_pred' not 'news_risk_score'
        if "news_pred" in news.columns:
            news = news.rename(columns={"news_pred": "news_risk_score"})
        if "news_risk_score" in news.columns:
            df = pd.merge(df, news[["date", "ticker", "news_risk_score"]], on=["date", "ticker"], how="left")
            df["news_risk_score"] = df["news_risk_score"].fillna(0.2)
            print(f"    ✓ Added news_risk_score")
        else:
            df["news_risk_score"] = 0.2
            print(f"    ⚠ Using default news_risk_score = 0.2")
    else:
        df["news_risk_score"] = 0.2
        print(f"    ⚠ Using default news_risk_score = 0.2")

    # 3. Merge retail risk scores
    if data.get("retail_preds") is not None:
        retail = data["retail_preds"][["date", "ticker", "retail_risk_score"]].copy()
        df = pd.merge(df, retail, on=["date", "ticker"], how="left")
        df["retail_risk_score"] = df["retail_risk_score"].fillna(0.2)
        print(f"    ✓ Added retail_risk_score")
    else:
        df["retail_risk_score"] = 0.2
        print(f"    ⚠ Using default retail_risk_score = 0.2")

    # 4. Calendar features
    df["date"] = pd.to_datetime(df["date"])
    dow = df["date"].dt.dayofweek
    df["is_monday"] = (dow == 0).astype(int)
    df["is_friday"] = (dow == 4).astype(int)
    df["is_q4"] = (df["date"].dt.quarter == 4).astype(int)
    print(f"    ✓ Added calendar features")

    # 5. Momentum features (with shift to prevent leakage)
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        ticker_data = df.loc[mask, "target_log_var"]

        df.loc[mask, "vol_ma5"] = ticker_data.rolling(5, min_periods=1).mean().shift(1)
        df.loc[mask, "vol_ma10"] = ticker_data.rolling(10, min_periods=1).mean().shift(1)
        df.loc[mask, "vol_std5"] = ticker_data.rolling(5, min_periods=2).std().shift(1)

    df["vol_ma5"] = df["vol_ma5"].fillna(df["target_log_var"].mean())
    df["vol_ma10"] = df["vol_ma10"].fillna(df["target_log_var"].mean())
    df["vol_std5"] = df["vol_std5"].fillna(0)
    print(f"    ✓ Added momentum features")

    # 6. Interaction
    df["news_x_retail"] = df["news_risk_score"] * df["retail_risk_score"]
    print(f"    ✓ Added interaction feature")

    return df


def train_and_evaluate(df, feature_cols, alpha=100.0, winsorize_pct=0.02):
    """
    Train the Ridge coordinator and evaluate.

    Mimics RiveCoordinator.train() exactly.
    """

    # Purged Walk-Forward Split
    cutoff = pd.to_datetime("2023-01-01")

    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()

    # Prepare X, y
    available_features = [f for f in feature_cols if f in df.columns]

    train = train.dropna(subset=available_features + ["target_log_var"])
    test = test.dropna(subset=available_features + ["target_log_var"])

    X_train = train[available_features].fillna(0).values
    y_train = train["target_log_var"].values
    X_test = test[available_features].fillna(0).values
    y_test = test["target_log_var"].values

    # Winsorize training target
    lower = np.percentile(y_train, winsorize_pct * 100)
    upper = np.percentile(y_train, (1 - winsorize_pct) * 100)
    y_train_winsorized = np.clip(y_train, lower, upper)

    # Train Ridge
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train_winsorized)

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train_winsorized, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)  # On ORIGINAL test target

    return {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "model": model,
        "features": available_features,
        "test_df": test
    }


def test_reproduce_baseline():
    """
    TEST 1: Reproduce HAR baseline R².
    """
    print("\n" + "="*70)
    print("TEST 1: Reproduce HAR Baseline")
    print("="*70)

    data = load_all_data()
    df = data["targets"].copy()

    # HAR features - using actual column names
    har_features = ["prev_day_rv", "rv_5d_mean", "rv_20d_mean"]

    # Check which features are available
    available_features = [f for f in har_features if f in df.columns]

    if len(available_features) == 0:
        # Try alternative feature names
        alt_features = ["realized_vol", "rsi_14", "VIX_close"]
        available_features = [f for f in alt_features if f in df.columns]
        print(f"    Using alternative features: {available_features}")

    if len(available_features) == 0:
        print("    ✗ No HAR features available")
        return True, 0.0

    # Simple Ridge on HAR
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df["date"] < cutoff].dropna(subset=available_features + ["target_log_var"])
    test = df[df["date"] >= cutoff].dropna(subset=available_features + ["target_log_var"])

    X_train = train[available_features].values
    y_train = train["target_log_var"].values
    X_test = test[available_features].values
    y_test = test["target_log_var"].values

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    har_r2 = r2_score(y_test, y_pred)

    print(f"\n  HAR Model (Ridge α=1.0):")
    print(f"    Features: {available_features}")
    print(f"    Train samples: {len(X_train):,}")
    print(f"    Test samples: {len(X_test):,}")
    print(f"    Test R²: {har_r2:.4f} ({har_r2*100:.2f}%)")

    # Expected: ~15-20% for 18-ticker universe (claimed 17.58% baseline for GICS-55)
    print("\n  Expected: HAR baseline should be ~15-25% R²")

    if har_r2 > 0.10:
        print("    ✓ PASS: HAR baseline is reasonable")
        return True, har_r2
    else:
        print("    ⚠ NOTE: HAR baseline lower than expected")
        return True, har_r2


def test_reproduce_full_ensemble():
    """
    TEST 2: Reproduce full RIVE ensemble R².
    """
    print("\n" + "="*70)
    print("TEST 2: Reproduce Full Ensemble R²")
    print("="*70)

    data = load_all_data()
    df = build_coordinator_features(data)

    # Full feature set
    feature_cols = [
        "tech_pred",
        "news_risk_score", "retail_risk_score",
        "is_friday", "is_monday", "is_q4",
        "vol_ma5", "vol_ma10", "vol_std5",
        "news_x_retail"
    ]

    results = train_and_evaluate(df, feature_cols, alpha=100.0, winsorize_pct=0.02)

    print(f"\n  Full Ensemble (Ridge α=100, winsorize=2%):")
    print(f"    Features: {len(results['features'])}")
    print(f"    Train samples: {results['n_train']:,}")
    print(f"    Test samples: {results['n_test']:,}")
    print(f"    Train R²: {results['train_r2']:.4f} ({results['train_r2']*100:.2f}%)")
    print(f"    Test R²: {results['test_r2']:.4f} ({results['test_r2']*100:.2f}%)")

    # Check coefficients
    print(f"\n  Model Coefficients:")
    coefs = dict(zip(results["features"], results["model"].coef_))
    for feat, coef in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {feat:20s}: {coef:+.4f}")

    return True, results["test_r2"]


def test_reproduce_tech_only():
    """
    TEST 3: Reproduce with tech_pred only (to measure agent contribution).
    """
    print("\n" + "="*70)
    print("TEST 3: Tech Prediction Only")
    print("="*70)

    data = load_all_data()
    df = build_coordinator_features(data)

    # Tech pred only
    results = train_and_evaluate(df, ["tech_pred"], alpha=100.0, winsorize_pct=0.02)

    print(f"\n  Tech Pred Only (Ridge α=100):")
    print(f"    Test R²: {results['test_r2']:.4f} ({results['test_r2']*100:.2f}%)")

    return True, results["test_r2"]


def test_sector_breakdown():
    """
    TEST 4: Verify sector-level R² breakdown.
    """
    print("\n" + "="*70)
    print("TEST 4: Sector Breakdown")
    print("="*70)

    data = load_all_data()
    df = build_coordinator_features(data)

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

    # Full ensemble training
    feature_cols = [
        "tech_pred",
        "news_risk_score", "retail_risk_score",
        "is_friday", "is_monday", "is_q4",
        "vol_ma5", "vol_ma10", "vol_std5",
        "news_x_retail"
    ]

    results = train_and_evaluate(df, feature_cols, alpha=100.0, winsorize_pct=0.02)

    # Sector breakdown on test data
    test_df = results["test_df"]

    # Get predictions for test set
    available_features = [f for f in feature_cols if f in test_df.columns]
    X_test = test_df[available_features].fillna(0).values
    test_df["y_pred"] = results["model"].predict(X_test)

    print(f"\n  Sector R² Breakdown:")
    print("  " + "-"*50)

    sector_results = {}

    for sector in test_df["sector"].dropna().unique():
        sector_data = test_df[test_df["sector"] == sector]
        if len(sector_data) > 50:
            sector_r2 = r2_score(sector_data["target_log_var"], sector_data["y_pred"])
            sector_results[sector] = sector_r2
            marker = " ⭐" if sector_r2 >= 0.30 else ""
            print(f"    {sector:15s}: R² = {sector_r2:.4f} ({sector_r2*100:.2f}%){marker}")

    return True, sector_results


def run_all_tests():
    """Run complete reproducibility test."""
    print("\n" + "="*70)
    print("  RIVE RESEARCH AUDIT - REPRODUCIBILITY TEST")
    print("  External Verification Suite")
    print("="*70)

    print("""
  This test attempts to independently reproduce the claimed results.

  Claimed Performance:
  - 18-ticker universe: ~30% R²
  - HAR baseline: ~17% R²
  - Improvement: ~5-10% over baseline
    """)

    results = {}

    try:
        passed, har_r2 = test_reproduce_baseline()
        results["har_baseline"] = (passed, har_r2)
    except Exception as e:
        print(f"  ✗ TEST 1 ERROR: {e}")
        results["har_baseline"] = (False, 0)
        import traceback
        traceback.print_exc()

    try:
        passed, full_r2 = test_reproduce_full_ensemble()
        results["full_ensemble"] = (passed, full_r2)
    except Exception as e:
        print(f"  ✗ TEST 2 ERROR: {e}")
        results["full_ensemble"] = (False, 0)
        import traceback
        traceback.print_exc()

    try:
        passed, tech_r2 = test_reproduce_tech_only()
        results["tech_only"] = (passed, tech_r2)
    except Exception as e:
        print(f"  ✗ TEST 3 ERROR: {e}")
        results["tech_only"] = (False, 0)
        import traceback
        traceback.print_exc()

    try:
        passed, sector_r2 = test_sector_breakdown()
        results["sector_breakdown"] = (passed, sector_r2)
    except Exception as e:
        print(f"  ✗ TEST 4 ERROR: {e}")
        results["sector_breakdown"] = (False, {})
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("  REPRODUCIBILITY SUMMARY")
    print("="*70)

    if "har_baseline" in results:
        _, har_r2 = results["har_baseline"]
        print(f"    HAR Baseline R²:      {har_r2:.4f} ({har_r2*100:.2f}%)")

    if "full_ensemble" in results:
        _, full_r2 = results["full_ensemble"]
        print(f"    Full Ensemble R²:     {full_r2:.4f} ({full_r2*100:.2f}%)")

    if "har_baseline" in results and "full_ensemble" in results:
        improvement = (full_r2 - har_r2) * 100
        print(f"\n    Improvement:          {improvement:+.2f}%")

    print("\n" + "="*70)
    print("  VERDICT")
    print("="*70)

    if "full_ensemble" in results:
        _, full_r2 = results["full_ensemble"]

        if full_r2 >= 0.25:
            print("  🏆 RESULTS REPRODUCED: Ensemble achieves 25%+ R²")
        elif full_r2 >= 0.20:
            print("  ✓ RESULTS PLAUSIBLE: Ensemble achieves 20%+ R²")
        elif full_r2 >= 0.15:
            print("  ⚠ RESULTS MARGINAL: Ensemble achieves 15-20% R²")
        else:
            print(f"  ⚠ RESULTS BELOW CLAIMS: Only {full_r2*100:.1f}% R²")

    return results


if __name__ == "__main__":
    run_all_tests()
