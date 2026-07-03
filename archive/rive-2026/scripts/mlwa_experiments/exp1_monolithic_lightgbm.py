"""
Experiment 1: Monolithic LightGBM Regressor on the Full Feature Set
====================================================================

Central question answered:
    "Why not just use one strong tabular model on the same information set?"

This trains a SINGLE LightGBM regressor using every end-of-day feature
available to RIVE (technical + news + retail + calendar + short-horizon vol)
and compares directly against RIVE's modular ensemble.

If RIVE beats the monolithic LightGBM, the benefit is from the modular
regime-aware design, not just from "using ML."

Usage:
    python scripts/mlwa_experiments/exp1_monolithic_lightgbm.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import mlflow

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlwa_experiments.assemble_features import (
    assemble_full_features, split_train_test, SECTOR_MAP
)


def train_monolithic_lightgbm(df, feature_cols, verbose=True):
    """
    Train a single LightGBM regressor on the full feature set.

    Uses the same hyperparameter style as RIVE's sub-agents (conservative,
    regularized) to ensure a fair comparison.

    Returns
    -------
    dict with model, predictions, and metrics
    """
    train, test = split_train_test(df)

    X_train = train[feature_cols].fillna(0)
    y_train = train["target_log_var"]
    X_test = test[feature_cols].fillna(0)
    y_test = test["target_log_var"]

    if verbose:
        print(f"\n{'=' * 70}")
        print("EXPERIMENT 1: MONOLITHIC LightGBM REGRESSOR")
        print(f"{'=' * 70}")
        print(f"  Features:  {len(feature_cols)}")
        print(f"  Train:     {len(X_train):,} samples")
        print(f"  Test:      {len(X_test):,} samples")

    # Conservative hyperparameters — mirrors the regularization style of
    # RIVE's LightGBM classifiers, adapted for regression.
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=20,
        colsample_bytree=0.7,
        subsample=0.8,
        subsample_freq=1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
    )

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Directional accuracy
    y_diff_true = np.diff(y_test.values)
    y_diff_pred = np.diff(y_test_pred)
    dir_acc = np.mean(np.sign(y_diff_true) == np.sign(y_diff_pred)) * 100

    if verbose:
        print(f"\n  {'Metric':<25} {'Train':>12} {'Test':>12}")
        print("  " + "-" * 51)
        print(f"  {'R²':<25} {train_r2:>12.4f} {test_r2:>12.4f}")
        print(f"  {'RMSE':<25} {'':>12} {test_rmse:>12.4f}")
        print(f"  {'MAE':<25} {'':>12} {test_mae:>12.4f}")
        print(f"  {'Directional Accuracy':<25} {'':>12} {dir_acc:>11.1f}%")

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    if verbose:
        print(f"\n  Top-10 Feature Importances:")
        print("  " + "-" * 45)
        for _, row in importance.head(10).iterrows():
            print(f"    {row['feature']:<30}: {row['importance']:>6}")

    # Sector breakdown
    test = test.copy()
    test["y_pred"] = y_test_pred
    sector_r2 = {}
    for sector in test["sector"].dropna().unique():
        mask = test["sector"] == sector
        if mask.sum() > 50:
            sector_r2[sector] = r2_score(
                y_test.values[mask.values], y_test_pred[mask.values]
            )

    if verbose:
        print(f"\n  Sector R²:")
        print("  " + "-" * 45)
        for s, r2 in sorted(sector_r2.items(), key=lambda x: x[1], reverse=True):
            print(f"    {s:15s}: {r2:.4f} ({r2*100:.2f}%)")

    return {
        "model": model,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "dir_acc": dir_acc,
        "sector_r2": sector_r2,
        "importance": importance,
        "y_test": y_test.values,
        "y_test_pred": y_test_pred,
    }


def main():
    # Assemble features
    df, feature_cols = assemble_full_features()

    # Train monolithic LightGBM
    results = train_monolithic_lightgbm(df, feature_cols)

    # Comparison table
    print(f"\n{'=' * 70}")
    print("COMPARISON: Monolithic LightGBM vs RIVE")
    print(f"{'=' * 70}")
    print(f"""
  +-----------------------------+------------+
  | Model                       | Test R²    |
  +-----------------------------+------------+
  | Monolithic LightGBM         | {results['test_r2']:>10.4f} |
  | RIVE (modular ensemble)     |     0.2304 |
  +-----------------------------+------------+
  | Difference (RIVE - LightGBM)| {0.2304 - results['test_r2']:>+10.4f} |
  +-----------------------------+------------+
""")

    if results["test_r2"] < 0.2304:
        print("  --> RIVE's modular design outperforms the monolithic baseline.")
        print("      The benefit comes from regime-aware modularity, not just ML.")
    else:
        print("  --> The monolithic LightGBM matches or beats RIVE.")
        print("      Consider whether the modular design adds interpretability value.")

    # MLflow logging
    mlflow.end_run()
    mlflow.set_experiment("mlwa_baselines")
    with mlflow.start_run(run_name="exp1_monolithic_lightgbm"):
        mlflow.log_params({
            "model": "LGBMRegressor",
            "n_features": len(feature_cols),
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 4,
        })
        mlflow.log_metrics({
            "train_r2": results["train_r2"],
            "test_r2": results["test_r2"],
            "test_rmse": results["test_rmse"],
            "test_mae": results["test_mae"],
            "dir_acc": results["dir_acc"],
        })
        mlflow.sklearn.log_model(results["model"], "monolithic_lightgbm")
    mlflow.end_run()

    print("  MLflow run logged to experiment 'mlwa_baselines'.")

    return results


if __name__ == "__main__":
    main()
