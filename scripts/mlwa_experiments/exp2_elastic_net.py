"""
Experiment 2: Elastic Net Baseline on the Full Feature Set
===========================================================

Central question answered:
    "Is the benefit from modularity or from adding more features?"

This trains an Elastic Net (linear ML baseline) on the exact same full
feature pool available to RIVE.  Together with the monolithic LightGBM,
this creates a clean three-way comparison:

    Elastic Net     = linear monolithic learner
    LightGBM        = nonlinear monolithic learner
    RIVE            = modular regime-aware learner

If both monolithic baselines trail RIVE, the modular design is justified.
If LightGBM matches RIVE but Elastic Net does not, the benefit is from
nonlinearity, not from modularity.

Usage:
    python scripts/mlwa_experiments/exp2_elastic_net.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import mlflow

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlwa_experiments.assemble_features import (
    assemble_full_features, split_train_test
)


def train_elastic_net(df, feature_cols, verbose=True):
    """
    Train Elastic Net with built-in cross-validation on the full feature set.

    Uses StandardScaler (fit on train only) since Elastic Net is sensitive
    to feature scales.  ElasticNetCV selects the best alpha and l1_ratio
    via 5-fold time-series-aware CV.

    Returns
    -------
    dict with model, predictions, and metrics
    """
    train, test = split_train_test(df)

    X_train_raw = train[feature_cols].fillna(0).values
    y_train = train["target_log_var"].values
    X_test_raw = test[feature_cols].fillna(0).values
    y_test = test["target_log_var"].values

    if verbose:
        print(f"\n{'=' * 70}")
        print("EXPERIMENT 2: ELASTIC NET (LINEAR ML BASELINE)")
        print(f"{'=' * 70}")
        print(f"  Features:  {len(feature_cols)}")
        print(f"  Train:     {len(X_train_raw):,} samples")
        print(f"  Test:      {len(X_test_raw):,} samples")

    # Standardize features (fit on train, transform both)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # ElasticNetCV: automatic alpha + l1_ratio selection
    # l1_ratio=1.0 → Lasso, l1_ratio=0.0 → Ridge, between → Elastic Net
    alphas = np.logspace(-4, 2, 50)

    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
        alphas=alphas,
        cv=5,
        max_iter=10000,
        random_state=42,
        n_jobs=-1,
    )

    if verbose:
        print(f"\n  Fitting ElasticNetCV (5-fold CV, 7 l1_ratio values)...")

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Directional accuracy
    y_diff_true = np.diff(y_test)
    y_diff_pred = np.diff(y_test_pred)
    dir_acc = np.mean(np.sign(y_diff_true) == np.sign(y_diff_pred)) * 100

    if verbose:
        print(f"\n  Selected hyperparameters:")
        print(f"    alpha:    {model.alpha_:.6f}")
        print(f"    l1_ratio: {model.l1_ratio_:.2f}")

        print(f"\n  {'Metric':<25} {'Train':>12} {'Test':>12}")
        print("  " + "-" * 51)
        print(f"  {'R²':<25} {train_r2:>12.4f} {test_r2:>12.4f}")
        print(f"  {'RMSE':<25} {'':>12} {test_rmse:>12.4f}")
        print(f"  {'MAE':<25} {'':>12} {test_mae:>12.4f}")
        print(f"  {'Directional Accuracy':<25} {'':>12} {dir_acc:>11.1f}%")

    # Non-zero coefficients
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_,
    })
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    n_nonzero = (coef_df["abs_coef"] > 1e-8).sum()
    n_total = len(feature_cols)

    if verbose:
        print(f"\n  Feature selection: {n_nonzero}/{n_total} non-zero coefficients")
        print(f"\n  Top-10 Coefficients (standardized):")
        print("  " + "-" * 50)
        for _, row in coef_df.head(10).iterrows():
            print(f"    {row['feature']:<30}: {row['coefficient']:>+10.4f}")

    # Sector breakdown
    test_df = test.copy()
    test_df["y_pred"] = y_test_pred
    sector_r2 = {}
    for sector in test_df["sector"].dropna().unique():
        mask = test_df["sector"] == sector
        if mask.sum() > 50:
            sector_r2[sector] = r2_score(
                y_test[mask.values], y_test_pred[mask.values]
            )

    if verbose:
        print(f"\n  Sector R²:")
        print("  " + "-" * 45)
        for s, r2 in sorted(sector_r2.items(), key=lambda x: x[1], reverse=True):
            print(f"    {s:15s}: {r2:.4f} ({r2*100:.2f}%)")

    return {
        "model": model,
        "scaler": scaler,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "dir_acc": dir_acc,
        "sector_r2": sector_r2,
        "coef_df": coef_df,
        "n_nonzero": n_nonzero,
        "alpha": model.alpha_,
        "l1_ratio": model.l1_ratio_,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
    }


def main():
    # Assemble features
    df, feature_cols = assemble_full_features()

    # Train Elastic Net
    results = train_elastic_net(df, feature_cols)

    # Comparison table
    print(f"\n{'=' * 70}")
    print("COMPARISON: Elastic Net vs RIVE")
    print(f"{'=' * 70}")
    print(f"""
  +-----------------------------+------------+
  | Model                       | Test R²    |
  +-----------------------------+------------+
  | Elastic Net (linear)        | {results['test_r2']:>10.4f} |
  | RIVE (modular ensemble)     |     0.2304 |
  +-----------------------------+------------+
  | Difference (RIVE - ElasticNet)| {0.2304 - results['test_r2']:>+8.4f} |
  +-----------------------------+------------+
""")

    # MLflow logging
    mlflow.end_run()
    mlflow.set_experiment("mlwa_baselines")
    with mlflow.start_run(run_name="exp2_elastic_net"):
        mlflow.log_params({
            "model": "ElasticNetCV",
            "n_features": len(feature_cols),
            "n_nonzero": results["n_nonzero"],
            "alpha": results["alpha"],
            "l1_ratio": results["l1_ratio"],
        })
        mlflow.log_metrics({
            "train_r2": results["train_r2"],
            "test_r2": results["test_r2"],
            "test_rmse": results["test_rmse"],
            "test_mae": results["test_mae"],
            "dir_acc": results["dir_acc"],
        })
    mlflow.end_run()

    print("  MLflow run logged to experiment 'mlwa_baselines'.")

    return results


if __name__ == "__main__":
    main()
