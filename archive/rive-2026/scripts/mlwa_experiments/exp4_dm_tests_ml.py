"""
Experiment 4: Diebold-Mariano Tests Between the Three ML Models
================================================================

Tests whether fixed-split R² differences among RIVE, LightGBM, and
Elastic Net are statistically significant.

Comparisons:
    1. RIVE  vs  Monolithic LightGBM
    2. RIVE  vs  Elastic Net
    3. Elastic Net  vs  LightGBM

Uses the same DM methodology (Newey-West HAC) as the February audit.

Usage:
    python scripts/mlwa_experiments/exp4_dm_tests_ml.py
"""

import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlwa_experiments.assemble_features import (
    assemble_full_features, split_train_test
)


def diebold_mariano(e1, e2, h=1, loss="MSE"):
    """
    Diebold-Mariano test for equal predictive accuracy.

    H0: E[d_t] = 0   (equal accuracy)
    H1: E[d_t] != 0   (different accuracy)

    Positive DM means model 1 has larger loss (model 2 is better).
    """
    if loss == "MSE":
        d = e1 ** 2 - e2 ** 2
    elif loss == "MAE":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    T = len(d)
    d_bar = d.mean()

    # Newey-West HAC variance
    bandwidth = max(h - 1, 0)
    gamma_0 = np.mean((d - d_bar) ** 2)
    gamma_sum = 0.0
    for k in range(1, bandwidth + 1):
        weight = 1 - k / (bandwidth + 1)
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += 2 * weight * gamma_k

    var_d = (gamma_0 + gamma_sum) / T
    if var_d <= 0:
        var_d = np.var(d, ddof=1) / T

    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return dm_stat, p_value, d_bar


def _train_rive_coordinator(df, feature_cols_rive):
    """Train RIVE-style Ridge coordinator and return test predictions."""
    train, test = split_train_test(df)

    X_train = train[feature_cols_rive].fillna(0)
    y_train = train["target_log_var"]
    X_test = test[feature_cols_rive].fillna(0)

    # Winsorize (same as RiveCoordinator)
    lo = y_train.quantile(0.02)
    hi = y_train.quantile(0.98)
    y_w = y_train.clip(lower=lo, upper=hi)

    model = Ridge(alpha=100.0)
    model.fit(X_train, y_w)
    return model.predict(X_test)


def main():
    from scripts.mlwa_experiments.exp1_monolithic_lightgbm import train_monolithic_lightgbm
    from scripts.mlwa_experiments.exp2_elastic_net import train_elastic_net

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: DIEBOLD-MARIANO TESTS BETWEEN ML MODELS")
    print("=" * 70)

    # Assemble features and train all three models
    df, feature_cols = assemble_full_features()

    # --- LightGBM ---
    res_lgbm = train_monolithic_lightgbm(df, feature_cols, verbose=False)

    # --- Elastic Net ---
    res_enet = train_elastic_net(df, feature_cols, verbose=False)

    # --- RIVE (Ridge coordinator on same data) ---
    # Use the RIVE-style feature set
    rive_feats = [f for f in [
        "rv_lag_1", "rv_lag_5", "rv_lag_22", "returns_sq_lag_1",
        "VIX_close", "rsi_14",
        "news_memory", "shock_memory", "sentiment_memory", "shock_vix_memory",
        "sentiment_avg", "novelty_score", "shock_index", "news_count",
        "volume_shock", "hype_zscore", "price_acceleration",
        "is_friday", "is_monday", "is_q4",
        "vol_ma5", "vol_ma10", "vol_std5",
    ] if f in df.columns]

    y_pred_rive = _train_rive_coordinator(df, rive_feats)
    _, test = split_train_test(df)
    y_test = test["target_log_var"].values

    y_pred_lgbm = res_lgbm["y_test_pred"]
    y_pred_enet = res_enet["y_test_pred"]

    # Verify alignment
    assert len(y_test) == len(y_pred_lgbm) == len(y_pred_enet) == len(y_pred_rive), \
        "Prediction arrays not aligned!"

    # R² check
    r2_rive = r2_score(y_test, y_pred_rive)
    r2_lgbm = r2_score(y_test, y_pred_lgbm)
    r2_enet = r2_score(y_test, y_pred_enet)

    print(f"\n  Fixed-Split R² (test set, n={len(y_test):,}):")
    print(f"    RIVE (Ridge coord.): {r2_rive:.4f} ({r2_rive*100:.2f}%)")
    print(f"    Monolithic LightGBM: {r2_lgbm:.4f} ({r2_lgbm*100:.2f}%)")
    print(f"    Elastic Net:         {r2_enet:.4f} ({r2_enet*100:.2f}%)")

    # Forecast errors
    e_rive = y_test - y_pred_rive
    e_lgbm = y_test - y_pred_lgbm
    e_enet = y_test - y_pred_enet

    # DM tests
    print(f"\n{'=' * 70}")
    print("  DIEBOLD-MARIANO TEST RESULTS")
    print("  H0: equal predictive accuracy (two-sided)")
    print(f"{'=' * 70}")

    comparisons = [
        ("RIVE vs LightGBM",   e_lgbm, e_rive, "Model 2 better = RIVE better"),
        ("RIVE vs Elastic Net", e_enet, e_rive, "Model 2 better = RIVE better"),
        ("LightGBM vs Elastic Net", e_enet, e_lgbm, "Model 2 better = LGBM better"),
    ]

    dm_results = []
    for label, e1, e2, interpret in comparisons:
        dm_mse, p_mse, d_mse = diebold_mariano(e1, e2, h=1, loss="MSE")
        dm_mae, p_mae, d_mae = diebold_mariano(e1, e2, h=1, loss="MAE")

        sig_mse = "***" if p_mse < 0.001 else "**" if p_mse < 0.01 else "*" if p_mse < 0.05 else "n.s."
        sig_mae = "***" if p_mae < 0.001 else "**" if p_mae < 0.01 else "*" if p_mae < 0.05 else "n.s."

        # Determine winner
        if d_mse > 0:
            winner_mse = label.split(" vs ")[0]
        else:
            winner_mse = label.split(" vs ")[1]

        print(f"\n  {label}")
        print(f"  {'─' * 55}")
        print(f"    MSE: DM = {dm_mse:+.4f},  p = {p_mse:.6f}  {sig_mse}")
        print(f"         mean(d_t) = {d_mse:+.6f}  "
              f"(lower MSE: {winner_mse})")
        print(f"    MAE: DM = {dm_mae:+.4f},  p = {p_mae:.6f}  {sig_mae}")
        print(f"         mean(d_t) = {d_mae:+.6f}")

        dm_results.append({
            "Comparison": label,
            "DM_MSE": dm_mse, "p_MSE": p_mse, "sig_MSE": sig_mse,
            "DM_MAE": dm_mae, "p_MAE": p_mae, "sig_MAE": sig_mae,
            "mean_d_MSE": d_mse,
        })

    # Verdict
    print(f"\n{'=' * 70}")
    print("  VERDICT")
    print(f"{'=' * 70}")

    all_ns = all(r["sig_MSE"] == "n.s." for r in dm_results)
    if all_ns:
        print("""
  All pairwise DM tests are NOT statistically significant (p > 0.05).

  Recommended paper statement:
    "Fixed-split performance differences among the three ML models
     (RIVE, monolithic LightGBM, Elastic Net) are not statistically
     significant (DM test, p > 0.05 in all pairwise comparisons).
     RIVE is statistically comparable in fixed-split accuracy but
     demonstrates superior stability under walk-forward evaluation."
""")
    else:
        sig_pairs = [r["Comparison"] for r in dm_results if r["sig_MSE"] != "n.s."]
        print(f"\n  Significant differences found for: {sig_pairs}")
        print("  Report the specific DM statistics and p-values in the paper.")

    # Save
    out_df = pd.DataFrame(dm_results)
    out_path = Path(__file__).parent / "dm_test_ml_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Results saved to: {out_path}")

    return dm_results


if __name__ == "__main__":
    main()
