"""
Experiment 8: Mincer-Zarnowitz Calibration for All 3 ML Models
================================================================

Tests forecast unbiasedness and efficiency using the Mincer-Zarnowitz
regression:
    actual_t = alpha + beta * forecast_t + epsilon_t

Under the null hypothesis of a well-calibrated forecast:
    H0: alpha = 0, beta = 1

This extends the February audit (which only covered RIVE, HAR-RV-X,
GARCH) to include Elastic Net and LightGBM.

Usage:
    python scripts/mlwa_experiments/exp8_calibration.py
"""

import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlwa_experiments.assemble_features import (
    assemble_full_features, split_train_test
)


def mincer_zarnowitz(actual, forecast):
    """
    Mincer-Zarnowitz regression: actual_t = alpha + beta * forecast_t + eps_t

    Tests H0: alpha=0, beta=1 (unbiased, efficient forecast)

    Returns: alpha, beta, f_stat, f_pvalue, r2
    """
    X = forecast.reshape(-1, 1)
    y = actual

    reg = LinearRegression().fit(X, y)
    alpha = reg.intercept_
    beta = reg.coef_[0]
    r2 = reg.score(X, y)

    y_pred = reg.predict(X)
    resid = y - y_pred
    n = len(y)
    SSR_unrestricted = np.sum(resid ** 2)

    # Restricted: actual = 0 + 1*forecast + eps
    SSR_restricted = np.sum((actual - forecast) ** 2)

    q = 2  # two restrictions
    k = 2  # intercept + slope
    f_stat = ((SSR_restricted - SSR_unrestricted) / q) / (SSR_unrestricted / (n - k))
    f_pvalue = 1 - stats.f.cdf(f_stat, q, n - k)

    return alpha, beta, f_stat, f_pvalue, r2


def main():
    from scripts.mlwa_experiments.exp1_monolithic_lightgbm import train_monolithic_lightgbm
    from scripts.mlwa_experiments.exp2_elastic_net import train_elastic_net

    print("\n" + "=" * 70)
    print("EXPERIMENT 8: MINCER-ZARNOWITZ CALIBRATION COMPARISON")
    print("=" * 70)

    df, feature_cols = assemble_full_features()

    # Train all models
    res_lgbm = train_monolithic_lightgbm(df, feature_cols, verbose=False)
    res_enet = train_elastic_net(df, feature_cols, verbose=False)

    # RIVE
    rive_feats = [f for f in [
        "rv_lag_1", "rv_lag_5", "rv_lag_22", "returns_sq_lag_1",
        "VIX_close", "rsi_14",
        "news_memory", "shock_memory", "sentiment_memory", "shock_vix_memory",
        "sentiment_avg", "novelty_score", "shock_index", "news_count",
        "volume_shock", "hype_zscore", "price_acceleration",
        "is_friday", "is_monday", "is_q4",
        "vol_ma5", "vol_ma10", "vol_std5",
    ] if f in df.columns]

    train, test = split_train_test(df)
    y_train = train["target_log_var"]
    lo = y_train.quantile(0.02)
    hi = y_train.quantile(0.98)
    model_rive = Ridge(alpha=100.0)
    model_rive.fit(train[rive_feats].fillna(0), y_train.clip(lower=lo, upper=hi))
    y_pred_rive = model_rive.predict(test[rive_feats].fillna(0))

    y_test = test["target_log_var"].values

    models = {
        "RIVE (Ridge coord.)": y_pred_rive,
        "Monolithic LightGBM": res_lgbm["y_test_pred"],
        "Elastic Net": res_enet["y_test_pred"],
    }

    print(f"\n  Test set: {len(y_test):,} observations")
    print(f"\n  Mincer-Zarnowitz: actual_t = alpha + beta * forecast_t + eps_t")
    print(f"  H0: alpha = 0, beta = 1 (unbiased, efficient forecast)")

    print(f"\n{'=' * 70}")
    print(f"  {'Model':<25} {'alpha':>8} {'beta':>8} {'R²':>8} "
          f"{'F-stat':>10} {'p-value':>10} {'Verdict':>15}")
    print(f"  {'─' * 87}")

    mz_results = []
    for name, y_pred in models.items():
        alpha, beta, f_stat, f_pval, r2 = mincer_zarnowitz(y_test, y_pred)
        verdict = "REJECT H0" if f_pval < 0.05 else "PASS"

        print(f"  {name:<25} {alpha:>+8.4f} {beta:>8.4f} {r2:>8.4f} "
              f"{f_stat:>10.2f} {f_pval:>10.6f} {verdict:>15}")

        mz_results.append({
            "Model": name,
            "alpha": alpha, "beta": beta, "R2": r2,
            "F_stat": f_stat, "p_value": f_pval, "verdict": verdict,
        })

    print(f"  {'─' * 87}")

    # Interpretation
    print(f"\n{'=' * 70}")
    print("  CALIBRATION INTERPRETATION")
    print(f"{'=' * 70}")

    for r in mz_results:
        print(f"\n  {r['Model']}:")
        if r["verdict"] == "PASS":
            print(f"    Forecast is well-calibrated (cannot reject alpha=0, beta=1)")
        else:
            if abs(r["alpha"]) > 0.5:
                print(f"    Systematic bias: alpha = {r['alpha']:+.4f} (non-zero intercept)")
            if abs(r["beta"] - 1.0) > 0.1:
                direction = "over-reactive" if r["beta"] > 1.0 else "under-reactive"
                print(f"    Slope beta = {r['beta']:.4f} (forecast is {direction})")
            else:
                print(f"    Mild miscalibration (F = {r['F_stat']:.2f}, p = {r['p_value']:.4f})")

    # Best calibrated model
    best_cal = min(mz_results, key=lambda x: abs(x["beta"] - 1.0) + abs(x["alpha"]))
    print(f"\n  Best calibrated: {best_cal['Model']} "
          f"(alpha={best_cal['alpha']:+.4f}, beta={best_cal['beta']:.4f})")

    # Save
    out_df = pd.DataFrame(mz_results)
    out_path = Path(__file__).parent / "calibration_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to: {out_path}")

    return mz_results


if __name__ == "__main__":
    main()
