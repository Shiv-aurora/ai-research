"""
Experiment 5: Tail-Event Metrics for the Final Forecast
=========================================================

RIVE's architecture is specifically designed for extreme events:
  - NewsAgent detects extreme residuals (top 20%)
  - RetailAgent detects high-attention regimes

This experiment tests whether RIVE is actually better than monolithic
baselines at capturing tail events, even if average R² is similar.

Metrics:
  1. Precision / Recall / F1 for top-decile realized volatility days
  2. ROC-AUC and PR-AUC for binary high-vol target
  3. Severe underprediction rate in top decile / quintile
  4. Conditional R² on high-vol vs low-vol subsets

Usage:
    python scripts/mlwa_experiments/exp5_tail_metrics.py
"""

import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    r2_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlwa_experiments.assemble_features import (
    assemble_full_features, split_train_test
)


def _get_all_predictions(df, feature_cols):
    """Train all three models and return aligned predictions."""
    from scripts.mlwa_experiments.exp1_monolithic_lightgbm import train_monolithic_lightgbm
    from scripts.mlwa_experiments.exp2_elastic_net import train_elastic_net

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
    X_train = train[rive_feats].fillna(0)
    y_train = train["target_log_var"]
    lo = y_train.quantile(0.02)
    hi = y_train.quantile(0.98)

    model = Ridge(alpha=100.0)
    model.fit(X_train, y_train.clip(lower=lo, upper=hi))
    y_pred_rive = model.predict(test[rive_feats].fillna(0))

    y_test = test["target_log_var"].values

    return {
        "y_test": y_test,
        "y_pred_rive": y_pred_rive,
        "y_pred_lgbm": res_lgbm["y_test_pred"],
        "y_pred_enet": res_enet["y_test_pred"],
        "test_df": test,
    }


def compute_tail_metrics(y_test, y_pred, percentile_threshold, label=""):
    """
    Compute tail-event metrics at a given percentile threshold.

    A "tail event" is defined as actual target_log_var > threshold.
    We check how well the model's predicted ranking identifies these events.
    """
    # Define binary high-vol target based on actual values
    threshold = np.percentile(y_test, percentile_threshold)
    y_binary = (y_test > threshold).astype(int)
    n_pos = y_binary.sum()
    n_neg = len(y_binary) - n_pos

    # Use predicted values as a score (higher pred = more likely high-vol)
    # For classification metrics, threshold at the same percentile of predictions
    pred_threshold = np.percentile(y_pred, percentile_threshold)
    y_pred_binary = (y_pred > pred_threshold).astype(int)

    precision = precision_score(y_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_binary, y_pred_binary, zero_division=0)

    # AUC metrics (using continuous predictions as scores)
    roc_auc = roc_auc_score(y_binary, y_pred)
    pr_auc = average_precision_score(y_binary, y_pred)

    # Severe underprediction rate:
    # Among actual top-decile days, what fraction did the model
    # predict as BELOW the median?
    median_pred = np.median(y_pred)
    tail_mask = y_binary == 1
    if tail_mask.sum() > 0:
        severe_underpred = np.mean(y_pred[tail_mask] < median_pred)
    else:
        severe_underpred = 0.0

    # Conditional R² on high-vol subset
    if tail_mask.sum() > 10:
        r2_tail = r2_score(y_test[tail_mask], y_pred[tail_mask])
    else:
        r2_tail = float("nan")

    # Conditional R² on low-vol subset
    low_mask = ~tail_mask
    if low_mask.sum() > 10:
        r2_low = r2_score(y_test[low_mask], y_pred[low_mask])
    else:
        r2_low = float("nan")

    return {
        "label": label,
        "threshold_pct": percentile_threshold,
        "n_tail": n_pos,
        "n_normal": n_neg,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "severe_underpred_rate": severe_underpred,
        "r2_tail": r2_tail,
        "r2_normal": r2_low,
    }


def main():
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: TAIL-EVENT METRICS")
    print("=" * 70)

    df, feature_cols = assemble_full_features()
    preds = _get_all_predictions(df, feature_cols)

    y_test = preds["y_test"]
    models = {
        "RIVE": preds["y_pred_rive"],
        "LightGBM": preds["y_pred_lgbm"],
        "ElasticNet": preds["y_pred_enet"],
    }

    # ---- Top Decile (90th percentile) ----
    print(f"\n{'=' * 70}")
    print("  TOP-DECILE TAIL EVENTS (>= 90th percentile of actual vol)")
    print(f"{'=' * 70}")

    decile_results = []
    for name, y_pred in models.items():
        m = compute_tail_metrics(y_test, y_pred, 90, label=name)
        decile_results.append(m)

    print(f"\n  {'Model':<15} {'Prec':>8} {'Recall':>8} {'F1':>8} "
          f"{'ROC-AUC':>9} {'PR-AUC':>9} {'Underpred':>10} "
          f"{'R²(tail)':>10} {'R²(norm)':>10}")
    print("  " + "-" * 100)
    for r in decile_results:
        print(f"  {r['label']:<15} {r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['f1']:>8.4f} {r['roc_auc']:>9.4f} {r['pr_auc']:>9.4f} "
              f"{r['severe_underpred_rate']:>10.1%} "
              f"{r['r2_tail']:>10.4f} {r['r2_normal']:>10.4f}")

    # ---- Top Quintile (80th percentile) ----
    print(f"\n{'=' * 70}")
    print("  TOP-QUINTILE TAIL EVENTS (>= 80th percentile of actual vol)")
    print(f"{'=' * 70}")

    quintile_results = []
    for name, y_pred in models.items():
        m = compute_tail_metrics(y_test, y_pred, 80, label=name)
        quintile_results.append(m)

    print(f"\n  {'Model':<15} {'Prec':>8} {'Recall':>8} {'F1':>8} "
          f"{'ROC-AUC':>9} {'PR-AUC':>9} {'Underpred':>10} "
          f"{'R²(tail)':>10} {'R²(norm)':>10}")
    print("  " + "-" * 100)
    for r in quintile_results:
        print(f"  {r['label']:<15} {r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['f1']:>8.4f} {r['roc_auc']:>9.4f} {r['pr_auc']:>9.4f} "
              f"{r['severe_underpred_rate']:>10.1%} "
              f"{r['r2_tail']:>10.4f} {r['r2_normal']:>10.4f}")

    # ---- Interpretation ----
    print(f"\n{'=' * 70}")
    print("  INTERPRETATION")
    print(f"{'=' * 70}")

    # Find best model on tail metrics
    best_f1_decile = max(decile_results, key=lambda x: x["f1"])
    best_auc_decile = max(decile_results, key=lambda x: x["roc_auc"])
    best_underpred = min(decile_results, key=lambda x: x["severe_underpred_rate"])

    print(f"\n  Top-decile best F1:           {best_f1_decile['label']} ({best_f1_decile['f1']:.4f})")
    print(f"  Top-decile best ROC-AUC:      {best_auc_decile['label']} ({best_auc_decile['roc_auc']:.4f})")
    print(f"  Lowest severe underprediction: {best_underpred['label']} ({best_underpred['severe_underpred_rate']:.1%})")

    rive_d = next(r for r in decile_results if r["label"] == "RIVE")
    lgbm_d = next(r for r in decile_results if r["label"] == "LightGBM")

    if rive_d["roc_auc"] > lgbm_d["roc_auc"]:
        print(f"\n  RIVE has better tail-event detection (ROC-AUC) than LightGBM.")
        print(f"  This supports the modular architecture's focus on extreme events.")
    elif abs(rive_d["roc_auc"] - lgbm_d["roc_auc"]) < 0.01:
        print(f"\n  RIVE and LightGBM have comparable tail-event detection.")
    else:
        print(f"\n  LightGBM has better tail-event detection than RIVE.")

    # Save
    all_results = decile_results + quintile_results
    out_df = pd.DataFrame(all_results)
    out_path = Path(__file__).parent / "tail_metrics_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to: {out_path}")

    return {"decile": decile_results, "quintile": quintile_results}


if __name__ == "__main__":
    main()
