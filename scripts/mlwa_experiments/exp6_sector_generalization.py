"""
Experiment 6: Leave-One-Sector-Out Generalization
==================================================

Tests cross-sectional generalization by repeatedly holding out one
sector at a time. This is much stronger than a single random 80/20
ticker split.

Protocol:
    For each of the 6 GICS sectors in the 18-ticker universe:
        1. Hold out ALL tickers from that sector (3 tickers)
        2. Train all three models on the remaining 15 tickers
        3. Evaluate on the held-out sector
        4. Record R² for each model

This answers: "Does the model generalize to unseen stocks, or is it
memorizing ticker-specific patterns?"

Usage:
    python scripts/mlwa_experiments/exp6_sector_generalization.py
"""

import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlwa_experiments.assemble_features import (
    assemble_full_features, split_train_test, SECTOR_MAP, TRAIN_CUTOFF
)


def _train_and_predict(model_type, X_train, y_train, X_test):
    """Train model and return predictions."""
    if model_type == "rive":
        lo = y_train.quantile(0.02)
        hi = y_train.quantile(0.98)
        model = Ridge(alpha=100.0)
        model.fit(X_train.fillna(0), y_train.clip(lower=lo, upper=hi))
        return model.predict(X_test.fillna(0))

    elif model_type == "lgbm":
        model = LGBMRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4,
            num_leaves=15, min_child_samples=20,
            colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42,
            verbose=-1, n_jobs=-1,
        )
        model.fit(X_train.fillna(0), y_train)
        return model.predict(X_test.fillna(0))

    elif model_type == "enet":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train.fillna(0))
        Xte = scaler.transform(X_test.fillna(0))
        alphas = np.logspace(-4, 2, 50)
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 0.99], alphas=alphas,
            cv=3, max_iter=10000, random_state=42, n_jobs=-1,
        )
        model.fit(Xtr, y_train)
        return model.predict(Xte)


def main():
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: LEAVE-ONE-SECTOR-OUT GENERALIZATION")
    print("=" * 70)

    df, feature_cols = assemble_full_features()

    # RIVE feature set
    rive_feats = [f for f in [
        "rv_lag_1", "rv_lag_5", "rv_lag_22", "returns_sq_lag_1",
        "VIX_close", "rsi_14",
        "news_memory", "shock_memory", "sentiment_memory", "shock_vix_memory",
        "sentiment_avg", "novelty_score", "shock_index", "news_count",
        "volume_shock", "hype_zscore", "price_acceleration",
        "is_friday", "is_monday", "is_q4",
        "vol_ma5", "vol_ma10", "vol_std5",
    ] if f in df.columns]

    sectors = sorted(df["sector"].dropna().unique())
    sectors = [s for s in sectors if s != "Other"]

    print(f"\n  Sectors: {sectors}")
    print(f"  Protocol: Train on 5 sectors, test on 1 held-out sector")
    print(f"  Time constraint: Train < 2023, Test >= 2023 (within each split)")

    results = []

    for held_out in sectors:
        held_tickers = [t for t, s in SECTOR_MAP.items() if s == held_out]

        # Split: train on other sectors, test on held-out sector
        # But also respect the time split
        train_mask = (df["sector"] != held_out) & (df["date"] < TRAIN_CUTOFF)
        test_mask = (df["sector"] == held_out) & (df["date"] >= TRAIN_CUTOFF)

        if train_mask.sum() < 100 or test_mask.sum() < 50:
            print(f"  {held_out}: skipped (insufficient data)")
            continue

        train_df = df[train_mask]
        test_df = df[test_mask]

        y_train = train_df["target_log_var"]
        y_test = test_df["target_log_var"].values

        row = {"sector": held_out, "tickers": held_tickers,
               "n_train": len(train_df), "n_test": len(test_df)}

        model_type_map = {"RIVE": "rive", "LightGBM": "lgbm", "ElasticNet": "enet"}
        for model_name, feats in [("RIVE", rive_feats),
                                   ("LightGBM", feature_cols),
                                   ("ElasticNet", feature_cols)]:
            X_train = train_df[feats]
            X_test = test_df[feats]
            y_pred = _train_and_predict(model_type_map[model_name],
                                        X_train, y_train, X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            row[f"r2_{model_name}"] = r2
            row[f"rmse_{model_name}"] = rmse

        results.append(row)

        print(f"  {held_out:12s} (n_test={row['n_test']:,}): "
              f"RIVE={row['r2_RIVE']:+.4f}  "
              f"LGBM={row['r2_LightGBM']:+.4f}  "
              f"ENET={row['r2_ElasticNet']:+.4f}")

    results_df = pd.DataFrame(results)

    # Aggregate
    print(f"\n{'=' * 70}")
    print("  AGGREGATE RESULTS")
    print(f"{'=' * 70}")

    print(f"\n  {'Metric':<25} {'RIVE':>10} {'LightGBM':>10} {'ElasticNet':>10}")
    print("  " + "-" * 57)

    for stat_name, func in [("Mean R²", "mean"), ("Std R²", "std"),
                             ("Min R²", "min"), ("Max R²", "max")]:
        vals = {}
        for m in ["RIVE", "LightGBM", "ElasticNet"]:
            vals[m] = getattr(results_df[f"r2_{m}"], func)()
        print(f"  {stat_name:<25} {vals['RIVE']:>10.4f} {vals['LightGBM']:>10.4f} "
              f"{vals['ElasticNet']:>10.4f}")

    # Win count
    print(f"\n  Sector Win Count:")
    for m in ["RIVE", "LightGBM", "ElasticNet"]:
        wins = sum(1 for _, r in results_df.iterrows()
                   if r[f"r2_{m}"] == max(r["r2_RIVE"], r["r2_LightGBM"], r["r2_ElasticNet"]))
        print(f"    {m:<15}: {wins}/{len(results_df)} sectors")

    # All-positive check
    print(f"\n  R² > 0 in all held-out sectors:")
    for m in ["RIVE", "LightGBM", "ElasticNet"]:
        all_pos = (results_df[f"r2_{m}"] > 0).all()
        print(f"    {m:<15}: {'Yes' if all_pos else 'No'}")
        if not all_pos:
            neg = results_df[results_df[f"r2_{m}"] <= 0]["sector"].tolist()
            print(f"                    Negative sectors: {neg}")

    # Save
    out_path = Path(__file__).parent / "sector_generalization_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to: {out_path}")

    return results_df


if __name__ == "__main__":
    main()
