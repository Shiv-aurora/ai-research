"""
RIVE Research Audit - Test 03: Conditional Regime Performance
==============================================================
Do the news/retail agent signals help SPECIFICALLY during stress periods,
even if their aggregate contribution averages to ~zero?

Methodology:
  1. Train two models on pre-2023 data:
     - RIVE-Full (10 features, with agent signals)
     - RIVE-Core (7 features, momentum + calendar + tech_pred only)
  2. Generate predictions on test data (2023-2024)
  3. Stratify test observations into regimes:
     - By news activity (quartiles of news_risk_score)
     - By retail activity (quartiles of retail_risk_score)
     - By realized volatility level (quartiles of lagged vol)
     - By combined stress (news AND retail both elevated)
  4. Compare R² and MAE of Full vs Core within each regime
  5. Test if the difference is significant (paired t-test on squared errors)

Author: External Audit
Date: February 2026
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_and_prepare():
    """Load and prepare the full dataset."""
    data_path = PROJECT_ROOT / "data" / "processed"

    targets = pd.read_parquet(data_path / "targets.parquet")
    residuals = pd.read_parquet(data_path / "residuals.parquet")

    targets["date"] = pd.to_datetime(targets["date"]).dt.tz_localize(None)
    residuals["date"] = pd.to_datetime(residuals["date"]).dt.tz_localize(None)

    try:
        retail_preds = pd.read_parquet(data_path / "retail_predictions.parquet")
        retail_preds["date"] = pd.to_datetime(retail_preds["date"]).dt.tz_localize(None)
    except Exception:
        retail_preds = None

    try:
        news_preds = pd.read_parquet(data_path / "news_predictions.parquet")
        news_preds["date"] = pd.to_datetime(news_preds["date"]).dt.tz_localize(None)
        if "news_pred" in news_preds.columns:
            news_preds = news_preds.rename(columns={"news_pred": "news_risk_score"})
    except Exception:
        news_preds = None

    df = targets.copy()
    df = pd.merge(df, residuals[["date", "ticker", "pred_tech"]],
                  on=["date", "ticker"], how="left")
    df["tech_pred"] = df["pred_tech"]

    if news_preds is not None and "news_risk_score" in news_preds.columns:
        df = pd.merge(df, news_preds[["date", "ticker", "news_risk_score"]],
                      on=["date", "ticker"], how="left")
        df["news_risk_score"] = df["news_risk_score"].fillna(0.2)
    else:
        df["news_risk_score"] = 0.2

    if retail_preds is not None:
        df = pd.merge(df, retail_preds[["date", "ticker", "retail_risk_score"]],
                      on=["date", "ticker"], how="left")
        df["retail_risk_score"] = df["retail_risk_score"].fillna(0.2)
    else:
        df["retail_risk_score"] = 0.2

    dow = df["date"].dt.dayofweek
    df["is_monday"] = (dow == 0).astype(int)
    df["is_friday"] = (dow == 4).astype(int)
    df["is_q4"] = (df["date"].dt.quarter == 4).astype(int)

    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        ticker_data = df.loc[mask, "target_log_var"]
        df.loc[mask, "vol_ma5"] = ticker_data.rolling(5, min_periods=1).mean().shift(1)
        df.loc[mask, "vol_ma10"] = ticker_data.rolling(10, min_periods=1).mean().shift(1)
        df.loc[mask, "vol_std5"] = ticker_data.rolling(5, min_periods=2).std().shift(1)

    df["vol_ma5"] = df["vol_ma5"].fillna(df["target_log_var"].mean())
    df["vol_ma10"] = df["vol_ma10"].fillna(df["target_log_var"].mean())
    df["vol_std5"] = df["vol_std5"].fillna(0)
    df["news_x_retail"] = df["news_risk_score"] * df["retail_risk_score"]

    return df


def train_models(df):
    """
    Train RIVE-Full (10 feat) and RIVE-Core (7 feat, no agent signals)
    on pre-2023 data. Return predictions on test data.
    """
    cutoff = pd.to_datetime("2023-01-01")

    all_features = [
        "tech_pred", "news_risk_score", "retail_risk_score",
        "is_friday", "is_monday", "is_q4",
        "vol_ma5", "vol_ma10", "vol_std5", "news_x_retail",
    ]
    core_features = [
        "tech_pred",
        "is_friday", "is_monday", "is_q4",
        "vol_ma5", "vol_ma10", "vol_std5",
    ]

    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()

    train_df = train_df.dropna(subset=all_features + ["target_log_var"])
    test_df = test_df.dropna(subset=all_features + ["target_log_var"])

    y_train = train_df["target_log_var"].values
    y_test = test_df["target_log_var"].values

    valid_train = np.isfinite(y_train)
    valid_test = np.isfinite(y_test)
    train_df = train_df[valid_train]
    test_df = test_df[valid_test]
    y_train = y_train[valid_train]
    y_test = y_test[valid_test]

    # Winsorize
    lower = np.percentile(y_train, 2)
    upper = np.percentile(y_train, 98)
    y_train_w = np.clip(y_train, lower, upper)

    # --- RIVE-Full (10 features) ---
    model_full = Ridge(alpha=100.0)
    model_full.fit(train_df[all_features].fillna(0).values, y_train_w)
    pred_full = model_full.predict(test_df[all_features].fillna(0).values)

    # --- RIVE-Core (7 features, no agent signals) ---
    model_core = Ridge(alpha=100.0)
    model_core.fit(train_df[core_features].fillna(0).values, y_train_w)
    pred_core = model_core.predict(test_df[core_features].fillna(0).values)

    test_df = test_df.copy()
    test_df["pred_full"] = pred_full
    test_df["pred_core"] = pred_core
    test_df["actual"] = y_test

    # Squared errors
    test_df["se_full"] = (test_df["actual"] - test_df["pred_full"]) ** 2
    test_df["se_core"] = (test_df["actual"] - test_df["pred_core"]) ** 2
    # Absolute errors
    test_df["ae_full"] = np.abs(test_df["actual"] - test_df["pred_full"])
    test_df["ae_core"] = np.abs(test_df["actual"] - test_df["pred_core"])

    print(f"  Train: {len(train_df):,} obs")
    print(f"  Test:  {len(test_df):,} obs")
    print(f"  Full R²:  {r2_score(y_test, pred_full)*100:.2f}%")
    print(f"  Core R²:  {r2_score(y_test, pred_core)*100:.2f}%")

    return test_df, model_full, model_core


def evaluate_regime(test_df, mask, regime_name):
    """Evaluate Full vs Core within a regime subset."""
    sub = test_df[mask]
    n = len(sub)
    if n < 50:
        return None

    actual = sub["actual"].values
    pred_full = sub["pred_full"].values
    pred_core = sub["pred_core"].values

    r2_full = r2_score(actual, pred_full)
    r2_core = r2_score(actual, pred_core)
    mae_full = mean_absolute_error(actual, pred_full)
    mae_core = mean_absolute_error(actual, pred_core)

    # Paired t-test on squared errors (is Full significantly better than Core?)
    se_diff = sub["se_core"].values - sub["se_full"].values  # positive = Full is better
    t_stat, p_value = stats.ttest_1samp(se_diff, 0)

    # One-sided: Full better than Core
    p_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2

    return {
        "regime": regime_name,
        "n": n,
        "r2_full": r2_full * 100,
        "r2_core": r2_core * 100,
        "r2_diff": (r2_full - r2_core) * 100,
        "mae_full": mae_full,
        "mae_core": mae_core,
        "mae_diff": mae_core - mae_full,  # positive = Full is better
        "t_stat": t_stat,
        "p_value": p_value,
        "p_one_sided": p_one_sided,
        "full_better": r2_full > r2_core,
    }


def run_stratified_analysis(test_df):
    """Run the conditional regime analysis."""

    results = []

    # ── 1. Overall (baseline) ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  OVERALL")
    print("=" * 70)
    res = evaluate_regime(test_df, np.ones(len(test_df), dtype=bool), "Overall")
    results.append(res)
    print(f"  Full R²: {res['r2_full']:.2f}%  |  Core R²: {res['r2_core']:.2f}%  |  "
          f"Diff: {res['r2_diff']:+.2f}pp  |  t={res['t_stat']:.2f}  p={res['p_value']:.4f}")

    # ── 2. By news activity quartiles ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STRATIFIED BY NEWS ACTIVITY (news_risk_score quartiles)")
    print("=" * 70)

    news_q = pd.qcut(test_df["news_risk_score"], q=4, labels=False, duplicates="drop")
    news_labels = {0: "Q1 (Low News)", 1: "Q2", 2: "Q3", 3: "Q4 (High News)"}

    for q in sorted(news_q.unique()):
        mask = news_q == q
        label = news_labels.get(q, f"Q{q+1}")
        res = evaluate_regime(test_df, mask, f"News {label}")
        if res:
            results.append(res)
            sig = "***" if res["p_one_sided"] < 0.001 else \
                  "**" if res["p_one_sided"] < 0.01 else \
                  "*" if res["p_one_sided"] < 0.05 else ""
            better = "FULL" if res["full_better"] else "CORE"
            print(f"  {label:20s}: N={res['n']:>6,}  |  Full {res['r2_full']:6.2f}%  "
                  f"Core {res['r2_core']:6.2f}%  |  Diff {res['r2_diff']:+6.2f}pp  "
                  f"|  t={res['t_stat']:+6.2f} {sig:3s}  |  {better} wins")

    # ── 3. By retail activity quartiles ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STRATIFIED BY RETAIL ACTIVITY (retail_risk_score quartiles)")
    print("=" * 70)

    retail_q = pd.qcut(test_df["retail_risk_score"], q=4, labels=False, duplicates="drop")
    retail_labels = {0: "Q1 (Low Retail)", 1: "Q2", 2: "Q3", 3: "Q4 (High Retail)"}

    for q in sorted(retail_q.unique()):
        mask = retail_q == q
        label = retail_labels.get(q, f"Q{q+1}")
        res = evaluate_regime(test_df, mask, f"Retail {label}")
        if res:
            results.append(res)
            sig = "***" if res["p_one_sided"] < 0.001 else \
                  "**" if res["p_one_sided"] < 0.01 else \
                  "*" if res["p_one_sided"] < 0.05 else ""
            better = "FULL" if res["full_better"] else "CORE"
            print(f"  {label:20s}: N={res['n']:>6,}  |  Full {res['r2_full']:6.2f}%  "
                  f"Core {res['r2_core']:6.2f}%  |  Diff {res['r2_diff']:+6.2f}pp  "
                  f"|  t={res['t_stat']:+6.2f} {sig:3s}  |  {better} wins")

    # ── 4. By realized volatility level ───────────────────────────────────
    print("\n" + "=" * 70)
    print("  STRATIFIED BY VOLATILITY LEVEL (lagged vol_ma5 quartiles)")
    print("=" * 70)

    vol_q = pd.qcut(test_df["vol_ma5"], q=4, labels=False, duplicates="drop")
    vol_labels = {0: "Q1 (Low Vol)", 1: "Q2", 2: "Q3", 3: "Q4 (High Vol)"}

    for q in sorted(vol_q.unique()):
        mask = vol_q == q
        label = vol_labels.get(q, f"Q{q+1}")
        res = evaluate_regime(test_df, mask, f"Vol {label}")
        if res:
            results.append(res)
            sig = "***" if res["p_one_sided"] < 0.001 else \
                  "**" if res["p_one_sided"] < 0.01 else \
                  "*" if res["p_one_sided"] < 0.05 else ""
            better = "FULL" if res["full_better"] else "CORE"
            print(f"  {label:20s}: N={res['n']:>6,}  |  Full {res['r2_full']:6.2f}%  "
                  f"Core {res['r2_core']:6.2f}%  |  Diff {res['r2_diff']:+6.2f}pp  "
                  f"|  t={res['t_stat']:+6.2f} {sig:3s}  |  {better} wins")

    # ── 5. Combined stress regimes ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMBINED REGIME ANALYSIS")
    print("=" * 70)

    # Define stress as top quartile of each signal
    news_high = test_df["news_risk_score"] >= test_df["news_risk_score"].quantile(0.75)
    retail_high = test_df["retail_risk_score"] >= test_df["retail_risk_score"].quantile(0.75)
    vol_high = test_df["vol_ma5"] >= test_df["vol_ma5"].quantile(0.75)

    combined_regimes = {
        "Calm (all low)":            ~news_high & ~retail_high & ~vol_high,
        "News-driven only":           news_high & ~retail_high,
        "Retail-driven only":        ~news_high &  retail_high,
        "Dual signal (N+R)":          news_high &  retail_high,
        "High vol only":              vol_high  & ~news_high & ~retail_high,
        "High vol + news":            vol_high  &  news_high,
        "High vol + retail":          vol_high  &  retail_high,
        "Triple stress (all high)":   vol_high  &  news_high & retail_high,
    }

    for regime_name, mask in combined_regimes.items():
        res = evaluate_regime(test_df, mask, regime_name)
        if res:
            results.append(res)
            sig = "***" if res["p_one_sided"] < 0.001 else \
                  "**" if res["p_one_sided"] < 0.01 else \
                  "*" if res["p_one_sided"] < 0.05 else ""
            better = "FULL" if res["full_better"] else "CORE"
            print(f"  {regime_name:28s}: N={res['n']:>6,}  |  Full {res['r2_full']:6.2f}%  "
                  f"Core {res['r2_core']:6.2f}%  |  Diff {res['r2_diff']:+6.2f}pp  "
                  f"|  t={res['t_stat']:+6.2f} {sig:3s}  |  {better} wins")
        else:
            print(f"  {regime_name:28s}: N < 50, skipped")

    # ── 6. Monotonicity test: does agent benefit increase with stress? ────
    print("\n" + "=" * 70)
    print("  MONOTONICITY: Does agent benefit increase with stress level?")
    print("=" * 70)

    # Create a composite stress score = news_z + retail_z + vol_z
    news_z = (test_df["news_risk_score"] - test_df["news_risk_score"].mean()) / test_df["news_risk_score"].std()
    retail_z = (test_df["retail_risk_score"] - test_df["retail_risk_score"].mean()) / test_df["retail_risk_score"].std()
    vol_z = (test_df["vol_ma5"] - test_df["vol_ma5"].mean()) / test_df["vol_ma5"].std()
    stress_score = news_z + retail_z + vol_z

    stress_q = pd.qcut(stress_score, q=5, labels=False, duplicates="drop")
    stress_labels = {0: "Q1 (Calmest)", 1: "Q2", 2: "Q3", 3: "Q4", 4: "Q5 (Most Stressed)"}

    quintile_diffs = []
    for q in sorted(stress_q.unique()):
        mask = stress_q == q
        label = stress_labels.get(q, f"Q{q+1}")
        res = evaluate_regime(test_df, mask, f"Stress {label}")
        if res:
            results.append(res)
            quintile_diffs.append(res["r2_diff"])
            sig = "***" if res["p_one_sided"] < 0.001 else \
                  "**" if res["p_one_sided"] < 0.01 else \
                  "*" if res["p_one_sided"] < 0.05 else ""
            better = "FULL" if res["full_better"] else "CORE"
            print(f"  {label:20s}: N={res['n']:>6,}  |  Full {res['r2_full']:6.2f}%  "
                  f"Core {res['r2_core']:6.2f}%  |  Diff {res['r2_diff']:+6.2f}pp  "
                  f"|  t={res['t_stat']:+6.2f} {sig:3s}  |  {better} wins")

    if len(quintile_diffs) >= 3:
        # Spearman correlation: stress quintile vs R² advantage
        quintile_ranks = np.arange(len(quintile_diffs))
        rho, p_mono = stats.spearmanr(quintile_ranks, quintile_diffs)
        print(f"\n  Spearman correlation (stress quintile vs Full advantage):")
        print(f"    rho = {rho:+.3f}, p = {p_mono:.4f}")
        if rho > 0 and p_mono < 0.05:
            print("    → Agent signals provide INCREASING benefit under stress ✓")
        elif rho > 0:
            print("    → Positive trend but not statistically significant")
        else:
            print("    → No evidence of increasing benefit under stress")

    return results


def save_results(results):
    """Save results to CSV."""
    output_path = Path(__file__).parent / "conditional_regime_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"\n  Results saved to {output_path}")


def print_summary(results):
    """Print a clear summary."""
    print("\n" + "=" * 70)
    print("  SUMMARY: Where do agent signals help?")
    print("=" * 70)

    full_wins = [r for r in results if r and r["full_better"]]
    core_wins = [r for r in results if r and not r["full_better"]]
    sig_full = [r for r in results if r and r["full_better"] and r["p_one_sided"] < 0.05]
    sig_core = [r for r in results if r and not r["full_better"] and r["p_one_sided"] < 0.05]

    print(f"\n  Total regime slices tested: {len(results)}")
    print(f"  Full wins (R² higher):  {len(full_wins)} "
          f"({len(sig_full)} significant at p<0.05)")
    print(f"  Core wins (R² higher):  {len(core_wins)} "
          f"({len(sig_core)} significant at p<0.05)")

    if sig_full:
        print(f"\n  Regimes where Full SIGNIFICANTLY outperforms Core (p<0.05):")
        for r in sig_full:
            print(f"    {r['regime']:30s}: +{r['r2_diff']:.2f}pp  "
                  f"(t={r['t_stat']:+.2f}, p={r['p_one_sided']:.4f})")

    if sig_core:
        print(f"\n  Regimes where Core SIGNIFICANTLY outperforms Full (p<0.05):")
        for r in sig_core:
            print(f"    {r['regime']:30s}: {r['r2_diff']:.2f}pp  "
                  f"(t={r['t_stat']:+.2f}, p={r['p_one_sided']:.4f})")

    # Key finding for paper
    print("\n" + "-" * 70)
    print("  KEY FINDING FOR PAPER:")
    print("-" * 70)

    # Check the hypothesis: Full beats Core in high-stress regimes
    stress_results = [r for r in results
                      if r and any(kw in r["regime"] for kw in
                                   ["High", "Q4", "Q5", "Dual", "Triple", "stress"])]
    stress_full_wins = [r for r in stress_results if r["full_better"]]

    if len(stress_full_wins) > len(stress_results) / 2:
        print("  HYPOTHESIS SUPPORTED: Agent signals tend to help during stress periods.")
        print(f"  Full model wins in {len(stress_full_wins)}/{len(stress_results)} "
              "stress-related regimes.")
    else:
        print("  HYPOTHESIS NOT SUPPORTED: Agent signals do not consistently")
        print(f"  help during stress. Full wins in only "
              f"{len(stress_full_wins)}/{len(stress_results)} stress regimes.")


if __name__ == "__main__":
    print("=" * 70)
    print("  TEST 03: CONDITIONAL REGIME PERFORMANCE")
    print("  Do agent signals help during stress periods?")
    print("=" * 70)

    print("\nLoading data...")
    df = load_and_prepare()
    print(f"  {len(df):,} rows")

    print("\nTraining Full (10 feat) vs Core (7 feat) models...")
    test_df, model_full, model_core = train_models(df)

    print("\nRunning stratified analysis...")
    results = run_stratified_analysis(test_df)

    save_results(results)
    print_summary(results)

    print("\n" + "=" * 70)
    print("  TEST 03 COMPLETE")
    print("=" * 70)
