"""
Brutal Decomposition: Isolate Alpha from Beta

This test answers the critical question:
"How much of our 30% R² is from REAL agent intelligence vs just calendar/VIX effects?"

The Waterfall:
- Model A: Pure Inertia (HAR baseline)
- Model B: Systematic Beta (Calendar + VIX - the "anyone could do this")
- Model C: Titan V8 Full (Agents added)

Alpha = Model C - Model B
If Alpha > 3%, our agents are REAL
If Alpha < 1%, our agents are NOISE

Usage:
    python scripts/final_decomposition.py
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent.parent))

np.random.seed(42)


def load_and_prepare_data():
    """Load targets and prepare all features."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    targets = pd.read_parquet("data/processed/targets.parquet")
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    news_features = pd.read_parquet("data/processed/news_features.parquet")
    
    for df in [targets, residuals, news_features]:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        if "ticker" in df.columns and df["ticker"].dtype.name == "category":
            df["ticker"] = df["ticker"].astype(str)
    
    df = targets.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # HAR features for tech_pred
    df["rv_lag_1"] = df.groupby("ticker")["realized_vol"].shift(1)
    df["rv_lag_5"] = df.groupby("ticker")["realized_vol"].transform(
        lambda x: x.rolling(5).mean()
    ).shift(1)
    df["rv_lag_22"] = df.groupby("ticker")["realized_vol"].transform(
        lambda x: x.rolling(22).mean()
    ).shift(1)
    df["returns_sq_lag_1"] = (df["close"].pct_change() ** 2).shift(1)
    
    df["VIX_close"] = df["VIX_close"].ffill().fillna(15)
    df["rsi_14"] = df["rsi_14"].ffill().fillna(50)
    
    # Calendar features
    df["is_friday"] = (df["date"].dt.dayofweek == 4).astype(int)
    df["is_monday"] = (df["date"].dt.dayofweek == 0).astype(int)
    df["is_q4"] = (df["date"].dt.quarter == 4).astype(int)
    
    # Generate tech_pred using HAR model trained on training set
    tech_features = ["rv_lag_1", "rv_lag_5", "rv_lag_22", "returns_sq_lag_1", "VIX_close", "rsi_14"]
    cutoff = pd.to_datetime("2023-01-01")
    
    train_tech = df[df["date"] < cutoff].dropna(subset=tech_features + ["target_log_var"])
    tech_model = Ridge(alpha=1.0)
    tech_model.fit(train_tech[tech_features], train_tech["target_log_var"])
    df["tech_pred"] = tech_model.predict(df[tech_features].fillna(0))
    
    # Merge residuals and news
    df = pd.merge(df, residuals[["date", "ticker", "resid_tech"]], on=["date", "ticker"], how="left")
    df = pd.merge(df, news_features, on=["date", "ticker"], how="left", suffixes=("", "_news"))
    
    # Create lagged news features for news_pred
    for col in ["news_count", "shock_index", "sentiment_avg"]:
        if col in df.columns:
            for lag in [1, 3, 5]:
                df[f"{col}_lag{lag}"] = df.groupby("ticker")[col].shift(lag)
    
    # News prediction
    from lightgbm import LGBMRegressor
    pca_cols = [c for c in df.columns if c.startswith("news_pca_")][:10]
    lag_cols = [c for c in df.columns if "_lag" in c and "news" in c]
    news_features_list = ["shock_index", "news_count", "sentiment_avg"] + pca_cols + lag_cols
    news_features_list = [f for f in news_features_list if f in df.columns]
    
    train_news = df[(df["date"] < cutoff) & df["resid_tech"].notna()].dropna(subset=news_features_list)
    if len(train_news) > 50:
        news_model = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.01, verbose=-1)
        news_model.fit(train_news[news_features_list], train_news["resid_tech"])
        df["news_pred"] = news_model.predict(df[news_features_list].fillna(0))
    else:
        df["news_pred"] = 0
    
    # Fund prediction
    df["debt_to_equity"] = df["debt_to_equity"].fillna(0)
    df["days_to_ex_div"] = df["days_to_ex_div"].fillna(365)
    df["debt_vix_interaction"] = df["debt_to_equity"] * df["VIX_close"]
    
    fund_features = ["debt_to_equity", "days_to_ex_div", "VIX_close", "debt_vix_interaction"]
    train_fund = df[(df["date"] < cutoff) & df["resid_tech"].notna()].dropna(subset=fund_features)
    if len(train_fund) > 50:
        fund_model = LGBMRegressor(n_estimators=200, max_depth=2, learning_rate=0.02, verbose=-1)
        fund_model.fit(train_fund[fund_features], train_fund["resid_tech"])
        df["fund_pred"] = fund_model.predict(df[fund_features].fillna(0))
    else:
        df["fund_pred"] = 0
    
    df["retail_pred"] = 0
    df = df.dropna(subset=["target_log_var"])
    
    print(f"   Loaded {len(df):,} rows")
    print(f"   Tickers: {df['ticker'].unique().tolist()}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


def train_model(X_train, y_train):
    """Train ElasticNet model."""
    model = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, name):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return {
        "name": name,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "coefficients": dict(zip(X_test.columns, model.coef_))
    }


def main():
    print("\n" + "=" * 70)
    print("BRUTAL DECOMPOSITION: ISOLATE ALPHA FROM BETA")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_and_prepare_data()
    
    # Train/Test Split
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    
    print(f"\n   Train: {len(train):,} samples (< 2023-01-01)")
    print(f"   Test:  {len(test):,} samples (>= 2023-01-01)")
    
    y_train = train["target_log_var"]
    y_test = test["target_log_var"]
    
    # ================================================================
    # MODEL A: Pure Inertia (HAR only)
    # ================================================================
    print("\n" + "=" * 70)
    print("MODEL A: PURE INERTIA (HAR Baseline)")
    print("=" * 70)
    
    features_a = ["tech_pred"]
    
    X_train_a = train[features_a].fillna(0)
    X_test_a = test[features_a].fillna(0)
    
    model_a = train_model(X_train_a, y_train)
    results_a = evaluate_model(model_a, X_test_a, y_test, "Model A: Pure Inertia")
    
    print(f"   Features: {features_a}")
    print(f"   Test R²:  {results_a['r2']:.4f} ({results_a['r2']*100:.2f}%)")
    print(f"   RMSE:     {results_a['rmse']:.4f}")
    
    # ================================================================
    # MODEL B: Systematic Beta (Calendar + VIX)
    # ================================================================
    print("\n" + "=" * 70)
    print("MODEL B: SYSTEMATIC BETA (Calendar + VIX)")
    print("=" * 70)
    print("   This is what ANY quant could build with public data.")
    
    features_b = ["tech_pred", "VIX_close", "is_friday", "is_monday", "is_q4"]
    
    X_train_b = train[features_b].fillna(0)
    X_test_b = test[features_b].fillna(0)
    
    model_b = train_model(X_train_b, y_train)
    results_b = evaluate_model(model_b, X_test_b, y_test, "Model B: Systematic Beta")
    
    lift_a_to_b = results_b['r2'] - results_a['r2']
    
    print(f"   Features: {features_b}")
    print(f"   Test R²:  {results_b['r2']:.4f} ({results_b['r2']*100:.2f}%)")
    print(f"   RMSE:     {results_b['rmse']:.4f}")
    print(f"\n   Lift from Model A: +{lift_a_to_b:.4f} (+{lift_a_to_b*100:.2f}%)")
    
    print("\n   Coefficients:")
    for feat, coef in results_b['coefficients'].items():
        print(f"      {feat:15s}: {coef:+.4f}")
    
    # ================================================================
    # MODEL C: Titan V8 Full (With Agents)
    # ================================================================
    print("\n" + "=" * 70)
    print("MODEL C: TITAN V8 (Full Ensemble with Agents)")
    print("=" * 70)
    print("   This includes our proprietary News and Fundamental agents.")
    
    features_c = ["tech_pred", "VIX_close", "is_friday", "is_monday", "is_q4", 
                  "news_pred", "fund_pred"]
    
    X_train_c = train[features_c].fillna(0)
    X_test_c = test[features_c].fillna(0)
    
    model_c = train_model(X_train_c, y_train)
    results_c = evaluate_model(model_c, X_test_c, y_test, "Model C: Titan V8")
    
    lift_b_to_c = results_c['r2'] - results_b['r2']
    lift_a_to_c = results_c['r2'] - results_a['r2']
    
    print(f"   Features: {features_c}")
    print(f"   Test R²:  {results_c['r2']:.4f} ({results_c['r2']*100:.2f}%)")
    print(f"   RMSE:     {results_c['rmse']:.4f}")
    print(f"\n   Lift from Model B: +{lift_b_to_c:.4f} (+{lift_b_to_c*100:.2f}%) <- THIS IS OUR ALPHA")
    print(f"   Lift from Model A: +{lift_a_to_c:.4f} (+{lift_a_to_c*100:.2f}%)")
    
    print("\n   Coefficients:")
    for feat, coef in results_c['coefficients'].items():
        marker = ""
        if feat in ["news_pred", "fund_pred"]:
            marker = " <- AGENT"
        print(f"      {feat:15s}: {coef:+.4f}{marker}")
    
    # ================================================================
    # WATERFALL SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("WATERFALL DECOMPOSITION")
    print("=" * 70)
    
    print(f"""
   ┌─────────────────────────────────────────────────────────────────────┐
   │                    BRUTAL DECOMPOSITION TABLE                       │
   ├─────────────────────────────────────────────────────────────────────┤
   │ Layer                  │ Features              │ R²      │ Lift    │
   ├────────────────────────┼───────────────────────┼─────────┼─────────┤
   │ A: Pure Inertia        │ tech_pred             │ {results_a['r2']:>6.2%}  │   -     │
   │                        │                       │         │         │
   │ B: Systematic Beta     │ + VIX, Calendar       │ {results_b['r2']:>6.2%}  │ {lift_a_to_b:>+6.2%}  │
   │    (Public Knowledge)  │                       │         │         │
   │                        │                       │         │         │
   │ C: Titan V8 Full       │ + news_pred, fund_pred│ {results_c['r2']:>6.2%}  │ {lift_b_to_c:>+6.2%}  │
   │    (Proprietary Alpha) │                       │         │         │
   └────────────────────────┴───────────────────────┴─────────┴─────────┘
    """)
    
    # ================================================================
    # VALIDATION CHECK
    # ================================================================
    print("=" * 70)
    print("ALPHA VALIDATION")
    print("=" * 70)
    
    alpha = lift_b_to_c
    
    print(f"\n   ALPHA (Lift from B to C): {alpha:.4f} ({alpha*100:.2f}%)")
    print()
    
    if alpha > 0.03:
        verdict = "AGENTS ARE REAL"
        emoji = "🏆"
        explanation = "The News and Fundamental agents add >3% R² beyond calendar/VIX."
        status = "VALIDATED"
    elif alpha > 0.01:
        verdict = "AGENTS ADD MARGINAL VALUE"
        emoji = "✅"
        explanation = "The agents add 1-3% R², which is meaningful but modest."
        status = "PARTIAL"
    else:
        verdict = "AGENTS ARE NOISE"
        emoji = "❌"
        explanation = "The agents add <1% R². The signal is from calendar/VIX, not agents."
        status = "FAILED"
    
    print(f"   {emoji} {verdict}")
    print(f"   {explanation}")
    
    # ================================================================
    # BREAKDOWN OF VALUE
    # ================================================================
    print("\n" + "=" * 70)
    print("VALUE ATTRIBUTION")
    print("=" * 70)
    
    total_lift = lift_a_to_c
    if total_lift > 0:
        beta_pct = (lift_a_to_b / total_lift) * 100
        alpha_pct = (lift_b_to_c / total_lift) * 100
    else:
        beta_pct = 0
        alpha_pct = 0
    
    print(f"""
   Total improvement over baseline: {total_lift:.4f} ({total_lift*100:.2f}%)
   
   Breakdown:
   ├── Beta (Calendar + VIX):     {lift_a_to_b:.4f} ({beta_pct:.1f}% of total lift)
   └── Alpha (Agent Intelligence): {lift_b_to_c:.4f} ({alpha_pct:.1f}% of total lift)
    """)
    
    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    print(f"""
   Model A (Inertia):      {results_a['r2']*100:>6.2f}% R²
   Model B (Beta):         {results_b['r2']*100:>6.2f}% R²  (+{lift_a_to_b*100:.2f}% from calendar/VIX)
   Model C (Titan V8):     {results_c['r2']*100:>6.2f}% R²  (+{lift_b_to_c*100:.2f}% from agents)
   
   {emoji} STATUS: {status}
   
   The 30% R² decomposes as:
   • {results_a['r2']*100:.1f}% from volatility inertia (HAR baseline)
   • {lift_a_to_b*100:.1f}% from systematic beta (VIX + Calendar effects)
   • {lift_b_to_c*100:.1f}% from proprietary alpha (News + Fundamental agents)
    """)
    
    print("=" * 70)
    
    # Save report
    report_path = Path("results/decomposition_report.md")
    with open(report_path, "w") as f:
        f.write(f"""# Brutal Decomposition Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

| Model | Features | Test R² | Lift |
|-------|----------|---------|------|
| A: Pure Inertia | tech_pred | {results_a['r2']:.4f} | - |
| B: Systematic Beta | + VIX, Calendar | {results_b['r2']:.4f} | +{lift_a_to_b:.4f} |
| C: Titan V8 | + Agents | {results_c['r2']:.4f} | +{lift_b_to_c:.4f} |

---

## Alpha Validation

**Alpha (Lift from B to C):** {alpha:.4f} ({alpha*100:.2f}%)

**Verdict:** {emoji} {verdict}

{explanation}

---

## Value Attribution

Total improvement over baseline: {total_lift:.4f} ({total_lift*100:.2f}%)

| Source | Contribution | % of Total |
|--------|--------------|------------|
| Beta (Calendar + VIX) | {lift_a_to_b:.4f} | {beta_pct:.1f}% |
| Alpha (Agents) | {lift_b_to_c:.4f} | {alpha_pct:.1f}% |

---

## Model Coefficients (Titan V8)

| Feature | Coefficient |
|---------|-------------|
""")
        for feat, coef in results_c['coefficients'].items():
            f.write(f"| {feat} | {coef:+.4f} |\n")
        
        f.write(f"""
---

## Conclusion

The 30% R² decomposes as:
- **{results_a['r2']*100:.1f}%** from volatility inertia (HAR baseline)
- **{lift_a_to_b*100:.1f}%** from systematic beta (VIX + Calendar effects)
- **{lift_b_to_c*100:.1f}%** from proprietary alpha (News + Fundamental agents)

---

*Generated by Brutal Decomposition Analysis*
""")
    
    print(f"   Report saved to: {report_path}")
    print("=" * 70)
    
    return {
        "model_a": results_a,
        "model_b": results_b,
        "model_c": results_c,
        "lift_a_to_b": lift_a_to_b,
        "lift_b_to_c": lift_b_to_c,
        "alpha": alpha,
        "status": status
    }


if __name__ == "__main__":
    main()

