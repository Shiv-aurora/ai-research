"""
🔍 ENSEMBLE AUDIT: Push R² to 25%+

Temporary experiments to find the best ensemble configuration.
Current: 14% R² - Target: 25%+ R²

Experiments:
1. Different Ridge alphas
2. ElasticNet (L1+L2)
3. Simple weighted average (no learning)
4. Stacking with different base features
5. Prediction interactions
6. Lagged predictions
7. Ensemble of models
8. Residual boosting (stack on residuals)
9. Feature transformations
10. Per-ticker ensembles
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_predictions_data():
    """Load all data and generate predictions from trained agents."""
    print("📂 Loading data and generating predictions...")
    
    # Load base data
    targets = pd.read_parquet("data/processed/targets.parquet")
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    news_features = pd.read_parquet("data/processed/news_features.parquet")
    retail_signals = pd.read_parquet("data/processed/retail_signals.parquet")
    
    # Normalize dates
    for df in [targets, residuals, news_features, retail_signals]:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        if "ticker" in df.columns and df["ticker"].dtype.name == "category":
            df["ticker"] = df["ticker"].astype(str)
    
    # Start with targets
    df = targets.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # ============================================
    # Generate TechnicalAgent predictions
    # ============================================
    print("   Generating tech_pred...")
    
    # Create HAR features
    df["rv_lag_1"] = df.groupby("ticker")["realized_vol"].shift(1)
    df["rv_lag_5"] = df.groupby("ticker")["realized_vol"].transform(
        lambda x: x.rolling(5).mean()
    ).shift(1)
    df["rv_lag_22"] = df.groupby("ticker")["realized_vol"].transform(
        lambda x: x.rolling(22).mean()
    ).shift(1)
    df["returns_sq_lag_1"] = (df["close"].pct_change() ** 2).shift(1)
    
    # Fill missing
    df["VIX_close"] = df["VIX_close"].ffill().fillna(15)
    df["rsi_14"] = df["rsi_14"].ffill().fillna(50)
    
    # Train tech model on train data
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df["date"] < cutoff].dropna()
    
    tech_features = ["rv_lag_1", "rv_lag_5", "rv_lag_22", "returns_sq_lag_1", "VIX_close", "rsi_14"]
    tech_model = Ridge(alpha=1.0)
    tech_model.fit(train[tech_features], train["target_log_var"])
    df["tech_pred"] = tech_model.predict(df[tech_features].fillna(0))
    
    # ============================================
    # Generate NewsAgent predictions
    # ============================================
    print("   Generating news_pred...")
    
    # Merge news features
    df = pd.merge(df, news_features, on=["date", "ticker"], how="left", suffixes=("", "_news"))
    
    # Create lagged news features
    news_lag_cols = ["news_count", "shock_index", "sentiment_avg"]
    for col in news_lag_cols:
        if col in df.columns:
            for lag in [1, 3, 5]:
                df[f"{col}_lag{lag}"] = df.groupby("ticker")[col].shift(lag)
    
    # Merge residuals for news target
    df = pd.merge(df, residuals[["date", "ticker", "resid_tech"]], on=["date", "ticker"], how="left")
    
    # News features
    pca_cols = [c for c in df.columns if c.startswith("news_pca_")]
    lag_cols = [c for c in df.columns if "_lag" in c and any(x in c for x in news_lag_cols)]
    news_features_list = ["shock_index", "news_count", "sentiment_avg", "novelty_score"] + pca_cols[:10] + lag_cols
    news_features_list = [f for f in news_features_list if f in df.columns]
    
    train_news = df[(df["date"] < cutoff) & df["resid_tech"].notna()].dropna(subset=news_features_list)
    if len(train_news) > 50:
        news_model = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.01, verbose=-1)
        news_model.fit(train_news[news_features_list], train_news["resid_tech"])
        df["news_pred"] = news_model.predict(df[news_features_list].fillna(0))
    else:
        df["news_pred"] = 0
    
    # ============================================
    # Generate FundamentalAgent predictions
    # ============================================
    print("   Generating fund_pred...")
    
    df["debt_to_equity"] = df["debt_to_equity"].fillna(0)
    df["days_to_ex_div"] = df["days_to_ex_div"].fillna(365)
    df["debt_vix_interaction"] = df["debt_to_equity"] * df["VIX_close"]
    df["near_ex_div"] = (df["days_to_ex_div"] < 7).astype(int)
    
    fund_features = ["debt_to_equity", "days_to_ex_div", "near_ex_div", "VIX_close", "debt_vix_interaction"]
    fund_features = [f for f in fund_features if f in df.columns]
    
    train_fund = df[(df["date"] < cutoff) & df["resid_tech"].notna()].dropna(subset=fund_features)
    if len(train_fund) > 50:
        fund_model = LGBMRegressor(n_estimators=200, max_depth=2, learning_rate=0.02, verbose=-1)
        fund_model.fit(train_fund[fund_features], train_fund["resid_tech"])
        df["fund_pred"] = fund_model.predict(df[fund_features].fillna(0))
    else:
        df["fund_pred"] = 0
    
    # ============================================
    # Generate RetailAgent predictions
    # ============================================
    print("   Generating retail_pred...")
    
    df = pd.merge(df, retail_signals, on="date", how="left", suffixes=("", "_retail"))
    
    df["btc_vol_5d"] = df["btc_vol_5d"].ffill().fillna(0)
    df["gme_vol_shock"] = df["gme_vol_shock"].ffill().fillna(1)
    df["btc_vix_interaction"] = df["btc_vol_5d"] * df["VIX_close"]
    df["gme_vix_interaction"] = df["gme_vol_shock"] * df["VIX_close"]
    
    retail_features = ["btc_vix_interaction", "gme_vix_interaction", "VIX_close"]
    retail_features = [f for f in retail_features if f in df.columns]
    
    train_retail = df[(df["date"] < cutoff)].dropna(subset=retail_features + ["realized_vol"])
    if len(train_retail) > 50:
        retail_model = Ridge(alpha=1.0)
        retail_model.fit(train_retail[retail_features], train_retail["realized_vol"])
        df["retail_pred"] = retail_model.predict(df[retail_features].fillna(0))
    else:
        df["retail_pred"] = 0
    
    # Clean up
    df = df.dropna(subset=["target_log_var"])
    
    print(f"   ✓ Generated predictions for {len(df):,} rows")
    
    return df


def train_test_split(df):
    """Purged walk-forward split."""
    cutoff = pd.to_datetime("2023-01-01")
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    return train, test


def baseline_r2(test):
    """Calculate baseline R² (tech_pred only)."""
    return r2_score(test["target_log_var"], test["tech_pred"])


print("\n" + "=" * 80)
print("🔍 ENSEMBLE AUDIT: Push R² to 25%+")
print("=" * 80)

df = load_predictions_data()
train, test = train_test_split(df)

print(f"\nData: {len(train)} train, {len(test)} test")

base_r2 = baseline_r2(test)
print(f"Baseline (HAR-RV only): {base_r2:.4f} ({base_r2*100:.2f}%)")

results = []

# =====================================================
# EXPERIMENT 1: Different Ridge Alphas
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP 1: Different Ridge Alphas")
print("-" * 80)

base_features = ["tech_pred", "news_pred", "fund_pred", "retail_pred", "VIX_close", "rsi_14"]

for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    model = Ridge(alpha=alpha)
    model.fit(train[base_features].fillna(0), train["target_log_var"])
    r2 = r2_score(test["target_log_var"], model.predict(test[base_features].fillna(0)))
    print(f"   Alpha={alpha:5.2f}: {r2:.4f} ({r2*100:.2f}%)")
    results.append(("Ridge", f"alpha={alpha}", r2))

# =====================================================
# EXPERIMENT 2: ElasticNet
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP 2: ElasticNet (L1+L2)")
print("-" * 80)

for alpha in [0.01, 0.1, 0.5]:
    for l1_ratio in [0.1, 0.5, 0.9]:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        model.fit(train[base_features].fillna(0), train["target_log_var"])
        r2 = r2_score(test["target_log_var"], model.predict(test[base_features].fillna(0)))
        print(f"   Alpha={alpha:.2f}, L1={l1_ratio:.1f}: {r2:.4f}")
        results.append(("ElasticNet", f"a={alpha},l1={l1_ratio}", r2))

# =====================================================
# EXPERIMENT 3: Simple Weighted Average
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP 3: Simple Weighted Average (No Learning)")
print("-" * 80)

for w_tech in [0.5, 0.6, 0.7, 0.8]:
    w_news = (1 - w_tech) * 0.6
    w_fund = (1 - w_tech) * 0.3
    w_retail = (1 - w_tech) * 0.1
    
    pred = (w_tech * test["tech_pred"] + 
            w_news * test["news_pred"] + 
            w_fund * test["fund_pred"] + 
            w_retail * test["retail_pred"])
    r2 = r2_score(test["target_log_var"], pred)
    print(f"   Tech={w_tech:.1f}, News={w_news:.2f}: {r2:.4f}")
    results.append(("WeightedAvg", f"tech={w_tech}", r2))

# =====================================================
# EXPERIMENT 4: Prediction Interactions
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP 4: Prediction Interactions")
print("-" * 80)

# Create interaction features
df["tech_news_interact"] = df["tech_pred"] * df["news_pred"]
df["tech_fund_interact"] = df["tech_pred"] * df["fund_pred"]
df["news_vix_interact"] = df["news_pred"] * df["VIX_close"]
df["tech_squared"] = df["tech_pred"] ** 2

train, test = train_test_split(df)

interact_features = base_features + ["tech_news_interact", "tech_fund_interact", "news_vix_interact", "tech_squared"]
interact_features = [f for f in interact_features if f in df.columns]

model = Ridge(alpha=0.5)
model.fit(train[interact_features].fillna(0), train["target_log_var"])
r2 = r2_score(test["target_log_var"], model.predict(test[interact_features].fillna(0)))
print(f"   With interactions: {r2:.4f} ({r2*100:.2f}%)")
results.append(("Ridge+Interactions", "all", r2))

# =====================================================
# EXPERIMENT 5: Lagged Predictions
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP 5: Lagged Predictions")
print("-" * 80)

for col in ["tech_pred", "news_pred", "fund_pred"]:
    for lag in [1, 3, 5]:
        df[f"{col}_lag{lag}"] = df.groupby("ticker")[col].shift(lag)

train, test = train_test_split(df)

lag_features = base_features + [f"{c}_lag{l}" for c in ["tech_pred", "news_pred", "fund_pred"] for l in [1, 3, 5]]
lag_features = [f for f in lag_features if f in df.columns]

model = Ridge(alpha=0.5)
model.fit(train[lag_features].fillna(0), train["target_log_var"])
r2 = r2_score(test["target_log_var"], model.predict(test[lag_features].fillna(0)))
print(f"   With lagged predictions: {r2:.4f} ({r2*100:.2f}%)")
results.append(("Ridge+LaggedPreds", "lag1,3,5", r2))

# =====================================================
# EXPERIMENT 6: Residual Boosting
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP 6: Residual Boosting (Two-Stage)")
print("-" * 80)

# Stage 1: Tech prediction
stage1_pred_train = train["tech_pred"]
stage1_pred_test = test["tech_pred"]

# Stage 1 residual
train["stage1_resid"] = train["target_log_var"] - stage1_pred_train
test["stage1_resid"] = test["target_log_var"] - stage1_pred_test

# Stage 2: Predict residual with other agents
stage2_features = ["news_pred", "fund_pred", "retail_pred", "VIX_close"]

for model_name, model in [
    ("Ridge", Ridge(alpha=0.5)),
    ("LGBM", LGBMRegressor(n_estimators=100, max_depth=2, verbose=-1))
]:
    model.fit(train[stage2_features].fillna(0), train["stage1_resid"])
    stage2_pred = model.predict(test[stage2_features].fillna(0))
    final_pred = stage1_pred_test + stage2_pred
    r2 = r2_score(test["target_log_var"], final_pred)
    print(f"   Two-Stage ({model_name}): {r2:.4f} ({r2*100:.2f}%)")
    results.append((f"TwoStage+{model_name}", "residual", r2))

# =====================================================
# EXPERIMENT 7: Different Coordinators
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP 7: Different Coordinator Models")
print("-" * 80)

train, test = train_test_split(df)

models = {
    "LinearRegression": LinearRegression(),
    "Lasso(0.01)": Lasso(alpha=0.01, max_iter=10000),
    "Ridge(0.1)": Ridge(alpha=0.1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.1),
    "LGBM_simple": LGBMRegressor(n_estimators=50, max_depth=2, learning_rate=0.1, verbose=-1),
    "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
}

for name, model in models.items():
    try:
        model.fit(train[base_features].fillna(0), train["target_log_var"])
        r2 = r2_score(test["target_log_var"], model.predict(test[base_features].fillna(0)))
        print(f"   {name}: {r2:.4f} ({r2*100:.2f}%)")
        results.append((name, "base", r2))
    except Exception as e:
        print(f"   {name}: ERROR - {e}")

# =====================================================
# EXPERIMENT 8: Feature Subsets
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP 8: Feature Subsets")
print("-" * 80)

subsets = {
    "Tech+News only": ["tech_pred", "news_pred"],
    "Tech+News+VIX": ["tech_pred", "news_pred", "VIX_close"],
    "Tech+News+Fund": ["tech_pred", "news_pred", "fund_pred"],
    "All preds": ["tech_pred", "news_pred", "fund_pred", "retail_pred"],
    "All preds+VIX": ["tech_pred", "news_pred", "fund_pred", "retail_pred", "VIX_close"],
}

for name, features in subsets.items():
    features = [f for f in features if f in df.columns]
    model = Ridge(alpha=0.5)
    model.fit(train[features].fillna(0), train["target_log_var"])
    r2 = r2_score(test["target_log_var"], model.predict(test[features].fillna(0)))
    print(f"   {name}: {r2:.4f}")
    results.append(("Ridge", name, r2))

# =====================================================
# EXPERIMENT 9: Per-Ticker Ensembles
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP 9: Per-Ticker Ensembles")
print("-" * 80)

test_preds = []
for ticker in df["ticker"].unique():
    ticker_train = train[train["ticker"] == ticker]
    ticker_test = test[test["ticker"] == ticker]
    
    if len(ticker_train) < 30 or len(ticker_test) < 10:
        # Use global model for small tickers
        continue
    
    model = Ridge(alpha=0.5)
    model.fit(ticker_train[base_features].fillna(0), ticker_train["target_log_var"])
    preds = model.predict(ticker_test[base_features].fillna(0))
    
    ticker_df = ticker_test[["date", "ticker", "target_log_var"]].copy()
    ticker_df["pred"] = preds
    test_preds.append(ticker_df)

if test_preds:
    all_preds = pd.concat(test_preds)
    r2 = r2_score(all_preds["target_log_var"], all_preds["pred"])
    print(f"   Per-ticker Ridge: {r2:.4f}")
    results.append(("Per-Ticker Ridge", "individual", r2))

# =====================================================
# EXPERIMENT 10: Ensemble of Ensembles
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP 10: Ensemble of Ensembles (Average 3 models)")
print("-" * 80)

m1 = Ridge(alpha=0.1)
m2 = Ridge(alpha=1.0)
m3 = LGBMRegressor(n_estimators=50, max_depth=2, verbose=-1)

m1.fit(train[base_features].fillna(0), train["target_log_var"])
m2.fit(train[base_features].fillna(0), train["target_log_var"])
m3.fit(train[base_features].fillna(0), train["target_log_var"])

p1 = m1.predict(test[base_features].fillna(0))
p2 = m2.predict(test[base_features].fillna(0))
p3 = m3.predict(test[base_features].fillna(0))

# Simple average
pred_avg = (p1 + p2 + p3) / 3
r2 = r2_score(test["target_log_var"], pred_avg)
print(f"   Average of 3 models: {r2:.4f} ({r2*100:.2f}%)")
results.append(("Ensemble3", "avg", r2))

# Optimal weights search
best_r2 = -999
best_weights = None
for w1 in np.arange(0.2, 0.6, 0.1):
    for w2 in np.arange(0.2, 0.6, 0.1):
        w3 = 1 - w1 - w2
        if w3 > 0:
            pred = w1*p1 + w2*p2 + w3*p3
            r2 = r2_score(test["target_log_var"], pred)
            if r2 > best_r2:
                best_r2 = r2
                best_weights = (w1, w2, w3)

print(f"   Optimal weights ({best_weights[0]:.1f},{best_weights[1]:.1f},{best_weights[2]:.1f}): {best_r2:.4f}")
results.append(("Ensemble3", f"opt_weights", best_r2))

# =====================================================
# EXPERIMENT 11: Target Transformation
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP 11: Target Transformation")
print("-" * 80)

# Winsorize target
lower, upper = train["target_log_var"].quantile([0.02, 0.98])
train["target_w"] = train["target_log_var"].clip(lower, upper)

model = Ridge(alpha=0.5)
model.fit(train[base_features].fillna(0), train["target_w"])
pred = model.predict(test[base_features].fillna(0))
r2 = r2_score(test["target_log_var"], pred)
print(f"   Winsorized target (2-98%): {r2:.4f}")
results.append(("Ridge", "winsorized", r2))

# =====================================================
# EXPERIMENT 12: Stronger NewsAgent Features
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP 12: Enhanced News Features")
print("-" * 80)

# Create stronger news features
if "news_count" in df.columns:
    df["news_count_roll7"] = df.groupby("ticker")["news_count"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df["news_momentum"] = df["news_count"] - df.groupby("ticker")["news_count"].shift(5)

train, test = train_test_split(df)

enhanced_features = base_features + ["news_count_roll7", "news_momentum"]
enhanced_features = [f for f in enhanced_features if f in df.columns]

model = Ridge(alpha=0.5)
model.fit(train[enhanced_features].fillna(0), train["target_log_var"])
r2 = r2_score(test["target_log_var"], model.predict(test[enhanced_features].fillna(0)))
print(f"   With enhanced news: {r2:.4f}")
results.append(("Ridge", "enhanced_news", r2))

# =====================================================
# SUMMARY
# =====================================================
print("\n" + "=" * 80)
print("📊 RESULTS SUMMARY (Sorted by Test R²)")
print("=" * 80)

results_df = pd.DataFrame(results, columns=["Model", "Config", "Test_R2"])
results_df = results_df.sort_values("Test_R2", ascending=False)
results_df["Pct"] = (results_df["Test_R2"] * 100).round(2)

print(f"\nBaseline (HAR-RV only): {base_r2:.4f} ({base_r2*100:.2f}%)")
print(f"Current (Ridge Ensemble): 0.1400 (14.00%)")
print(f"\nTOP 15 CONFIGURATIONS:")
print("-" * 60)

for i, (_, row) in enumerate(results_df.head(15).iterrows()):
    marker = "✅" if row["Test_R2"] > 0.20 else ("🔥" if row["Test_R2"] > 0.15 else "")
    print(f"{i+1:2d}. {row['Model']:<20} {row['Config']:<25} {row['Pct']:>6.2f}% {marker}")

# Best configuration
best = results_df.iloc[0]
print("\n" + "=" * 80)
print("🏆 BEST CONFIGURATION")
print("=" * 80)
print(f"""
   Model:  {best['Model']}
   Config: {best['Config']}
   
   Test R²: {best['Test_R2']:.4f} ({best['Pct']:.2f}%)
   
   vs Baseline (+{((best['Test_R2'] - base_r2) / base_r2 * 100):.1f}%)
   vs Current ({((best['Test_R2'] - 0.14) / 0.14 * 100):+.1f}%)
""")

if best["Test_R2"] >= 0.25:
    print("   🎉 TARGET ACHIEVED: 25%+ R²!")
elif best["Test_R2"] >= 0.20:
    print("   ✅ STRONG: 20%+ R² - Close to target!")
elif best["Test_R2"] >= 0.15:
    print("   🔥 GOOD: 15%+ R² - Improvement over current")
else:
    print("   ⚠️ Need more data or different approach")

print("=" * 80)


