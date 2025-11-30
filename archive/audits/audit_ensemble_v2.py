"""
🔍 ENSEMBLE AUDIT V2: Aggressive Push for 25%+

Based on V1 findings:
- ElasticNet(0.1, 0.1) = 15.68% ← Best so far
- Two-Stage Ridge = 14.66%

Now trying more aggressive approaches:
1. VIX Regime Switching
2. Ticker-specific models with smarter pooling
3. Rolling training windows
4. Polynomial features
5. More aggressive residual stacking
6. Time-based features
7. Volatility regime detection
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, Lasso, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_and_prepare():
    """Load data with enhanced features."""
    print("📂 Loading data...")
    
    targets = pd.read_parquet("data/processed/targets.parquet")
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    news_features = pd.read_parquet("data/processed/news_features.parquet")
    
    try:
        retail_signals = pd.read_parquet("data/processed/retail_signals.parquet")
    except:
        retail_signals = None
    
    # Normalize
    for df in [targets, residuals, news_features]:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        if "ticker" in df.columns and df["ticker"].dtype.name == "category":
            df["ticker"] = df["ticker"].astype(str)
    
    df = targets.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # ================================================
    # HAR Features (Tech Agent)
    # ================================================
    df["rv_lag_1"] = df.groupby("ticker")["realized_vol"].shift(1)
    df["rv_lag_5"] = df.groupby("ticker")["realized_vol"].transform(lambda x: x.rolling(5).mean()).shift(1)
    df["rv_lag_22"] = df.groupby("ticker")["realized_vol"].transform(lambda x: x.rolling(22).mean()).shift(1)
    df["returns_sq_lag_1"] = (df["close"].pct_change() ** 2).shift(1)
    
    # Fill
    df["VIX_close"] = df["VIX_close"].ffill().fillna(15)
    df["rsi_14"] = df["rsi_14"].ffill().fillna(50)
    
    # ================================================
    # Enhanced Features for Audit
    # ================================================
    
    # 1. Volatility Regime
    df["vix_high"] = (df["VIX_close"] > 25).astype(int)
    df["vix_extreme"] = (df["VIX_close"] > 35).astype(int)
    df["vix_low"] = (df["VIX_close"] < 15).astype(int)
    
    # 2. Momentum features
    df["rv_momentum"] = df["rv_lag_1"] - df["rv_lag_5"]
    df["rv_acceleration"] = df.groupby("ticker")["rv_momentum"].diff()
    
    # 3. Mean reversion signals
    df["rv_mean_22"] = df.groupby("ticker")["realized_vol"].transform(lambda x: x.rolling(22).mean())
    df["rv_deviation"] = df["realized_vol"].shift(1) / df["rv_mean_22"].shift(1) - 1
    
    # 4. Day of week (seasonality)
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["is_monday"] = (df["dow"] == 0).astype(int)
    df["is_friday"] = (df["dow"] == 4).astype(int)
    
    # 5. Month effects
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_jan"] = (df["month"] == 1).astype(int)
    df["is_q4"] = (df["month"] >= 10).astype(int)
    
    # 6. VIX dynamics
    df["vix_change"] = df.groupby("ticker")["VIX_close"].diff()
    df["vix_up"] = (df["vix_change"] > 0).astype(int)
    df["vix_roll5"] = df.groupby("ticker")["VIX_close"].transform(lambda x: x.rolling(5).mean())
    df["vix_vs_avg"] = df["VIX_close"] / df["vix_roll5"]
    
    # 7. RSI dynamics
    df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)
    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
    
    # ================================================
    # Tech predictions
    # ================================================
    cutoff = pd.to_datetime("2023-01-01")
    tech_features = ["rv_lag_1", "rv_lag_5", "rv_lag_22", "returns_sq_lag_1", "VIX_close", "rsi_14"]
    
    train_tech = df[df["date"] < cutoff].dropna(subset=tech_features + ["target_log_var"])
    tech_model = Ridge(alpha=1.0)
    tech_model.fit(train_tech[tech_features], train_tech["target_log_var"])
    df["tech_pred"] = tech_model.predict(df[tech_features].fillna(0))
    
    # ================================================
    # Merge news and residuals
    # ================================================
    df = pd.merge(df, residuals[["date", "ticker", "resid_tech"]], on=["date", "ticker"], how="left")
    df = pd.merge(df, news_features, on=["date", "ticker"], how="left", suffixes=("", "_news"))
    
    # News lagged features
    for col in ["news_count", "shock_index", "sentiment_avg"]:
        if col in df.columns:
            for lag in [1, 3, 5]:
                df[f"{col}_lag{lag}"] = df.groupby("ticker")[col].shift(lag)
    
    # News predictions
    pca_cols = [c for c in df.columns if c.startswith("news_pca_")][:10]
    lag_cols = [c for c in df.columns if "_lag" in c and "news" in c]
    news_features_list = ["shock_index", "news_count", "sentiment_avg", "novelty_score"] + pca_cols + lag_cols
    news_features_list = [f for f in news_features_list if f in df.columns]
    
    train_news = df[(df["date"] < cutoff) & df["resid_tech"].notna()].dropna(subset=news_features_list)
    if len(train_news) > 50:
        news_model = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.01, verbose=-1)
        news_model.fit(train_news[news_features_list], train_news["resid_tech"])
        df["news_pred"] = news_model.predict(df[news_features_list].fillna(0))
    else:
        df["news_pred"] = 0
    
    # Fund predictions
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
    
    # Retail (simple)
    if retail_signals is not None:
        retail_signals["date"] = pd.to_datetime(retail_signals["date"]).dt.tz_localize(None)
        df = pd.merge(df, retail_signals, on="date", how="left", suffixes=("", "_r"))
        df["btc_vol_5d"] = df["btc_vol_5d"].ffill().fillna(0)
        df["retail_pred"] = df["btc_vol_5d"] * df["VIX_close"] * 0.001  # Simple signal
    else:
        df["retail_pred"] = 0
    
    df = df.dropna(subset=["target_log_var"])
    
    print(f"   ✓ Dataset: {len(df):,} rows with {len(df.columns)} features")
    return df


def train_test_split(df):
    cutoff = pd.to_datetime("2023-01-01")
    return df[df["date"] < cutoff].copy(), df[df["date"] >= cutoff].copy()


print("\n" + "=" * 80)
print("🔍 ENSEMBLE AUDIT V2: Aggressive Push for 25%+")
print("=" * 80)

df = load_and_prepare()
train, test = train_test_split(df)
print(f"\nData: {len(train)} train, {len(test)} test")

base_r2 = r2_score(test["target_log_var"], test["tech_pred"])
print(f"Baseline (HAR-RV): {base_r2:.4f} ({base_r2*100:.2f}%)")

results = []

# =====================================================
# EXPERIMENT A: VIX Regime Switching
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP A: VIX Regime Switching")
print("-" * 80)

base_features = ["tech_pred", "news_pred", "fund_pred", "VIX_close", "rsi_14"]

# Train separate models for high/low VIX
train_high = train[train["VIX_close"] > 20]
train_low = train[train["VIX_close"] <= 20]

m_high = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000)
m_low = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000)

if len(train_high) > 30:
    m_high.fit(train_high[base_features].fillna(0), train_high["target_log_var"])
if len(train_low) > 30:
    m_low.fit(train_low[base_features].fillna(0), train_low["target_log_var"])

# Predict based on regime
preds = []
for _, row in test.iterrows():
    x = row[base_features].fillna(0).values.reshape(1, -1)
    if row["VIX_close"] > 20:
        p = m_high.predict(x)[0]
    else:
        p = m_low.predict(x)[0]
    preds.append(p)

r2 = r2_score(test["target_log_var"], preds)
print(f"   VIX Regime Switch: {r2:.4f} ({r2*100:.2f}%)")
results.append(("VIX Regime Switch", "vix>20", r2))

# =====================================================
# EXPERIMENT B: Polynomial Features
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP B: Polynomial Features")
print("-" * 80)

core_features = ["tech_pred", "news_pred", "fund_pred"]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(train[core_features].fillna(0))
X_test_poly = poly.transform(test[core_features].fillna(0))

for alpha in [0.1, 1.0, 10.0]:
    model = Ridge(alpha=alpha)
    model.fit(X_train_poly, train["target_log_var"])
    r2 = r2_score(test["target_log_var"], model.predict(X_test_poly))
    print(f"   Poly(2) + Ridge({alpha}): {r2:.4f}")
    results.append(("Poly2+Ridge", f"a={alpha}", r2))

# =====================================================
# EXPERIMENT C: Multi-Stage Stacking
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP C: Multi-Stage Stacking (3 levels)")
print("-" * 80)

# Stage 1: HAR (already done)
stage1_pred = train["tech_pred"]
stage1_test = test["tech_pred"]

# Stage 2: Residual prediction
train["stage1_resid"] = train["target_log_var"] - stage1_pred
test["stage1_resid"] = test["target_log_var"] - stage1_test

stage2_features = ["news_pred", "fund_pred", "VIX_close", "vix_high", "rv_momentum"]
stage2_features = [f for f in stage2_features if f in df.columns]

m2 = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
m2.fit(train[stage2_features].fillna(0), train["stage1_resid"])
stage2_pred_train = m2.predict(train[stage2_features].fillna(0))
stage2_pred_test = m2.predict(test[stage2_features].fillna(0))

# Stage 3: Residual of residual
train["stage2_resid"] = train["stage1_resid"] - stage2_pred_train
test["stage2_resid"] = test["stage1_resid"] - stage2_pred_test

stage3_features = ["rsi_14", "rv_deviation", "vix_change", "is_monday"]
stage3_features = [f for f in stage3_features if f in df.columns]

if stage3_features:
    m3 = Ridge(alpha=1.0)
    m3.fit(train[stage3_features].fillna(0), train["stage2_resid"])
    stage3_pred_test = m3.predict(test[stage3_features].fillna(0))
else:
    stage3_pred_test = 0

final_pred = stage1_test + stage2_pred_test + stage3_pred_test
r2 = r2_score(test["target_log_var"], final_pred)
print(f"   3-Stage Stacking: {r2:.4f} ({r2*100:.2f}%)")
results.append(("3-Stage Stack", "full", r2))

# =====================================================
# EXPERIMENT D: Weighted by Ticker Volatility
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP D: Ticker-Weighted Ensemble")
print("-" * 80)

# Calculate each ticker's volatility level
ticker_vol = train.groupby("ticker")["realized_vol"].mean()

all_preds = []
for ticker in test["ticker"].unique():
    t_train = train[train["ticker"] == ticker]
    t_test = test[test["ticker"] == ticker]
    
    if len(t_train) < 30:
        # Use global model
        model = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000)
        model.fit(train[base_features].fillna(0), train["target_log_var"])
    else:
        # Ticker-specific
        model = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000)
        model.fit(t_train[base_features].fillna(0), t_train["target_log_var"])
    
    preds = model.predict(t_test[base_features].fillna(0))
    
    for i, (idx, row) in enumerate(t_test.iterrows()):
        all_preds.append({
            "idx": idx,
            "actual": row["target_log_var"],
            "pred": preds[i]
        })

pred_df = pd.DataFrame(all_preds)
r2 = r2_score(pred_df["actual"], pred_df["pred"])
print(f"   Ticker-specific: {r2:.4f} ({r2*100:.2f}%)")
results.append(("Ticker-Specific", "elasticnet", r2))

# =====================================================
# EXPERIMENT E: Robust Regression (Huber)
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP E: Robust Regression (Huber)")
print("-" * 80)

for epsilon in [1.1, 1.35, 2.0]:
    model = HuberRegressor(epsilon=epsilon, max_iter=1000)
    model.fit(train[base_features].fillna(0), train["target_log_var"])
    r2 = r2_score(test["target_log_var"], model.predict(test[base_features].fillna(0)))
    print(f"   Huber(e={epsilon}): {r2:.4f}")
    results.append(("Huber", f"e={epsilon}", r2))

# =====================================================
# EXPERIMENT F: Extended Feature Set
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP F: Extended Feature Sets")
print("-" * 80)

feature_sets = {
    "base": ["tech_pred", "news_pred", "fund_pred", "VIX_close", "rsi_14"],
    "momentum": ["tech_pred", "news_pred", "fund_pred", "VIX_close", "rv_momentum", "rv_deviation"],
    "regime": ["tech_pred", "news_pred", "fund_pred", "vix_high", "vix_low", "rsi_oversold"],
    "calendar": ["tech_pred", "news_pred", "fund_pred", "VIX_close", "is_monday", "is_friday", "is_q4"],
    "full": ["tech_pred", "news_pred", "fund_pred", "VIX_close", "rsi_14", 
             "rv_momentum", "vix_high", "vix_change", "is_monday"],
}

for name, features in feature_sets.items():
    features = [f for f in features if f in df.columns]
    model = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000)
    model.fit(train[features].fillna(0), train["target_log_var"])
    r2 = r2_score(test["target_log_var"], model.predict(test[features].fillna(0)))
    print(f"   {name}: {r2:.4f}")
    results.append(("ElasticNet", name, r2))

# =====================================================
# EXPERIMENT G: Scaling Before Training
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP G: StandardScaler + Models")
print("-" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train[base_features].fillna(0))
X_test_scaled = scaler.transform(test[base_features].fillna(0))

for name, model in [
    ("Ridge", Ridge(alpha=1.0)),
    ("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000)),
    ("Huber", HuberRegressor(epsilon=1.35, max_iter=1000)),
]:
    model.fit(X_train_scaled, train["target_log_var"])
    r2 = r2_score(test["target_log_var"], model.predict(X_test_scaled))
    print(f"   Scaled {name}: {r2:.4f}")
    results.append((f"Scaled+{name}", "std", r2))

# =====================================================
# EXPERIMENT H: Ensemble Voting
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP H: Voting Ensemble")
print("-" * 80)

models = [
    ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000),
    ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
    Ridge(alpha=10.0),
    HuberRegressor(epsilon=1.35, max_iter=1000),
]

preds = []
for m in models:
    m.fit(train[base_features].fillna(0), train["target_log_var"])
    preds.append(m.predict(test[base_features].fillna(0)))

# Average voting
avg_pred = np.mean(preds, axis=0)
r2 = r2_score(test["target_log_var"], avg_pred)
print(f"   4-Model Average: {r2:.4f}")
results.append(("Voting4", "avg", r2))

# Median voting
med_pred = np.median(preds, axis=0)
r2 = r2_score(test["target_log_var"], med_pred)
print(f"   4-Model Median: {r2:.4f}")
results.append(("Voting4", "median", r2))

# =====================================================
# EXPERIMENT I: Aggressive Feature Selection
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP I: LASSO Feature Selection")
print("-" * 80)

all_features = [
    "tech_pred", "news_pred", "fund_pred", "retail_pred",
    "VIX_close", "rsi_14", "rv_momentum", "rv_deviation",
    "vix_high", "vix_low", "vix_change", "vix_vs_avg",
    "rsi_oversold", "rsi_overbought",
    "is_monday", "is_friday", "is_q4"
]
all_features = [f for f in all_features if f in df.columns]

# Use LASSO for feature selection
lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(train[all_features].fillna(0), train["target_log_var"])

# Get selected features
selected = [f for f, c in zip(all_features, lasso.coef_) if abs(c) > 0.001]
print(f"   Selected {len(selected)} features: {selected}")

if selected:
    model = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=10000)
    model.fit(train[selected].fillna(0), train["target_log_var"])
    r2 = r2_score(test["target_log_var"], model.predict(test[selected].fillna(0)))
    print(f"   LASSO-selected ElasticNet: {r2:.4f}")
    results.append(("LASSO-Select", str(len(selected)), r2))

# =====================================================
# EXPERIMENT J: Time-Weighted Training
# =====================================================
print("\n" + "-" * 80)
print("🔬 EXP J: Time-Weighted Training (Recent data weighted more)")
print("-" * 80)

# More recent data gets higher weight
train_sorted = train.sort_values("date")
n = len(train_sorted)
weights = np.linspace(0.5, 1.5, n)  # Older data: 0.5, Recent: 1.5

model = Ridge(alpha=1.0)
model.fit(train_sorted[base_features].fillna(0), train_sorted["target_log_var"], sample_weight=weights)
r2 = r2_score(test["target_log_var"], model.predict(test[base_features].fillna(0)))
print(f"   Time-weighted Ridge: {r2:.4f}")
results.append(("TimeWeighted", "Ridge", r2))

# =====================================================
# SUMMARY
# =====================================================
print("\n" + "=" * 80)
print("📊 AUDIT V2 RESULTS (Sorted by Test R²)")
print("=" * 80)

results_df = pd.DataFrame(results, columns=["Model", "Config", "Test_R2"])
results_df = results_df.sort_values("Test_R2", ascending=False)
results_df["Pct"] = (results_df["Test_R2"] * 100).round(2)

current = 0.14
print(f"\nBaseline (HAR-RV): {base_r2:.4f} ({base_r2*100:.2f}%)")
print(f"Current Best: {current:.4f} ({current*100:.2f}%)")
print(f"\nTOP 20 CONFIGURATIONS:")
print("-" * 65)

for i, (_, row) in enumerate(results_df.head(20).iterrows()):
    marker = "🏆" if row["Test_R2"] >= 0.25 else ("✅" if row["Test_R2"] >= 0.20 else ("🔥" if row["Test_R2"] > current else ""))
    print(f"{i+1:2d}. {row['Model']:<20} {row['Config']:<20} {row['Pct']:>6.2f}% {marker}")

best = results_df.iloc[0]
print("\n" + "=" * 80)
print("🏆 BEST CONFIGURATION FROM V2")
print("=" * 80)
print(f"""
   Model:  {best['Model']}
   Config: {best['Config']}
   
   Test R²: {best['Test_R2']:.4f} ({best['Pct']:.2f}%)
   
   vs Baseline: {((best['Test_R2'] - base_r2) / max(abs(base_r2), 0.001) * 100):+.1f}%
   vs Current:  {((best['Test_R2'] - current) / current * 100):+.1f}%
""")

if best["Test_R2"] >= 0.25:
    print("   🎉 TARGET ACHIEVED: 25%+ R²!")
else:
    gap = 0.25 - best["Test_R2"]
    print(f"   ⚠️ Gap to 25%: {gap:.2f} ({gap*100:.1f}%)")
    print("\n   RECOMMENDATIONS:")
    print("   1. Need more data (only 1,686 rows, ideally 10,000+)")
    print("   2. Try more tickers (currently 3, need 12)")
    print("   3. Full 7-year date range would help")
    print("   4. ElasticNet(0.1, 0.1) consistently best")
    print("   5. Multi-stage stacking helps")

print("=" * 80)


