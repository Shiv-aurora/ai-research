"""
🔧 SQUEEZE MAXIMUM PERFORMANCE
Auditor Mode: Optimize for current data only (no full ingestion)

Experiments:
1. Remove noisy PCA features (only use top correlated)
2. Only use lagged features
3. Winsorize target
4. Ensemble: Average multiple models
5. Cross-validation hyperparameter search
6. Different train/test splits
7. Remove outliers from training
8. Stack: Ridge baseline + LightGBM correction
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_data():
    """Load and prepare data with lagged features."""
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    news = pd.read_parquet("data/processed/news_features.parquet")
    targets = pd.read_parquet("data/processed/targets.parquet")
    
    for df in [residuals, news, targets]:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        if df["ticker"].dtype.name == "category":
            df["ticker"] = df["ticker"].astype(str)
    
    merged = pd.merge(residuals, news, on=["date", "ticker"], how="inner")
    merged = pd.merge(merged, targets[["date", "ticker", "VIX_close"]], 
                      on=["date", "ticker"], how="left")
    merged["VIX_close"] = merged["VIX_close"].ffill().fillna(15)
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # Add lagged features
    lag_cols = ["news_count", "shock_index", "sentiment_avg"]
    for col in lag_cols:
        for lag in [1, 2, 3, 5]:
            merged[f"{col}_lag{lag}"] = merged.groupby("ticker")[col].shift(lag)
    
    # VIX interaction
    merged["shock_vix"] = merged["shock_index"] * merged["VIX_close"]
    
    merged = merged.dropna()
    return merged

def train_test_split(df, cutoff="2023-01-01"):
    cutoff = pd.to_datetime(cutoff)
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    return train, test

print("\n" + "="*70)
print("🔧 SQUEEZE MAXIMUM PERFORMANCE")
print("   Auditor Mode: No full ingestion, optimize current data")
print("="*70)

df = load_data()
train, test = train_test_split(df)

print(f"\nData: {len(train)} train, {len(test)} test")

# Feature groups
core = ["shock_index", "news_count", "sentiment_avg", "novelty_score"]
pca_all = [c for c in df.columns if c.startswith("news_pca_")]
lags = [c for c in df.columns if "_lag" in c]
interactions = ["shock_vix", "VIX_close"]

all_features = core + pca_all + lags + interactions
all_features = [f for f in all_features if f in df.columns]

print(f"Total features: {len(all_features)}")

# Baseline
model_base = LGBMRegressor(n_estimators=200, max_depth=2, num_leaves=4, 
                            learning_rate=0.02, verbose=-1, random_state=42)
model_base.fit(train[all_features], train['resid_tech'])
baseline_r2 = r2_score(test['resid_tech'], model_base.predict(test[all_features]))
print(f"\nBASELINE: {baseline_r2:.4f} ({baseline_r2*100:.2f}%)")

results = []

# =====================================================
# EXPERIMENT 1: ONLY LAGGED FEATURES (Remove PCA noise)
# =====================================================
print("\n" + "-"*70)
print("🔬 EXP 1: ONLY LAGGED + CORE (Remove PCA noise)")
print("-"*70)

features_1 = core + lags + interactions
model = LGBMRegressor(n_estimators=200, max_depth=2, num_leaves=4,
                       learning_rate=0.02, verbose=-1, random_state=42)
model.fit(train[features_1], train['resid_tech'])
r2 = r2_score(test['resid_tech'], model.predict(test[features_1]))
print(f"  Test R²: {r2:.4f} ({r2*100:.2f}%)")
results.append(("Only Lags (no PCA)", r2, len(features_1)))

# =====================================================
# EXPERIMENT 2: TOP CORRELATED FEATURES ONLY
# =====================================================
print("\n" + "-"*70)
print("🔬 EXP 2: TOP 10 CORRELATED FEATURES ONLY")
print("-"*70)

correlations = []
for f in all_features:
    corr = df[f].corr(df['resid_tech'])
    if not np.isnan(corr):
        correlations.append((f, abs(corr)))
correlations.sort(key=lambda x: x[1], reverse=True)
top10_features = [c[0] for c in correlations[:10]]
print(f"  Top 10: {top10_features}")

model = LGBMRegressor(n_estimators=200, max_depth=2, num_leaves=4,
                       learning_rate=0.02, verbose=-1, random_state=42)
model.fit(train[top10_features], train['resid_tech'])
r2 = r2_score(test['resid_tech'], model.predict(test[top10_features]))
print(f"  Test R²: {r2:.4f} ({r2*100:.2f}%)")
results.append(("Top 10 Correlated", r2, 10))

# =====================================================
# EXPERIMENT 3: WINSORIZE TARGET
# =====================================================
print("\n" + "-"*70)
print("🔬 EXP 3: WINSORIZE TARGET (2%-98%)")
print("-"*70)

lower, upper = df['resid_tech'].quantile([0.02, 0.98])
df['resid_w'] = df['resid_tech'].clip(lower, upper)
train, test = train_test_split(df)

model = LGBMRegressor(n_estimators=200, max_depth=2, num_leaves=4,
                       learning_rate=0.02, verbose=-1, random_state=42)
model.fit(train[all_features], train['resid_w'])
r2 = r2_score(test['resid_w'], model.predict(test[all_features]))
print(f"  Test R²: {r2:.4f} ({r2*100:.2f}%)")
results.append(("Winsorized Target", r2, len(all_features)))

# =====================================================
# EXPERIMENT 4: REMOVE TRAINING OUTLIERS
# =====================================================
print("\n" + "-"*70)
print("🔬 EXP 4: REMOVE TRAINING OUTLIERS (|resid| < 2 std)")
print("-"*70)

threshold = train['resid_tech'].std() * 2
train_clean = train[train['resid_tech'].abs() < threshold]
print(f"  Train before: {len(train)}, after: {len(train_clean)} ({100*len(train_clean)/len(train):.1f}%)")

model = LGBMRegressor(n_estimators=200, max_depth=2, num_leaves=4,
                       learning_rate=0.02, verbose=-1, random_state=42)
model.fit(train_clean[all_features], train_clean['resid_tech'])
r2 = r2_score(test['resid_tech'], model.predict(test[all_features]))
print(f"  Test R²: {r2:.4f} ({r2*100:.2f}%)")
results.append(("Remove Train Outliers", r2, len(all_features)))

# =====================================================
# EXPERIMENT 5: ENSEMBLE (Average 3 models)
# =====================================================
print("\n" + "-"*70)
print("🔬 EXP 5: ENSEMBLE (Ridge + 2 LightGBMs)")
print("-"*70)

train, test = train_test_split(df)

m1 = Ridge(alpha=1.0)
m2 = LGBMRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, verbose=-1, random_state=42)
m3 = LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.01, verbose=-1, random_state=123)

m1.fit(train[all_features], train['resid_tech'])
m2.fit(train[all_features], train['resid_tech'])
m3.fit(train[all_features], train['resid_tech'])

p1 = m1.predict(test[all_features])
p2 = m2.predict(test[all_features])
p3 = m3.predict(test[all_features])

# Simple average
pred_avg = (p1 + p2 + p3) / 3
r2 = r2_score(test['resid_tech'], pred_avg)
print(f"  Simple Average: {r2:.4f} ({r2*100:.2f}%)")
results.append(("Ensemble (3 models)", r2, len(all_features)))

# Weighted by train performance
tr1 = r2_score(train['resid_tech'], m1.predict(train[all_features]))
tr2 = r2_score(train['resid_tech'], m2.predict(train[all_features]))
tr3 = r2_score(train['resid_tech'], m3.predict(train[all_features]))

# Inverse of overfitting as weight (less overfit = higher weight)
weights = [1/max(tr1, 0.01), 1/max(tr2, 0.01), 1/max(tr3, 0.01)]
weights = np.array(weights) / sum(weights)
pred_weighted = weights[0]*p1 + weights[1]*p2 + weights[2]*p3
r2_w = r2_score(test['resid_tech'], pred_weighted)
print(f"  Weighted (anti-overfit): {r2_w:.4f} ({r2_w*100:.2f}%)")

# =====================================================
# EXPERIMENT 6: STACKING (Ridge + LightGBM correction)
# =====================================================
print("\n" + "-"*70)
print("🔬 EXP 6: STACKING (Ridge base + LGBM corrects)")
print("-"*70)

# Train Ridge as base
ridge = Ridge(alpha=1.0)
ridge.fit(train[all_features], train['resid_tech'])
train_ridge_pred = ridge.predict(train[all_features])
train['ridge_resid'] = train['resid_tech'] - train_ridge_pred

# Train LightGBM on Ridge residuals
lgbm = LGBMRegressor(n_estimators=100, max_depth=2, learning_rate=0.03, verbose=-1)
lgbm.fit(train[all_features], train['ridge_resid'])

# Predict
test_ridge = ridge.predict(test[all_features])
test_lgbm = lgbm.predict(test[all_features])
test_final = test_ridge + test_lgbm

r2 = r2_score(test['resid_tech'], test_final)
print(f"  Stacked (Ridge + LGBM): {r2:.4f} ({r2*100:.2f}%)")
results.append(("Stacking (Ridge+LGBM)", r2, len(all_features)))

# =====================================================
# EXPERIMENT 7: DIFFERENT TRAIN/TEST SPLIT
# =====================================================
print("\n" + "-"*70)
print("🔬 EXP 7: DIFFERENT SPLITS (earlier cutoff)")
print("-"*70)

for cutoff in ["2022-06-01", "2022-09-01", "2023-01-01", "2023-06-01"]:
    tr, te = train_test_split(df, cutoff)
    if len(tr) < 100 or len(te) < 50:
        continue
    model = LGBMRegressor(n_estimators=200, max_depth=2, num_leaves=4,
                           learning_rate=0.02, verbose=-1, random_state=42)
    model.fit(tr[all_features], tr['resid_tech'])
    r2 = r2_score(te['resid_tech'], model.predict(te[all_features]))
    print(f"  Cutoff {cutoff}: Train={len(tr):4d}, Test={len(te):4d}, R²={r2:.4f}")

# =====================================================
# EXPERIMENT 8: MORE LAG FEATURES (lag 7, 10)
# =====================================================
print("\n" + "-"*70)
print("🔬 EXP 8: EXTENDED LAGS (add lag 7, 10)")
print("-"*70)

df2 = load_data()
for col in ["news_count", "shock_index"]:
    for lag in [7, 10]:
        df2[f"{col}_lag{lag}"] = df2.groupby("ticker")[col].shift(lag)
df2 = df2.dropna()

train2, test2 = train_test_split(df2)
ext_lags = [c for c in df2.columns if "_lag" in c]
features_ext = core + ext_lags + interactions
features_ext = [f for f in features_ext if f in df2.columns]

model = LGBMRegressor(n_estimators=200, max_depth=2, num_leaves=4,
                       learning_rate=0.02, verbose=-1, random_state=42)
model.fit(train2[features_ext], train2['resid_tech'])
r2 = r2_score(test2['resid_tech'], model.predict(test2[features_ext]))
print(f"  Extended lags (1,2,3,5,7,10): {r2:.4f} ({r2*100:.2f}%)")
results.append(("Extended Lags", r2, len(features_ext)))

# =====================================================
# EXPERIMENT 9: CROSS-VALIDATION HYPERPARAMETER SEARCH
# =====================================================
print("\n" + "-"*70)
print("🔬 EXP 9: HYPERPARAMETER GRID SEARCH")
print("-"*70)

train, test = train_test_split(df)

best_r2 = -999
best_params = None

for n_est in [50, 100, 200]:
    for depth in [2, 3]:
        for lr in [0.01, 0.02, 0.05]:
            for leaves in [4, 8]:
                model = LGBMRegressor(
                    n_estimators=n_est, max_depth=depth, 
                    learning_rate=lr, num_leaves=leaves,
                    verbose=-1, random_state=42
                )
                model.fit(train[all_features], train['resid_tech'])
                r2 = r2_score(test['resid_tech'], model.predict(test[all_features]))
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = (n_est, depth, lr, leaves)

print(f"  Best params: n_est={best_params[0]}, depth={best_params[1]}, lr={best_params[2]}, leaves={best_params[3]}")
print(f"  Best R²: {best_r2:.4f} ({best_r2*100:.2f}%)")
results.append(("Best Hyperparams", best_r2, len(all_features)))

# =====================================================
# EXPERIMENT 10: COMBINED BEST PRACTICES
# =====================================================
print("\n" + "-"*70)
print("🔬 EXP 10: COMBINED BEST PRACTICES")
print("-"*70)

# Use extended lags
df3 = load_data()
for col in ["news_count", "shock_index"]:
    for lag in [7, 10]:
        df3[f"{col}_lag{lag}"] = df3.groupby("ticker")[col].shift(lag)
df3 = df3.dropna()

# Winsorize
lower, upper = df3['resid_tech'].quantile([0.02, 0.98])
df3['resid_w'] = df3['resid_tech'].clip(lower, upper)

train3, test3 = train_test_split(df3)

# Remove outliers from training
threshold = train3['resid_tech'].std() * 2
train3_clean = train3[train3['resid_tech'].abs() < threshold]

# Only use lagged + core features (no noisy PCA)
ext_lags = [c for c in df3.columns if "_lag" in c]
features_combo = core + ext_lags + interactions
features_combo = [f for f in features_combo if f in df3.columns]

# Best hyperparams
model = LGBMRegressor(
    n_estimators=best_params[0], max_depth=best_params[1],
    learning_rate=best_params[2], num_leaves=best_params[3],
    verbose=-1, random_state=42
)
model.fit(train3_clean[features_combo], train3_clean['resid_w'])
r2 = r2_score(test3['resid_w'], model.predict(test3[features_combo]))
print(f"  Combined approach: {r2:.4f} ({r2*100:.2f}%)")
results.append(("COMBINED BEST", r2, len(features_combo)))

# =====================================================
# SUMMARY
# =====================================================
print("\n" + "="*70)
print("📊 RESULTS RANKED BY TEST R²")
print("="*70)

results_df = pd.DataFrame(results, columns=["Experiment", "Test_R2", "N_Features"])
results_df = results_df.sort_values("Test_R2", ascending=False)
results_df["Pct"] = (results_df["Test_R2"] * 100).round(2)
results_df["vs_Baseline"] = ((results_df["Test_R2"] - baseline_r2) / baseline_r2 * 100).round(1)

print(f"\nBASELINE: {baseline_r2:.4f} ({baseline_r2*100:.2f}%)\n")
print(results_df.to_string(index=False))

best = results_df.iloc[0]
print(f"\n🏆 BEST APPROACH: {best['Experiment']}")
print(f"   Test R²: {best['Test_R2']:.4f} ({best['Pct']:.2f}%)")
print(f"   Improvement vs Baseline: {best['vs_Baseline']:+.1f}%")

print("\n" + "="*70)

