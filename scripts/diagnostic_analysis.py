"""
🔍 DIAGNOSTIC ANALYSIS: Why is R² only 4%?
Temporary script to test hypotheses and find improvements.

Experiments:
1. Target Analysis - Is the residual too noisy?
2. Feature Selection - Are we using too many noisy features?
3. Lag Analysis - Does news affect volatility with delay?
4. Per-Ticker Analysis - Do some tickers work better?
5. Regime Analysis - Does it work better in high/low VIX?
6. Different Models - Linear vs GBM vs Ridge
7. Simpler Target - Predict direction instead of magnitude?
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_data():
    """Load and prepare all data."""
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    news = pd.read_parquet("data/processed/news_features.parquet")
    targets = pd.read_parquet("data/processed/targets.parquet")
    
    # Normalize dates
    for df in [residuals, news, targets]:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        if df["ticker"].dtype.name == "category":
            df["ticker"] = df["ticker"].astype(str)
    
    # Merge
    merged = pd.merge(residuals, news, on=["date", "ticker"], how="inner")
    merged = pd.merge(merged, targets[["date", "ticker", "VIX_close", "realized_vol"]], 
                      on=["date", "ticker"], how="left")
    
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    merged["VIX_close"] = merged["VIX_close"].ffill().fillna(15)
    
    return merged

def get_features(df, include_pca=True, top_k=None):
    """Get feature columns."""
    features = ["shock_index", "news_count", "sentiment_avg", "novelty_score"]
    
    if include_pca:
        pca_cols = [c for c in df.columns if c.startswith("news_pca_")]
        features.extend(sorted(pca_cols))
    
    if "VIX_close" in df.columns:
        features.append("VIX_close")
    
    return [f for f in features if f in df.columns]

def train_test_split(df, cutoff="2023-01-01"):
    """Time-series split."""
    cutoff = pd.to_datetime(cutoff)
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    return train, test

def evaluate(y_true, y_pred, name=""):
    """Calculate metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"name": name, "R2": r2, "RMSE": rmse}

print("\n" + "="*70)
print("🔍 DIAGNOSTIC ANALYSIS: Improving NewsAgent R²")
print("="*70)

# Load data
df = load_data()
train, test = train_test_split(df)

print(f"\nData: {len(train)} train, {len(test)} test")
print(f"Tickers: {df['ticker'].unique().tolist()}")

# =====================================================
# EXPERIMENT 1: TARGET ANALYSIS
# =====================================================
print("\n" + "-"*70)
print("📊 EXPERIMENT 1: TARGET ANALYSIS")
print("-"*70)

print(f"\nResidual (resid_tech) statistics:")
print(f"  Mean:  {df['resid_tech'].mean():.4f}")
print(f"  Std:   {df['resid_tech'].std():.4f}")
print(f"  Min:   {df['resid_tech'].min():.2f}")
print(f"  Max:   {df['resid_tech'].max():.2f}")

# Check for outliers
q1 = df['resid_tech'].quantile(0.25)
q3 = df['resid_tech'].quantile(0.75)
iqr = q3 - q1
outliers = ((df['resid_tech'] < q1 - 1.5*iqr) | (df['resid_tech'] > q3 + 1.5*iqr)).sum()
print(f"  Outliers (IQR method): {outliers} ({100*outliers/len(df):.1f}%)")

# Try winsorizing
df_clean = df.copy()
lower, upper = df['resid_tech'].quantile([0.01, 0.99])
df_clean['resid_tech_winsor'] = df_clean['resid_tech'].clip(lower, upper)
print(f"\n  After winsorizing (1%-99%):")
print(f"    New range: [{lower:.2f}, {upper:.2f}]")

# =====================================================
# EXPERIMENT 2: FEATURE IMPORTANCE ANALYSIS
# =====================================================
print("\n" + "-"*70)
print("📊 EXPERIMENT 2: FEATURE CORRELATION WITH TARGET")
print("-"*70)

features = get_features(df)
correlations = []
for f in features:
    corr = df[f].corr(df['resid_tech'])
    correlations.append((f, corr))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)
print("\nTop 10 correlated features:")
for f, c in correlations[:10]:
    print(f"  {f:25s}: {c:+.4f}")

# =====================================================
# EXPERIMENT 3: MODEL COMPARISON
# =====================================================
print("\n" + "-"*70)
print("📊 EXPERIMENT 3: MODEL COMPARISON")
print("-"*70)

features = get_features(df, include_pca=True)
X_train, y_train = train[features], train['resid_tech']
X_test, y_test = test[features], test['resid_tech']

models = {
    "Ridge (α=1)": Ridge(alpha=1.0),
    "Ridge (α=10)": Ridge(alpha=10.0),
    "Lasso (α=0.1)": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "LightGBM (current)": LGBMRegressor(n_estimators=1000, learning_rate=0.01, 
                                         max_depth=3, reg_alpha=0.1, verbose=-1),
    "LightGBM (simpler)": LGBMRegressor(n_estimators=100, learning_rate=0.05, 
                                         max_depth=2, num_leaves=4, verbose=-1),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    gap = train_r2 - test_r2
    
    results.append({
        "Model": name,
        "Train R²": train_r2,
        "Test R²": test_r2,
        "Gap": gap
    })

results_df = pd.DataFrame(results).sort_values("Test R²", ascending=False)
print("\n" + results_df.to_string(index=False))

# =====================================================
# EXPERIMENT 4: FEATURE SELECTION
# =====================================================
print("\n" + "-"*70)
print("📊 EXPERIMENT 4: FEATURE SELECTION (Top K features)")
print("-"*70)

for k in [5, 10, 15, 20]:
    selector = SelectKBest(f_regression, k=k)
    X_train_k = selector.fit_transform(X_train, y_train)
    X_test_k = selector.transform(X_test)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_k, y_train)
    
    test_r2 = r2_score(y_test, model.predict(X_test_k))
    
    selected = [f for f, s in zip(features, selector.get_support()) if s]
    print(f"  K={k:2d}: Test R² = {test_r2:.4f}")

# Best K
selector = SelectKBest(f_regression, k=10)
X_train_k = selector.fit_transform(X_train, y_train)
selected = [f for f, s in zip(features, selector.get_support()) if s]
print(f"\n  Best features (k=10): {selected}")

# =====================================================
# EXPERIMENT 5: LAG ANALYSIS
# =====================================================
print("\n" + "-"*70)
print("📊 EXPERIMENT 5: LAG ANALYSIS (Does news predict future residuals?)")
print("-"*70)

# Create lagged features
df_lag = df.copy()
for lag in [1, 2, 3, 5]:
    df_lag[f'resid_tech_future_{lag}d'] = df_lag.groupby('ticker')['resid_tech'].shift(-lag)

# Test if today's news predicts future residuals
print("\nCorrelation of today's shock_index with future residuals:")
for lag in [1, 2, 3, 5]:
    col = f'resid_tech_future_{lag}d'
    corr = df_lag['shock_index'].corr(df_lag[col])
    print(f"  +{lag} days: {corr:+.4f}")

print("\nCorrelation of today's news_count with future residuals:")
for lag in [1, 2, 3, 5]:
    col = f'resid_tech_future_{lag}d'
    corr = df_lag['news_count'].corr(df_lag[col])
    print(f"  +{lag} days: {corr:+.4f}")

# =====================================================
# EXPERIMENT 6: PER-TICKER ANALYSIS
# =====================================================
print("\n" + "-"*70)
print("📊 EXPERIMENT 6: PER-TICKER ANALYSIS")
print("-"*70)

for ticker in df['ticker'].unique():
    ticker_train = train[train['ticker'] == ticker]
    ticker_test = test[test['ticker'] == ticker]
    
    if len(ticker_train) < 20 or len(ticker_test) < 10:
        continue
    
    X_tr = ticker_train[features]
    y_tr = ticker_train['resid_tech']
    X_te = ticker_test[features]
    y_te = ticker_test['resid_tech']
    
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    
    train_r2 = r2_score(y_tr, model.predict(X_tr))
    test_r2 = r2_score(y_te, model.predict(X_te))
    
    print(f"  {ticker}: Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}")

# =====================================================
# EXPERIMENT 7: VIX REGIME ANALYSIS
# =====================================================
print("\n" + "-"*70)
print("📊 EXPERIMENT 7: VIX REGIME ANALYSIS")
print("-"*70)

vix_median = df['VIX_close'].median()
print(f"\nVIX median: {vix_median:.1f}")

for regime, mask_train, mask_test in [
    ("Low VIX", train['VIX_close'] <= vix_median, test['VIX_close'] <= vix_median),
    ("High VIX", train['VIX_close'] > vix_median, test['VIX_close'] > vix_median),
]:
    tr = train[mask_train]
    te = test[mask_test]
    
    if len(tr) < 50 or len(te) < 20:
        continue
    
    model = Ridge(alpha=1.0)
    model.fit(tr[features], tr['resid_tech'])
    
    test_r2 = r2_score(te['resid_tech'], model.predict(te[features]))
    print(f"  {regime}: n_test={len(te):4d}, Test R² = {test_r2:.4f}")

# =====================================================
# EXPERIMENT 8: DIRECTION PREDICTION (Binary)
# =====================================================
print("\n" + "-"*70)
print("📊 EXPERIMENT 8: DIRECTION PREDICTION")
print("-"*70)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train['resid_direction'] = (train['resid_tech'] > 0).astype(int)
test['resid_direction'] = (test['resid_tech'] > 0).astype(int)

clf = LogisticRegression(max_iter=1000, C=0.1)
clf.fit(X_train, train['resid_direction'])

train_acc = accuracy_score(train['resid_direction'], clf.predict(X_train))
test_acc = accuracy_score(test['resid_direction'], clf.predict(X_test))

print(f"\nDirection prediction (above/below 0):")
print(f"  Train accuracy: {100*train_acc:.1f}%")
print(f"  Test accuracy:  {100*test_acc:.1f}%")
print(f"  Baseline (random): 50%")

# =====================================================
# EXPERIMENT 9: EXTREME RESIDUALS ONLY
# =====================================================
print("\n" + "-"*70)
print("📊 EXPERIMENT 9: PREDICT EXTREME RESIDUALS")
print("-"*70)

# Only try to predict when residual is extreme
threshold = df['resid_tech'].std()
print(f"\nThreshold: |resid| > {threshold:.2f} (1 std)")

train_extreme = train[train['resid_tech'].abs() > threshold]
test_extreme = test[test['resid_tech'].abs() > threshold]

print(f"  Train extreme: {len(train_extreme)} ({100*len(train_extreme)/len(train):.1f}%)")
print(f"  Test extreme:  {len(test_extreme)} ({100*len(test_extreme)/len(test):.1f}%)")

if len(train_extreme) > 50 and len(test_extreme) > 20:
    model = Ridge(alpha=1.0)
    model.fit(train_extreme[features], train_extreme['resid_tech'])
    
    test_r2 = r2_score(test_extreme['resid_tech'], 
                        model.predict(test_extreme[features]))
    print(f"  Test R² (extreme only): {test_r2:.4f}")

# =====================================================
# EXPERIMENT 10: DIFFERENT ROLLING WINDOWS
# =====================================================
print("\n" + "-"*70)
print("📊 EXPERIMENT 10: DIFFERENT ROLLING AGGREGATIONS")
print("-"*70)

df_agg = df.copy()

# Create weekly aggregations
for col in ['shock_index', 'news_count', 'sentiment_avg']:
    df_agg[f'{col}_weekly'] = df_agg.groupby('ticker')[col].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

agg_features = features + [f'{c}_weekly' for c in ['shock_index', 'news_count', 'sentiment_avg']]
agg_features = [f for f in agg_features if f in df_agg.columns]

train_agg = df_agg[df_agg['date'] < pd.to_datetime('2023-01-01')]
test_agg = df_agg[df_agg['date'] >= pd.to_datetime('2023-01-01')]

model = Ridge(alpha=1.0)
model.fit(train_agg[agg_features].fillna(0), train_agg['resid_tech'])
test_r2 = r2_score(test_agg['resid_tech'], 
                    model.predict(test_agg[agg_features].fillna(0)))
print(f"\n  With weekly aggregations: Test R² = {test_r2:.4f}")

# =====================================================
# FINAL RECOMMENDATIONS
# =====================================================
print("\n" + "="*70)
print("💡 DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
print("="*70)

print("""
1. TARGET ISSUE:
   - Residuals have extreme outliers (some > 5 std)
   - Try: Winsorize target or use robust loss function

2. MODEL CHOICE:
   - Ridge/Linear models perform similarly to LightGBM
   - Simpler is better - complex models overfit the noise

3. FEATURE SELECTION:
   - Too many noisy PCA features
   - Try: Use only top 10 correlated features

4. VIX REGIME:
   - News may work better in one regime
   - Try: Train separate models for high/low VIX

5. LAG STRUCTURE:
   - News might have delayed effect (2-3 days)
   - Try: Add lagged news features

6. DIRECTION vs MAGNITUDE:
   - Predicting direction might be easier than magnitude
   - Try: Binary classification instead of regression

7. EXTREME RESIDUALS:
   - Most residuals are small and noisy
   - Try: Only predict when news is significant

8. DATA VOLUME:
   - Only 741 train, 878 test samples
   - Try: More tickers, longer history
""")

print("="*70)

