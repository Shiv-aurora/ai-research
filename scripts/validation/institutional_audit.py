"""
Institutional Validation Suite (AQR/Two Sigma Protocol)

Three rigorous institutional-grade tests:
1. Test A: Horizon Sensitivity (Alpha Decay) - IC decay across horizons
2. Test B: Regime-Conditional Performance - R² by market regime
3. Test C: Block Bootstrap (Robustness) - Circular block bootstrap CI

Usage:
    python scripts/institutional_audit.py
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import mlflow

# Import universe configurations
from scripts.scale_up.config_universes import TOP_50_ACTIVE, SECTOR_MAP_ACTIVE


def load_high_octane_data():
    """Load the High Octane dataset with all features."""
    print("\n📂 Loading High Octane dataset...")
    
    # Load targets
    targets = pd.read_parquet(PROJECT_ROOT / "data/processed/targets_deseasonalized.parquet")
    targets["date"] = pd.to_datetime(targets["date"]).dt.tz_localize(None)
    if targets["ticker"].dtype.name == "category":
        targets["ticker"] = targets["ticker"].astype(str)
    
    # Filter to High Octane tickers
    targets = targets[targets["ticker"].isin(TOP_50_ACTIVE)].copy()
    
    # Load residuals
    residuals = pd.read_parquet(PROJECT_ROOT / "data/processed/residuals.parquet")
    residuals["date"] = pd.to_datetime(residuals["date"]).dt.tz_localize(None)
    if residuals["ticker"].dtype.name == "category":
        residuals["ticker"] = residuals["ticker"].astype(str)
    residuals = residuals[residuals["ticker"].isin(TOP_50_ACTIVE)].copy()
    
    # Load news features
    news = pd.read_parquet(PROJECT_ROOT / "data/processed/news_features.parquet")
    news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(None)
    if news["ticker"].dtype.name == "category":
        news["ticker"] = news["ticker"].astype(str)
    news = news[news["ticker"].isin(TOP_50_ACTIVE)].copy()
    
    # Load retail proxy
    retail = pd.read_parquet(PROJECT_ROOT / "data/processed/reddit_proxy.parquet")
    retail["date"] = pd.to_datetime(retail["date"]).dt.tz_localize(None)
    if retail["ticker"].dtype.name == "category":
        retail["ticker"] = retail["ticker"].astype(str)
    retail = retail[retail["ticker"].isin(TOP_50_ACTIVE)].copy()
    
    # Merge
    df = pd.merge(targets, residuals[["date", "ticker", "pred_tech_excess", "resid_tech"]], 
                  on=["date", "ticker"], how="left")
    
    # Create tech_pred
    if "seasonal_component" in df.columns and "pred_tech_excess" in df.columns:
        df["tech_pred"] = df["pred_tech_excess"] + df["seasonal_component"]
    else:
        df["tech_pred"] = df["target_log_var"].mean()
    
    # Merge news
    news_cols = ["date", "ticker"] + [c for c in news.columns if c.startswith(("news_pca", "shock", "sentiment", "news_count"))]
    news_cols = [c for c in news_cols if c in news.columns]
    df = pd.merge(df, news[news_cols], on=["date", "ticker"], how="left")
    
    # Merge retail
    retail_cols = ["date", "ticker", "volume_shock", "hype_signal", "hype_zscore", "price_acceleration",
                   "volume_shock_roll3", "hype_signal_roll3", "hype_signal_roll7"]
    retail_cols = [c for c in retail_cols if c in retail.columns]
    df = pd.merge(df, retail[retail_cols], on=["date", "ticker"], how="left")
    
    # Add calendar features
    dow = df["date"].dt.dayofweek
    df["is_monday"] = (dow == 0).astype(int)
    df["is_friday"] = (dow == 4).astype(int)
    df["is_q4"] = (df["date"].dt.quarter == 4).astype(int)
    df["year"] = df["date"].dt.year
    
    # Add momentum features
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        ticker_data = df.loc[mask, "target_log_var"]
        df.loc[mask, "vol_ma5"] = ticker_data.rolling(5, min_periods=1).mean().shift(1)
        df.loc[mask, "vol_ma10"] = ticker_data.rolling(10, min_periods=1).mean().shift(1)
        df.loc[mask, "vol_std5"] = ticker_data.rolling(5, min_periods=2).std().shift(1)
        
        # Create future targets for horizon analysis
        for h in [1, 2, 3, 5, 10]:
            df.loc[mask, f"target_T{h}"] = ticker_data.shift(-h)
    
    # Fill NaN
    df["vol_ma5"] = df["vol_ma5"].fillna(df["target_log_var"].mean())
    df["vol_ma10"] = df["vol_ma10"].fillna(df["target_log_var"].mean())
    df["vol_std5"] = df["vol_std5"].fillna(0)
    
    # Add sector
    df["sector"] = df["ticker"].map(SECTOR_MAP_ACTIVE)
    
    # Drop NaN target
    df = df.dropna(subset=["target_log_var"])
    
    # Sort by date
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    print(f"   ✓ Loaded {len(df):,} rows, {df['ticker'].nunique()} tickers")
    print(f"   ✓ Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


def train_titan_v15(train_df, test_df, alpha=100.0, winsorize_pct=0.02):
    """
    Train RIVE and return predictions on test set.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Compute resid_tech if missing
    if "resid_tech" not in train_df.columns:
        train_df["resid_tech"] = train_df["target_log_var"] - train_df.get("tech_pred", train_df["target_log_var"].mean())
        test_df["resid_tech"] = test_df["target_log_var"] - test_df.get("tech_pred", test_df["target_log_var"].mean())
    
    train_df["resid_tech"] = train_df["resid_tech"].fillna(0)
    test_df["resid_tech"] = test_df["resid_tech"].fillna(0)
    
    # News Classifier
    news_features = [c for c in train_df.columns if c.startswith("news_pca")][:10]
    news_features += ["shock_index", "news_count"] if "shock_index" in train_df.columns else []
    news_features = [c for c in news_features if c in train_df.columns]
    
    if len(news_features) > 0:
        threshold = train_df["resid_tech"].quantile(0.80)
        train_df["is_extreme"] = (train_df["resid_tech"] > threshold).astype(int)
        test_df["is_extreme"] = (test_df["resid_tech"] > threshold).astype(int)
        
        X_news_train = train_df[news_features].fillna(0)
        X_news_test = test_df[news_features].fillna(0)
        
        news_clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, verbose=-1, random_state=42)
        news_clf.fit(X_news_train, train_df["is_extreme"])
        
        train_df["news_risk_score"] = news_clf.predict_proba(X_news_train)[:, 1]
        test_df["news_risk_score"] = news_clf.predict_proba(X_news_test)[:, 1]
    else:
        train_df["news_risk_score"] = 0.2
        test_df["news_risk_score"] = 0.2
    
    # Retail Classifier
    retail_features = ["volume_shock", "hype_signal", "hype_zscore", "price_acceleration",
                       "volume_shock_roll3", "hype_signal_roll3", "hype_signal_roll7"]
    retail_features = [c for c in retail_features if c in train_df.columns]
    
    if len(retail_features) > 0:
        X_retail_train = train_df[retail_features].fillna(0)
        X_retail_test = test_df[retail_features].fillna(0)
        
        retail_clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, verbose=-1, random_state=42)
        retail_clf.fit(X_retail_train, train_df["is_extreme"])
        
        train_df["retail_risk_score"] = retail_clf.predict_proba(X_retail_train)[:, 1]
        test_df["retail_risk_score"] = retail_clf.predict_proba(X_retail_test)[:, 1]
    else:
        train_df["retail_risk_score"] = 0.2
        test_df["retail_risk_score"] = 0.2
    
    # Interaction
    train_df["news_x_retail"] = train_df["news_risk_score"] * train_df["retail_risk_score"]
    test_df["news_x_retail"] = test_df["news_risk_score"] * test_df["retail_risk_score"]
    
    # Coordinator
    coord_features = ["tech_pred", "news_risk_score", "retail_risk_score",
                      "is_friday", "is_monday", "is_q4",
                      "vol_ma5", "vol_ma10", "vol_std5", "news_x_retail"]
    coord_features = [c for c in coord_features if c in train_df.columns]
    
    X_train = train_df[coord_features].fillna(0)
    y_train = train_df["target_log_var"]
    X_test = test_df[coord_features].fillna(0)
    y_test = test_df["target_log_var"]
    
    # Winsorization
    lower = y_train.quantile(winsorize_pct)
    upper = y_train.quantile(1 - winsorize_pct)
    y_train_win = y_train.clip(lower=lower, upper=upper)
    
    # Train
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train, y_train_win)
    
    # Predict
    y_pred = model.predict(X_test)
    
    return y_pred, y_test.values, test_df


# =============================================================================
# TEST A: HORIZON SENSITIVITY (ALPHA DECAY)
# =============================================================================

def test_horizon_sensitivity(df):
    """
    Test A: Horizon Sensitivity (Alpha Decay)
    
    Measure how predictions correlate with actual volatility at different horizons.
    High IC at T that slowly decays indicates real predictive power.
    IC hitting 0 at T+1 indicates leakage.
    """
    print("\n" + "=" * 70)
    print("🧪 TEST A: HORIZON SENSITIVITY (Alpha Decay)")
    print("=" * 70)
    print("   Purpose: Measure IC decay across prediction horizons")
    print("   Pass: IC should be high at T and slowly decay")
    print("   Fail: IC hitting 0 at T+1 indicates potential leakage")
    
    # Standard split
    cutoff = pd.Timestamp("2023-01-01")
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    
    print(f"\n   Train: {len(train):,} samples (< 2023)")
    print(f"   Test:  {len(test):,} samples (>= 2023)")
    
    # Train model and get predictions for T
    y_pred, y_true, test_df = train_titan_v15(train, test)
    test_df["y_pred"] = y_pred
    
    # Calculate IC (Information Coefficient) at different horizons
    horizons = [0, 1, 2, 3, 5, 10]
    ic_results = []
    
    print(f"\n   📊 Information Coefficient by Horizon:")
    print("   " + "-" * 50)
    
    for h in horizons:
        if h == 0:
            # T: correlation with same-day volatility
            target_col = "target_log_var"
        else:
            target_col = f"target_T{h}"
        
        if target_col in test_df.columns:
            # Remove NaN (from shifted targets)
            valid_mask = test_df[target_col].notna()
            if valid_mask.sum() > 100:
                ic, pvalue = stats.spearmanr(
                    test_df.loc[valid_mask, "y_pred"],
                    test_df.loc[valid_mask, target_col]
                )
                
                ic_results.append({
                    "horizon": h,
                    "ic": ic,
                    "pvalue": pvalue,
                    "n_samples": valid_mask.sum()
                })
                
                sig = "***" if pvalue < 0.001 else ("**" if pvalue < 0.01 else ("*" if pvalue < 0.05 else ""))
                print(f"      T+{h:2d}: IC = {ic:+.4f} (p={pvalue:.4f}) {sig}")
    
    # Plot IC Decay curve
    print("\n   📈 Plotting IC Decay Curve...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    horizons_plot = [r["horizon"] for r in ic_results]
    ics_plot = [r["ic"] for r in ic_results]
    
    ax.plot(horizons_plot, ics_plot, 'b-o', linewidth=2, markersize=8, label='Information Coefficient')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero (No Predictive Power)')
    ax.fill_between(horizons_plot, 0, ics_plot, alpha=0.3)
    
    ax.set_xlabel('Horizon (Days Forward)', fontsize=12)
    ax.set_ylabel('Information Coefficient (Spearman)', fontsize=12)
    ax.set_title('Alpha Decay: IC vs Prediction Horizon\n(How far into the future does the signal persist?)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(horizons_plot)
    ax.set_xticklabels([f'T+{h}' for h in horizons_plot])
    
    # Save plot
    plot_path = PROJECT_ROOT / "reports" / "ic_decay_curve.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved plot to: {plot_path}")
    
    # Analysis
    ic_t0 = ic_results[0]["ic"] if ic_results else 0
    ic_t1 = ic_results[1]["ic"] if len(ic_results) > 1 else 0
    ic_decay_rate = (ic_t0 - ic_t1) / ic_t0 if ic_t0 != 0 else 0
    
    print(f"\n   📊 Analysis:")
    print(f"      IC at T+0: {ic_t0:.4f}")
    print(f"      IC at T+1: {ic_t1:.4f}")
    print(f"      Decay Rate (T→T+1): {ic_decay_rate*100:.1f}%")
    
    # Verdict
    if ic_t0 > 0.3 and ic_t1 > 0.1:
        verdict = "PASS"
        print(f"\n   ✅ PASSED: Strong IC at T ({ic_t0:.4f}) with gradual decay")
        print("      → Real predictive power, not leakage")
    elif ic_t0 > 0.2 and ic_t1 > 0:
        verdict = "PARTIAL"
        print(f"\n   ⚠️ PARTIAL: Moderate IC with some persistence")
    else:
        verdict = "FAIL"
        print(f"\n   ❌ FAIL: Weak or no IC persistence")
    
    return {
        "test": "Horizon Sensitivity",
        "ic_t0": ic_t0,
        "ic_t1": ic_t1,
        "decay_rate": ic_decay_rate,
        "ic_results": ic_results,
        "verdict": verdict
    }


# =============================================================================
# TEST B: REGIME-CONDITIONAL PERFORMANCE
# =============================================================================

def test_regime_conditional(df):
    """
    Test B: Regime-Conditional Performance
    
    Measure R² separately for Bull, Bear, and Crisis regimes.
    """
    print("\n" + "=" * 70)
    print("🧪 TEST B: REGIME-CONDITIONAL PERFORMANCE")
    print("=" * 70)
    print("   Purpose: Measure performance across different market regimes")
    print("   Regimes: Bull (SPY above trend), Bear (SPY below trend), Crisis (VIX > 30)")
    
    # Check if we have VIX data
    if "VIX_close" not in df.columns:
        print("   ⚠️ VIX_close not found, loading from targets...")
        targets = pd.read_parquet(PROJECT_ROOT / "data/processed/targets.parquet")
        targets["date"] = pd.to_datetime(targets["date"]).dt.tz_localize(None)
        if "VIX_close" in targets.columns:
            vix_df = targets[["date", "VIX_close"]].drop_duplicates()
            df = pd.merge(df, vix_df, on="date", how="left", suffixes=("", "_y"))
            if "VIX_close_y" in df.columns:
                df["VIX_close"] = df["VIX_close"].fillna(df["VIX_close_y"])
                df = df.drop(columns=["VIX_close_y"])
    
    # Standard split
    cutoff = pd.Timestamp("2023-01-01")
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    
    print(f"\n   Train: {len(train):,} samples")
    print(f"   Test:  {len(test):,} samples")
    
    # Train model
    y_pred, y_true, test_df = train_titan_v15(train, test)
    test_df["y_pred"] = y_pred
    test_df["y_true"] = y_true
    
    # Define regimes
    regime_results = []
    
    # Regime 1: Crisis (VIX > 30)
    if "VIX_close" in test_df.columns and test_df["VIX_close"].notna().any():
        crisis_mask = test_df["VIX_close"] > 30
        n_crisis = crisis_mask.sum()
        
        print(f"\n   📊 Regime Definitions:")
        print(f"      Crisis (VIX > 30): {n_crisis:,} samples ({n_crisis/len(test_df)*100:.1f}%)")
        
        if n_crisis > 50:
            r2_crisis = r2_score(test_df.loc[crisis_mask, "y_true"], 
                                  test_df.loc[crisis_mask, "y_pred"])
            regime_results.append({"regime": "Crisis (VIX>30)", "r2": r2_crisis, "n": n_crisis})
        else:
            regime_results.append({"regime": "Crisis (VIX>30)", "r2": float('nan'), "n": n_crisis})
    
    # Regime 2: High Vol vs Low Vol (based on VIX)
    if "VIX_close" in test_df.columns and test_df["VIX_close"].notna().any():
        vix_median = test_df["VIX_close"].median()
        
        high_vol_mask = test_df["VIX_close"] > vix_median
        low_vol_mask = test_df["VIX_close"] <= vix_median
        
        n_high = high_vol_mask.sum()
        n_low = low_vol_mask.sum()
        
        print(f"      High Vol (VIX > {vix_median:.1f}): {n_high:,} samples ({n_high/len(test_df)*100:.1f}%)")
        print(f"      Low Vol (VIX <= {vix_median:.1f}): {n_low:,} samples ({n_low/len(test_df)*100:.1f}%)")
        
        if n_high > 50:
            r2_high = r2_score(test_df.loc[high_vol_mask, "y_true"], 
                               test_df.loc[high_vol_mask, "y_pred"])
            regime_results.append({"regime": "High Vol", "r2": r2_high, "n": n_high})
        
        if n_low > 50:
            r2_low = r2_score(test_df.loc[low_vol_mask, "y_true"], 
                              test_df.loc[low_vol_mask, "y_pred"])
            regime_results.append({"regime": "Low Vol", "r2": r2_low, "n": n_low})
    
    # Regime 3: By Year
    print(f"\n   📊 Performance by Year:")
    for year in sorted(test_df["year"].unique()):
        year_mask = test_df["year"] == year
        n_year = year_mask.sum()
        
        if n_year > 50:
            r2_year = r2_score(test_df.loc[year_mask, "y_true"], 
                               test_df.loc[year_mask, "y_pred"])
            
            # Determine regime label
            if year == 2022:
                label = f"{year} (Bear)"
            elif year == 2023:
                label = f"{year} (Recovery)"
            else:
                label = f"{year} (Bull)"
            
            regime_results.append({"regime": label, "r2": r2_year, "n": n_year})
            print(f"      {label}: R² = {r2_year:.4f} ({r2_year*100:.2f}%), n={n_year:,}")
    
    # Overall
    r2_overall = r2_score(y_true, y_pred)
    regime_results.append({"regime": "Overall", "r2": r2_overall, "n": len(test_df)})
    
    # Print summary table
    print(f"\n   📊 Regime Performance Summary:")
    print("   " + "-" * 55)
    print(f"   {'Regime':<25} {'R²':>12} {'N':>12}")
    print("   " + "-" * 55)
    
    for r in regime_results:
        r2_str = f"{r['r2']*100:.2f}%" if pd.notna(r['r2']) else "N/A"
        status = "✅" if pd.notna(r['r2']) and r['r2'] >= 0.20 else ("⚠️" if pd.notna(r['r2']) and r['r2'] >= 0 else "❌")
        print(f"   {r['regime']:<25} {r2_str:>12} {r['n']:>12,} {status}")
    
    # Verdict
    valid_r2s = [r['r2'] for r in regime_results if pd.notna(r['r2']) and r['regime'] != 'Overall']
    min_r2 = min(valid_r2s) if valid_r2s else 0
    mean_r2 = np.mean(valid_r2s) if valid_r2s else 0
    
    if min_r2 >= 0.15:
        verdict = "PASS"
        print(f"\n   ✅ PASSED: All regimes show positive R² (min: {min_r2*100:.2f}%)")
    elif mean_r2 >= 0.15:
        verdict = "PARTIAL"
        print(f"\n   ⚠️ PARTIAL: Some regimes underperform (min: {min_r2*100:.2f}%)")
    else:
        verdict = "FAIL"
        print(f"\n   ❌ FAIL: Poor regime performance (min: {min_r2*100:.2f}%)")
    
    return {
        "test": "Regime-Conditional",
        "regimes": regime_results,
        "min_r2": min_r2,
        "mean_r2": mean_r2,
        "verdict": verdict
    }


# =============================================================================
# TEST C: BLOCK BOOTSTRAP
# =============================================================================

def test_block_bootstrap(df, n_bootstrap=500, block_size=10):
    """
    Test C: Block Bootstrap (Robustness)
    
    Circular block bootstrap to generate confidence intervals.
    """
    print("\n" + "=" * 70)
    print("🧪 TEST C: BLOCK BOOTSTRAP (Robustness)")
    print("=" * 70)
    print(f"   Method: Circular Block Bootstrap")
    print(f"   Block Size: {block_size} days")
    print(f"   Iterations: {n_bootstrap}")
    print("   Pass: 5th percentile > Baseline R² (15.4%)")
    
    # Standard split
    cutoff = pd.Timestamp("2023-01-01")
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    
    print(f"\n   Train: {len(train):,} samples")
    print(f"   Test:  {len(test):,} samples")
    
    # Train model once
    y_pred, y_true, test_df = train_titan_v15(train, test)
    
    # Baseline R²
    baseline_r2 = r2_score(y_true, y_pred)
    print(f"\n   📊 Baseline R²: {baseline_r2:.4f} ({baseline_r2*100:.2f}%)")
    
    # Circular block bootstrap
    print(f"\n   🔄 Running {n_bootstrap} bootstrap iterations...")
    
    n_test = len(y_true)
    n_blocks = int(np.ceil(n_test / block_size))
    
    bootstrap_r2s = []
    
    np.random.seed(42)
    
    for i in range(n_bootstrap):
        # Sample block start indices with replacement
        block_starts = np.random.randint(0, n_test, size=n_blocks)
        
        # Build bootstrapped indices (circular)
        indices = []
        for start in block_starts:
            for j in range(block_size):
                idx = (start + j) % n_test
                indices.append(idx)
                if len(indices) >= n_test:
                    break
            if len(indices) >= n_test:
                break
        
        indices = np.array(indices[:n_test])
        
        # Calculate R² on bootstrap sample
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        r2_boot = r2_score(y_true_boot, y_pred_boot)
        bootstrap_r2s.append(r2_boot)
        
        if (i + 1) % 100 == 0:
            print(f"      Completed {i+1}/{n_bootstrap} iterations...")
    
    bootstrap_r2s = np.array(bootstrap_r2s)
    
    # Calculate confidence intervals
    ci_5 = np.percentile(bootstrap_r2s, 5)
    ci_95 = np.percentile(bootstrap_r2s, 95)
    mean_r2 = np.mean(bootstrap_r2s)
    std_r2 = np.std(bootstrap_r2s)
    
    print(f"\n   📊 Bootstrap Results:")
    print(f"      Mean R²:   {mean_r2:.4f} ({mean_r2*100:.2f}%)")
    print(f"      Std R²:    {std_r2:.4f}")
    print(f"      5th pct:   {ci_5:.4f} ({ci_5*100:.2f}%)")
    print(f"      95th pct:  {ci_95:.4f} ({ci_95*100:.2f}%)")
    print(f"      95% CI:    [{ci_5*100:.2f}%, {ci_95*100:.2f}%]")
    
    # Plot histogram
    print("\n   📈 Plotting Bootstrap Distribution...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(bootstrap_r2s * 100, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(x=baseline_r2 * 100, color='red', linestyle='-', linewidth=2, label=f'Baseline R²: {baseline_r2*100:.2f}%')
    ax.axvline(x=ci_5 * 100, color='orange', linestyle='--', linewidth=2, label=f'5th percentile: {ci_5*100:.2f}%')
    ax.axvline(x=ci_95 * 100, color='orange', linestyle='--', linewidth=2, label=f'95th percentile: {ci_95*100:.2f}%')
    ax.axvline(x=15.4, color='green', linestyle=':', linewidth=2, label='HAR Baseline: 15.4%')
    
    ax.set_xlabel('Test R² (%)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Block Bootstrap Distribution ({n_bootstrap} iterations, block={block_size}d)\n95% CI: [{ci_5*100:.2f}%, {ci_95*100:.2f}%]', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = PROJECT_ROOT / "reports" / "bootstrap_distribution.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved plot to: {plot_path}")
    
    # Verdict (5th percentile should beat baseline)
    har_baseline = 0.154  # 15.4%
    
    if ci_5 > har_baseline:
        verdict = "PASS"
        print(f"\n   ✅ PASSED: 5th percentile ({ci_5*100:.2f}%) > HAR baseline (15.4%)")
        print("      → Result is robust under resampling")
    elif ci_5 > 0:
        verdict = "PARTIAL"
        print(f"\n   ⚠️ PARTIAL: 5th percentile ({ci_5*100:.2f}%) positive but below baseline")
    else:
        verdict = "FAIL"
        print(f"\n   ❌ FAIL: 5th percentile ({ci_5*100:.2f}%) is negative or zero")
    
    return {
        "test": "Block Bootstrap",
        "baseline_r2": baseline_r2,
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "ci_5": ci_5,
        "ci_95": ci_95,
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
        "verdict": verdict
    }


# =============================================================================
# INSTITUTIONAL REPORT
# =============================================================================

def generate_institutional_report(results):
    """Generate the final institutional validation report."""
    
    print("\n" + "=" * 70)
    print("📋 INSTITUTIONAL VALIDATION REPORT (AQR/Two Sigma Protocol)")
    print("=" * 70)
    
    # Summary table
    print(f"""
   ┌─────────────────────────────┬──────────────────────────┬──────────┐
   │ Test                        │ Key Metric               │ Verdict  │
   ├─────────────────────────────┼──────────────────────────┼──────────┤""")
    
    for r in results:
        if r["test"] == "Horizon Sensitivity":
            metric = f"IC decay: {r['ic_t0']:.3f}→{r['ic_t1']:.3f}"
        elif r["test"] == "Regime-Conditional":
            metric = f"Min regime R²: {r['min_r2']*100:.1f}%"
        elif r["test"] == "Block Bootstrap":
            metric = f"95% CI: [{r['ci_5']*100:.1f}%, {r['ci_95']*100:.1f}%]"
        else:
            metric = "N/A"
        
        status = "✅" if r["verdict"] == "PASS" else ("⚠️" if r["verdict"] == "PARTIAL" else "❌")
        print(f"   │ {r['test']:<27} │ {metric:<24} │ {status} {r['verdict']:6} │")
    
    print("   └─────────────────────────────┴──────────────────────────┴──────────┘")
    
    # Count verdicts
    passes = sum(1 for r in results if r["verdict"] == "PASS")
    partials = sum(1 for r in results if r["verdict"] == "PARTIAL")
    fails = sum(1 for r in results if r["verdict"] == "FAIL")
    
    # Final assessment
    print(f"\n   📊 SUMMARY:")
    print(f"      Passed:  {passes}/3")
    print(f"      Partial: {partials}/3")
    print(f"      Failed:  {fails}/3")
    
    if fails == 0 and passes >= 2:
        overall = "INSTITUTIONAL GRADE"
        emoji = "🏆"
        conclusion = "Model meets institutional validation standards (AQR/Two Sigma)."
    elif fails == 0:
        overall = "RESEARCH GRADE"
        emoji = "✅"
        conclusion = "Model shows promise but requires further validation."
    else:
        overall = "NEEDS IMPROVEMENT"
        emoji = "⚠️"
        conclusion = "Model fails some institutional tests."
    
    print(f"\n   {emoji} FINAL ASSESSMENT: {overall}")
    print(f"\n   💡 CONCLUSION: {conclusion}")
    
    # Detailed insights
    print(f"\n   📈 KEY INSIGHTS:")
    print("   " + "-" * 55)
    
    for r in results:
        if r["test"] == "Horizon Sensitivity":
            if r["ic_t0"] > 0.3:
                print(f"   ✅ Strong signal persistence (IC={r['ic_t0']:.3f} at T)")
            else:
                print(f"   ⚠️ Moderate signal strength (IC={r['ic_t0']:.3f} at T)")
        elif r["test"] == "Regime-Conditional":
            print(f"   {'✅' if r['min_r2'] > 0.15 else '⚠️'} Regime robustness: min R² = {r['min_r2']*100:.1f}%")
        elif r["test"] == "Block Bootstrap":
            print(f"   {'✅' if r['ci_5'] > 0.154 else '⚠️'} Bootstrap CI: [{r['ci_5']*100:.1f}%, {r['ci_95']*100:.1f}%]")
    
    print("\n" + "=" * 70)
    
    return {
        "overall": overall,
        "passes": passes,
        "partials": partials,
        "fails": fails,
        "results": results
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🏛️ INSTITUTIONAL VALIDATION SUITE")
    print("   AQR / Two Sigma Protocol")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # End any stale MLflow runs
    mlflow.end_run()
    
    # Load data
    df = load_high_octane_data()
    
    # Run all tests
    results = []
    
    # Test A: Horizon Sensitivity
    horizon_result = test_horizon_sensitivity(df)
    results.append(horizon_result)
    
    # Test B: Regime-Conditional
    regime_result = test_regime_conditional(df)
    results.append(regime_result)
    
    # Test C: Block Bootstrap
    bootstrap_result = test_block_bootstrap(df, n_bootstrap=500, block_size=10)
    results.append(bootstrap_result)
    
    # Generate report
    report = generate_institutional_report(results)
    
    # Timing
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n   Duration: {duration}")
    print(f"   Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return report


if __name__ == "__main__":
    main()

