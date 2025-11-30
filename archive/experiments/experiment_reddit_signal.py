"""
🔍 REDDIT/RETAIL SIGNAL AUDIT: Finding Hidden Value

The Reddit proxy contains:
- volume_shock: Volume / 20-day MA (detects unusual trading activity)
- hype_signal: volume_shock × |price_acceleration| (retail frenzy indicator)
- hype_zscore: Normalized hype signal
- price_acceleration: Second derivative of log price

Hypotheses to Test:
1. Direct Correlation - Does hype predict next-day volatility?
2. Extreme Events - Does high hype predict extreme vol days?
3. Regime Interaction - Does hype matter more in high-VIX periods?
4. Sector Sensitivity - Which sectors respond to retail activity?
5. Lagged Effects - Does hype have delayed impact (1-3 days)?
6. Combined Signal - Does hype add value on top of news?
7. Classification - Predict high-vol days from hype
8. Coordinator Integration - Add hype to the ensemble

NO PERMANENT CHANGES - EXPERIMENTS ONLY
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score
from lightgbm import LGBMRegressor, LGBMClassifier


def load_all_data():
    """Load all required data."""
    print("\n📂 Loading data...")
    
    # Reddit proxy
    reddit = pd.read_parquet("data/processed/reddit_proxy.parquet")
    reddit["date"] = pd.to_datetime(reddit["date"]).dt.tz_localize(None)
    print(f"   ✓ Reddit proxy: {len(reddit):,} rows")
    
    # Targets
    targets = pd.read_parquet("data/processed/targets_deseasonalized.parquet")
    targets["date"] = pd.to_datetime(targets["date"]).dt.tz_localize(None)
    print(f"   ✓ Targets: {len(targets):,} rows")
    
    # News features
    news = pd.read_parquet("data/processed/news_features.parquet")
    news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(None)
    print(f"   ✓ News: {len(news):,} rows")
    
    # Residuals
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    residuals["date"] = pd.to_datetime(residuals["date"]).dt.tz_localize(None)
    print(f"   ✓ Residuals: {len(residuals):,} rows")
    
    # Normalize tickers
    for df in [reddit, targets, news, residuals]:
        if df["ticker"].dtype.name == "category":
            df["ticker"] = df["ticker"].astype(str)
    
    # Merge all
    df = pd.merge(targets, reddit, on=["date", "ticker"], how="inner")
    df = pd.merge(df, news[["date", "ticker", "shock_index", "news_count", "sentiment_avg"]], 
                  on=["date", "ticker"], how="left")
    df = pd.merge(df, residuals[["date", "ticker", "resid_tech", "pred_tech_excess"]], 
                  on=["date", "ticker"], how="left")
    
    # Fill NaN
    for col in ["shock_index", "news_count", "sentiment_avg"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Add VIX
    try:
        vix = pd.read_parquet("data/processed/vix.parquet")
        vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None)
        df = pd.merge(df, vix[["date", "VIX_close"]], on="date", how="left")
        df["VIX_close"] = df["VIX_close"].fillna(20)
    except:
        df["VIX_close"] = 20.0
    
    # Add sector
    SECTOR_MAP = {
        'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech',
        'JPM': 'Finance', 'BAC': 'Finance', 'V': 'Finance',
        'CAT': 'Industrial', 'GE': 'Industrial', 'BA': 'Industrial',
        'WMT': 'Consumer', 'MCD': 'Consumer', 'COST': 'Consumer',
        'XOM': 'Energy', 'CVX': 'Energy', 'SLB': 'Energy',
        'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare'
    }
    df["sector"] = df["ticker"].map(SECTOR_MAP)
    
    # Calendar features
    df["is_friday"] = (df["date"].dt.dayofweek == 4).astype(int)
    df["is_monday"] = (df["date"].dt.dayofweek == 0).astype(int)
    
    df = df.dropna(subset=["target_excess", "volume_shock"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    print(f"\n   📊 Merged: {len(df):,} rows")
    
    return df


def main():
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🔍 REDDIT/RETAIL SIGNAL AUDIT")
    print("   Finding Hidden Value in Volume/Hype Signals")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    df = load_all_data()
    
    # Time split
    cutoff = pd.to_datetime("2023-01-01")
    train_mask = df["date"] < cutoff
    test_mask = df["date"] >= cutoff
    
    all_results = []
    
    # ==========================================================================
    # EXPERIMENT 1: DIRECT CORRELATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 1: DIRECT CORRELATION")
    print("   Do retail signals correlate with next-day volatility?")
    print("=" * 70)
    
    reddit_features = ["volume_shock", "hype_signal", "hype_zscore", 
                       "price_acceleration", "volume_shock_roll3", "hype_signal_roll7"]
    
    targets_to_check = ["target_excess", "resid_tech", "realized_vol"]
    
    print(f"\n   CORRELATIONS (Test Set):")
    print(f"   {'Feature':<25} {'target_excess':>15} {'resid_tech':>15} {'realized_vol':>15}")
    print("   " + "-" * 73)
    
    test_df = df[test_mask].copy()
    
    for feat in reddit_features:
        if feat not in test_df.columns:
            continue
        corrs = []
        for target in targets_to_check:
            if target in test_df.columns:
                corr = test_df[feat].corr(test_df[target])
                corrs.append(corr)
            else:
                corrs.append(np.nan)
        
        print(f"   {feat:<25} {corrs[0]:>+15.4f} {corrs[1]:>+15.4f} {corrs[2]:>+15.4f}")
    
    # ==========================================================================
    # EXPERIMENT 2: REDDIT ONLY REGRESSION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 2: REDDIT ONLY REGRESSION")
    print("   Can Reddit features alone predict volatility?")
    print("=" * 70)
    
    reddit_feats = [f for f in reddit_features if f in df.columns]
    
    X_train = df.loc[train_mask, reddit_feats].fillna(0)
    X_test = df.loc[test_mask, reddit_feats].fillna(0)
    y_train = df.loc[train_mask, "target_excess"]
    y_test = df.loc[test_mask, "target_excess"]
    
    # Test different targets
    for target in ["target_excess", "resid_tech"]:
        if target not in df.columns:
            continue
        
        y_train_t = df.loc[train_mask, target].fillna(0)
        y_test_t = df.loc[test_mask, target].fillna(0)
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train_t)
        
        train_r2 = r2_score(y_train_t, model.predict(X_train))
        test_r2 = r2_score(y_test_t, model.predict(X_test))
        
        print(f"\n   Target: {target}")
        print(f"   Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
        
        all_results.append({"exp": f"Reddit → {target}", "r2": test_r2})
    
    # ==========================================================================
    # EXPERIMENT 3: EXTREME EVENTS CLASSIFICATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 3: EXTREME EVENTS CLASSIFICATION")
    print("   Can Reddit signals predict extreme volatility days?")
    print("=" * 70)
    
    # Create extreme event target
    df["is_extreme_vol"] = (df["target_excess"] > df["target_excess"].quantile(0.80)).astype(int)
    df["is_high_hype"] = (df["hype_zscore"] > 1.0).astype(int)
    
    # Cross-tabulation
    train_df = df[train_mask]
    ct = pd.crosstab(train_df["is_high_hype"], train_df["is_extreme_vol"], normalize="index")
    
    print(f"\n   Cross-tabulation (Train Set):")
    print(f"   P(Extreme Vol | High Hype):  {ct.loc[1, 1]*100:.1f}%")
    print(f"   P(Extreme Vol | Normal):     {ct.loc[0, 1]*100:.1f}%")
    print(f"   Lift: {(ct.loc[1, 1] / ct.loc[0, 1]):.2f}x")
    
    # Classification model
    X_train_c = df.loc[train_mask, reddit_feats].fillna(0)
    X_test_c = df.loc[test_mask, reddit_feats].fillna(0)
    y_train_c = df.loc[train_mask, "is_extreme_vol"]
    y_test_c = df.loc[test_mask, "is_extreme_vol"]
    
    clf = LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=-1)
    clf.fit(X_train_c, y_train_c)
    
    y_pred_proba = clf.predict_proba(X_test_c)[:, 1]
    auc = roc_auc_score(y_test_c, y_pred_proba)
    
    print(f"\n   Classification (Extreme Vol from Reddit):")
    print(f"   Test AUC: {auc:.4f}")
    
    if auc > 0.55:
        print(f"   ✅ Reddit signals have predictive power for extreme events!")
    else:
        print(f"   ⚠️ Weak signal for extreme events")
    
    all_results.append({"exp": "Reddit → Extreme Vol (AUC)", "r2": auc - 0.5})
    
    # ==========================================================================
    # EXPERIMENT 4: SECTOR ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 4: SECTOR SENSITIVITY")
    print("   Which sectors respond most to retail activity?")
    print("=" * 70)
    
    print(f"\n   {'Sector':<15} {'N Test':>10} {'Correlation':>12} {'R²':>12}")
    print("   " + "-" * 52)
    
    sector_results = []
    
    for sector in df["sector"].dropna().unique():
        sector_df = df[df["sector"] == sector]
        sector_test = sector_df[sector_df["date"] >= cutoff]
        
        if len(sector_test) < 50:
            continue
        
        corr = sector_test["hype_zscore"].corr(sector_test["target_excess"])
        
        # Quick regression
        X = sector_df.loc[sector_df["date"] < cutoff, ["hype_zscore", "volume_shock"]].fillna(0)
        y = sector_df.loc[sector_df["date"] < cutoff, "target_excess"]
        X_test_s = sector_test[["hype_zscore", "volume_shock"]].fillna(0)
        y_test_s = sector_test["target_excess"]
        
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        r2 = r2_score(y_test_s, model.predict(X_test_s))
        
        print(f"   {sector:<15} {len(sector_test):>10,} {corr:>+12.4f} {r2:>12.4f}")
        sector_results.append({"sector": sector, "corr": corr, "r2": r2})
    
    best_sector = max(sector_results, key=lambda x: x["r2"])
    print(f"\n   Best sector: {best_sector['sector']} (R² = {best_sector['r2']:.4f})")
    
    # ==========================================================================
    # EXPERIMENT 5: LAGGED EFFECTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 5: LAGGED EFFECTS")
    print("   Does hype have delayed impact (1-3 days)?")
    print("=" * 70)
    
    df_lag = df.copy()
    
    # Create lagged hype features
    for ticker in df_lag["ticker"].unique():
        mask = df_lag["ticker"] == ticker
        for lag in [1, 2, 3]:
            df_lag.loc[mask, f"hype_lag{lag}"] = df_lag.loc[mask, "hype_zscore"].shift(lag)
    
    df_lag = df_lag.dropna(subset=["hype_lag1", "hype_lag2", "hype_lag3"])
    
    train_lag = df_lag[df_lag["date"] < cutoff]
    test_lag = df_lag[df_lag["date"] >= cutoff]
    
    print(f"\n   {'Lag':>6} {'Correlation':>15} {'R² (alone)':>15}")
    print("   " + "-" * 38)
    
    for lag in [0, 1, 2, 3]:
        if lag == 0:
            feat = "hype_zscore"
        else:
            feat = f"hype_lag{lag}"
        
        corr = test_lag[feat].corr(test_lag["target_excess"])
        
        X_train = train_lag[[feat]].fillna(0)
        X_test = test_lag[[feat]].fillna(0)
        y_train = train_lag["target_excess"]
        y_test = test_lag["target_excess"]
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        
        marker = " ⭐" if r2 > 0 else ""
        print(f"   {lag:>6} {corr:>+15.4f} {r2:>15.4f}{marker}")
        
        all_results.append({"exp": f"Hype Lag {lag}", "r2": r2})
    
    # ==========================================================================
    # EXPERIMENT 6: COMBINED WITH NEWS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 6: REDDIT + NEWS COMBINED")
    print("   Does hype add value on top of news features?")
    print("=" * 70)
    
    # Feature sets
    news_only = ["shock_index", "news_count", "sentiment_avg"]
    reddit_only = ["hype_zscore", "volume_shock"]
    combined = news_only + reddit_only
    
    available_news = [f for f in news_only if f in df.columns]
    available_reddit = [f for f in reddit_only if f in df.columns]
    available_combined = available_news + available_reddit
    
    train_df = df[train_mask]
    test_df = df[test_mask]
    
    print(f"\n   {'Feature Set':<25} {'Test R²':>15}")
    print("   " + "-" * 42)
    
    feature_sets = {
        "News Only": available_news,
        "Reddit Only": available_reddit,
        "News + Reddit": available_combined,
    }
    
    for name, feats in feature_sets.items():
        if not feats:
            continue
        
        X_train = train_df[feats].fillna(0)
        X_test = test_df[feats].fillna(0)
        y_train = train_df["target_excess"]
        y_test = test_df["target_excess"]
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        
        print(f"   {name:<25} {r2:>15.4f}")
        all_results.append({"exp": name, "r2": r2})
    
    # ==========================================================================
    # EXPERIMENT 7: ADD TO COORDINATOR
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 7: COORDINATOR INTEGRATION")
    print("   Does hype improve the full ensemble?")
    print("=" * 70)
    
    # Add momentum and calendar
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        df.loc[mask, "vol_ma5"] = df.loc[mask, "target_log_var"].rolling(5, min_periods=1).mean().shift(1)
    
    df = df.dropna(subset=["vol_ma5", "pred_tech_excess"])
    
    # Re-split after adding features
    train_df = df[df["date"] < cutoff]
    test_df = df[df["date"] >= cutoff]
    
    # Feature sets to compare
    base_features = ["pred_tech_excess", "VIX_close", "is_friday", "is_monday", "vol_ma5"]
    with_hype = base_features + ["hype_zscore", "volume_shock"]
    with_all = with_hype + ["shock_index"]
    
    available_base = [f for f in base_features if f in df.columns]
    available_hype = [f for f in with_hype if f in df.columns]
    available_all = [f for f in with_all if f in df.columns]
    
    print(f"\n   {'Configuration':<35} {'Test R² (total)':>18}")
    print("   " + "-" * 55)
    
    configs = {
        "Coordinator (Base)": available_base,
        "Coordinator + Hype": available_hype,
        "Coordinator + Hype + News": available_all,
    }
    
    for name, feats in configs.items():
        X_train = train_df[feats].fillna(0)
        X_test = test_df[feats].fillna(0)
        y_train = train_df["target_log_var"]
        y_test = test_df["target_log_var"]
        
        model = Ridge(alpha=0.1)
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        
        marker = " ⭐" if "Hype" in name and r2 > 0.15 else ""
        print(f"   {name:<35} {r2:>18.4f}{marker}")
        all_results.append({"exp": name, "r2": r2})
    
    # ==========================================================================
    # EXPERIMENT 8: HYPE SPIKE EVENTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 8: HYPE SPIKE EVENTS")
    print("   What happens after a hype spike?")
    print("=" * 70)
    
    # Define hype spike (zscore > 2)
    df["hype_spike"] = (df["hype_zscore"] > 2).astype(int)
    
    # Get next-day volatility after spikes
    spike_days = df[df["hype_spike"] == 1]
    normal_days = df[df["hype_spike"] == 0]
    
    print(f"\n   Number of hype spikes: {len(spike_days):,} ({len(spike_days)/len(df)*100:.1f}%)")
    
    if len(spike_days) > 50:
        # Compare average vol
        spike_vol = spike_days["target_excess"].mean()
        normal_vol = normal_days["target_excess"].mean()
        
        print(f"\n   Average target_excess after hype spike: {spike_vol:.4f}")
        print(f"   Average target_excess on normal days:   {normal_vol:.4f}")
        print(f"   Difference: {spike_vol - normal_vol:+.4f}")
        
        if spike_vol > normal_vol:
            print(f"\n   ✅ Hype spikes ARE followed by higher volatility!")
        else:
            print(f"\n   ⚠️ No clear pattern after hype spikes")
    
    # ==========================================================================
    # EXPERIMENT 9: MOMENTUM INTERACTION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 9: HYPE × MOMENTUM INTERACTION")
    print("   Does hype matter more when volatility is already rising?")
    print("=" * 70)
    
    # Create interaction
    df["hype_x_volma"] = df["hype_zscore"] * df["vol_ma5"]
    df["hype_x_vix"] = df["hype_zscore"] * (df["VIX_close"] / 20)
    
    train_df = df[df["date"] < cutoff]
    test_df = df[df["date"] >= cutoff]
    
    interaction_features = ["hype_zscore", "vol_ma5", "VIX_close", 
                            "hype_x_volma", "hype_x_vix"]
    
    X_train = train_df[interaction_features].fillna(0)
    X_test = test_df[interaction_features].fillna(0)
    y_train = train_df["target_excess"]
    y_test = test_df["target_excess"]
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    
    print(f"\n   Hype with interactions R²: {r2:.4f}")
    
    # Print coefficients
    print(f"\n   Coefficients:")
    for i, feat in enumerate(interaction_features):
        print(f"      {feat:<20}: {model.coef_[i]:+.4f}")
    
    all_results.append({"exp": "Hype × Interactions", "r2": r2})
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🏆 REDDIT SIGNAL AUDIT SUMMARY")
    print("=" * 70)
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("r2", ascending=False)
    
    print(f"\n   {'Rank':<6} {'Experiment':<40} {'Score':>12}")
    print("   " + "-" * 60)
    
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        marker = " ⭐" if row["r2"] > 0 else ""
        print(f"   {i:<6} {row['exp']:<40} {row['r2']:>12.4f}{marker}")
    
    # Key findings
    positive_results = results_df[results_df["r2"] > 0]
    
    print("\n" + "=" * 70)
    print("💡 KEY FINDINGS")
    print("=" * 70)
    
    if len(positive_results) > 0:
        print(f"""
   ✅ REDDIT SIGNALS HAVE VALUE!
   
   {len(positive_results)} experiments showed positive contribution:
""")
        for _, row in positive_results.iterrows():
            print(f"      • {row['exp']}: {row['r2']:.4f}")
        
        print(f"""
   RECOMMENDATIONS:
   
   1. Add 'hype_zscore' to the Coordinator feature set
   2. Use 'volume_shock' as an additional risk indicator
   3. Consider hype spikes as regime flags
        """)
    else:
        print(f"""
   ⚠️ Reddit signals show limited standalone value
   
   BUT: May still add marginal value in combination with other features
        """)
    
    # Timing
    end_time = datetime.now()
    print(f"\n   Duration: {end_time - start_time}")
    print("=" * 70)
    
    return results_df


if __name__ == "__main__":
    main()


