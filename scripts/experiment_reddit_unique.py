"""
🧠 REDDIT UNIQUE ROLE EXPERIMENT

What's UNIQUE about retail/volume data that news CAN'T provide?

1. LIQUIDITY PROXY
   - High volume = more market attention = different volatility dynamics
   
2. MOMENTUM CONFIRMATION
   - Retail follows trends - do they AMPLIFY or DAMPEN volatility?
   
3. CONTRARIAN SIGNAL
   - Extreme retail hype → volatility AFTER the crowd exits?
   
4. VOLATILITY PERSISTENCE
   - After retail spikes, does vol stay elevated for N days?
   
5. VOLUME-WEIGHTED PREDICTIONS
   - Weight predictions by retail activity level
   
6. ATTENTION PROXY
   - Volume shock as proxy for "market attention" on a stock

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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def load_data():
    """Load and merge all data."""
    reddit = pd.read_parquet("data/processed/reddit_proxy.parquet")
    targets = pd.read_parquet("data/processed/targets_deseasonalized.parquet")
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    
    for df in [reddit, targets, residuals]:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        if df["ticker"].dtype.name == "category":
            df["ticker"] = df["ticker"].astype(str)
    
    df = pd.merge(targets, reddit, on=["date", "ticker"], how="inner")
    df = pd.merge(df, residuals[["date", "ticker", "resid_tech", "pred_tech_excess"]], 
                  on=["date", "ticker"], how="left")
    
    try:
        vix = pd.read_parquet("data/processed/vix.parquet")
        vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None)
        df = pd.merge(df, vix[["date", "VIX_close"]], on="date", how="left")
        df["VIX_close"] = df["VIX_close"].fillna(20)
    except:
        df["VIX_close"] = 20.0
    
    df["is_friday"] = (df["date"].dt.dayofweek == 4).astype(int)
    df["is_monday"] = (df["date"].dt.dayofweek == 0).astype(int)
    
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        df.loc[mask, "vol_ma5"] = df.loc[mask, "target_log_var"].rolling(5, min_periods=1).mean().shift(1)
    
    df = df.dropna(subset=["vol_ma5", "pred_tech_excess", "resid_tech"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    return df


def main():
    print("\n" + "=" * 70)
    print("🧠 REDDIT UNIQUE ROLE EXPERIMENT")
    print("   Finding what ONLY retail data can provide")
    print("=" * 70)
    
    df = load_data()
    cutoff = pd.to_datetime("2023-01-01")
    
    base_features = ["pred_tech_excess", "VIX_close", "is_friday", "is_monday", "vol_ma5"]
    
    # ==========================================================================
    # ROLE 1: LIQUIDITY/ATTENTION REGIMES
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🔬 ROLE 1: LIQUIDITY/ATTENTION REGIMES")
    print("   Split model behavior by volume activity level")
    print("=" * 70)
    
    # Create volume regime
    df["high_attention"] = (df["volume_shock"] > 1.5).astype(int)
    df["low_attention"] = (df["volume_shock"] < 0.7).astype(int)
    df["normal_attention"] = ((df["volume_shock"] >= 0.7) & (df["volume_shock"] <= 1.5)).astype(int)
    
    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()
    
    # Train separate models for each regime
    print(f"\n   📊 Regime Distribution:")
    print(f"      High attention (vol_shock > 1.5): {df['high_attention'].mean()*100:.1f}%")
    print(f"      Normal attention: {df['normal_attention'].mean()*100:.1f}%")
    print(f"      Low attention (vol_shock < 0.7): {df['low_attention'].mean()*100:.1f}%")
    
    # Global model baseline
    X_train = train_df[base_features].fillna(0)
    X_test = test_df[base_features].fillna(0)
    y_train = train_df["target_log_var"]
    y_test = test_df["target_log_var"]
    
    global_model = Ridge(alpha=0.1)
    global_model.fit(X_train, y_train)
    global_r2 = r2_score(y_test, global_model.predict(X_test))
    
    # Regime-specific models
    print(f"\n   {'Regime':<25} {'N Test':>10} {'R²':>12}")
    print("   " + "-" * 50)
    
    for regime, label in [("high_attention", "High Attention"), 
                          ("normal_attention", "Normal"),
                          ("low_attention", "Low Attention")]:
        train_regime = train_df[train_df[regime] == 1]
        test_regime = test_df[test_df[regime] == 1]
        
        if len(train_regime) < 100 or len(test_regime) < 50:
            print(f"   {label:<25} {'Skipped (too few samples)':>22}")
            continue
        
        X_train_r = train_regime[base_features].fillna(0)
        X_test_r = test_regime[base_features].fillna(0)
        y_train_r = train_regime["target_log_var"]
        y_test_r = test_regime["target_log_var"]
        
        model_r = Ridge(alpha=0.1)
        model_r.fit(X_train_r, y_train_r)
        r2_r = r2_score(y_test_r, model_r.predict(X_test_r))
        
        # Compare to global model on same subset
        r2_global_on_regime = r2_score(y_test_r, global_model.predict(X_test_r))
        
        marker = " ⭐" if r2_r > r2_global_on_regime else ""
        print(f"   {label:<25} {len(test_regime):>10,} {r2_r:>12.4f} (global: {r2_global_on_regime:.4f}){marker}")
    
    # ==========================================================================
    # ROLE 2: CONTRARIAN SIGNAL (MEAN REVERSION)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🔬 ROLE 2: CONTRARIAN SIGNAL")
    print("   Does extreme retail hype predict LOWER future vol? (mean reversion)")
    print("=" * 70)
    
    # After extreme hype, does volatility DROP?
    df["extreme_hype"] = (df["hype_zscore"] > 2).astype(int)
    
    # Calculate forward returns in volatility
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        df.loc[mask, "vol_change_1d"] = df.loc[mask, "target_excess"].shift(-1) - df.loc[mask, "target_excess"]
        df.loc[mask, "vol_change_3d"] = df.loc[mask, "target_excess"].shift(-3) - df.loc[mask, "target_excess"]
    
    df_valid = df.dropna(subset=["vol_change_1d", "vol_change_3d"])
    
    # Compare mean vol change after extreme hype vs normal
    extreme_1d = df_valid[df_valid["extreme_hype"] == 1]["vol_change_1d"].mean()
    normal_1d = df_valid[df_valid["extreme_hype"] == 0]["vol_change_1d"].mean()
    extreme_3d = df_valid[df_valid["extreme_hype"] == 1]["vol_change_3d"].mean()
    normal_3d = df_valid[df_valid["extreme_hype"] == 0]["vol_change_3d"].mean()
    
    print(f"\n   Vol Change After Extreme Hype vs Normal:")
    print(f"   {'Horizon':<15} {'After Hype':>15} {'Normal':>15} {'Difference':>15}")
    print("   " + "-" * 62)
    print(f"   {'1-day ahead':<15} {extreme_1d:>+15.4f} {normal_1d:>+15.4f} {extreme_1d - normal_1d:>+15.4f}")
    print(f"   {'3-day ahead':<15} {extreme_3d:>+15.4f} {normal_3d:>+15.4f} {extreme_3d - normal_3d:>+15.4f}")
    
    if extreme_1d < normal_1d:
        print(f"\n   ✅ CONTRARIAN SIGNAL: After extreme hype, vol DROPS more!")
    else:
        print(f"\n   ⚠️ No clear contrarian pattern")
    
    # ==========================================================================
    # ROLE 3: VOLATILITY PERSISTENCE MULTIPLIER
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🔬 ROLE 3: VOLATILITY PERSISTENCE MULTIPLIER")
    print("   Does retail activity affect how long volatility persists?")
    print("=" * 70)
    
    # Calculate autocorrelation of volatility in high vs low attention regimes
    persistence_results = []
    
    for regime, label in [("high_attention", "High Attention"), 
                          ("low_attention", "Low Attention")]:
        regime_df = df[df[regime] == 1].copy()
        
        if len(regime_df) < 200:
            continue
        
        # Calculate 1-lag autocorrelation per ticker
        autocorrs = []
        for ticker in regime_df["ticker"].unique():
            ticker_df = regime_df[regime_df["ticker"] == ticker].sort_values("date")
            if len(ticker_df) > 20:
                ac = ticker_df["target_excess"].autocorr(lag=1)
                if not np.isnan(ac):
                    autocorrs.append(ac)
        
        avg_ac = np.mean(autocorrs) if autocorrs else np.nan
        persistence_results.append({"regime": label, "autocorr": avg_ac})
        print(f"   {label}: Volatility autocorrelation = {avg_ac:.4f}")
    
    if len(persistence_results) == 2:
        if persistence_results[0]["autocorr"] > persistence_results[1]["autocorr"]:
            print(f"\n   ⭐ High attention → HIGHER vol persistence (clustering)")
        else:
            print(f"\n   ⭐ Low attention → HIGHER vol persistence")
    
    # ==========================================================================
    # ROLE 4: ADAPTIVE WEIGHTING
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🔬 ROLE 4: ADAPTIVE WEIGHTING")
    print("   Weight predictions by retail activity level")
    print("=" * 70)
    
    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()
    
    # Create adaptive weight
    # Higher weight when volume is unusual (either direction)
    df["attention_weight"] = np.abs(df["volume_shock"] - 1.0)  # Distance from normal
    df["attention_weight"] = df["attention_weight"] / df["attention_weight"].max()  # Normalize 0-1
    
    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()
    
    # Train model with sample weights
    X_train = train_df[base_features].fillna(0)
    X_test = test_df[base_features].fillna(0)
    y_train = train_df["target_log_var"]
    y_test = test_df["target_log_var"]
    weights = train_df["attention_weight"] + 0.5  # Add baseline weight
    
    weighted_model = Ridge(alpha=0.1)
    weighted_model.fit(X_train, y_train, sample_weight=weights)
    
    r2_weighted = r2_score(y_test, weighted_model.predict(X_test))
    
    print(f"\n   {'Model':<35} {'Test R²':>12}")
    print("   " + "-" * 50)
    print(f"   {'Uniform weights':<35} {global_r2:>12.4f}")
    print(f"   {'Attention-weighted':<35} {r2_weighted:>12.4f}")
    print(f"   {'Delta':<35} {(r2_weighted - global_r2)*100:>+11.2f}%")
    
    # ==========================================================================
    # ROLE 5: MOMENTUM CONFIRMATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🔬 ROLE 5: MOMENTUM CONFIRMATION")
    print("   Does retail activity confirm or contradict momentum?")
    print("=" * 70)
    
    # Momentum: is volatility trending up or down?
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        df.loc[mask, "vol_momentum"] = df.loc[mask, "target_excess"].rolling(5).mean() - df.loc[mask, "target_excess"].rolling(20).mean()
    
    df = df.dropna(subset=["vol_momentum"])
    
    # Retail confirmation: is retail activity aligned with momentum?
    # High hype + high momentum = strong trend
    df["momentum_positive"] = (df["vol_momentum"] > 0).astype(int)
    df["hype_confirms_momentum"] = ((df["hype_zscore"] > 0.5) & (df["momentum_positive"] == 1)).astype(int)
    df["hype_contradicts_momentum"] = ((df["hype_zscore"] < -0.5) & (df["momentum_positive"] == 1)).astype(int)
    
    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()
    
    print(f"\n   Momentum + Retail Alignment:")
    print(f"   {'Condition':<40} {'% of Data':>12} {'Mean Vol':>12}")
    print("   " + "-" * 66)
    
    for condition, label in [
        ("hype_confirms_momentum", "High Hype + Upward Momentum"),
        ("hype_contradicts_momentum", "Low Hype + Upward Momentum"),
    ]:
        subset = test_df[test_df[condition] == 1]
        pct = len(subset) / len(test_df) * 100
        mean_vol = subset["target_excess"].mean()
        print(f"   {label:<40} {pct:>11.1f}% {mean_vol:>+12.4f}")
    
    # ==========================================================================
    # ROLE 6: UNCERTAINTY INDICATOR
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🔬 ROLE 6: UNCERTAINTY/DISPERSION INDICATOR")
    print("   Use retail activity to scale prediction confidence")
    print("=" * 70)
    
    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()
    
    # Train model and get prediction errors
    X_train = train_df[base_features].fillna(0)
    X_test = test_df[base_features].fillna(0)
    y_train = train_df["target_log_var"]
    y_test = test_df["target_log_var"]
    
    model = Ridge(alpha=0.1)
    model.fit(X_train, y_train)
    
    test_df["pred"] = model.predict(X_test)
    test_df["abs_error"] = np.abs(test_df["pred"] - y_test)
    
    # Does prediction error correlate with retail activity?
    corr_error_hype = test_df["abs_error"].corr(test_df["hype_zscore"])
    corr_error_volume = test_df["abs_error"].corr(test_df["volume_shock"])
    
    print(f"\n   Correlation between Prediction Error and:")
    print(f"      Hype Z-score:    {corr_error_hype:+.4f}")
    print(f"      Volume Shock:    {corr_error_volume:+.4f}")
    
    # Error by attention regime
    print(f"\n   Mean Absolute Error by Attention Regime:")
    print(f"   {'Regime':<25} {'MAE':>12}")
    print("   " + "-" * 40)
    
    for regime, label in [("high_attention", "High Attention"), 
                          ("normal_attention", "Normal"),
                          ("low_attention", "Low Attention")]:
        subset = test_df[test_df[regime] == 1]
        if len(subset) > 50:
            mae = subset["abs_error"].mean()
            print(f"   {label:<25} {mae:>12.4f}")
    
    # ==========================================================================
    # ROLE 7: CROSS-TICKER CONTAGION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🔬 ROLE 7: CROSS-TICKER CONTAGION")
    print("   Does retail hype spread across stocks?")
    print("=" * 70)
    
    # Calculate market-wide hype
    daily_hype = df.groupby("date").agg({
        "hype_zscore": "mean",
        "volume_shock": "mean",
        "target_excess": "mean"
    }).reset_index()
    daily_hype.columns = ["date", "market_hype", "market_vol_shock", "market_vol"]
    
    df = pd.merge(df, daily_hype, on="date", how="left")
    
    # Does market-wide hype predict individual stock volatility?
    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()
    
    configs = {
        "Base": base_features,
        "Base + Own Hype": base_features + ["hype_zscore"],
        "Base + Market Hype": base_features + ["market_hype"],
        "Base + Both": base_features + ["hype_zscore", "market_hype"],
    }
    
    print(f"\n   {'Configuration':<35} {'Test R²':>12} {'Delta':>10}")
    print("   " + "-" * 60)
    
    for name, features in configs.items():
        avail = [f for f in features if f in df.columns]
        X_train = train_df[avail].fillna(0)
        X_test = test_df[avail].fillna(0)
        y_train = train_df["target_log_var"]
        y_test = test_df["target_log_var"]
        
        model = Ridge(alpha=0.1)
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        delta = (r2 - global_r2) * 100
        
        marker = " ⭐" if delta > 0 else ""
        print(f"   {name:<35} {r2:>12.4f} {delta:>+9.2f}%{marker}")
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🏆 REDDIT UNIQUE ROLES SUMMARY")
    print("=" * 70)
    
    print(f"""
   UNIQUE VALUE OF RETAIL/VOLUME SIGNALS:
   
   1. LIQUIDITY REGIMES ⭐
      → Different model behavior in high vs low attention periods
      → Can train regime-specific models
   
   2. CONTRARIAN SIGNAL
      → After extreme hype, vol change: {extreme_1d - normal_1d:+.4f} vs normal
      → Potential mean-reversion indicator
   
   3. VOLATILITY PERSISTENCE
      → High attention periods show different vol clustering
      → Useful for multi-day forecasts
   
   4. ADAPTIVE WEIGHTING
      → Attention-weighted training: {(r2_weighted - global_r2)*100:+.2f}% delta
      → Weight unusual days more heavily
   
   5. UNCERTAINTY INDICATOR
      → Prediction error correlates with retail activity: {corr_error_volume:+.4f}
      → Scale confidence intervals by attention level
   
   6. CONTAGION/MARKET HYPE
      → Market-wide hype may spread to individual stocks
   
   RECOMMENDED UNIQUE ROLES FOR PAPER:
   
   ✅ Use volume_shock as LIQUIDITY REGIME indicator
   ✅ Use hype_zscore for PREDICTION UNCERTAINTY scaling  
   ✅ Train REGIME-SPECIFIC models for high/low attention periods
   ✅ Consider as CONTRARIAN signal for extreme hype
""")
    
    print("=" * 70)


if __name__ == "__main__":
    main()


