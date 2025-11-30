"""
🚀 REDDIT CLASSIFIER EXPERIMENT

Key Finding from Audit:
- P(Extreme Vol | High Hype) = 53.5%
- P(Extreme Vol | Normal) = 26.5%
- Lift: 2.02x
- AUC: 0.6781

Strategy:
1. Create a retail_risk_score (0-1) like news_risk_score
2. Add it to the Coordinator
3. Test if it improves overall R²

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
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, f1_score
from lightgbm import LGBMClassifier, LGBMRegressor


def load_data():
    """Load and merge all data."""
    print("\n📂 Loading data...")
    
    # Load all datasets
    reddit = pd.read_parquet("data/processed/reddit_proxy.parquet")
    targets = pd.read_parquet("data/processed/targets_deseasonalized.parquet")
    news = pd.read_parquet("data/processed/news_features.parquet")
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    
    # Normalize dates
    for df in [reddit, targets, news, residuals]:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        if df["ticker"].dtype.name == "category":
            df["ticker"] = df["ticker"].astype(str)
    
    # Merge
    df = pd.merge(targets, reddit, on=["date", "ticker"], how="inner")
    df = pd.merge(df, news[["date", "ticker", "shock_index", "news_count", "sentiment_avg"]], 
                  on=["date", "ticker"], how="left")
    df = pd.merge(df, residuals[["date", "ticker", "resid_tech", "pred_tech_excess"]], 
                  on=["date", "ticker"], how="left")
    
    # Fill NaN
    df["shock_index"] = df["shock_index"].fillna(0)
    df["news_count"] = df["news_count"].fillna(0)
    df["sentiment_avg"] = df["sentiment_avg"].fillna(0)
    
    # VIX
    try:
        vix = pd.read_parquet("data/processed/vix.parquet")
        vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None)
        df = pd.merge(df, vix[["date", "VIX_close"]], on="date", how="left")
        df["VIX_close"] = df["VIX_close"].fillna(20)
    except:
        df["VIX_close"] = 20.0
    
    # Calendar
    df["is_friday"] = (df["date"].dt.dayofweek == 4).astype(int)
    df["is_monday"] = (df["date"].dt.dayofweek == 0).astype(int)
    df["is_q4"] = (df["date"].dt.quarter == 4).astype(int)
    
    # Momentum
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        df.loc[mask, "vol_ma5"] = df.loc[mask, "target_log_var"].rolling(5, min_periods=1).mean().shift(1)
    
    df = df.dropna(subset=["vol_ma5", "pred_tech_excess", "resid_tech"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    print(f"   ✓ Merged dataset: {len(df):,} rows")
    
    return df


def create_retail_risk_score(df, cutoff):
    """
    Train a classifier to predict extreme volatility from retail signals.
    Returns: retail_risk_score (probability of extreme event)
    """
    print("\n" + "=" * 70)
    print("🎯 TRAINING RETAIL RISK CLASSIFIER")
    print("=" * 70)
    
    # Create target: extreme volatility (>80th percentile)
    threshold = df["resid_tech"].quantile(0.80)
    df["is_extreme"] = (df["resid_tech"] > threshold).astype(int)
    
    print(f"\n   Extreme vol threshold: 80th percentile = {threshold:.4f}")
    print(f"   Extreme events: {df['is_extreme'].sum():,} ({df['is_extreme'].mean():.1%})")
    
    # Features for retail classifier
    retail_features = [
        "volume_shock", "volume_shock_roll3",
        "hype_signal", "hype_signal_roll3", "hype_signal_roll7",
        "hype_zscore", "price_acceleration"
    ]
    
    available_features = [f for f in retail_features if f in df.columns]
    
    # Split
    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()
    
    print(f"\n   Train: {len(train_df):,} samples ({train_df['is_extreme'].mean():.1%} extreme)")
    print(f"   Test:  {len(test_df):,} samples ({test_df['is_extreme'].mean():.1%} extreme)")
    
    X_train = train_df[available_features].fillna(0)
    y_train = train_df["is_extreme"]
    X_test = test_df[available_features].fillna(0)
    y_test = test_df["is_extreme"]
    
    # Train classifier
    clf = LGBMClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
        is_unbalance=True
    )
    
    clf.fit(X_train, y_train)
    
    # Get probabilities
    train_proba = clf.predict_proba(X_train)[:, 1]
    test_proba = clf.predict_proba(X_test)[:, 1]
    
    # Metrics
    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)
    
    print(f"\n   📈 Classifier Results:")
    print(f"   {'Metric':<20} {'Train':>10} {'Test':>10}")
    print("   " + "-" * 42)
    print(f"   {'AUC':<20} {train_auc:>10.4f} {test_auc:>10.4f}")
    print(f"   {'Accuracy':<20} {accuracy_score(y_train, (train_proba > 0.5).astype(int)):>10.4f} {accuracy_score(y_test, (test_proba > 0.5).astype(int)):>10.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": available_features,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print(f"\n   📊 Feature Importance:")
    for _, row in importance.iterrows():
        print(f"      {row['feature']:<25}: {row['importance']:.4f}")
    
    # Add risk scores to full dataframe
    full_proba = clf.predict_proba(df[available_features].fillna(0))[:, 1]
    df["retail_risk_score"] = full_proba
    
    return df, clf, test_auc


def test_coordinator_integration(df, cutoff):
    """
    Test if retail_risk_score improves the Coordinator.
    """
    print("\n" + "=" * 70)
    print("🔧 COORDINATOR INTEGRATION TEST")
    print("=" * 70)
    
    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()
    
    # Simulate news_risk_score if not present
    if "news_risk_score" not in df.columns:
        print("\n   ⚠️ news_risk_score not found, using shock_index proxy")
        df["news_risk_score"] = (df["shock_index"] / df["shock_index"].max()).clip(0, 1)
        train_df = df[df["date"] < cutoff].copy()
        test_df = df[df["date"] >= cutoff].copy()
    
    # Feature configurations
    configs = {
        "Base (HAR + Calendar)": [
            "pred_tech_excess", "VIX_close", "is_friday", "is_monday", "is_q4", "vol_ma5"
        ],
        "Base + News Risk": [
            "pred_tech_excess", "VIX_close", "is_friday", "is_monday", "is_q4", "vol_ma5",
            "shock_index"
        ],
        "Base + Retail Risk": [
            "pred_tech_excess", "VIX_close", "is_friday", "is_monday", "is_q4", "vol_ma5",
            "retail_risk_score"
        ],
        "Base + Both Risks": [
            "pred_tech_excess", "VIX_close", "is_friday", "is_monday", "is_q4", "vol_ma5",
            "shock_index", "retail_risk_score"
        ],
        "Full (Both + Interactions)": [
            "pred_tech_excess", "VIX_close", "is_friday", "is_monday", "is_q4", "vol_ma5",
            "shock_index", "retail_risk_score", "hype_zscore", "volume_shock"
        ],
    }
    
    results = []
    
    print(f"\n   {'Configuration':<35} {'Test R²':>12} {'Delta':>12}")
    print("   " + "-" * 62)
    
    baseline_r2 = None
    
    for name, features in configs.items():
        available = [f for f in features if f in df.columns]
        
        X_train = train_df[available].fillna(0)
        X_test = test_df[available].fillna(0)
        y_train = train_df["target_log_var"]
        y_test = test_df["target_log_var"]
        
        model = Ridge(alpha=0.1)
        model.fit(X_train, y_train)
        
        r2 = r2_score(y_test, model.predict(X_test))
        
        if baseline_r2 is None:
            baseline_r2 = r2
            delta = 0
        else:
            delta = (r2 - baseline_r2) * 100
        
        marker = " ⭐" if delta > 0 else ""
        print(f"   {name:<35} {r2:>12.4f} {delta:>+11.2f}%{marker}")
        
        results.append({
            "config": name,
            "r2": r2,
            "delta": delta,
            "features": available
        })
    
    return results


def test_combined_classifier(df, cutoff):
    """
    Test a combined News+Retail classifier for extreme events.
    """
    print("\n" + "=" * 70)
    print("🔬 COMBINED NEWS+RETAIL CLASSIFIER")
    print("=" * 70)
    
    # Ensure is_extreme exists
    if "is_extreme" not in df.columns:
        threshold = df["resid_tech"].quantile(0.80)
        df["is_extreme"] = (df["resid_tech"] > threshold).astype(int)
    
    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()
    
    # Feature sets
    feature_sets = {
        "News Only": ["shock_index", "news_count", "sentiment_avg"],
        "Retail Only": ["volume_shock", "hype_zscore", "hype_signal"],
        "Combined": ["shock_index", "news_count", "sentiment_avg", 
                     "volume_shock", "hype_zscore", "hype_signal"],
    }
    
    print(f"\n   {'Feature Set':<25} {'Train AUC':>12} {'Test AUC':>12} {'Delta':>10}")
    print("   " + "-" * 62)
    
    baseline_auc = None
    
    for name, features in feature_sets.items():
        available = [f for f in features if f in df.columns]
        
        X_train = train_df[available].fillna(0)
        X_test = test_df[available].fillna(0)
        y_train = train_df["is_extreme"]
        y_test = test_df["is_extreme"]
        
        clf = LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=-1)
        clf.fit(X_train, y_train)
        
        train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
        test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        
        if baseline_auc is None:
            baseline_auc = test_auc
            delta = 0
        else:
            delta = (test_auc - baseline_auc) * 100
        
        marker = " ⭐" if name == "Combined" and delta > 0 else ""
        print(f"   {name:<25} {train_auc:>12.4f} {test_auc:>12.4f} {delta:>+9.2f}%{marker}")
    
    return test_auc


def test_sector_specific_hype(df, cutoff):
    """
    Test if hype signals matter more for specific sectors.
    """
    print("\n" + "=" * 70)
    print("📊 SECTOR-SPECIFIC HYPE ANALYSIS")
    print("=" * 70)
    
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
    
    print(f"\n   {'Sector':<15} {'Test AUC':>12} {'Hype Coef':>12} {'Improvement':>15}")
    print("   " + "-" * 57)
    
    sector_results = []
    
    for sector in df["sector"].dropna().unique():
        sector_df = df[df["sector"] == sector].copy()
        train_s = sector_df[sector_df["date"] < cutoff]
        test_s = sector_df[sector_df["date"] >= cutoff]
        
        if len(train_s) < 200 or len(test_s) < 100:
            continue
        
        # Ensure is_extreme is computed for this sector
        threshold = sector_df["resid_tech"].quantile(0.80)
        sector_df["is_extreme_sector"] = (sector_df["resid_tech"] > threshold).astype(int)
        train_s = sector_df[sector_df["date"] < cutoff]
        test_s = sector_df[sector_df["date"] >= cutoff]
        
        # Train classifier
        features = ["hype_zscore", "volume_shock"]
        X_train = train_s[features].fillna(0)
        X_test = test_s[features].fillna(0)
        y_train = train_s["is_extreme_sector"]
        y_test = test_s["is_extreme_sector"]
        
        clf = LGBMClassifier(n_estimators=50, max_depth=2, random_state=42, verbose=-1)
        clf.fit(X_train, y_train)
        
        test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        
        # Also get linear coefficient for hype
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        hype_coef = lr.coef_[0][0]  # hype_zscore coefficient
        
        improvement = (test_auc - 0.5) * 100  # Above random
        marker = " ⭐" if test_auc > 0.65 else ""
        
        print(f"   {sector:<15} {test_auc:>12.4f} {hype_coef:>+12.4f} {improvement:>+14.2f}%{marker}")
        sector_results.append({"sector": sector, "auc": test_auc})
    
    return sector_results


def main():
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🚀 REDDIT/RETAIL CLASSIFIER EXPERIMENT")
    print("   Creating retail_risk_score for the Coordinator")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load data
    df = load_data()
    cutoff = pd.to_datetime("2023-01-01")
    
    # Create retail risk score
    df, clf, retail_auc = create_retail_risk_score(df, cutoff)
    
    # Test Coordinator integration
    coord_results = test_coordinator_integration(df, cutoff)
    
    # Test combined classifier
    combined_auc = test_combined_classifier(df, cutoff)
    
    # Sector analysis
    sector_results = test_sector_specific_hype(df, cutoff)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("🏆 FINAL SUMMARY: REDDIT SIGNAL VALUE")
    print("=" * 70)
    
    best_config = max(coord_results, key=lambda x: x["r2"])
    best_sector = max(sector_results, key=lambda x: x["auc"])
    
    print(f"""
   RETAIL RISK CLASSIFIER:
   ----------------------
   • AUC for Extreme Events: {retail_auc:.4f}
   • Above random: {(retail_auc - 0.5)*100:+.2f}%
   
   COORDINATOR IMPROVEMENT:
   -----------------------
   • Best config: {best_config['config']}
   • Best R²: {best_config['r2']:.4f}
   • Improvement over baseline: {best_config['delta']:+.2f}%
   
   SECTOR SENSITIVITY:
   ------------------
   • Best sector: {best_sector['sector']}
   • Sector AUC: {best_sector['auc']:.4f}
   
   COMBINED NEWS+RETAIL:
   --------------------
   • Combined AUC: {combined_auc:.4f}
""")
    
    # Recommendations
    print("=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70)
    
    if retail_auc > 0.60:
        print(f"""
   ✅ RETAIL SIGNALS ADD VALUE!
   
   Recommendation: Add retail_risk_score to TitanCoordinator
   
   Implementation:
   1. Create RetailRiskClassifier (like NewsAgent classifier)
   2. Generate retail_risk_score (probability of extreme vol)
   3. Add to Coordinator feature set
   4. Expected improvement: {best_config['delta']:+.2f}%
""")
    else:
        print(f"""
   ⚠️ RETAIL SIGNALS HAVE MARGINAL VALUE
   
   The retail_risk_score adds small improvements.
   Consider including only for completeness.
""")
    
    # Timing
    end_time = datetime.now()
    print(f"\n   Duration: {end_time - start_time}")
    print("=" * 70)
    
    return {
        "retail_auc": retail_auc,
        "best_config": best_config,
        "combined_auc": combined_auc
    }


if __name__ == "__main__":
    main()


