"""
NewsAgent: News-Based Residual Prediction (Optimized)
Phase 3.5: Overfitting Fix & Signal Enhancement

This agent predicts the RESIDUALS from the TechnicalAgent (HAR-RV),
with optimizations to reduce overfitting and improve generalization.

Key Optimizations:
1. Rolling features (3-day, 7-day) to capture narrative persistence
2. Interaction features with VIX (bad news matters more when scared)
3. Heavy regularization (L1, L2) to prevent memorization
4. Shallow trees and high min_child_samples

Usage:
    python -m src.agents.news_agent
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.tracker import MLTracker


class NewsAgent:
    """
    Optimized NewsAgent for residual prediction.
    
    Features:
        - Original: shock_index, news_count, sentiment_avg, novelty_score, news_pca_0..19
        - Rolling: 3-day and 7-day means for key features (narrative persistence)
        - Interactions: shock * VIX, news_count * VIX (regime context)
    
    Target:
        - resid_tech: Residual from TechnicalAgent
    
    Regularization:
        - L1 (reg_alpha) and L2 (reg_lambda) to prevent overfitting
        - Shallow trees (max_depth=3) and min_child_samples=50
    """
    
    def __init__(self, experiment_name: str = "titan_v8_news_agent"):
        """
        Initialize the optimized NewsAgent.
        
        Heavy regularization to combat overfitting:
        - Lower learning rate (0.01)
        - Shallow trees (max_depth=3)
        - L1/L2 regularization
        - High min_child_samples (50)
        """
        self.model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,       # Very low for robustness
            max_depth=3,              # Shallow trees
            num_leaves=15,            # Simple structure
            min_child_samples=50,     # Broad patterns only
            reg_alpha=0.1,            # L1 regularization
            reg_lambda=0.1,           # L2 regularization
            subsample=0.8,            # Bagging for robustness
            colsample_bytree=0.8,     # Feature bagging
            random_state=42,
            verbose=-1
        )
        
        self.tracker = MLTracker(experiment_name)
        self.feature_cols = None
        self.target_col = "resid_tech"
        self.train_metrics = None
        self.test_metrics = None
        self.df = None
        
    def load_and_merge_data(self) -> pd.DataFrame:
        """
        Load data and perform on-the-fly feature engineering.
        
        Feature Engineering:
        A. Rolling features (narrative persistence)
        B. Interaction features with VIX (regime context)
        
        Returns:
            Merged DataFrame with engineered features
        """
        print("\n📂 Loading and merging data...")
        
        # Load residuals from TechnicalAgent
        residuals_path = Path("data/processed/residuals.parquet")
        if not residuals_path.exists():
            raise FileNotFoundError(
                "Residuals file not found! Run TechnicalAgent first."
            )
        
        residuals_df = pd.read_parquet(residuals_path)
        print(f"   ✓ Residuals: {len(residuals_df):,} rows")
        
        # Load news features
        news_path = Path("data/processed/news_features.parquet")
        news_df = pd.read_parquet(news_path)
        print(f"   ✓ News features: {len(news_df):,} rows")
        
        # Load targets for VIX (needed for interaction features)
        targets_path = Path("data/processed/targets.parquet")
        targets_df = pd.read_parquet(targets_path)
        vix_df = targets_df[["date", "ticker", "VIX_close"]].copy()
        print(f"   ✓ VIX data loaded")
        
        # Normalize dates
        residuals_df["date"] = pd.to_datetime(residuals_df["date"]).dt.tz_localize(None)
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.tz_localize(None)
        vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.tz_localize(None)
        
        # Convert ticker to string
        for df in [residuals_df, news_df, vix_df]:
            if df["ticker"].dtype.name == "category":
                df["ticker"] = df["ticker"].astype(str)
        
        # Merge residuals + news
        merged = pd.merge(residuals_df, news_df, on=["date", "ticker"], how="inner")
        
        # Merge with VIX
        merged = pd.merge(merged, vix_df, on=["date", "ticker"], how="left")
        merged["VIX_close"] = merged["VIX_close"].ffill().fillna(15)  # Default VIX
        
        print(f"   ✓ Merged: {len(merged):,} rows")
        
        # Sort for proper rolling calculations
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        # =============================================
        # FEATURE ENGINEERING
        # =============================================
        print("\n   🔧 Engineering features...")
        
        # A. ROLLING FEATURES (Narrative Persistence)
        # News themes persist - a 3-day trend is stronger than a 1-day blip
        roll_cols = ["news_pca_0", "news_pca_1", "news_pca_2", "news_pca_3", 
                     "news_pca_4", "sentiment_avg"]
        
        for col in roll_cols:
            if col in merged.columns:
                # Group by ticker for proper rolling
                merged[f"{col}_roll3"] = merged.groupby("ticker")[col].transform(
                    lambda x: x.rolling(3, min_periods=1).mean()
                )
                merged[f"{col}_roll7"] = merged.groupby("ticker")[col].transform(
                    lambda x: x.rolling(7, min_periods=1).mean()
                )
        
        print(f"      ✓ Added {len(roll_cols) * 2} rolling features")
        
        # B. INTERACTION FEATURES (Regime Context)
        # Bad news matters more when market is already scared (high VIX)
        merged["shock_vix_interaction"] = merged["shock_index"] * merged["VIX_close"]
        merged["news_vix_interaction"] = merged["news_count"] * merged["VIX_close"]
        merged["sentiment_vix_interaction"] = merged["sentiment_avg"] * merged["VIX_close"]
        
        print(f"      ✓ Added 3 interaction features")
        
        # C. VOLATILITY REGIME FEATURE
        # Is VIX above/below median? (binary regime)
        vix_median = merged["VIX_close"].median()
        merged["high_vix_regime"] = (merged["VIX_close"] > vix_median).astype(int)
        
        print(f"      ✓ Added VIX regime feature (median={vix_median:.1f})")
        
        # D. CLEAN UP - Drop NaN from rolling windows
        before = len(merged)
        merged = merged.dropna(subset=[self.target_col])
        after = len(merged)
        if before > after:
            print(f"      ✓ Dropped {before - after} rows with NaN")
        
        print(f"\n   📊 Final dataset: {len(merged):,} rows")
        print(f"   📊 Total features available: {len([c for c in merged.columns if c not in ['date', 'ticker', 'resid_tech', 'target_log_var', 'pred_tech']])}")
        
        self.df = merged
        return merged
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Get all feature column names including engineered features.
        """
        # Original core features
        features = ["shock_index", "news_count", "sentiment_avg", "novelty_score"]
        
        # PCA columns
        pca_cols = [col for col in df.columns if col.startswith("news_pca_")]
        features.extend(sorted(pca_cols))
        
        # Rolling features
        roll_cols = [col for col in df.columns if "_roll3" in col or "_roll7" in col]
        features.extend(sorted(roll_cols))
        
        # Interaction features
        interaction_cols = [col for col in df.columns if "_interaction" in col]
        features.extend(sorted(interaction_cols))
        
        # Regime feature
        if "high_vix_regime" in df.columns:
            features.append("high_vix_regime")
        
        # VIX itself
        if "VIX_close" in df.columns:
            features.append("VIX_close")
        
        # Remove duplicates and non-existent columns
        features = [f for f in features if f in df.columns]
        features = list(dict.fromkeys(features))  # Remove duplicates, preserve order
        
        return features
    
    def train(self, df: pd.DataFrame = None) -> dict:
        """
        Train the optimized LightGBM model on residuals.
        """
        if df is None:
            df = self.df
            
        print("\n🎯 Training Optimized NewsAgent on RESIDUALS...")
        
        # Define features
        self.feature_cols = self.get_feature_columns(df)
        
        print(f"\n   Features: {len(self.feature_cols)}")
        print(f"   Target: {self.target_col}")
        print(f"\n   Feature breakdown:")
        print(f"      - Original:     {len([f for f in self.feature_cols if not ('_roll' in f or '_interaction' in f or 'regime' in f or 'VIX' in f)])}")
        print(f"      - Rolling:      {len([f for f in self.feature_cols if '_roll' in f])}")
        print(f"      - Interaction:  {len([f for f in self.feature_cols if '_interaction' in f])}")
        print(f"      - VIX/Regime:   {len([f for f in self.feature_cols if 'VIX' in f or 'regime' in f])}")
        
        # Time-series split
        train_cutoff = pd.to_datetime("2023-01-01")
        
        train_mask = df["date"] < train_cutoff
        test_mask = df["date"] >= train_cutoff
        
        if train_mask.sum() < 50 or test_mask.sum() < 20:
            print("\n   ⚠️ Limited date range - using 70/30 split")
            split_idx = int(len(df) * 0.7)
            train_mask = df.index < split_idx
            test_mask = df.index >= split_idx
            train_cutoff = df.iloc[split_idx]["date"]
        
        X_train = df.loc[train_mask, self.feature_cols]
        y_train = df.loc[train_mask, self.target_col]
        X_test = df.loc[test_mask, self.feature_cols]
        y_test = df.loc[test_mask, self.target_col]
        
        print(f"\n   📊 Split:")
        print(f"      Train: {len(X_train):,} samples")
        print(f"      Test:  {len(X_test):,} samples")
        
        # Start MLflow run
        with self.tracker.start_run(run_name="news_agent_optimized"):
            # Log parameters
            self.tracker.log_params({
                "model": "LGBMRegressor_Optimized",
                "target": "resid_tech",
                "n_estimators": 1000,
                "learning_rate": 0.01,
                "max_depth": 3,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "min_child_samples": 50,
                "n_features": len(self.feature_cols),
                "n_rolling_features": len([f for f in self.feature_cols if '_roll' in f]),
                "n_interaction_features": len([f for f in self.feature_cols if '_interaction' in f]),
            })
            
            # Train model with early stopping
            print("\n   🔧 Fitting Optimized LightGBM...")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                eval_metric="rmse"
            )
            
            # Predictions
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # Calculate metrics
            print("\n   📈 Evaluating...")
            
            self.train_metrics = self.tracker.log_metrics(
                y_train.values, y_train_pred, step=0
            )
            
            self.test_metrics = self.tracker.log_metrics(
                y_test.values, y_test_pred, step=1
            )
            
            # Print comparison
            print(f"\n   {'Metric':<20} {'Train':>10} {'Test':>10} {'Gap':>10}")
            print("   " + "-" * 52)
            
            r2_gap = self.train_metrics['R2'] - self.test_metrics['R2']
            print(f"   {'RMSE':<20} {self.train_metrics['RMSE']:>10.4f} {self.test_metrics['RMSE']:>10.4f}")
            print(f"   {'MAE':<20} {self.train_metrics['MAE']:>10.4f} {self.test_metrics['MAE']:>10.4f}")
            print(f"   {'R²':<20} {self.train_metrics['R2']:>10.4f} {self.test_metrics['R2']:>10.4f} {r2_gap:>10.4f}")
            print(f"   {'Dir. Accuracy':<20} {self.train_metrics['Directional_Accuracy']:>9.1f}% {self.test_metrics['Directional_Accuracy']:>9.1f}%")
            
            # Log model
            self.tracker.log_model(self.model, "news_agent_optimized")
        
        return {
            "train": self.train_metrics,
            "test": self.test_metrics,
            "r2_gap": r2_gap
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importances sorted by importance."""
        if self.model is None or self.feature_cols is None:
            raise ValueError("Model not trained yet!")
        
        importance_df = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_
        })
        
        importance_df = importance_df.sort_values(
            "importance", ascending=False
        ).reset_index(drop=True)
        
        total = importance_df["importance"].sum()
        importance_df["pct"] = (importance_df["importance"] / total * 100).round(1)
        
        # Add feature type
        def get_feature_type(f):
            if "_roll" in f:
                return "rolling"
            elif "_interaction" in f:
                return "interaction"
            elif "VIX" in f or "regime" in f:
                return "regime"
            elif "pca" in f:
                return "pca"
            else:
                return "original"
        
        importance_df["type"] = importance_df["feature"].apply(get_feature_type)
        
        return importance_df
    
    def predict(self, df: pd.DataFrame = None) -> pd.Series:
        """Generate residual predictions."""
        if df is None:
            df = self.df
        
        X = df[self.feature_cols]
        predictions = self.model.predict(X)
        
        return pd.Series(predictions, index=df.index)


def main():
    """Run optimized NewsAgent."""
    print("\n" + "=" * 60)
    print("🚀 TITAN V8 NEWS AGENT (OPTIMIZED)")
    print("    Phase 3.5: Overfitting Fix & Signal Enhancement")
    print("=" * 60)
    
    # Initialize agent
    agent = NewsAgent(experiment_name="titan_v8_news_agent")
    
    # Load and merge data
    df = agent.load_and_merge_data()
    
    # Train model
    metrics = agent.train(df)
    
    # Get feature importance
    importance = agent.get_feature_importance()
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 TOP 15 MOST IMPORTANT FEATURES")
    print("=" * 60)
    print(importance.head(15).to_string(index=False))
    
    # Analyze feature types in top 10
    top10 = importance.head(10)
    type_counts = top10["type"].value_counts()
    
    print("\n   Feature type breakdown (Top 10):")
    for ftype, count in type_counts.items():
        print(f"      {ftype}: {count}")
    
    # Check for engineered features in top 10
    engineered_in_top10 = top10[top10["type"].isin(["rolling", "interaction", "regime"])]
    
    print("\n" + "=" * 60)
    print("📈 OVERFITTING ANALYSIS")
    print("=" * 60)
    
    train_r2 = metrics['train']['R2']
    test_r2 = metrics['test']['R2']
    r2_gap = metrics['r2_gap']
    
    print(f"\n   Train R²: {train_r2:.4f}")
    print(f"   Test R²:  {test_r2:.4f}")
    print(f"   Gap:      {r2_gap:.4f}")
    
    # Assessment
    if r2_gap < 0.1:
        gap_verdict = "✅ EXCELLENT - Minimal overfitting!"
    elif r2_gap < 0.3:
        gap_verdict = "✅ GOOD - Acceptable generalization."
    elif r2_gap < 0.5:
        gap_verdict = "⚠️ MODERATE - Some overfitting, but manageable."
    else:
        gap_verdict = "❌ HIGH - Still overfitting significantly."
    
    print(f"\n   {gap_verdict}")
    
    # Final verdict
    print("\n" + "=" * 60)
    print("🔬 FINAL ASSESSMENT")
    print("=" * 60)
    
    if test_r2 > 0.05 and r2_gap < 0.3:
        verdict = "✅ SUCCESS - Optimizations worked!"
        action = "Proceed to Hybrid Ensemble (Phase 4)."
    elif test_r2 > 0 and r2_gap < 0.5:
        verdict = "⚠️ PARTIAL SUCCESS - Signal exists but weak."
        action = "Consider additional feature engineering or data."
    else:
        verdict = "❌ NEEDS MORE WORK - Signal not robust."
        action = "Try different features or model architecture."
    
    print(f"\n   {verdict}")
    print(f"\n   Test R²: {test_r2:.4f}")
    print(f"   R² Gap:  {r2_gap:.4f}")
    print(f"\n   Action: {action}")
    
    if len(engineered_in_top10) > 0:
        print(f"\n   💡 Engineered features in top 10: {len(engineered_in_top10)}")
        for _, row in engineered_in_top10.iterrows():
            print(f"      - {row['feature']} ({row['type']}, {row['pct']}%)")
    
    print("=" * 60)
    
    print("\n✅ Optimized NewsAgent training complete!")
    print("   Results logged to MLflow: news_agent_optimized")


if __name__ == "__main__":
    main()
