"""
NewsAgent: Exponential Decay Kernel Implementation (Phase 6)

FIX: Replaced discrete lags with Exponential Decay Kernel
that cuts off BEFORE the weekly cycle repeats (stops at shift(4)).

This eliminates the "Weekly Echo" anomaly where lag5 was acting
as a proxy for the Friday effect.

Decay Kernel Formula:
    news_memory = 0.50 * shift(1) + 0.25 * shift(2) + 0.15 * shift(3) + 0.10 * shift(4)

Key Features:
- news_memory: Weighted sum of recent news_count
- shock_memory: Weighted sum of recent shock_index
- NO lag5 (avoids weekly seasonality contamination)

Model: LightGBM with robust parameters

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


def calculate_decay_kernel(series: pd.Series, group_col: pd.Series = None) -> pd.Series:
    """
    Calculate exponential decay kernel for a time series.
    
    Formula: 0.50 * shift(1) + 0.25 * shift(2) + 0.15 * shift(3) + 0.10 * shift(4)
    
    IMPORTANT: We stop at shift(4) to avoid touching shift(5) 
    which can contaminate with weekly seasonality (Friday effect).
    
    Weights sum to 1.0 for interpretability.
    
    Args:
        series: The time series to apply decay kernel to
        group_col: Optional groupby column (e.g., ticker)
        
    Returns:
        Series with decay-weighted memory of past values
    """
    weights = [0.50, 0.25, 0.15, 0.10]  # Exponential decay, sum=1.0
    
    if group_col is not None:
        # Group-aware shifting
        df = pd.DataFrame({"value": series, "group": group_col})
        
        result = pd.Series(0.0, index=series.index)
        for i, w in enumerate(weights, 1):
            shifted = df.groupby("group")["value"].shift(i)
            result += w * shifted.fillna(0)
        
        return result
    else:
        # Simple shifting
        result = pd.Series(0.0, index=series.index)
        for i, w in enumerate(weights, 1):
            result += w * series.shift(i).fillna(0)
        
        return result


class NewsAgent:
    """
    NewsAgent with Exponential Decay Kernel (Phase 6 Fix).
    Phase 7: De-seasonalized news features support.
    
    Key Changes:
    - REMOVED: All discrete lags (_lag1, _lag2, _lag3, _lag5)
    - REMOVED: Rolling features (_roll3, _roll7)
    - ADDED: Decay kernel features (news_memory, shock_memory)
    - Phase 7: De-seasonalize news_count by day_of_week
    
    The decay kernel captures the persistence of news effects
    without contaminating with weekly seasonality.
    
    Features:
        - news_memory: Decay kernel of news_count (or news_count_excess)
        - shock_memory: Decay kernel of shock_index
        - sentiment_avg: Current sentiment
        - PCA features: Topic embeddings
    """
    
    def __init__(self, experiment_name: str = "titan_v8_news_agent_v2",
                 use_deseasonalized: bool = False):
        """Initialize NewsAgent with decay kernel approach."""
        # Robust LightGBM parameters
        self.model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=3,
            num_leaves=8,
            min_child_samples=20,
            colsample_bytree=0.8,
            subsample=0.8,
            reg_alpha=0.05,
            reg_lambda=0.05,
            random_state=42,
            verbose=-1
        )
        
        self.tracker = MLTracker(experiment_name)
        self.feature_cols = None
        self.target_col = "resid_tech"
        self.train_metrics = None
        self.test_metrics = None
        self.df = None
        
        # Phase 7: De-seasonalization support
        self.use_deseasonalized = use_deseasonalized
        self.news_seasonal_map = None
        
    def load_and_merge_data(self) -> pd.DataFrame:
        """
        Load data and engineer DECAY KERNEL features.
        
        Phase 6 Fix:
        - NO discrete lags (removed _lag1, _lag2, _lag3, _lag5)
        - NO rolling features (removed _roll3, _roll7)
        - YES decay kernel features (news_memory, shock_memory)
        
        Phase 7 Addition:
        - De-seasonalize news_count by (ticker, day_of_week)
        
        Returns:
            DataFrame with decay kernel features
        """
        print("\n📂 Loading and merging data...")
        
        # Load residuals
        residuals_path = Path("data/processed/residuals.parquet")
        if not residuals_path.exists():
            raise FileNotFoundError("Residuals file not found! Run TechnicalAgent first.")
        
        residuals_df = pd.read_parquet(residuals_path)
        print(f"   ✓ Residuals: {len(residuals_df):,} rows")
        
        # Load news features
        news_path = Path("data/processed/news_features.parquet")
        news_df = pd.read_parquet(news_path)
        print(f"   ✓ News features: {len(news_df):,} rows")
        
        # Load targets for VIX
        targets_path = Path("data/processed/targets.parquet")
        targets_df = pd.read_parquet(targets_path)
        vix_df = targets_df[["date", "ticker", "VIX_close"]].copy()
        
        # Normalize dates
        for df in [residuals_df, news_df, vix_df]:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            if df["ticker"].dtype.name == "category":
                df["ticker"] = df["ticker"].astype(str)
        
        # Merge
        merged = pd.merge(residuals_df, news_df, on=["date", "ticker"], how="inner")
        merged = pd.merge(merged, vix_df, on=["date", "ticker"], how="left")
        merged["VIX_close"] = merged["VIX_close"].ffill().fillna(15)
        
        # Sort for proper kernel calculations
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        print(f"   ✓ Merged: {len(merged):,} rows")
        
        # =============================================
        # PHASE 7: DE-SEASONALIZE NEWS FEATURES
        # =============================================
        if self.use_deseasonalized:
            print("\n   🔧 De-seasonalizing news features...")
            
            # Add day of week
            merged["day_of_week"] = merged["date"].dt.dayofweek
            
            # Calculate median news_count per (ticker, day_of_week)
            news_seasonal = merged.groupby(["ticker", "day_of_week"])["news_count"].median()
            self.news_seasonal_map = news_seasonal.to_dict()
            
            # Apply de-seasonalization
            def get_news_seasonal(row):
                return self.news_seasonal_map.get((row["ticker"], row["day_of_week"]), row["news_count"])
            
            merged["news_count_seasonal"] = merged.apply(get_news_seasonal, axis=1)
            merged["news_count_excess"] = merged["news_count"] - merged["news_count_seasonal"]
            
            print(f"      ✓ Created news_count_excess (de-seasonalized)")
            print(f"      Original news_count std: {merged['news_count'].std():.4f}")
            print(f"      Excess news_count std:   {merged['news_count_excess'].std():.4f}")
        
        # =============================================
        # FEATURE ENGINEERING: DECAY KERNEL (Phase 6)
        # =============================================
        print("\n   🔧 Engineering DECAY KERNEL features...")
        print("      Formula: 0.50*t-1 + 0.25*t-2 + 0.15*t-3 + 0.10*t-4")
        print("      (Stops at t-4 to avoid weekly echo at t-5)")
        
        # Phase 7: Use de-seasonalized news_count if available
        news_col = "news_count_excess" if self.use_deseasonalized and "news_count_excess" in merged.columns else "news_count"
        
        if self.use_deseasonalized:
            print(f"      ✓ Using de-seasonalized: {news_col}")
        
        # Apply decay kernel to key features
        merged["news_memory"] = calculate_decay_kernel(
            merged[news_col], 
            group_col=merged["ticker"]
        )
        
        merged["shock_memory"] = calculate_decay_kernel(
            merged["shock_index"],
            group_col=merged["ticker"]
        )
        
        # Optional: sentiment memory (less critical)
        merged["sentiment_memory"] = calculate_decay_kernel(
            merged["sentiment_avg"],
            group_col=merged["ticker"]
        )
        
        print(f"      ✓ Created: news_memory, shock_memory, sentiment_memory")
        
        # VIX interaction with memory (regime-aware)
        merged["shock_vix_memory"] = merged["shock_memory"] * merged["VIX_close"]
        print(f"      ✓ Created: shock_vix_memory (regime interaction)")
        
        # Drop NaN rows (first 4 rows per ticker due to kernel)
        before = len(merged)
        merged = merged.dropna(subset=["news_memory", "shock_memory", self.target_col])
        after = len(merged)
        print(f"      ✓ Dropped {before - after} rows with NaN (kernel warmup)")
        
        print(f"\n   📊 Final dataset: {len(merged):,} rows")
        
        # Stats
        print(f"\n   📊 Residual Stats:")
        print(f"      Mean:  {merged[self.target_col].mean():.4f}")
        print(f"      Std:   {merged[self.target_col].std():.4f}")
        
        self.df = merged
        return merged
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get feature column names for decay kernel model."""
        features = []
        
        # Decay kernel features (PRIMARY)
        kernel_features = ["news_memory", "shock_memory", "sentiment_memory", "shock_vix_memory"]
        features.extend([f for f in kernel_features if f in df.columns])
        
        # Current values (secondary)
        current_features = ["sentiment_avg", "novelty_score"]
        features.extend([f for f in current_features if f in df.columns])
        
        # PCA columns (topic embeddings)
        pca_cols = [c for c in df.columns if c.startswith("news_pca_")]
        features.extend(sorted(pca_cols))
        
        # VIX for regime context
        if "VIX_close" in df.columns:
            features.append("VIX_close")
        
        # Remove duplicates
        features = list(dict.fromkeys(features))
        
        return features
    
    def train(self, df: pd.DataFrame = None) -> dict:
        """Train LightGBM with decay kernel features."""
        if df is None:
            df = self.df
            
        print("\n🎯 Training NewsAgent with DECAY KERNEL...")
        
        # Define features
        self.feature_cols = self.get_feature_columns(df)
        
        # Count feature types
        n_kernel = len([f for f in self.feature_cols if "memory" in f])
        n_pca = len([f for f in self.feature_cols if "pca" in f])
        n_other = len(self.feature_cols) - n_kernel - n_pca
        
        print(f"\n   Features: {len(self.feature_cols)}")
        print(f"      - Kernel:  {n_kernel} (decay memory)")
        print(f"      - PCA:     {n_pca}")
        print(f"      - Other:   {n_other}")
        print(f"   Target: {self.target_col}")
        
        # Time-series split
        train_cutoff = pd.to_datetime("2023-01-01")
        
        train_mask = df["date"] < train_cutoff
        test_mask = df["date"] >= train_cutoff
        
        if train_mask.sum() < 50 or test_mask.sum() < 20:
            print("\n   ⚠️ Limited date range - using 70/30 split")
            split_idx = int(len(df) * 0.7)
            train_mask = pd.Series([i < split_idx for i in range(len(df))], index=df.index)
            test_mask = ~train_mask
        
        X_train = df.loc[train_mask, self.feature_cols]
        y_train = df.loc[train_mask, self.target_col]
        X_test = df.loc[test_mask, self.feature_cols]
        y_test = df.loc[test_mask, self.target_col]
        
        print(f"\n   📊 Split:")
        print(f"      Train: {len(X_train):,} samples")
        print(f"      Test:  {len(X_test):,} samples")
        
        # Start MLflow run
        with self.tracker.start_run(run_name="news_agent_decay_kernel"):
            # Log parameters
            self.tracker.log_params({
                "model": "LGBMRegressor_DecayKernel",
                "target": "resid_tech",
                "kernel_weights": "0.50,0.25,0.15,0.10",
                "kernel_max_lag": 4,
                "n_features": len(self.feature_cols),
                "n_kernel_features": n_kernel,
            })
            
            # Train model
            print("\n   🔧 Fitting LightGBM with Decay Kernel features...")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
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
            print(f"\n   {'Metric':<25} {'Train':>10} {'Test':>10} {'Gap':>10}")
            print("   " + "-" * 57)
            
            r2_gap = self.train_metrics['R2'] - self.test_metrics['R2']
            print(f"   {'RMSE':<25} {self.train_metrics['RMSE']:>10.4f} {self.test_metrics['RMSE']:>10.4f}")
            print(f"   {'MAE':<25} {self.train_metrics['MAE']:>10.4f} {self.test_metrics['MAE']:>10.4f}")
            print(f"   {'R²':<25} {self.train_metrics['R2']:>10.4f} {self.test_metrics['R2']:>10.4f} {r2_gap:>10.4f}")
            print(f"   {'Directional Accuracy':<25} {self.train_metrics['Directional_Accuracy']:>9.1f}% {self.test_metrics['Directional_Accuracy']:>9.1f}%")
            
            # Log model
            self.tracker.log_model(self.model, "news_agent_decay_kernel")
        
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
        if total > 0:
            importance_df["pct"] = (importance_df["importance"] / total * 100).round(1)
        else:
            importance_df["pct"] = 0.0
        
        # Add feature type
        def get_type(f):
            if "memory" in f:
                return "KERNEL"
            elif "pca" in f:
                return "pca"
            elif "VIX" in f:
                return "regime"
            else:
                return "other"
        
        importance_df["type"] = importance_df["feature"].apply(get_type)
        
        return importance_df
    
    def predict(self, df: pd.DataFrame = None) -> pd.Series:
        """Generate residual predictions."""
        if df is None:
            df = self.df
        
        X = df[self.feature_cols]
        predictions = self.model.predict(X)
        
        return pd.Series(predictions, index=df.index)


def main():
    """Run NewsAgent with decay kernel features."""
    print("\n" + "=" * 65)
    print("🚀 TITAN V8 NEWS AGENT (DECAY KERNEL - Phase 6)")
    print("    Fix: Replaced discrete lags with exponential decay kernel")
    print("=" * 65)
    
    # Initialize agent
    agent = NewsAgent()
    
    # Load and merge data
    df = agent.load_and_merge_data()
    
    # Train model
    metrics = agent.train(df)
    
    # Get feature importance
    importance = agent.get_feature_importance()
    
    # Print feature importance
    print("\n" + "=" * 65)
    print("📊 FEATURE IMPORTANCE (Decay Kernel Model)")
    print("=" * 65)
    print(importance.to_string(index=False))
    
    # Final assessment
    print("\n" + "=" * 65)
    print("📈 FINAL RESULTS")
    print("=" * 65)
    
    test_r2 = metrics['test']['R2']
    test_dir = metrics['test']['Directional_Accuracy']
    
    print(f"\n   Test R²:              {test_r2:.4f} ({test_r2*100:.2f}%)")
    print(f"   Directional Accuracy: {test_dir:.1f}%")
    
    print("\n" + "=" * 65)
    print("✅ NewsAgent with Decay Kernel complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
