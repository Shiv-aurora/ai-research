"""
NewsAgent: News-Only Volatility Prediction
Phase 1.3: Feasibility Test

This agent tests whether news features alone can predict volatility.
Uses LightGBM with MLflow tracking.

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
    Agent that predicts volatility using only news-derived features.
    
    This is a feasibility test to determine if news signals
    have predictive power for next-day volatility.
    
    Features used:
        - shock_index: Keyword-based severity score
        - news_count: Number of headlines per day
        - sentiment_avg: Average sentiment
        - novelty_score: Cosine distance from previous day
        - news_pca_0..19: 20 news theme dimensions
    
    Target:
        - target_log_var: log(next_day_realized_variance)
    """
    
    def __init__(self, experiment_name: str = "titan_v8_news_agent"):
        """
        Initialize the NewsAgent with LightGBM model.
        
        Args:
            experiment_name: MLflow experiment name
        """
        self.model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        
        self.tracker = MLTracker(experiment_name)
        self.feature_cols = None
        self.train_metrics = None
        self.test_metrics = None
        
    def load_and_merge_data(self) -> pd.DataFrame:
        """
        Load and merge price targets with news features.
        
        Returns:
            Merged DataFrame ready for training
        """
        print("\n📂 Loading and merging data...")
        
        # Load targets (price-based volatility)
        targets_path = Path("data/processed/targets.parquet")
        targets_df = pd.read_parquet(targets_path)
        print(f"   ✓ Targets: {len(targets_df):,} rows")
        
        # Load news features
        news_path = Path("data/processed/news_features.parquet")
        news_df = pd.read_parquet(news_path)
        print(f"   ✓ News features: {len(news_df):,} rows")
        
        # Ensure date columns are compatible
        targets_df["date"] = pd.to_datetime(targets_df["date"]).dt.tz_localize(None)
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.tz_localize(None)
        
        # Convert ticker to string if categorical
        if targets_df["ticker"].dtype.name == "category":
            targets_df["ticker"] = targets_df["ticker"].astype(str)
        if news_df["ticker"].dtype.name == "category":
            news_df["ticker"] = news_df["ticker"].astype(str)
        
        # Merge on date and ticker
        merged = pd.merge(
            targets_df,
            news_df,
            on=["date", "ticker"],
            how="inner"
        )
        print(f"   ✓ Merged: {len(merged):,} rows")
        
        # Drop rows with NaN target
        before = len(merged)
        merged = merged.dropna(subset=["target_log_var"])
        after = len(merged)
        if before > after:
            print(f"   ✓ Dropped {before - after} rows with NaN target")
        
        # Sort by date
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        print(f"   ✓ Date range: {merged['date'].min()} to {merged['date'].max()}")
        print(f"   ✓ Tickers: {merged['ticker'].unique().tolist()}")
        
        return merged
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Get the feature column names.
        
        Args:
            df: DataFrame with features
        
        Returns:
            List of feature column names
        """
        # Core features
        features = ["shock_index", "news_count", "sentiment_avg", "novelty_score"]
        
        # Add all PCA columns
        pca_cols = [col for col in df.columns if col.startswith("news_pca_")]
        features.extend(sorted(pca_cols))
        
        return features
    
    def train(self, df: pd.DataFrame) -> dict:
        """
        Train the LightGBM model with time-series split.
        
        Args:
            df: Merged DataFrame with features and target
        
        Returns:
            Dictionary with train/test metrics
        """
        print("\n🎯 Training NewsAgent...")
        
        # Define features and target
        self.feature_cols = self.get_feature_columns(df)
        target_col = "target_log_var"
        
        print(f"   Features: {len(self.feature_cols)}")
        print(f"   Target: {target_col}")
        
        # Time-series split
        train_cutoff = pd.to_datetime("2023-01-01")
        
        train_mask = df["date"] < train_cutoff
        test_mask = df["date"] >= train_cutoff
        
        X_train = df.loc[train_mask, self.feature_cols]
        y_train = df.loc[train_mask, target_col]
        X_test = df.loc[test_mask, self.feature_cols]
        y_test = df.loc[test_mask, target_col]
        
        print(f"\n   📊 Split:")
        print(f"      Train: {len(X_train):,} samples (< 2023-01-01)")
        print(f"      Test:  {len(X_test):,} samples (>= 2023-01-01)")
        
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Train or test set is empty! Check date range.")
        
        # Start MLflow run
        with self.tracker.start_run(run_name="news_agent_feasibility"):
            # Log parameters
            self.tracker.log_params({
                "model": "LGBMRegressor",
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 6,
                "n_features": len(self.feature_cols),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_cutoff": str(train_cutoff.date())
            })
            
            # Train model
            print("\n   🔧 Fitting LightGBM...")
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
            
            # Train metrics
            self.train_metrics = self.tracker.log_metrics(
                y_train.values, y_train_pred, step=0
            )
            print(f"\n   Train Metrics:")
            print(f"      RMSE: {self.train_metrics['RMSE']:.4f}")
            print(f"      MAE:  {self.train_metrics['MAE']:.4f}")
            print(f"      R²:   {self.train_metrics['R2']:.4f}")
            
            # Test metrics (the important ones)
            self.test_metrics = self.tracker.log_metrics(
                y_test.values, y_test_pred, step=1
            )
            print(f"\n   Test Metrics:")
            print(f"      RMSE: {self.test_metrics['RMSE']:.4f}")
            print(f"      MAE:  {self.test_metrics['MAE']:.4f}")
            print(f"      R²:   {self.test_metrics['R2']:.4f}")
            print(f"      Directional Accuracy: {self.test_metrics['Directional_Accuracy']:.1f}%")
            
            # Log model
            self.tracker.log_model(self.model, "news_agent_lgbm")
        
        return {
            "train": self.train_metrics,
            "test": self.test_metrics
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances sorted by importance.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None or self.feature_cols is None:
            raise ValueError("Model not trained yet!")
        
        importance_df = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_
        })
        
        importance_df = importance_df.sort_values(
            "importance", ascending=False
        ).reset_index(drop=True)
        
        return importance_df


def main():
    """Run NewsAgent feasibility test."""
    print("\n" + "=" * 60)
    print("🚀 TITAN V8 NEWS AGENT FEASIBILITY TEST")
    print("    Phase 1.3: Can News Predict Volatility?")
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
    print("📊 TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 60)
    print(importance.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("📈 FINAL RESULTS")
    print("=" * 60)
    print(f"\nTest Set Performance:")
    print(f"   RMSE: {metrics['test']['RMSE']:.4f}")
    print(f"   MAE:  {metrics['test']['MAE']:.4f}")
    print(f"   R²:   {metrics['test']['R2']:.4f}")
    print(f"   Directional Accuracy: {metrics['test']['Directional_Accuracy']:.1f}%")
    
    # Feasibility assessment
    print("\n" + "=" * 60)
    print("🔬 FEASIBILITY ASSESSMENT")
    print("=" * 60)
    
    r2 = metrics['test']['R2']
    dir_acc = metrics['test']['Directional_Accuracy']
    
    if r2 > 0.1 and dir_acc > 55:
        verdict = "✅ SUCCESS - News features show predictive signal!"
        recommendation = "Proceed to integrate with price features."
    elif r2 > 0 and dir_acc > 50:
        verdict = "⚠️ MARGINAL - Weak but positive signal detected."
        recommendation = "News may add value when combined with other features."
    else:
        verdict = "❌ INSUFFICIENT - News alone is not predictive."
        recommendation = "Focus on price/volatility features; news as secondary."
    
    print(f"\n{verdict}")
    print(f"\nRecommendation: {recommendation}")
    print("=" * 60)
    
    print("\n✅ NewsAgent feasibility test complete!")
    print("   Results logged to MLflow experiment: titan_v8_news_agent")


if __name__ == "__main__":
    main()

