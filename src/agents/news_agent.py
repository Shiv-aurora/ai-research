"""
NewsAgent: News-Based Residual Prediction
Phase 3: News Agent Integration

This agent predicts the RESIDUALS from the TechnicalAgent (HAR-RV),
attempting to explain what technical features cannot.

Key Insight:
- Phase 1.3 showed news alone can't predict raw volatility (R² < 0)
- But news might explain the "unexplained" variance (residuals)
- Even R² = 0.05 on residuals is valuable!

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
    Agent that predicts TechnicalAgent RESIDUALS using news features.
    
    After removing the "inertia" (what technical features explain),
    we test if news can capture the remaining signal.
    
    Features used:
        - shock_index: Keyword-based severity score
        - news_count: Number of headlines per day
        - sentiment_avg: Average sentiment
        - novelty_score: Cosine distance from previous day
        - news_pca_0..19: 20 news theme dimensions
    
    Target:
        - resid_tech: Residual from TechnicalAgent (HAR-RV)
        - resid_tech = target_log_var - pred_tech
    """
    
    def __init__(self, experiment_name: str = "titan_v8_news_agent"):
        """
        Initialize the NewsAgent with LightGBM model.
        
        Hyperparameters tuned for residual prediction:
        - Lower learning rate (0.03) to prevent overfitting noise
        - Shallower trees (max_depth=4) for generalization
        
        Args:
            experiment_name: MLflow experiment name
        """
        self.model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,  # Lower for residuals
            max_depth=4,         # Shallower to prevent overfitting
            num_leaves=15,
            min_child_samples=20,
            random_state=42,
            verbose=-1
        )
        
        self.tracker = MLTracker(experiment_name)
        self.feature_cols = None
        self.target_col = "resid_tech"  # Changed from target_log_var
        self.train_metrics = None
        self.test_metrics = None
        self.df = None
        
    def load_and_merge_data(self) -> pd.DataFrame:
        """
        Load residuals from TechnicalAgent and merge with news features.
        
        Returns:
            Merged DataFrame ready for training on residuals
        """
        print("\n📂 Loading and merging data...")
        
        # Load residuals from TechnicalAgent
        residuals_path = Path("data/processed/residuals.parquet")
        if not residuals_path.exists():
            raise FileNotFoundError(
                "Residuals file not found! Run TechnicalAgent first:\n"
                "  python -m src.agents.technical_agent"
            )
        
        residuals_df = pd.read_parquet(residuals_path)
        print(f"   ✓ Residuals: {len(residuals_df):,} rows")
        print(f"      Columns: {list(residuals_df.columns)}")
        
        # Load news features
        news_path = Path("data/processed/news_features.parquet")
        news_df = pd.read_parquet(news_path)
        print(f"   ✓ News features: {len(news_df):,} rows")
        
        # Ensure date columns are compatible
        residuals_df["date"] = pd.to_datetime(residuals_df["date"]).dt.tz_localize(None)
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.tz_localize(None)
        
        # Convert ticker to string if categorical
        if residuals_df["ticker"].dtype.name == "category":
            residuals_df["ticker"] = residuals_df["ticker"].astype(str)
        if news_df["ticker"].dtype.name == "category":
            news_df["ticker"] = news_df["ticker"].astype(str)
        
        # Merge on date and ticker
        merged = pd.merge(
            residuals_df,
            news_df,
            on=["date", "ticker"],
            how="inner"
        )
        print(f"   ✓ Merged: {len(merged):,} rows")
        
        # Drop rows with NaN in target (resid_tech)
        before = len(merged)
        merged = merged.dropna(subset=[self.target_col])
        after = len(merged)
        if before > after:
            print(f"   ✓ Dropped {before - after} rows with NaN target")
        
        # Sort by date
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        print(f"   ✓ Date range: {merged['date'].min()} to {merged['date'].max()}")
        print(f"   ✓ Tickers: {merged['ticker'].unique().tolist()}")
        
        # Show residual stats
        print(f"\n   📊 Residual Stats (what we're predicting):")
        print(f"      Mean:  {merged[self.target_col].mean():.4f}")
        print(f"      Std:   {merged[self.target_col].std():.4f}")
        print(f"      Range: [{merged[self.target_col].min():.2f}, {merged[self.target_col].max():.2f}]")
        
        self.df = merged
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
    
    def train(self, df: pd.DataFrame = None) -> dict:
        """
        Train the LightGBM model on residuals with time-series split.
        
        Args:
            df: Merged DataFrame with features and residual target
        
        Returns:
            Dictionary with train/test metrics
        """
        if df is None:
            df = self.df
            
        print("\n🎯 Training NewsAgent on RESIDUALS...")
        print("   (Predicting what TechnicalAgent couldn't explain)")
        
        # Define features and target
        self.feature_cols = self.get_feature_columns(df)
        
        print(f"\n   Features: {len(self.feature_cols)}")
        print(f"   Target: {self.target_col} (TechnicalAgent residuals)")
        
        # Time-series split
        train_cutoff = pd.to_datetime("2023-01-01")
        
        train_mask = df["date"] < train_cutoff
        test_mask = df["date"] >= train_cutoff
        
        # If date split doesn't work, use 70/30 split
        if train_mask.sum() < 50 or test_mask.sum() < 20:
            print("\n   ⚠️ Limited date range - using 70/30 split")
            split_idx = int(len(df) * 0.7)
            train_mask = df.index < split_idx
            test_mask = df.index >= split_idx
            train_cutoff = df.iloc[split_idx]["date"] if split_idx < len(df) else df["date"].max()
        
        X_train = df.loc[train_mask, self.feature_cols]
        y_train = df.loc[train_mask, self.target_col]
        X_test = df.loc[test_mask, self.feature_cols]
        y_test = df.loc[test_mask, self.target_col]
        
        print(f"\n   📊 Split:")
        print(f"      Train: {len(X_train):,} samples")
        print(f"      Test:  {len(X_test):,} samples")
        print(f"      Cutoff: {train_cutoff}")
        
        if len(X_train) < 10 or len(X_test) < 5:
            raise ValueError("Not enough data for training!")
        
        # Start MLflow run
        with self.tracker.start_run(run_name="news_agent_residual"):
            # Log parameters
            self.tracker.log_params({
                "model": "LGBMRegressor",
                "target": "resid_tech",
                "n_estimators": 500,
                "learning_rate": 0.03,
                "max_depth": 4,
                "n_features": len(self.feature_cols),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_cutoff": str(train_cutoff)
            })
            
            # Train model
            print("\n   🔧 Fitting LightGBM on residuals...")
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
            self.tracker.log_model(self.model, "news_agent_residual")
        
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
        
        # Add percentage
        total = importance_df["importance"].sum()
        importance_df["pct"] = (importance_df["importance"] / total * 100).round(1)
        
        return importance_df
    
    def predict(self, df: pd.DataFrame = None) -> pd.Series:
        """
        Generate residual predictions.
        
        Args:
            df: DataFrame with features (uses self.df if None)
        
        Returns:
            Series of predicted residuals
        """
        if df is None:
            df = self.df
        
        X = df[self.feature_cols]
        predictions = self.model.predict(X)
        
        return pd.Series(predictions, index=df.index)


def main():
    """Run NewsAgent on TechnicalAgent residuals."""
    print("\n" + "=" * 60)
    print("🚀 TITAN V8 NEWS AGENT (RESIDUAL PREDICTION)")
    print("    Phase 3: Can News Explain the Unexplained?")
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
    print("\n(Now that we removed volatility inertia, do news features matter?)\n")
    print(importance.head(10).to_string(index=False))
    
    # Check if shock_index or sentiment rose to the top
    top_5 = importance.head(5)["feature"].tolist()
    key_features_in_top5 = [f for f in ["shock_index", "sentiment_avg", "novelty_score"] if f in top_5]
    
    print("\n" + "=" * 60)
    print("📈 FINAL RESULTS")
    print("=" * 60)
    print(f"\nTarget: resid_tech (TechnicalAgent residuals)")
    print(f"\nTest Set Performance:")
    print(f"   RMSE: {metrics['test']['RMSE']:.4f}")
    print(f"   MAE:  {metrics['test']['MAE']:.4f}")
    print(f"   R²:   {metrics['test']['R2']:.4f}")
    print(f"   Directional Accuracy: {metrics['test']['Directional_Accuracy']:.1f}%")
    
    # Assessment
    print("\n" + "=" * 60)
    print("🔬 RESIDUAL PREDICTION ASSESSMENT")
    print("=" * 60)
    
    r2 = metrics['test']['R2']
    dir_acc = metrics['test']['Directional_Accuracy']
    
    if r2 > 0.05:
        verdict = "✅ SUCCESS - News explains some residual variance!"
        explanation = f"R²={r2:.3f} means news captures {r2*100:.1f}% of unexplained variance."
        action = "Proceed to build Hybrid Ensemble (Tech + News)."
    elif r2 > 0:
        verdict = "⚠️ MARGINAL - Tiny but positive signal detected."
        explanation = f"R²={r2:.4f} is small but positive."
        action = "News may add marginal value; consider feature selection."
    else:
        verdict = "❌ NO SIGNAL - News doesn't help explain residuals."
        explanation = f"R²={r2:.4f} is zero or negative."
        action = "Focus on other data sources (options, order flow, etc.)."
    
    print(f"\n{verdict}")
    print(f"\n{explanation}")
    print(f"\nAction: {action}")
    
    if key_features_in_top5:
        print(f"\n💡 Key news features in top 5: {key_features_in_top5}")
    else:
        print(f"\n💡 PCA themes dominate - specific events matter more than sentiment.")
    
    print("=" * 60)
    
    print("\n✅ NewsAgent residual training complete!")
    print("   Results logged to MLflow experiment: titan_v8_news_agent")
    print("   Model saved as: news_agent_residual")


if __name__ == "__main__":
    main()
