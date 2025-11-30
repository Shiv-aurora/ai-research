"""
NewsAgent: Classification-Based Risk Scorer (Phase 11)

PHASE 11: THE HYBRID PIVOT
Switched from Regression to Classification after experiments showed:
- Regression R² ≈ 0% (news cannot predict vol magnitude)
- Classification AUC ≈ 0.60 (news CAN predict extreme events)

This agent predicts "Extreme Volatility Events" (residuals > 80th percentile)
and outputs a risk probability score (0.0 to 1.0).

Target: is_extreme = 1 if resid_tech > 80th percentile
Model: LGBMClassifier with AUC metric
Output: news_risk_score (probability of extreme event)

Usage:
    from src.agents.news_agent import NewsAgent
    agent = NewsAgent()
    df = agent.load_and_merge_data()
    agent.train(df)
    risk_scores = agent.predict_proba(df)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.tracker import MLTracker


def calculate_decay_kernel(series: pd.Series, group_col: pd.Series = None) -> pd.Series:
    """
    Calculate exponential decay kernel for a time series.
    
    Formula: 0.50 * shift(1) + 0.25 * shift(2) + 0.15 * shift(3) + 0.10 * shift(4)
    
    Args:
        series: The time series to apply decay kernel to
        group_col: Optional groupby column (e.g., ticker)
        
    Returns:
        Series with decay-weighted memory of past values
    """
    weights = [0.50, 0.25, 0.15, 0.10]
    
    if group_col is not None:
        df = pd.DataFrame({"value": series, "group": group_col})
        
        result = pd.Series(0.0, index=series.index)
        for i, w in enumerate(weights, 1):
            shifted = df.groupby("group")["value"].shift(i)
            result += w * shifted.fillna(0)
        
        return result
    else:
        result = pd.Series(0.0, index=series.index)
        for i, w in enumerate(weights, 1):
            result += w * series.shift(i).fillna(0)
        
        return result


class NewsAgent:
    """
    NewsAgent: Classification-Based Risk Scorer (Phase 11).
    
    Predicts probability of "Extreme Volatility Events" from news features.
    
    Key Changes from Phase 6:
    - Model: LGBMClassifier (was LGBMRegressor)
    - Target: is_extreme binary (was resid_tech continuous)
    - Output: news_risk_score probability (was residual prediction)
    - Metric: AUC-ROC (was R²)
    
    Features:
        - news_memory: Decay kernel of news_count
        - shock_memory: Decay kernel of shock_index
        - sentiment_memory: Decay kernel of sentiment
        - PCA features: Topic embeddings
    """
    
    def __init__(self, experiment_name: str = "titan_v8_news_classifier",
                 extreme_percentile: float = 0.80):
        """
        Initialize NewsAgent classifier.
        
        Args:
            experiment_name: MLflow experiment name
            extreme_percentile: Percentile threshold for extreme events (default: 80th)
        """
        # LGBMClassifier with AUC optimization
        self.model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=8,
            min_child_samples=20,
            colsample_bytree=0.8,
            subsample=0.8,
            reg_alpha=0.05,
            reg_lambda=0.05,
            is_unbalance=True,  # Handle class imbalance
            objective='binary',
            metric='auc',
            random_state=42,
            verbose=-1
        )
        
        self.tracker = MLTracker(experiment_name)
        self.feature_cols = None
        self.target_col = "is_extreme"
        self.resid_col = "resid_tech"
        self.extreme_percentile = extreme_percentile
        self.extreme_threshold = None
        
        self.train_metrics = None
        self.test_metrics = None
        self.df = None
        
    def load_and_merge_data(self) -> pd.DataFrame:
        """
        Load data and create classification target.
        
        Target Definition:
        - is_extreme = 1 if resid_tech > 80th percentile
        - is_extreme = 0 otherwise
        
        Returns:
            DataFrame with features and classification target
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
        
        # Load VIX
        try:
            targets_path = Path("data/processed/targets.parquet")
            targets_df = pd.read_parquet(targets_path)
            vix_df = targets_df[["date", "ticker", "VIX_close"]].copy()
        except:
            vix_df = None
        
        # Normalize dates
        for df in [residuals_df, news_df]:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            if df["ticker"].dtype.name == "category":
                df["ticker"] = df["ticker"].astype(str)
        
        if vix_df is not None:
            vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.tz_localize(None)
            if vix_df["ticker"].dtype.name == "category":
                vix_df["ticker"] = vix_df["ticker"].astype(str)
        
        # Merge
        merged = pd.merge(residuals_df, news_df, on=["date", "ticker"], how="inner")
        
        if vix_df is not None:
            merged = pd.merge(merged, vix_df, on=["date", "ticker"], how="left")
            merged["VIX_close"] = merged["VIX_close"].ffill().fillna(20)
        else:
            merged["VIX_close"] = 20.0
        
        # Sort for proper kernel calculations
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        print(f"   ✓ Merged: {len(merged):,} rows")
        
        # =============================================
        # CREATE CLASSIFICATION TARGET
        # =============================================
        print(f"\n   🎯 Creating classification target...")
        print(f"      Threshold: {self.extreme_percentile*100:.0f}th percentile of resid_tech")
        
        # Calculate threshold on TRAINING data only (before 2023)
        train_cutoff = pd.to_datetime("2023-01-01")
        train_resid = merged.loc[merged["date"] < train_cutoff, self.resid_col]
        
        self.extreme_threshold = train_resid.quantile(self.extreme_percentile)
        print(f"      Threshold value: {self.extreme_threshold:.4f}")
        
        # Create binary target
        merged[self.target_col] = (merged[self.resid_col] > self.extreme_threshold).astype(int)
        
        # Class distribution
        n_extreme = merged[self.target_col].sum()
        n_normal = len(merged) - n_extreme
        pct_extreme = n_extreme / len(merged) * 100
        
        print(f"      Normal:  {n_normal:,} ({100-pct_extreme:.1f}%)")
        print(f"      Extreme: {n_extreme:,} ({pct_extreme:.1f}%)")
        
        # =============================================
        # FEATURE ENGINEERING: DECAY KERNEL
        # =============================================
        print("\n   🔧 Engineering DECAY KERNEL features...")
        
        # Apply decay kernel to key features
        merged["news_memory"] = calculate_decay_kernel(
            merged["news_count"], 
            group_col=merged["ticker"]
        )
        
        merged["shock_memory"] = calculate_decay_kernel(
            merged["shock_index"],
            group_col=merged["ticker"]
        )
        
        merged["sentiment_memory"] = calculate_decay_kernel(
            merged["sentiment_avg"],
            group_col=merged["ticker"]
        )
        
        print(f"      ✓ Created: news_memory, shock_memory, sentiment_memory")
        
        # VIX interaction
        merged["shock_vix_memory"] = merged["shock_memory"] * merged["VIX_close"] / 20
        print(f"      ✓ Created: shock_vix_memory (regime interaction)")
        
        # Drop NaN rows
        before = len(merged)
        merged = merged.dropna(subset=["news_memory", "shock_memory", self.resid_col])
        after = len(merged)
        print(f"      ✓ Dropped {before - after} rows with NaN")
        
        print(f"\n   📊 Final dataset: {len(merged):,} rows")
        
        self.df = merged
        return merged
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get feature column names."""
        features = []
        
        # Decay kernel features
        kernel_features = ["news_memory", "shock_memory", "sentiment_memory", "shock_vix_memory"]
        features.extend([f for f in kernel_features if f in df.columns])
        
        # Current values
        current_features = ["sentiment_avg", "novelty_score", "shock_index", "news_count"]
        features.extend([f for f in current_features if f in df.columns])
        
        # PCA columns
        pca_cols = [c for c in df.columns if c.startswith("news_pca_")]
        features.extend(sorted(pca_cols))
        
        # VIX
        if "VIX_close" in df.columns:
            features.append("VIX_close")
        
        # Remove duplicates
        features = list(dict.fromkeys(features))
        
        return features
    
    def train(self, df: pd.DataFrame = None) -> dict:
        """
        Train LGBMClassifier for extreme event prediction.
        
        Returns:
            Dictionary with train/test metrics including AUC
        """
        if df is None:
            df = self.df
            
        print("\n🎯 Training NewsAgent CLASSIFIER...")
        
        # Define features
        self.feature_cols = self.get_feature_columns(df)
        
        print(f"\n   Features: {len(self.feature_cols)}")
        print(f"   Target: {self.target_col} (Extreme Event)")
        
        # Time-series split
        train_cutoff = pd.to_datetime("2023-01-01")
        
        train_mask = df["date"] < train_cutoff
        test_mask = df["date"] >= train_cutoff
        
        X_train = df.loc[train_mask, self.feature_cols].fillna(0)
        y_train = df.loc[train_mask, self.target_col]
        X_test = df.loc[test_mask, self.feature_cols].fillna(0)
        y_test = df.loc[test_mask, self.target_col]
        
        print(f"\n   📊 Split:")
        print(f"      Train: {len(X_train):,} samples ({y_train.mean()*100:.1f}% extreme)")
        print(f"      Test:  {len(X_test):,} samples ({y_test.mean()*100:.1f}% extreme)")
        
        # Start MLflow run
        with self.tracker.start_run(run_name="news_classifier"):
            # Log parameters
            self.tracker.log_params({
                "model": "LGBMClassifier",
                "target": "is_extreme",
                "extreme_percentile": self.extreme_percentile,
                "extreme_threshold": self.extreme_threshold,
                "n_features": len(self.feature_cols),
            })
            
            # Train model
            print("\n   🔧 Fitting LGBMClassifier...")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric="auc"
            )
            
            # Predictions
            y_train_proba = self.model.predict_proba(X_train)[:, 1]
            y_test_proba = self.model.predict_proba(X_test)[:, 1]
            
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # Calculate metrics
            print("\n   📈 Evaluating...")
            
            self.train_metrics = {
                'AUC': roc_auc_score(y_train, y_train_proba),
                'Accuracy': accuracy_score(y_train, y_train_pred),
                'Precision': precision_score(y_train, y_train_pred, zero_division=0),
                'Recall': recall_score(y_train, y_train_pred, zero_division=0),
                'F1': f1_score(y_train, y_train_pred, zero_division=0)
            }
            
            self.test_metrics = {
                'AUC': roc_auc_score(y_test, y_test_proba),
                'Accuracy': accuracy_score(y_test, y_test_pred),
                'Precision': precision_score(y_test, y_test_pred, zero_division=0),
                'Recall': recall_score(y_test, y_test_pred, zero_division=0),
                'F1': f1_score(y_test, y_test_pred, zero_division=0)
            }
            
            # Log to MLflow
            try:
                import mlflow
                for key, val in self.test_metrics.items():
                    mlflow.log_metric(f"test_{key}", val)
            except:
                pass  # MLflow logging optional
            
            # Print comparison
            print(f"\n   {'Metric':<15} {'Train':>10} {'Test':>10}")
            print("   " + "-" * 37)
            
            for metric in ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']:
                train_val = self.train_metrics[metric]
                test_val = self.test_metrics[metric]
                print(f"   {metric:<15} {train_val:>10.4f} {test_val:>10.4f}")
            
            # AUC assessment
            auc = self.test_metrics['AUC']
            print(f"\n   🎯 Test AUC: {auc:.4f}")
            
            if auc >= 0.65:
                print(f"   ✅ EXCELLENT! Strong predictive signal")
            elif auc >= 0.60:
                print(f"   ✅ GOOD! Meaningful signal detected")
            elif auc >= 0.55:
                print(f"   ⚠️ WEAK signal, but better than random")
            else:
                print(f"   ❌ No significant signal")
            
            # Log model
            self.tracker.log_model(self.model, "news_classifier")
        
        return {
            "train": self.train_metrics,
            "test": self.test_metrics
        }
    
    def predict_proba(self, df: pd.DataFrame = None) -> pd.Series:
        """
        Generate probability of extreme event.
        
        Returns:
            Series with news_risk_score (0.0 to 1.0)
        """
        if df is None:
            df = self.df
        
        X = df[self.feature_cols].fillna(0)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return pd.Series(probabilities, index=df.index, name="news_risk_score")
    
    def predict(self, df: pd.DataFrame = None) -> pd.Series:
        """
        Generate binary predictions.
        
        Returns:
            Series with is_extreme predictions (0 or 1)
        """
        if df is None:
            df = self.df
        
        X = df[self.feature_cols].fillna(0)
        predictions = self.model.predict(X)
        
        return pd.Series(predictions, index=df.index, name="is_extreme_pred")
    
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
        
        return importance_df


def main():
    """Run NewsAgent classifier."""
    print("\n" + "=" * 65)
    print("🚀 TITAN V8 NEWS AGENT - CLASSIFIER (Phase 11)")
    print("    The Hybrid Pivot: Regression → Classification")
    print("=" * 65)
    
    # Initialize agent
    agent = NewsAgent(extreme_percentile=0.80)
    
    # Load and merge data
    df = agent.load_and_merge_data()
    
    # Train model
    metrics = agent.train(df)
    
    # Get predictions
    risk_scores = agent.predict_proba(df)
    
    # Get feature importance
    importance = agent.get_feature_importance()
    
    # Print feature importance
    print("\n" + "=" * 65)
    print("📊 FEATURE IMPORTANCE")
    print("=" * 65)
    print(importance.head(15).to_string(index=False))
    
    # Final assessment
    print("\n" + "=" * 65)
    print("📈 FINAL RESULTS")
    print("=" * 65)
    
    auc = metrics['test']['AUC']
    
    print(f"\n   Test AUC: {auc:.4f} ({auc*100:.2f}%)")
    print(f"   Goal: > 0.60")
    
    if auc >= 0.60:
        print(f"\n   ✅ Goal achieved! News can predict extreme events.")
    else:
        print(f"\n   ⚠️ Below goal, but still useful as risk signal.")
    
    print("\n" + "=" * 65)
    print("✅ NewsAgent Classifier complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
