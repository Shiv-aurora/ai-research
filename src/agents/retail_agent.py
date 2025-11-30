"""
RetailRegimeAgent: Volume/Hype-Based Regime Classifier

Phase 13: The Retail Regime Agent

Key Insight from Audit:
- Retail signals work as REGIME INDICATORS, not direct predictors
- High attention regime: Different volatility dynamics
- Low attention regime: Higher volatility persistence
- Extreme hype: Contrarian/mean-reversion signal

Architecture:
- Model: LGBMClassifier (predict extreme volatility events)
- Target: is_extreme_vol (Top 20% volatility)
- Features: volume_shock, hype_zscore, price_acceleration
- Outputs:
    - retail_risk_score (probability of extreme event, 0-1)
    - is_high_attention (binary regime flag)

Usage:
    from src.agents.retail_agent import RetailRegimeAgent
    agent = RetailRegimeAgent()
    agent.load_and_process_data()
    agent.train()
    risk_scores = agent.predict_proba(df)
    regime_flags = agent.predict_class(df)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.tracker import MLTracker


class RetailRegimeAgent:
    """
    Retail Regime Agent: Uses volume/hype signals as regime indicators.
    
    Unlike NewsAgent (which predicts volatility directly), this agent:
    1. Classifies market attention regimes (high/low)
    2. Provides contrarian signals (extreme hype → mean reversion)
    3. Outputs risk scores for coordinator integration
    
    Key Features:
    - volume_shock: Volume / 20-day MA (unusual trading activity)
    - hype_zscore: Normalized hype signal
    - price_acceleration: Second derivative of log price
    """
    
    def __init__(self, 
                 experiment_name: str = "titan_v8_retail_regime",
                 extreme_percentile: float = 0.80):
        """
        Initialize RetailRegimeAgent.
        
        Args:
            experiment_name: MLflow experiment name
            extreme_percentile: Percentile threshold for extreme volatility
        """
        self.model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=8,
            min_child_samples=20,
            colsample_bytree=0.8,
            subsample=0.8,
            reg_alpha=0.05,
            reg_lambda=0.05,
            random_state=42,
            verbose=-1,
            objective='binary',
            metric='auc',
            is_unbalance=True
        )
        
        self.tracker = MLTracker(experiment_name)
        self.feature_cols = None
        self.target_col = "is_extreme_vol"
        self.extreme_percentile = extreme_percentile
        self.extreme_threshold = None
        self.attention_threshold = None  # For regime flag
        self.train_metrics = None
        self.test_metrics = None
        self.df = None
        
    def load_and_process_data(self) -> pd.DataFrame:
        """
        Load retail proxy data and create targets.
        
        Returns:
            DataFrame with features and targets
        """
        print("\n📂 Loading retail data...")
        
        # Load reddit proxy
        reddit = pd.read_parquet("data/processed/reddit_proxy.parquet")
        reddit["date"] = pd.to_datetime(reddit["date"]).dt.tz_localize(None)
        if reddit["ticker"].dtype.name == "category":
            reddit["ticker"] = reddit["ticker"].astype(str)
        print(f"   ✓ Reddit proxy: {len(reddit):,} rows")
        
        # Load residuals (for target)
        residuals = pd.read_parquet("data/processed/residuals.parquet")
        residuals["date"] = pd.to_datetime(residuals["date"]).dt.tz_localize(None)
        if residuals["ticker"].dtype.name == "category":
            residuals["ticker"] = residuals["ticker"].astype(str)
        print(f"   ✓ Residuals: {len(residuals):,} rows")
        
        # Merge
        df = pd.merge(reddit, residuals[["date", "ticker", "resid_tech"]], 
                      on=["date", "ticker"], how="inner")
        print(f"   ✓ Merged: {len(df):,} rows")
        
        # Create extreme volatility target
        print(f"\n   🎯 Creating classification target...")
        self.extreme_threshold = df["resid_tech"].quantile(self.extreme_percentile)
        df["is_extreme_vol"] = (df["resid_tech"] > self.extreme_threshold).astype(int)
        
        print(f"      Threshold: {self.extreme_percentile:.0%} percentile = {self.extreme_threshold:.4f}")
        print(f"      Normal:  {(df['is_extreme_vol'] == 0).sum():,} ({(df['is_extreme_vol'] == 0).mean():.1%})")
        print(f"      Extreme: {(df['is_extreme_vol'] == 1).sum():,} ({(df['is_extreme_vol'] == 1).mean():.1%})")
        
        # Create attention regime flag
        self.attention_threshold = 1.5  # volume_shock > 1.5 = high attention
        df["is_high_attention"] = (df["volume_shock"] > self.attention_threshold).astype(int)
        print(f"\n   📊 Attention Regime:")
        print(f"      High Attention: {df['is_high_attention'].mean():.1%}")
        
        # Define features
        self.feature_cols = [
            "volume_shock",
            "volume_shock_roll3",
            "hype_signal",
            "hype_signal_roll3",
            "hype_signal_roll7",
            "hype_zscore",
            "price_acceleration"
        ]
        
        # Drop NaN
        df = df.dropna(subset=self.feature_cols + [self.target_col])
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        print(f"\n   📊 Final dataset: {len(df):,} rows")
        
        self.df = df
        return df
    
    def train(self, df: pd.DataFrame = None) -> dict:
        """
        Train the retail regime classifier.
        
        Args:
            df: DataFrame with features and target (optional, uses self.df)
            
        Returns:
            Dictionary with train/test metrics
        """
        if df is None:
            df = self.df
        
        print("\n" + "=" * 70)
        print("🎯 TRAINING RETAIL REGIME AGENT")
        print("=" * 70)
        
        print(f"\n   Features: {len(self.feature_cols)}")
        for f in self.feature_cols:
            print(f"      - {f}")
        print(f"   Target: {self.target_col}")
        
        # Time-based split
        cutoff = pd.to_datetime("2023-01-01")
        train = df[df["date"] < cutoff]
        test = df[df["date"] >= cutoff]
        
        print(f"\n   📊 Split:")
        print(f"      Train: {len(train):,} samples ({train['is_extreme_vol'].mean():.1%} extreme)")
        print(f"      Test:  {len(test):,} samples ({test['is_extreme_vol'].mean():.1%} extreme)")
        
        X_train = train[self.feature_cols].fillna(0)
        y_train = train[self.target_col]
        X_test = test[self.feature_cols].fillna(0)
        y_test = test[self.target_col]
        
        # Train
        print("\n   🔧 Fitting LGBMClassifier...")
        
        with self.tracker.start_run(run_name="retail_regime_classifier"):
            self.tracker.log_params({
                "model": "LGBMClassifier_RetailRegime",
                "target": self.target_col,
                "extreme_percentile": self.extreme_percentile,
                "extreme_threshold": self.extreme_threshold,
                "attention_threshold": self.attention_threshold,
                "n_features": len(self.feature_cols),
                "n_train": len(train),
                "n_test": len(test)
            })
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
            )
            
            # Predictions
            y_train_proba = self.model.predict_proba(X_train)[:, 1]
            y_test_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Metrics
            print("\n   📈 Evaluating...")
            
            self.train_metrics = {
                "AUC": roc_auc_score(y_train, y_train_proba),
                "Accuracy": accuracy_score(y_train, (y_train_proba > 0.5).astype(int)),
                "Precision": precision_score(y_train, (y_train_proba > 0.5).astype(int), zero_division=0),
                "Recall": recall_score(y_train, (y_train_proba > 0.5).astype(int), zero_division=0),
                "F1": f1_score(y_train, (y_train_proba > 0.5).astype(int), zero_division=0),
            }
            
            self.test_metrics = {
                "AUC": roc_auc_score(y_test, y_test_proba),
                "Accuracy": accuracy_score(y_test, (y_test_proba > 0.5).astype(int)),
                "Precision": precision_score(y_test, (y_test_proba > 0.5).astype(int), zero_division=0),
                "Recall": recall_score(y_test, (y_test_proba > 0.5).astype(int), zero_division=0),
                "F1": f1_score(y_test, (y_test_proba > 0.5).astype(int), zero_division=0),
            }
            
            # Log metrics using MLflow directly
            import mlflow
            for key, val in self.train_metrics.items():
                mlflow.log_metric(f"train_{key}", val)
            for key, val in self.test_metrics.items():
                mlflow.log_metric(f"test_{key}", val)
            
            # Print results
            print(f"\n   {'Metric':<20} {'Train':>10} {'Test':>10}")
            print("   " + "-" * 42)
            for key in self.train_metrics.keys():
                print(f"   {key:<20} {self.train_metrics[key]:>10.4f} {self.test_metrics[key]:>10.4f}")
            
            # Feature importance
            print(f"\n   📊 Feature Importance:")
            importance = pd.DataFrame({
                "feature": self.feature_cols,
                "importance": self.model.feature_importances_
            }).sort_values("importance", ascending=False)
            
            for _, row in importance.iterrows():
                print(f"      {row['feature']:<25}: {row['importance']:.4f}")
            
            # Goal check
            if self.test_metrics["AUC"] >= 0.60:
                print(f"\n   ✅ AUC {self.test_metrics['AUC']:.4f} >= 0.60 TARGET ACHIEVED!")
            else:
                print(f"\n   ⚠️ AUC {self.test_metrics['AUC']:.4f} < 0.60 (weak but useful)")
            
            self.tracker.log_model(self.model, "retail_regime_classifier")
        
        return {
            "train": self.train_metrics,
            "test": self.test_metrics
        }
    
    def predict_proba(self, df: pd.DataFrame = None) -> pd.Series:
        """
        Predict probability of extreme volatility (retail_risk_score).
        
        Args:
            df: DataFrame with features (optional, uses self.df)
            
        Returns:
            Series with retail_risk_score (0-1)
        """
        if df is None:
            df = self.df
        
        X = df[self.feature_cols].fillna(0)
        proba = self.model.predict_proba(X)[:, 1]
        
        return pd.Series(proba, index=df.index, name="retail_risk_score")
    
    def predict_class(self, df: pd.DataFrame = None) -> pd.Series:
        """
        Predict high attention regime flag.
        
        Uses volume_shock > attention_threshold as regime indicator.
        
        Args:
            df: DataFrame with features (optional, uses self.df)
            
        Returns:
            Series with is_high_attention (0 or 1)
        """
        if df is None:
            df = self.df
        
        # Use volume_shock for regime classification
        is_high_attention = (df["volume_shock"] > self.attention_threshold).astype(int)
        
        return pd.Series(is_high_attention.values, index=df.index, name="is_high_attention")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance DataFrame."""
        importance = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        total = importance["importance"].sum()
        if total > 0:
            importance["pct"] = (importance["importance"] / total * 100).round(1)
        else:
            importance["pct"] = 0.0
        
        return importance


def main():
    """Test the RetailRegimeAgent."""
    print("\n" + "=" * 70)
    print("🚀 RETAIL REGIME AGENT TEST")
    print("=" * 70)
    
    agent = RetailRegimeAgent()
    df = agent.load_and_process_data()
    metrics = agent.train(df)
    
    # Get predictions
    risk_scores = agent.predict_proba(df)
    regime_flags = agent.predict_class(df)
    
    print(f"\n📊 Predictions:")
    print(f"   retail_risk_score: mean={risk_scores.mean():.4f}, std={risk_scores.std():.4f}")
    print(f"   is_high_attention: {regime_flags.mean():.1%} high attention days")
    
    print("\n" + "=" * 70)
    print("✅ RETAIL REGIME AGENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

