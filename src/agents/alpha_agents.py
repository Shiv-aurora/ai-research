"""
Alpha Agents: FundamentalAgent and RedditAgent

Phase 4: Additional agents to capture unexplained volatility.

FundamentalAgent:
- Uses debt_to_equity, days_to_ex_div, and VIX interactions
- Predicts residuals from TechnicalAgent

RedditAgent:
- Uses volume/price anomalies as retail hype proxy
- Predicts residuals from TechnicalAgent

Both agents use robust LightGBM parameters to prevent overfitting.

Usage:
    python -m src.agents.alpha_agents
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.tracker import MLTracker


class FundamentalAgent:
    """
    Predicts volatility residuals using fundamental features.
    
    Features:
    - debt_to_equity: Financial leverage
    - days_to_ex_div: Proximity to dividend events
    - debt_vix_interaction: Leverage × Market Fear
    
    Target: resid_tech (unexplained volatility from TechnicalAgent)
    """
    
    def __init__(self, experiment_name: str = "titan_v8_fundamental"):
        """Initialize FundamentalAgent with robust LightGBM params."""
        # Robust hyperparameters (prevent overfitting)
        self.model = LGBMRegressor(
            n_estimators=200,
            max_depth=2,
            learning_rate=0.02,
            num_leaves=4,
            min_child_samples=30,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        
        self.tracker = MLTracker(experiment_name)
        self.feature_cols = None
        self.target_col = "resid_tech"
        self.train_metrics = None
        self.test_metrics = None
        
    def load_and_process_data(self) -> pd.DataFrame:
        """Load targets and residuals, create fundamental features."""
        print("\n📂 Loading data for FundamentalAgent...")
        
        # Load targets (has fundamental features)
        targets = pd.read_parquet("data/processed/targets.parquet")
        residuals = pd.read_parquet("data/processed/residuals.parquet")
        
        # Normalize dates
        for df in [targets, residuals]:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            if df["ticker"].dtype.name == "category":
                df["ticker"] = df["ticker"].astype(str)
        
        # Merge
        merged = pd.merge(residuals, targets, on=["date", "ticker"], how="inner")
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        print(f"   Merged: {len(merged):,} rows")
        
        # Feature engineering
        print("   Engineering fundamental features...")
        
        # Fill missing fundamental features
        merged["debt_to_equity"] = merged["debt_to_equity"].fillna(0)
        merged["days_to_ex_div"] = merged["days_to_ex_div"].fillna(365)  # Far from ex-div
        merged["VIX_close"] = merged["VIX_close"].ffill().fillna(15)
        
        # Interaction: High leverage + high fear = danger
        merged["debt_vix_interaction"] = merged["debt_to_equity"] * merged["VIX_close"]
        
        # Dividend proximity feature
        merged["near_ex_div"] = (merged["days_to_ex_div"] < 7).astype(int)
        
        # Lagged fundamental features
        merged["debt_lag5"] = merged.groupby("ticker")["debt_to_equity"].shift(5)
        merged["debt_lag5"] = merged["debt_lag5"].fillna(merged["debt_to_equity"])
        
        # Drop NaN
        merged = merged.dropna(subset=[self.target_col])
        
        print(f"   Final: {len(merged):,} rows")
        
        return merged
    
    def get_feature_columns(self) -> list:
        """Define fundamental feature columns."""
        return [
            "debt_to_equity",
            "days_to_ex_div",
            "near_ex_div",
            "VIX_close",
            "debt_vix_interaction",
            "debt_lag5"
        ]
    
    def train(self, df: pd.DataFrame = None) -> dict:
        """Train on residuals."""
        if df is None:
            df = self.load_and_process_data()
        
        print("\n🎯 Training FundamentalAgent...")
        
        self.feature_cols = self.get_feature_columns()
        self.feature_cols = [f for f in self.feature_cols if f in df.columns]
        
        print(f"   Features: {self.feature_cols}")
        print(f"   Target: {self.target_col}")
        
        # Time-series split
        cutoff = pd.to_datetime("2023-01-01")
        train = df[df["date"] < cutoff]
        test = df[df["date"] >= cutoff]
        
        if len(train) < 50 or len(test) < 20:
            split_idx = int(len(df) * 0.7)
            train = df.iloc[:split_idx]
            test = df.iloc[split_idx:]
        
        X_train = train[self.feature_cols]
        y_train = train[self.target_col]
        X_test = test[self.feature_cols]
        y_test = test[self.target_col]
        
        print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Train
        with self.tracker.start_run(run_name="fundamental_agent"):
            self.tracker.log_params({
                "model": "LGBMRegressor",
                "target": "resid_tech",
                "n_features": len(self.feature_cols)
            })
            
            self.model.fit(X_train, y_train)
            
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            self.train_metrics = self.tracker.log_metrics(y_train.values, y_train_pred, step=0)
            self.test_metrics = self.tracker.log_metrics(y_test.values, y_test_pred, step=1)
            
            self.tracker.log_model(self.model, "fundamental_agent")
        
        return {
            "train": self.train_metrics,
            "test": self.test_metrics
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importances."""
        importance = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        total = importance["importance"].sum()
        importance["pct"] = (importance["importance"] / total * 100).round(1)
        
        return importance


class RedditAgent:
    """
    Predicts volatility residuals using retail hype proxy.
    
    Features:
    - volume_shock: Volume anomaly
    - hype_signal_roll3: Smoothed hype signal
    - hype_zscore: Normalized hype
    
    Target: resid_tech (unexplained volatility from TechnicalAgent)
    """
    
    def __init__(self, experiment_name: str = "titan_v8_reddit"):
        """Initialize RedditAgent with robust LightGBM params."""
        self.model = LGBMRegressor(
            n_estimators=200,
            max_depth=2,
            learning_rate=0.02,
            num_leaves=4,
            min_child_samples=30,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        
        self.tracker = MLTracker(experiment_name)
        self.feature_cols = None
        self.target_col = "resid_tech"
        self.train_metrics = None
        self.test_metrics = None
        
    def load_and_process_data(self) -> pd.DataFrame:
        """Load reddit proxy and residuals."""
        print("\n📂 Loading data for RedditAgent...")
        
        reddit_path = Path("data/processed/reddit_proxy.parquet")
        if not reddit_path.exists():
            raise FileNotFoundError(
                "Reddit proxy not found! Run create_reddit_proxy first."
            )
        
        reddit = pd.read_parquet(reddit_path)
        residuals = pd.read_parquet("data/processed/residuals.parquet")
        
        # Normalize dates
        for df in [reddit, residuals]:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            if df["ticker"].dtype.name == "category":
                df["ticker"] = df["ticker"].astype(str)
        
        # Merge
        merged = pd.merge(residuals, reddit, on=["date", "ticker"], how="inner")
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        print(f"   Merged: {len(merged):,} rows")
        
        # Add lagged hype features
        merged["hype_lag1"] = merged.groupby("ticker")["hype_signal_roll3"].shift(1)
        merged["hype_lag3"] = merged.groupby("ticker")["hype_signal_roll3"].shift(3)
        merged["volume_shock_lag1"] = merged.groupby("ticker")["volume_shock"].shift(1)
        
        merged = merged.dropna()
        
        print(f"   Final: {len(merged):,} rows")
        
        return merged
    
    def get_feature_columns(self) -> list:
        """Define reddit proxy feature columns."""
        return [
            "volume_shock",
            "volume_shock_roll3",
            "volume_shock_lag1",
            "hype_signal_roll3",
            "hype_signal_roll7",
            "hype_zscore",
            "hype_lag1",
            "hype_lag3",
            "price_acceleration"
        ]
    
    def train(self, df: pd.DataFrame = None) -> dict:
        """Train on residuals."""
        if df is None:
            df = self.load_and_process_data()
        
        print("\n🎯 Training RedditAgent...")
        
        self.feature_cols = self.get_feature_columns()
        self.feature_cols = [f for f in self.feature_cols if f in df.columns]
        
        print(f"   Features: {self.feature_cols}")
        print(f"   Target: {self.target_col}")
        
        # Time-series split
        cutoff = pd.to_datetime("2023-01-01")
        train = df[df["date"] < cutoff]
        test = df[df["date"] >= cutoff]
        
        if len(train) < 50 or len(test) < 20:
            split_idx = int(len(df) * 0.7)
            train = df.iloc[:split_idx]
            test = df.iloc[split_idx:]
        
        X_train = train[self.feature_cols]
        y_train = train[self.target_col]
        X_test = test[self.feature_cols]
        y_test = test[self.target_col]
        
        print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Train
        with self.tracker.start_run(run_name="reddit_agent"):
            self.tracker.log_params({
                "model": "LGBMRegressor",
                "target": "resid_tech",
                "n_features": len(self.feature_cols)
            })
            
            self.model.fit(X_train, y_train)
            
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            self.train_metrics = self.tracker.log_metrics(y_train.values, y_train_pred, step=0)
            self.test_metrics = self.tracker.log_metrics(y_test.values, y_test_pred, step=1)
            
            self.tracker.log_model(self.model, "reddit_agent")
        
        return {
            "train": self.train_metrics,
            "test": self.test_metrics
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importances."""
        importance = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        total = importance["importance"].sum()
        importance["pct"] = (importance["importance"] / total * 100).round(1)
        
        return importance


def main():
    """Test both alpha agents."""
    print("\n" + "=" * 65)
    print("🚀 ALPHA AGENTS TEST")
    print("=" * 65)
    
    # Test FundamentalAgent
    print("\n" + "-" * 65)
    print("📊 FUNDAMENTAL AGENT")
    print("-" * 65)
    
    fund_agent = FundamentalAgent()
    fund_metrics = fund_agent.train()
    
    print(f"\n   Train R²: {fund_metrics['train']['R2']:.4f}")
    print(f"   Test R²:  {fund_metrics['test']['R2']:.4f}")
    
    print("\n   Top 3 Features:")
    print(fund_agent.get_feature_importance().head(3).to_string(index=False))
    
    # Test RedditAgent
    print("\n" + "-" * 65)
    print("📊 REDDIT AGENT")
    print("-" * 65)
    
    try:
        reddit_agent = RedditAgent()
        reddit_metrics = reddit_agent.train()
        
        print(f"\n   Train R²: {reddit_metrics['train']['R2']:.4f}")
        print(f"   Test R²:  {reddit_metrics['test']['R2']:.4f}")
        
        print("\n   Top 3 Features:")
        print(reddit_agent.get_feature_importance().head(3).to_string(index=False))
    except FileNotFoundError as e:
        print(f"   ⚠️ {e}")
        print("   Run: python -m src.pipeline.create_reddit_proxy first")
    
    print("\n" + "=" * 65)
    print("✅ ALPHA AGENTS COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()

