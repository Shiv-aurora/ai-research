"""
Alpha Agents: FundamentalAgent and RetailRiskAgent

Phase 4.5: Additional agents to capture unexplained volatility.

FundamentalAgent:
- Uses debt_to_equity, days_to_ex_div, and VIX interactions
- Predicts residuals from TechnicalAgent

RetailRiskAgent (replaced RedditAgent):
- Uses external market proxies (BTC, GME, IWM) for retail sentiment
- GLOBAL signals applied to all tickers
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


class RetailRiskAgent:
    """
    Predicts volatility residuals using external retail risk signals.
    
    Features (GLOBAL - same for all tickers):
    - btc_vol_5d: Bitcoin volatility (Crypto/Degen risk)
    - gme_vol_shock: GameStop volume anomaly (Meme mania)
    - small_cap_excess: IWM/SPY ratio (Risk-on sentiment)
    - retail_mania: Composite retail risk index
    - btc_vix_interaction: Crypto × Fear interaction
    
    Target: resid_tech (unexplained volatility from TechnicalAgent)
    """
    
    def __init__(self, experiment_name: str = "titan_v8_retail_risk"):
        """Initialize RetailRiskAgent with robust LightGBM params."""
        self.model = LGBMRegressor(
            n_estimators=200,
            max_depth=2,
            learning_rate=0.03,  # Slightly faster for global signals
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
        """Load retail signals and residuals, create features."""
        print("\n📂 Loading data for RetailRiskAgent...")
        
        retail_path = Path("data/processed/retail_signals.parquet")
        if not retail_path.exists():
            raise FileNotFoundError(
                "Retail signals not found! Run ingest_retail first."
            )
        
        # Load data
        retail = pd.read_parquet(retail_path)
        residuals = pd.read_parquet("data/processed/residuals.parquet")
        targets = pd.read_parquet("data/processed/targets.parquet")
        
        # Normalize dates
        for df in [retail, residuals, targets]:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            if "ticker" in df.columns and df["ticker"].dtype.name == "category":
                df["ticker"] = df["ticker"].astype(str)
        
        print(f"   Retail signals: {len(retail):,} rows")
        print(f"   Residuals: {len(residuals):,} rows")
        
        # Get VIX from targets
        vix_df = targets[["date", "ticker", "VIX_close"]].copy()
        
        # Merge residuals with VIX first
        merged = pd.merge(residuals, vix_df, on=["date", "ticker"], how="left")
        merged["VIX_close"] = merged["VIX_close"].ffill().fillna(15)
        
        # Left join retail signals (GLOBAL) onto residuals by date only
        merged = pd.merge(merged, retail, on="date", how="left")
        
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        print(f"   After merge: {len(merged):,} rows")
        
        # Feature engineering
        print("   Engineering retail risk features...")
        
        # Fill any missing retail signals
        retail_cols = ["btc_vol_5d", "btc_ret_5d", "btc_mom_20d", 
                       "gme_vol_shock", "gme_vol_5d", "gme_ret_5d",
                       "small_cap_excess", "small_cap_mom",
                       "retail_mania", "risk_on_signal"]
        
        for col in retail_cols:
            if col in merged.columns:
                merged[col] = merged[col].ffill().fillna(0)
        
        # Interaction features
        if "btc_vol_5d" in merged.columns:
            merged["btc_vix_interaction"] = merged["btc_vol_5d"] * merged["VIX_close"]
        
        if "gme_vol_shock" in merged.columns:
            merged["gme_vix_interaction"] = merged["gme_vol_shock"] * merged["VIX_close"]
        
        # Lagged retail signals
        for col in ["retail_mania", "btc_vol_5d"]:
            if col in merged.columns:
                merged[f"{col}_lag1"] = merged.groupby("ticker")[col].shift(1)
                merged[f"{col}_lag5"] = merged.groupby("ticker")[col].shift(5)
        
        # Drop NaN
        merged = merged.dropna(subset=[self.target_col])
        
        print(f"   Final: {len(merged):,} rows")
        
        return merged
    
    def get_feature_columns(self) -> list:
        """Define retail risk feature columns."""
        return [
            # Core retail signals
            "btc_vol_5d",
            "btc_ret_5d",
            "gme_vol_shock",
            "gme_vol_5d",
            "small_cap_excess",
            "small_cap_mom",
            # Composite
            "retail_mania",
            "risk_on_signal",
            # Interactions
            "btc_vix_interaction",
            "gme_vix_interaction",
            # VIX context
            "VIX_close",
            # Lagged
            "retail_mania_lag1",
            "retail_mania_lag5",
            "btc_vol_5d_lag1",
            "btc_vol_5d_lag5"
        ]
    
    def train(self, df: pd.DataFrame = None) -> dict:
        """Train on residuals."""
        if df is None:
            df = self.load_and_process_data()
        
        print("\n🎯 Training RetailRiskAgent...")
        
        self.feature_cols = self.get_feature_columns()
        self.feature_cols = [f for f in self.feature_cols if f in df.columns]
        
        print(f"   Features: {len(self.feature_cols)}")
        for f in self.feature_cols:
            print(f"      - {f}")
        print(f"   Target: {self.target_col}")
        
        # Time-series split
        cutoff = pd.to_datetime("2023-01-01")
        train = df[df["date"] < cutoff]
        test = df[df["date"] >= cutoff]
        
        if len(train) < 50 or len(test) < 20:
            split_idx = int(len(df) * 0.7)
            train = df.iloc[:split_idx]
            test = df.iloc[split_idx:]
        
        X_train = train[self.feature_cols].fillna(0)
        y_train = train[self.target_col]
        X_test = test[self.feature_cols].fillna(0)
        y_test = test[self.target_col]
        
        print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Train
        with self.tracker.start_run(run_name="retail_risk_agent"):
            self.tracker.log_params({
                "model": "LGBMRegressor",
                "target": "resid_tech",
                "n_features": len(self.feature_cols),
                "feature_type": "global_retail_signals"
            })
            
            self.model.fit(X_train, y_train)
            
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            self.train_metrics = self.tracker.log_metrics(y_train.values, y_train_pred, step=0)
            self.test_metrics = self.tracker.log_metrics(y_test.values, y_test_pred, step=1)
            
            self.tracker.log_model(self.model, "retail_risk_agent")
        
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
        if total > 0:
            importance["pct"] = (importance["importance"] / total * 100).round(1)
        else:
            importance["pct"] = 0.0
        
        return importance


# Keep RedditAgent for backwards compatibility (deprecated)
RedditAgent = RetailRiskAgent


def main():
    """Test alpha agents."""
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
    
    # Test RetailRiskAgent
    print("\n" + "-" * 65)
    print("📊 RETAIL RISK AGENT")
    print("-" * 65)
    
    try:
        retail_agent = RetailRiskAgent()
        retail_metrics = retail_agent.train()
        
        print(f"\n   Train R²: {retail_metrics['train']['R2']:.4f}")
        print(f"   Test R²:  {retail_metrics['test']['R2']:.4f}")
        
        print("\n   Top 3 Features:")
        print(retail_agent.get_feature_importance().head(3).to_string(index=False))
    except FileNotFoundError as e:
        print(f"   ⚠️ {e}")
        print("   Run: python -m src.pipeline.ingest_retail first")
    
    print("\n" + "=" * 65)
    print("✅ ALPHA AGENTS COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
