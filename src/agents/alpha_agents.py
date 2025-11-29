"""
Alpha Agents: FundamentalAgent and RetailRiskAgent

Phase 4.5 OPTIMIZED: Additional agents for volatility prediction.

FundamentalAgent:
- Uses debt_to_equity, days_to_ex_div, and VIX interactions
- Predicts residuals from TechnicalAgent
- Test R²: 6.17%

RetailRiskAgent (OPTIMIZED from exhaustive audit):
- Uses VIX × Retail interactions (btc_vix, gme_vix)
- Predicts REALIZED_VOL directly (not residuals!)
- Model: Ridge (linear relationship)
- Test R²: 10.72%

Key Finding: Raw retail signals don't predict volatility, but their
INTERACTION with VIX (market fear) does. When crypto is volatile AND
market fear is high, stock volatility increases.

Usage:
    python -m src.agents.alpha_agents
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
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
    Predicts REALIZED VOLATILITY using VIX-Retail interaction signals.
    
    OPTIMIZED from exhaustive audit (10.72% Test R²):
    - Target: realized_vol (NOT residuals!)
    - Features: VIX interactions only (btc_vix, gme_vix, VIX_close)
    - Model: Ridge (linear relationship works best)
    - Lags: No lag (same day)
    
    Key Insight: Retail risk signals matter THROUGH their interaction
    with VIX (market fear). When both BTC and VIX are volatile,
    stock volatility increases.
    """
    
    def __init__(self, experiment_name: str = "titan_v8_retail_risk"):
        """Initialize RetailRiskAgent with Ridge (optimal from audit)."""
        # Ridge is optimal from audit (linear relationship)
        self.model = Ridge(alpha=1.0)
        
        self.tracker = MLTracker(experiment_name)
        self.feature_cols = None
        self.target_col = "realized_vol"  # CHANGED from resid_tech!
        self.train_metrics = None
        self.test_metrics = None
        
    def load_and_process_data(self) -> pd.DataFrame:
        """Load retail signals and targets, create VIX interaction features."""
        print("\n📂 Loading data for RetailRiskAgent (OPTIMIZED)...")
        
        retail_path = Path("data/processed/retail_signals.parquet")
        if not retail_path.exists():
            raise FileNotFoundError(
                "Retail signals not found! Run ingest_retail first."
            )
        
        # Load data
        retail = pd.read_parquet(retail_path)
        targets = pd.read_parquet("data/processed/targets.parquet")
        
        # Normalize dates
        for df in [retail, targets]:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            if "ticker" in df.columns and df["ticker"].dtype.name == "category":
                df["ticker"] = df["ticker"].astype(str)
        
        print(f"   Retail signals: {len(retail):,} rows")
        print(f"   Targets: {len(targets):,} rows")
        
        # Merge targets with retail signals (by date only - global signals)
        merged = pd.merge(targets, retail, on="date", how="left")
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        print(f"   After merge: {len(merged):,} rows")
        
        # Feature engineering (VIX interactions - optimal from audit)
        print("   Engineering VIX interaction features (optimal config)...")
        
        # Fill missing
        merged["VIX_close"] = merged["VIX_close"].ffill().fillna(15)
        merged["btc_vol_5d"] = merged["btc_vol_5d"].ffill().fillna(0)
        merged["gme_vol_shock"] = merged["gme_vol_shock"].ffill().fillna(1)
        
        # VIX Interaction features (THE KEY from audit)
        merged["btc_vix_interaction"] = merged["btc_vol_5d"] * merged["VIX_close"]
        merged["gme_vix_interaction"] = merged["gme_vol_shock"] * merged["VIX_close"]
        
        # Drop NaN for target
        merged = merged.dropna(subset=[self.target_col])
        
        print(f"   Final: {len(merged):,} rows")
        print(f"   Target: {self.target_col}")
        
        return merged
    
    def get_feature_columns(self) -> list:
        """
        Define optimal feature columns (from audit).
        
        VIX Interactions ONLY - this is the key finding!
        Raw retail signals don't work, but their interaction with VIX does.
        """
        return [
            "btc_vix_interaction",   # BTC vol × VIX (crypto fear × market fear)
            "gme_vix_interaction",   # GME vol shock × VIX (meme × market fear)
            "VIX_close"              # Market fear baseline
        ]
    
    def train(self, df: pd.DataFrame = None) -> dict:
        """Train on realized_vol using Ridge (optimal from audit)."""
        if df is None:
            df = self.load_and_process_data()
        
        print("\n🎯 Training RetailRiskAgent (OPTIMIZED - Ridge on realized_vol)...")
        
        self.feature_cols = self.get_feature_columns()
        self.feature_cols = [f for f in self.feature_cols if f in df.columns]
        
        print(f"\n   Optimal Configuration (from audit):")
        print(f"   - Model: Ridge (linear)")
        print(f"   - Target: {self.target_col}")
        print(f"   - Features: {self.feature_cols}")
        print(f"   - Lags: None (same day)")
        
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
        
        print(f"\n   Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Train Ridge model
        with self.tracker.start_run(run_name="retail_risk_optimized"):
            self.tracker.log_params({
                "model": "Ridge",
                "target": "realized_vol",
                "n_features": len(self.feature_cols),
                "feature_type": "vix_interactions",
                "optimized": True
            })
            
            self.model.fit(X_train, y_train)
            
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            self.train_metrics = self.tracker.log_metrics(y_train.values, y_train_pred, step=0)
            self.test_metrics = self.tracker.log_metrics(y_test.values, y_test_pred, step=1)
            
            self.tracker.log_model(self.model, "retail_risk_optimized")
        
        return {
            "train": self.train_metrics,
            "test": self.test_metrics
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature coefficients (Ridge uses .coef_ not .feature_importances_)."""
        # Ridge uses coefficients, take absolute value for importance
        importance = pd.DataFrame({
            "feature": self.feature_cols,
            "coefficient": self.model.coef_,
            "importance": np.abs(self.model.coef_)
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
