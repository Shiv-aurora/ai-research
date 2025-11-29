"""
TitanCoordinator: The Final Ensemble Fusion Layer

Phase 5: Combines all agent predictions into a unified volatility forecast.

Architecture:
- Inputs: tech_pred (baseline) + news_pred, fund_pred, retail_pred (correctors)
- Gating Context: VIX, RSI, shock_index to weight agents dynamically
- Model: XGBoost Regressor
- Validation: Purged Walk-Forward (Train < 2023, Test >= 2023)

The coordinator learns WHEN to trust each agent based on market regime.

Usage:
    from src.coordinator.fusion import TitanCoordinator
    coordinator = TitanCoordinator()
    coordinator.train(predictions_df)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.tracker import MLTracker


class TitanCoordinator:
    """
    Ensemble Coordinator that fuses all agent predictions.
    
    Inputs:
    - tech_pred: TechnicalAgent prediction (HAR-RV baseline)
    - news_pred: NewsAgent residual prediction
    - fund_pred: FundamentalAgent residual prediction
    - retail_pred: RetailRiskAgent prediction
    
    Gating Context:
    - VIX_close: Market fear level
    - rsi_14: Momentum indicator
    - shock_index_lag5: News shock persistence
    
    The coordinator learns optimal weights for each agent
    conditioned on the current market regime.
    """
    
    def __init__(self, experiment_name: str = "titan_v8_coordinator"):
        """Initialize TitanCoordinator with XGBoost."""
        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0
        )
        
        self.tracker = MLTracker(experiment_name)
        self.feature_cols = None
        self.target_col = "target_log_var"
        self.train_metrics = None
        self.test_metrics = None
        self.baseline_metrics = None
        
    def get_feature_columns(self) -> list:
        """Define coordinator input features."""
        return [
            # Agent predictions (the ensemble members)
            "tech_pred",
            "news_pred",
            "fund_pred",
            "retail_pred",
            
            # Gating context (regime indicators)
            "VIX_close",
            "rsi_14",
            "shock_index_lag5",
            
            # Additional context
            "rv_lag_1",
            "rv_lag_5"
        ]
    
    def prepare_predictions_dataset(
        self,
        tech_agent,
        news_agent,
        fund_agent,
        retail_agent,
        targets_df: pd.DataFrame,
        news_features_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Prepare a unified dataset with all agent predictions.
        
        Strategy: Each agent generates predictions on its OWN data,
        then we merge all predictions by date/ticker.
        
        Args:
            tech_agent: Trained TechnicalAgent
            news_agent: Trained NewsAgent
            fund_agent: Trained FundamentalAgent
            retail_agent: Trained RetailRiskAgent
            targets_df: DataFrame with targets and features
            news_features_df: DataFrame with news features (for shock_index)
            
        Returns:
            DataFrame with all predictions and gating features
        """
        print("\n📊 Preparing unified predictions dataset...")
        
        df = targets_df.copy()
        
        # Normalize date
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        if df["ticker"].dtype.name == "category":
            df["ticker"] = df["ticker"].astype(str)
        
        # Sort
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        # Add tech_pred (from TechnicalAgent)
        # Use agent's internal dataframe which has all HAR features
        if hasattr(tech_agent, 'model') and tech_agent.model is not None:
            if hasattr(tech_agent, 'df') and tech_agent.df is not None:
                tech_df = tech_agent.df.copy()
                tech_df["date"] = pd.to_datetime(tech_df["date"]).dt.tz_localize(None)
                
                # Generate predictions on agent's data
                tech_preds = tech_agent.model.predict(
                    tech_df[tech_agent.feature_cols].fillna(0)
                )
                tech_df["tech_pred"] = tech_preds
                
                # Merge predictions back
                pred_df = tech_df[["date", "ticker", "tech_pred"]].copy()
                df = pd.merge(df, pred_df, on=["date", "ticker"], how="left")
                df["tech_pred"] = df["tech_pred"].fillna(df["tech_pred"].mean())
                print(f"   ✓ Added tech_pred (from agent's data)")
            else:
                df["tech_pred"] = 0
                print(f"   ⚠️ TechnicalAgent has no internal data")
        else:
            df["tech_pred"] = 0
            print(f"   ⚠️ TechnicalAgent not trained, using 0")
        
        # Add news_pred (from NewsAgent)
        # Use agent's internal dataframe if available (has all features)
        if hasattr(news_agent, 'model') and news_agent.model is not None:
            if hasattr(news_agent, 'df') and news_agent.df is not None:
                # Use agent's own data which has all features
                news_df = news_agent.df.copy()
                news_df["date"] = pd.to_datetime(news_df["date"]).dt.tz_localize(None)
                
                # Generate predictions on agent's data
                news_preds = news_agent.model.predict(
                    news_df[news_agent.feature_cols].fillna(0)
                )
                news_df["news_pred"] = news_preds
                
                # Merge predictions back
                pred_df = news_df[["date", "ticker", "news_pred"]].copy()
                df = pd.merge(df, pred_df, on=["date", "ticker"], how="left")
                df["news_pred"] = df["news_pred"].fillna(0)
                print(f"   ✓ Added news_pred (from agent's data)")
            else:
                df["news_pred"] = 0
                print(f"   ⚠️ NewsAgent has no internal data")
        else:
            df["news_pred"] = 0
            print(f"   ⚠️ NewsAgent not trained, using 0")
        
        # Add fund_pred (from FundamentalAgent)
        # FundamentalAgent loads its own data, so we need to generate predictions on that
        if hasattr(fund_agent, 'model') and fund_agent.model is not None:
            # Load FundamentalAgent's data (it creates its own merged dataset)
            try:
                fund_df = fund_agent.load_and_process_data()
                fund_df["date"] = pd.to_datetime(fund_df["date"]).dt.tz_localize(None)
                
                fund_preds = fund_agent.model.predict(
                    fund_df[fund_agent.feature_cols].fillna(0)
                )
                fund_df["fund_pred"] = fund_preds
                
                pred_df = fund_df[["date", "ticker", "fund_pred"]].copy()
                df = pd.merge(df, pred_df, on=["date", "ticker"], how="left")
                df["fund_pred"] = df["fund_pred"].fillna(0)
                print(f"   ✓ Added fund_pred (from agent's data)")
            except Exception as e:
                df["fund_pred"] = 0
                print(f"   ⚠️ FundamentalAgent error: {e}")
        else:
            df["fund_pred"] = 0
            print(f"   ⚠️ FundamentalAgent not trained, using 0")
        
        # Add retail_pred (from RetailRiskAgent)
        if hasattr(retail_agent, 'model') and retail_agent.model is not None:
            try:
                retail_df = retail_agent.load_and_process_data()
                retail_df["date"] = pd.to_datetime(retail_df["date"]).dt.tz_localize(None)
                
                retail_preds = retail_agent.model.predict(
                    retail_df[retail_agent.feature_cols].fillna(0)
                )
                retail_df["retail_pred"] = retail_preds
                
                pred_df = retail_df[["date", "ticker", "retail_pred"]].copy()
                df = pd.merge(df, pred_df, on=["date", "ticker"], how="left")
                df["retail_pred"] = df["retail_pred"].fillna(0)
                print(f"   ✓ Added retail_pred (from agent's data)")
            except Exception as e:
                df["retail_pred"] = 0
                print(f"   ⚠️ RetailRiskAgent error: {e}")
        else:
            df["retail_pred"] = 0
            print(f"   ⚠️ RetailRiskAgent not trained, using 0")
        
        # Add gating context
        df["VIX_close"] = df["VIX_close"].ffill().fillna(15)
        df["rsi_14"] = df["rsi_14"].ffill().fillna(50)
        
        # shock_index_lag5 from news features
        if "shock_index" in df.columns:
            df["shock_index_lag5"] = df.groupby("ticker")["shock_index"].shift(5)
        else:
            df["shock_index_lag5"] = 0
        
        df["shock_index_lag5"] = df["shock_index_lag5"].fillna(0)
        
        # Drop NaN in target
        df = df.dropna(subset=[self.target_col])
        
        print(f"\n   📊 Final dataset: {len(df):,} rows")
        
        return df
    
    def train(self, df: pd.DataFrame) -> dict:
        """
        Train the coordinator on combined predictions.
        
        Uses Purged Walk-Forward validation:
        - Train: data < 2023-01-01
        - Test: data >= 2023-01-01
        """
        print("\n" + "=" * 70)
        print("🎯 TRAINING TITAN COORDINATOR")
        print("=" * 70)
        
        self.feature_cols = self.get_feature_columns()
        self.feature_cols = [f for f in self.feature_cols if f in df.columns]
        
        print(f"\n   Input Features:")
        for f in self.feature_cols:
            print(f"      - {f}")
        print(f"   Target: {self.target_col}")
        
        # Purged Walk-Forward Split
        cutoff = pd.to_datetime("2023-01-01")
        train = df[df["date"] < cutoff].copy()
        test = df[df["date"] >= cutoff].copy()
        
        print(f"\n   📊 Purged Walk-Forward Split:")
        print(f"      Train: {len(train):,} samples (< 2023-01-01)")
        print(f"      Test:  {len(test):,} samples (>= 2023-01-01)")
        
        X_train = train[self.feature_cols].fillna(0)
        y_train = train[self.target_col]
        X_test = test[self.feature_cols].fillna(0)
        y_test = test[self.target_col]
        
        # Calculate baseline (tech_pred only)
        if "tech_pred" in df.columns:
            baseline_r2 = r2_score(y_test, test["tech_pred"])
            baseline_rmse = np.sqrt(mean_squared_error(y_test, test["tech_pred"]))
            self.baseline_metrics = {"R2": baseline_r2, "RMSE": baseline_rmse}
            print(f"\n   📈 Baseline (HAR-RV only):")
            print(f"      Test R²:  {baseline_r2:.4f} ({baseline_r2*100:.2f}%)")
            print(f"      Test RMSE: {baseline_rmse:.4f}")
        
        # Train coordinator
        print("\n   🔧 Training XGBoost Coordinator...")
        
        with self.tracker.start_run(run_name="titan_coordinator"):
            self.tracker.log_params({
                "model": "XGBRegressor",
                "n_estimators": 200,
                "max_depth": 3,
                "n_features": len(self.feature_cols),
                "n_train": len(train),
                "n_test": len(test)
            })
            
            self.model.fit(X_train, y_train)
            
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            self.train_metrics = self.tracker.log_metrics(
                y_train.values, y_train_pred, step=0
            )
            self.test_metrics = self.tracker.log_metrics(
                y_test.values, y_test_pred, step=1
            )
            
            self.tracker.log_model(self.model, "titan_coordinator")
        
        # Print results
        print(f"\n   📈 Coordinator Results:")
        print(f"   {'Metric':<25} {'Train':>10} {'Test':>10}")
        print("   " + "-" * 47)
        print(f"   {'RMSE':<25} {self.train_metrics['RMSE']:>10.4f} {self.test_metrics['RMSE']:>10.4f}")
        print(f"   {'MAE':<25} {self.train_metrics['MAE']:>10.4f} {self.test_metrics['MAE']:>10.4f}")
        print(f"   {'R²':<25} {self.train_metrics['R2']:>10.4f} {self.test_metrics['R2']:>10.4f}")
        print(f"   {'Directional Accuracy':<25} {self.train_metrics['Directional_Accuracy']:>9.1f}% {self.test_metrics['Directional_Accuracy']:>9.1f}%")
        
        return {
            "train": self.train_metrics,
            "test": self.test_metrics,
            "baseline": self.baseline_metrics
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importances from the coordinator."""
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
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions from the coordinator."""
        X = df[self.feature_cols].fillna(0)
        return self.model.predict(X)
    
    def print_comparison_table(self):
        """Print the final comparison table."""
        print("\n" + "=" * 70)
        print("💰 THE MONEY SHOT: FINAL COMPARISON")
        print("=" * 70)
        
        if self.baseline_metrics is None or self.test_metrics is None:
            print("   ⚠️ Train the coordinator first!")
            return
        
        baseline_r2 = self.baseline_metrics["R2"]
        titan_r2 = self.test_metrics["R2"]
        baseline_rmse = self.baseline_metrics["RMSE"]
        titan_rmse = self.test_metrics["RMSE"]
        
        improvement_r2 = ((titan_r2 - baseline_r2) / max(abs(baseline_r2), 0.001)) * 100
        improvement_rmse = ((baseline_rmse - titan_rmse) / baseline_rmse) * 100
        
        print(f"""
   ┌─────────────────────────┬────────────────┬────────────────┐
   │ Model                   │ Test R²        │ Test RMSE      │
   ├─────────────────────────┼────────────────┼────────────────┤
   │ Baseline (HAR-RV)       │ {baseline_r2:>12.4f}   │ {baseline_rmse:>12.4f}   │
   │ Titan V8 (Ensemble)     │ {titan_r2:>12.4f}   │ {titan_rmse:>12.4f}   │
   ├─────────────────────────┼────────────────┼────────────────┤
   │ Improvement             │ {improvement_r2:>+11.1f}%   │ {improvement_rmse:>+11.1f}%   │
   └─────────────────────────┴────────────────┴────────────────┘
        """)
        
        if titan_r2 > baseline_r2:
            print("   ✅ Titan V8 OUTPERFORMS the baseline!")
        else:
            print("   ⚠️ Baseline is still better - ensemble needs tuning")
        
        print("=" * 70)


def main():
    """Test the TitanCoordinator."""
    print("\n" + "=" * 70)
    print("🚀 TITAN COORDINATOR TEST")
    print("=" * 70)
    
    # This is a simple test - full usage is in run_full_pipeline.py
    coordinator = TitanCoordinator()
    print(f"   Features: {coordinator.get_feature_columns()}")
    print("   ✅ Coordinator initialized successfully")
    print("\n   Run scripts/run_full_pipeline.py for full training")


if __name__ == "__main__":
    main()

