"""
TechnicalAgent: HAR-RV Volatility Prediction Model
Phase 2: Technical Baseline (Anchor Model)

This agent implements the Heterogeneous Autoregressive Realized Volatility (HAR-RV)
model which captures volatility persistence across multiple time horizons.

HAR-RV is known for:
- Strong baseline performance for volatility forecasting
- Linear structure that captures volatility clustering
- Robust to overfitting (unlike GBMs on trending data)

Usage:
    python -m src.agents.technical_agent
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.tracker import MLTracker


class TechnicalAgent:
    """
    HAR-RV (Heterogeneous Autoregressive) Technical Agent.
    
    This model predicts volatility using lagged realized volatility
    at multiple horizons (daily, weekly, monthly) plus exogenous features.
    
    Features:
        - rv_lag_1: Previous day's realized volatility
        - rv_lag_5: Weekly average RV (5-day rolling mean)
        - rv_lag_22: Monthly average RV (22-day rolling mean)
        - returns_sq_lag_1: Squared returns (leverage effect)
        - VIX_close: Implied volatility (market fear gauge)
        - rsi_14: RSI momentum indicator
    
    Target:
        - target_log_var: log(next_day_realized_variance)
    
    Model:
        - Ridge Regression (L2 regularization)
        - Why Ridge: Volatility is autoregressive/linear; GBMs overfit trends
    """
    
    def __init__(self, experiment_name: str = "titan_v8_technical_agent"):
        """
        Initialize the TechnicalAgent with Ridge regression.
        
        Args:
            experiment_name: MLflow experiment name
        """
        self.model = Ridge(alpha=1.0)
        self.tracker = MLTracker(experiment_name)
        self.feature_cols = [
            'rv_lag_1', 'rv_lag_5', 'rv_lag_22',
            'returns_sq_lag_1', 'VIX_close', 'rsi_14'
        ]
        self.target_col = 'target_log_var'
        self.df = None
        self.train_metrics = None
        self.test_metrics = None
        
    def load_and_process_data(self) -> pd.DataFrame:
        """
        Load price data and engineer HAR-RV features.
        
        Returns:
            DataFrame with HAR features ready for training
        """
        print("\n📂 Loading and processing data...")
        
        # Load targets
        targets_path = Path("data/processed/targets.parquet")
        df = pd.read_parquet(targets_path)
        print(f"   ✓ Loaded {len(df):,} rows")
        
        # Ensure proper types
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        if df["ticker"].dtype.name == "category":
            df["ticker"] = df["ticker"].astype(str)
        
        # Sort by ticker and date for proper rolling calculations
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        print("\n   🔧 Engineering HAR-RV features...")
        
        # Process each ticker separately to avoid cross-ticker contamination
        processed_dfs = []
        
        for ticker in df["ticker"].unique():
            ticker_df = df[df["ticker"] == ticker].copy()
            ticker_df = ticker_df.sort_values("date").reset_index(drop=True)
            
            # =============================================
            # HAR-RV Core Features
            # =============================================
            
            # rv_lag_1: Previous day's realized volatility
            ticker_df["rv_lag_1"] = ticker_df["realized_vol"].shift(1)
            
            # rv_lag_5: Weekly average RV (5-day rolling mean, lagged)
            ticker_df["rv_lag_5"] = ticker_df["realized_vol"].rolling(
                window=5, min_periods=1
            ).mean().shift(1)
            
            # rv_lag_22: Monthly average RV (22-day rolling mean, lagged)
            ticker_df["rv_lag_22"] = ticker_df["realized_vol"].rolling(
                window=22, min_periods=1
            ).mean().shift(1)
            
            # =============================================
            # Leverage Effect Feature
            # =============================================
            
            # Daily returns
            ticker_df["daily_return"] = ticker_df["close"] / ticker_df["close"].shift(1) - 1
            
            # Squared returns (captures leverage effect - larger drops = higher vol)
            ticker_df["returns_sq_lag_1"] = (ticker_df["daily_return"] ** 2).shift(1)
            
            # Overnight return (if available, otherwise use daily)
            if "overnight_return" not in ticker_df.columns:
                ticker_df["overnight_return"] = ticker_df["daily_return"].shift(1)
            
            processed_dfs.append(ticker_df)
        
        df = pd.concat(processed_dfs, ignore_index=True)
        
        # =============================================
        # Exogenous Features
        # =============================================
        
        # VIX (fill missing with forward fill, then 0)
        if "VIX_close" in df.columns:
            df["VIX_close"] = df["VIX_close"].ffill().fillna(0)
        else:
            print("   ⚠️ VIX_close not found, filling with 0")
            df["VIX_close"] = 0
        
        # RSI (fill missing with 50 - neutral)
        if "rsi_14" in df.columns:
            df["rsi_14"] = df["rsi_14"].ffill().fillna(50)
        else:
            print("   ⚠️ rsi_14 not found, filling with 50")
            df["rsi_14"] = 50
        
        # =============================================
        # Clean up
        # =============================================
        
        # Drop first 22 rows per ticker (due to rolling window warm-up)
        df = df.groupby("ticker").apply(
            lambda x: x.iloc[22:] if len(x) > 22 else x.iloc[0:0]
        ).reset_index(drop=True)
        
        # Drop rows with NaN in features or target
        required_cols = self.feature_cols + [self.target_col]
        before = len(df)
        df = df.dropna(subset=required_cols)
        after = len(df)
        
        print(f"   ✓ Dropped {before - after} rows with NaN")
        print(f"   ✓ Final dataset: {len(df):,} rows")
        print(f"   ✓ Tickers: {df['ticker'].unique().tolist()}")
        print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Store for later use
        self.df = df
        
        return df
    
    def train(self, df: pd.DataFrame = None) -> dict:
        """
        Train the Ridge regression model with time-series split.
        
        Args:
            df: DataFrame with features (uses self.df if None)
        
        Returns:
            Dictionary with train/test metrics
        """
        if df is None:
            df = self.df
        
        if df is None:
            raise ValueError("No data loaded! Call load_and_process_data first.")
        
        print("\n🎯 Training TechnicalAgent (HAR-RV)...")
        
        # Time-series split
        train_cutoff = pd.to_datetime("2023-01-01")
        
        train_mask = df["date"] < train_cutoff
        test_mask = df["date"] >= train_cutoff
        
        # Handle case where split doesn't work well
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
        print(f"      Cutoff: {train_cutoff}")
        
        # Start MLflow run
        with self.tracker.start_run(run_name="technical_agent_harv"):
            # Log parameters
            self.tracker.log_params({
                "model": "Ridge",
                "alpha": 1.0,
                "features": str(self.feature_cols),
                "n_features": len(self.feature_cols),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_cutoff": str(train_cutoff)
            })
            
            # Train model
            print("\n   🔧 Fitting Ridge Regression...")
            self.model.fit(X_train, y_train)
            
            # Print coefficients
            print("\n   📈 Model Coefficients:")
            for feat, coef in zip(self.feature_cols, self.model.coef_):
                print(f"      {feat:20s}: {coef:+.4f}")
            print(f"      {'Intercept':20s}: {self.model.intercept_:+.4f}")
            
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
            self.tracker.log_model(self.model, "technical_agent_ridge")
        
        return {
            "train": self.train_metrics,
            "test": self.test_metrics
        }
    
    def predict(self, df: pd.DataFrame = None) -> pd.Series:
        """
        Generate predictions for the dataset.
        
        Args:
            df: DataFrame with features (uses self.df if None)
        
        Returns:
            Series of predictions
        """
        if df is None:
            df = self.df
        
        X = df[self.feature_cols]
        predictions = self.model.predict(X)
        
        return pd.Series(predictions, index=df.index)
    
    def save_residuals(self, output_path: str = "data/processed/residuals.parquet"):
        """
        Save residuals for downstream agents (e.g., NewsAgent).
        
        The residuals represent what the technical model couldn't explain,
        which the news agent will try to predict.
        
        Args:
            output_path: Path to save residuals parquet
        """
        print("\n💾 Saving residuals...")
        
        if self.df is None:
            raise ValueError("No data! Call load_and_process_data first.")
        
        # Generate predictions for entire dataset
        predictions = self.predict()
        
        # Calculate residuals
        residuals = self.df[self.target_col] - predictions
        
        # Create residuals DataFrame
        residuals_df = pd.DataFrame({
            "date": self.df["date"],
            "ticker": self.df["ticker"],
            "target_log_var": self.df[self.target_col],
            "pred_tech": predictions,
            "resid_tech": residuals
        })
        
        # Save to parquet
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        residuals_df.to_parquet(output_path, index=False, engine="pyarrow")
        
        print(f"   ✓ Saved to: {output_path}")
        print(f"   ✓ Residual stats:")
        print(f"      Mean: {residuals.mean():.4f}")
        print(f"      Std:  {residuals.std():.4f}")
        
        return residuals_df


def main():
    """Run TechnicalAgent benchmark."""
    print("\n" + "=" * 60)
    print("🚀 TITAN V8 TECHNICAL AGENT (HAR-RV)")
    print("    Phase 2: Building the Anchor Model")
    print("=" * 60)
    
    # Initialize agent
    agent = TechnicalAgent(experiment_name="titan_v8_technical_agent")
    
    # Load and process data
    df = agent.load_and_process_data()
    
    # Train model
    metrics = agent.train(df)
    
    # Print results
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Train':>12} {'Test':>12}")
    print("-" * 50)
    print(f"{'RMSE':<25} {metrics['train']['RMSE']:>12.4f} {metrics['test']['RMSE']:>12.4f}")
    print(f"{'MAE':<25} {metrics['train']['MAE']:>12.4f} {metrics['test']['MAE']:>12.4f}")
    print(f"{'R²':<25} {metrics['train']['R2']:>12.4f} {metrics['test']['R2']:>12.4f}")
    print(f"{'Directional Accuracy':<25} {metrics['train']['Directional_Accuracy']:>11.1f}% {metrics['test']['Directional_Accuracy']:>11.1f}%")
    
    # Assess performance
    print("\n" + "=" * 60)
    print("🔬 PERFORMANCE ASSESSMENT")
    print("=" * 60)
    
    test_r2 = metrics['test']['R2']
    test_dir_acc = metrics['test']['Directional_Accuracy']
    
    if test_r2 > 0.4:
        verdict = "✅ EXCELLENT - HAR-RV baseline is strong!"
    elif test_r2 > 0.2:
        verdict = "✅ GOOD - Solid baseline performance."
    elif test_r2 > 0:
        verdict = "⚠️ WEAK - Baseline has predictive power but limited."
    else:
        verdict = "❌ POOR - Model not learning meaningful patterns."
    
    print(f"\nTest R²: {test_r2:.4f}")
    print(f"Expected: > 0.40 for good volatility model")
    print(f"\n{verdict}")
    
    # Save residuals for Phase 3
    residuals_df = agent.save_residuals()
    
    print("\n" + "=" * 60)
    print("📦 RESIDUALS SAVED FOR PHASE 3")
    print("=" * 60)
    print(f"File: data/processed/residuals.parquet")
    print(f"Rows: {len(residuals_df):,}")
    print(f"\nThe NewsAgent (Phase 3) will train on resid_tech")
    print("to capture what technical features cannot explain.")
    print("=" * 60)
    
    print("\n✅ TechnicalAgent training complete!")
    print("   Results logged to MLflow experiment: titan_v8_technical_agent")


if __name__ == "__main__":
    main()


