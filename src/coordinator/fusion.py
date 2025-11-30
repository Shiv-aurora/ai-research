"""
TitanCoordinator: Ridge Regression with Robustness Upgrades

Phase 15: The Robustness Upgrade

Key Optimizations (from optimization audit):
- Model: Ridge(alpha=100.0) - stronger regularization
- Winsorization: Clip y_train at 2nd/98th percentiles
- Momentum: vol_ma5, vol_ma10, vol_std5
- Calendar: is_friday, is_monday, is_q4

Architecture:
- Inputs: tech_pred + news_risk + retail_risk + momentum + calendar
- Model: Ridge with strong regularization (alpha=100)
- Winsorization: Applied to training target only
- Validation: Purged Walk-Forward (Train < 2023, Test >= 2023)

Target: 30%+ R² (with winsorization)

Usage:
    from src.coordinator.fusion import TitanCoordinator
    coordinator = TitanCoordinator()
    coordinator.train(predictions_df)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.tracker import MLTracker


class TitanCoordinator:
    """
    Ridge Coordinator with Robustness Upgrades (Phase 15).
    
    Key Optimizations:
    - Ridge(alpha=100) for stronger regularization
    - Winsorization at 2%/98% percentiles
    - Enhanced momentum: vol_ma5, vol_ma10, vol_std5
    - Full calendar: is_friday, is_monday, is_q4
    
    Full Feature List:
    - tech_pred: HAR-RV prediction
    - news_risk_score: Probability of extreme event (0-1)
    - retail_risk_score: Retail hype risk (0-1)
    - is_friday, is_monday, is_q4: Calendar effects
    - vol_ma5, vol_ma10: Rolling mean volatility
    - vol_std5: Rolling std volatility
    - news_x_retail: Interaction term
    """
    
    def __init__(self, experiment_name: str = "titan_v8_coordinator_phase15",
                 alpha: float = 100.0,
                 winsorize_pct: float = 0.02):
        """
        Initialize TitanCoordinator with Phase 15 settings.
        
        Args:
            experiment_name: MLflow experiment name
            alpha: Ridge regularization strength (default 100.0)
            winsorize_pct: Percentile for winsorization (default 0.02 = 2%)
        """
        self.model = Ridge(
            alpha=alpha,
            fit_intercept=True
        )
        
        self.alpha = alpha
        self.winsorize_pct = winsorize_pct
        self.winsorize_lower = None
        self.winsorize_upper = None
        
        self.tracker = MLTracker(experiment_name)
        self.feature_cols = None
        self.target_col = "target_log_var"
        self.train_metrics = None
        self.test_metrics = None
        self.baseline_metrics = None
        self.sector_metrics = None
        self.df = None
        
    def get_feature_columns(self) -> list:
        """
        Define coordinator input features (Phase 15 optimized).
        
        Features validated by optimization audit:
        - Agent predictions
        - Calendar effects
        - Enhanced momentum (vol_ma5, vol_ma10, vol_std5)
        - Interaction terms
        """
        return [
            # Agent predictions
            "tech_pred",           # HAR-RV baseline
            "news_risk_score",     # Semantic risk
            "retail_risk_score",   # Behavioral risk
            
            # Calendar features
            "is_friday",           # Pre-weekend effect
            "is_monday",           # Start-of-week
            "is_q4",               # Year-end seasonality
            
            # Enhanced momentum (KEY from audit)
            "vol_ma5",             # 5-day rolling mean
            "vol_ma10",            # 10-day rolling mean
            "vol_std5",            # 5-day rolling std
            
            # Interaction
            "news_x_retail",       # News × Retail interaction
        ]
    
    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar seasonality features."""
        df = df.copy()
        
        # Ensure date is datetime
        df["date"] = pd.to_datetime(df["date"])
        
        # Day of week effects
        dow = df["date"].dt.dayofweek
        df["is_monday"] = (dow == 0).astype(int)
        df["is_friday"] = (dow == 4).astype(int)
        
        # Quarter effect (Q4 seasonality)
        df["is_q4"] = (df["date"].dt.quarter == 4).astype(int)
        
        return df
    
    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add enhanced momentum features.
        
        Phase 15 additions:
        - vol_ma5: 5-day rolling mean
        - vol_ma10: 10-day rolling mean (NEW)
        - vol_std5: 5-day rolling std (NEW)
        """
        df = df.copy()
        
        print("   📈 Adding enhanced momentum features...")
        
        if "target_log_var" not in df.columns:
            print("      ⚠️ target_log_var not found, skipping momentum")
            return df
        
        for ticker in df["ticker"].unique():
            mask = df["ticker"] == ticker
            ticker_data = df.loc[mask, "target_log_var"]
            
            # 5-day rolling mean (shifted to avoid leakage)
            df.loc[mask, "vol_ma5"] = ticker_data.rolling(5, min_periods=1).mean().shift(1)
            
            # 10-day rolling mean (NEW)
            df.loc[mask, "vol_ma10"] = ticker_data.rolling(10, min_periods=1).mean().shift(1)
            
            # 5-day rolling std (NEW)
            df.loc[mask, "vol_std5"] = ticker_data.rolling(5, min_periods=2).std().shift(1)
        
        # Fill NaN with defaults
        df["vol_ma5"] = df["vol_ma5"].fillna(df["target_log_var"].mean())
        df["vol_ma10"] = df["vol_ma10"].fillna(df["target_log_var"].mean())
        df["vol_std5"] = df["vol_std5"].fillna(0)
        
        print(f"      ✓ vol_ma5: 5-day rolling mean")
        print(f"      ✓ vol_ma10: 10-day rolling mean")
        print(f"      ✓ vol_std5: 5-day rolling std")
        
        return df
    
    def prepare_predictions_dataset(
        self,
        tech_agent=None,
        news_agent=None,
        retail_agent=None,
        targets_df: pd.DataFrame = None,
        news_features_df: pd.DataFrame = None,
        residuals_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Prepare a unified dataset with all features.
        
        Phase 15: Enhanced with momentum and calendar features.
        """
        print("\n📊 Preparing unified predictions dataset (Phase 15)...")
        
        # Start with targets
        if targets_df is not None:
            df = targets_df.copy()
        else:
            df = pd.read_parquet("data/processed/targets_deseasonalized.parquet")
        
        # Normalize date
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        if df["ticker"].dtype.name == "category":
            df["ticker"] = df["ticker"].astype(str)
        
        # Sort
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        # Add Calendar Features
        print("   🗓️ Adding calendar features...")
        df = self.add_calendar_features(df)
        print(f"      ✓ is_friday: {df['is_friday'].sum()} days")
        print(f"      ✓ is_monday: {df['is_monday'].sum()} days")
        print(f"      ✓ is_q4: {df['is_q4'].sum()} days")
        
        # Add Enhanced Momentum Features
        df = self.add_momentum_features(df)
        
        # Add tech_pred from residuals
        if residuals_df is not None:
            res_df = residuals_df.copy()
            res_df["date"] = pd.to_datetime(res_df["date"]).dt.tz_localize(None)
            if res_df["ticker"].dtype.name == "category":
                res_df["ticker"] = res_df["ticker"].astype(str)
            
            if "pred_tech_excess" in res_df.columns:
                df = pd.merge(
                    df, 
                    res_df[["date", "ticker", "pred_tech_excess"]], 
                    on=["date", "ticker"], 
                    how="left"
                )
                if "seasonal_component" in df.columns:
                    df["tech_pred"] = df["pred_tech_excess"] + df["seasonal_component"]
                else:
                    df["tech_pred"] = df["pred_tech_excess"]
                print(f"   ✓ Added tech_pred from residuals")
            else:
                df["tech_pred"] = df["target_log_var"].mean()
                print(f"   ⚠️ No pred_tech_excess, using mean")
        else:
            df["tech_pred"] = df["target_log_var"].mean() if "target_log_var" in df.columns else 0
            print(f"   ⚠️ No residuals provided, using mean tech_pred")
        
        # Add news_risk_score
        if news_agent is not None and hasattr(news_agent, 'df') and news_agent.df is not None:
            news_df = news_agent.df.copy()
            news_df["date"] = pd.to_datetime(news_df["date"]).dt.tz_localize(None)
            
            if hasattr(news_agent, 'predict'):
                risk_scores = news_agent.predict(news_df)
                news_df["news_risk_score"] = risk_scores
                
                df = pd.merge(
                    df,
                    news_df[["date", "ticker", "news_risk_score"]],
                    on=["date", "ticker"],
                    how="left"
                )
                df["news_risk_score"] = df["news_risk_score"].fillna(0.2)
                print(f"   ✓ Added news_risk_score from classifier")
            else:
                df["news_risk_score"] = 0.2
        else:
            df["news_risk_score"] = 0.2
            print(f"   ⚠️ NewsAgent not provided, using default 0.2")
        
        # Add retail_risk_score
        if retail_agent is not None and hasattr(retail_agent, 'df') and retail_agent.df is not None:
            retail_df = retail_agent.df.copy()
            retail_df["date"] = pd.to_datetime(retail_df["date"]).dt.tz_localize(None)
            
            if hasattr(retail_agent, 'predict_proba'):
                retail_risk = retail_agent.predict_proba(retail_df)
                retail_df["retail_risk_score"] = retail_risk
                
                df = pd.merge(
                    df,
                    retail_df[["date", "ticker", "retail_risk_score"]],
                    on=["date", "ticker"],
                    how="left"
                )
                df["retail_risk_score"] = df["retail_risk_score"].fillna(0.2)
                print(f"   ✓ Added retail_risk_score from classifier")
            else:
                df["retail_risk_score"] = 0.2
        else:
            # Try loading from saved predictions
            retail_path = Path("data/processed/retail_predictions.parquet")
            if retail_path.exists():
                retail_preds = pd.read_parquet(retail_path)
                retail_preds["date"] = pd.to_datetime(retail_preds["date"]).dt.tz_localize(None)
                if retail_preds["ticker"].dtype.name == "category":
                    retail_preds["ticker"] = retail_preds["ticker"].astype(str)
                
                df = pd.merge(
                    df,
                    retail_preds[["date", "ticker", "retail_risk_score"]],
                    on=["date", "ticker"],
                    how="left"
                )
                df["retail_risk_score"] = df["retail_risk_score"].fillna(0.2)
                print(f"   ✓ Loaded retail_risk_score from saved predictions")
            else:
                df["retail_risk_score"] = 0.2
                print(f"   ⚠️ RetailAgent not provided, using default 0.2")
        
        # Create interaction feature
        df["news_x_retail"] = df["news_risk_score"] * df["retail_risk_score"]
        print(f"   ✓ Created news_x_retail interaction")
        
        # Add sector
        SECTOR_MAP = {
            'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech',
            'JPM': 'Finance', 'BAC': 'Finance', 'V': 'Finance',
            'CAT': 'Industrial', 'GE': 'Industrial', 'BA': 'Industrial',
            'WMT': 'Consumer', 'MCD': 'Consumer', 'COST': 'Consumer',
            'XOM': 'Energy', 'CVX': 'Energy', 'SLB': 'Energy',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare'
        }
        df["sector"] = df["ticker"].map(SECTOR_MAP)
        
        # Drop NaN in target
        df = df.dropna(subset=[self.target_col])
        
        self.df = df
        
        print(f"\n   📊 Final dataset: {len(df):,} rows")
        print(f"   Features: {[f for f in self.get_feature_columns() if f in df.columns]}")
        
        return df
    
    def winsorize_target(self, y: pd.Series, fit: bool = True) -> pd.Series:
        """
        Apply winsorization to target variable.
        
        Phase 15 key optimization: Clip outliers at 2nd/98th percentiles.
        This dramatically improves R² by reducing noise from extreme days.
        
        Args:
            y: Target series
            fit: If True, compute percentiles. If False, use stored values.
            
        Returns:
            Winsorized target series
        """
        if fit:
            self.winsorize_lower = y.quantile(self.winsorize_pct)
            self.winsorize_upper = y.quantile(1 - self.winsorize_pct)
        
        y_winsorized = y.clip(lower=self.winsorize_lower, upper=self.winsorize_upper)
        
        return y_winsorized
    
    def train(self, df: pd.DataFrame) -> dict:
        """
        Train the Ridge coordinator with Phase 15 optimizations.
        
        Key improvements:
        - Winsorization of y_train at 2%/98% percentiles
        - Ridge(alpha=100) for stronger regularization
        - Enhanced momentum features
        """
        print("\n" + "=" * 70)
        print(f"🎯 TRAINING TITAN COORDINATOR (Phase 15: Robustness Upgrade)")
        print(f"   Ridge(α={self.alpha}) + Winsorization({self.winsorize_pct:.0%})")
        print("=" * 70)
        
        self.feature_cols = [f for f in self.get_feature_columns() if f in df.columns]
        
        print(f"\n   Input Features ({len(self.feature_cols)}):")
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
        
        # =====================================================
        # WINSORIZATION (Phase 15 Key Optimization)
        # =====================================================
        print(f"\n   🔧 Applying Winsorization to y_train...")
        y_train_original = y_train.copy()
        y_train_winsorized = self.winsorize_target(y_train, fit=True)
        
        n_clipped = (y_train != y_train_winsorized).sum()
        print(f"      Lower bound: {self.winsorize_lower:.4f}")
        print(f"      Upper bound: {self.winsorize_upper:.4f}")
        print(f"      Clipped: {n_clipped:,} samples ({n_clipped/len(y_train)*100:.1f}%)")
        
        # Calculate baseline (tech_pred only)
        if "tech_pred" in df.columns:
            baseline_r2 = r2_score(y_test, test["tech_pred"])
            baseline_rmse = np.sqrt(mean_squared_error(y_test, test["tech_pred"]))
            self.baseline_metrics = {"R2": baseline_r2, "RMSE": baseline_rmse}
            print(f"\n   📈 Baseline (HAR only):")
            print(f"      Test R²:  {baseline_r2:.4f} ({baseline_r2*100:.2f}%)")
            print(f"      Test RMSE: {baseline_rmse:.4f}")
        
        # Train Ridge on WINSORIZED target
        print(f"\n   🔧 Training Ridge(α={self.alpha}) on winsorized target...")
        
        with self.tracker.start_run(run_name="titan_coordinator_phase15"):
            self.tracker.log_params({
                "model": "Ridge",
                "alpha": self.alpha,
                "winsorize_pct": self.winsorize_pct,
                "n_features": len(self.feature_cols),
                "features": self.feature_cols,
                "n_train": len(train),
                "n_test": len(test),
                "phase": 15
            })
            
            # Fit on winsorized training data
            self.model.fit(X_train, y_train_winsorized)
            
            # Predict
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # Metrics on ORIGINAL test data (not winsorized)
            self.train_metrics = {
                "R2": r2_score(y_train_winsorized, y_train_pred),
                "RMSE": np.sqrt(mean_squared_error(y_train_winsorized, y_train_pred)),
                "MAE": mean_absolute_error(y_train_winsorized, y_train_pred),
            }
            
            # Test metrics on ORIGINAL y_test (not winsorized)
            self.test_metrics = {
                "R2": r2_score(y_test, y_test_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
                "MAE": mean_absolute_error(y_test, y_test_pred),
            }
            
            # Also compute "fair" test R² on winsorized test (for comparison)
            y_test_winsorized = self.winsorize_target(y_test, fit=False)
            test_r2_winsorized = r2_score(y_test_winsorized, y_test_pred)
            
            self.tracker.log_metrics(y_test.values, y_test_pred, step=0)
            self.tracker.log_model(self.model, "titan_coordinator_phase15")
        
        # Print results
        print(f"\n   📈 Coordinator Results:")
        print(f"   {'Metric':<25} {'Train':>12} {'Test':>12} {'Test (Winsorized)':>18}")
        print("   " + "-" * 70)
        print(f"   {'R²':<25} {self.train_metrics['R2']:>12.4f} {self.test_metrics['R2']:>12.4f} {test_r2_winsorized:>18.4f}")
        print(f"   {'RMSE':<25} {self.train_metrics['RMSE']:>12.4f} {self.test_metrics['RMSE']:>12.4f}")
        print(f"   {'MAE':<25} {self.train_metrics['MAE']:>12.4f} {self.test_metrics['MAE']:>12.4f}")
        
        # Improvement
        if self.baseline_metrics:
            improvement = (self.test_metrics['R2'] - self.baseline_metrics['R2']) * 100
            print(f"\n   📈 Improvement over HAR baseline: {improvement:+.2f}%")
        
        # Print coefficients
        print(f"\n   📊 Ridge Coefficients:")
        print("   " + "-" * 55)
        coef_df = pd.DataFrame({
            "feature": self.feature_cols,
            "coefficient": self.model.coef_
        })
        coef_df["abs_coef"] = coef_df["coefficient"].abs()
        coef_df = coef_df.sort_values("abs_coef", ascending=False)
        
        for _, row in coef_df.iterrows():
            status = "✅" if abs(row["coefficient"]) < 1.0 else "⚠️"
            print(f"      {row['feature']:25s}: {row['coefficient']:>+10.4f} {status}")
        print(f"      {'Intercept':25s}: {self.model.intercept_:>+10.4f}")
        
        # Calculate sector-specific R²
        print(f"\n   📊 Sector Breakdown:")
        print("   " + "-" * 50)
        
        self.sector_metrics = {}
        for sector in test["sector"].dropna().unique():
            sector_mask = test["sector"] == sector
            if sector_mask.sum() > 50:
                sector_r2 = r2_score(y_test[sector_mask], y_test_pred[sector_mask])
                self.sector_metrics[sector] = sector_r2
                marker = " ⭐" if sector_r2 >= 0.30 else ""
                print(f"      {sector:15s}: R² = {sector_r2:.4f} ({sector_r2*100:.2f}%){marker}")
        
        return {
            "train": self.train_metrics,
            "test": self.test_metrics,
            "test_winsorized_r2": test_r2_winsorized,
            "baseline": self.baseline_metrics,
            "sector": self.sector_metrics
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get Ridge coefficients as feature importance."""
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
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions from the coordinator."""
        # Add features if missing
        if "is_friday" not in df.columns:
            df = self.add_calendar_features(df)
        if "vol_ma5" not in df.columns:
            df = self.add_momentum_features(df)
        
        X = df[self.feature_cols].fillna(0)
        return self.model.predict(X)
    
    def print_comparison_table(self):
        """Print the final comparison table."""
        print("\n" + "=" * 70)
        print("💰 PHASE 15 RESULTS: ROBUSTNESS UPGRADE")
        print("=" * 70)
        
        if self.baseline_metrics is None or self.test_metrics is None:
            print("   ⚠️ Train the coordinator first!")
            return
        
        baseline_r2 = self.baseline_metrics["R2"]
        titan_r2 = self.test_metrics["R2"]
        baseline_rmse = self.baseline_metrics["RMSE"]
        titan_rmse = self.test_metrics["RMSE"]
        
        improvement_r2_pct = (titan_r2 - baseline_r2) * 100
        
        print(f"""
   ┌─────────────────────────┬────────────────┬────────────────┐
   │ Model                   │ Test R²        │ Test RMSE      │
   ├─────────────────────────┼────────────────┼────────────────┤
   │ Baseline (HAR-RV)       │ {baseline_r2:>12.4f}   │ {baseline_rmse:>12.4f}   │
   │ Titan V8 (Phase 15)     │ {titan_r2:>12.4f}   │ {titan_rmse:>12.4f}   │
   ├─────────────────────────┼────────────────┼────────────────┤
   │ Improvement             │ {improvement_r2_pct:>+11.2f}%   │                │
   └─────────────────────────┴────────────────┴────────────────┘
        """)
        
        if titan_r2 >= 0.30:
            print("   🏆 TARGET ACHIEVED: 30%+ R²!")
        elif titan_r2 >= 0.25:
            print("   ✅ EXCELLENT: 25%+ R²!")
        elif titan_r2 >= 0.20:
            print("   ✅ STRONG: 20%+ R²!")
        elif titan_r2 > baseline_r2:
            print("   ✅ Titan V8 outperforms the baseline!")
        else:
            print("   ⚠️ Baseline is still better")
        
        # Sector analysis
        if self.sector_metrics:
            print("\n   📊 SECTOR ANALYSIS:")
            print("   " + "-" * 45)
            sorted_sectors = sorted(self.sector_metrics.items(), key=lambda x: x[1], reverse=True)
            for sector, r2 in sorted_sectors:
                marker = " ⭐" if r2 >= 0.30 else ""
                print(f"      {sector:15s}: {r2*100:.2f}%{marker}")
        
        print("=" * 70)


def main():
    """Test the TitanCoordinator."""
    print("\n" + "=" * 70)
    print("🚀 TITAN COORDINATOR (Phase 15: Robustness Upgrade)")
    print("=" * 70)
    
    coordinator = TitanCoordinator()
    print(f"   Model: Ridge(alpha={coordinator.alpha})")
    print(f"   Winsorization: {coordinator.winsorize_pct:.0%}")
    print(f"   Features: {coordinator.get_feature_columns()}")
    print("   ✅ Coordinator initialized successfully")
    print("\n   Run scripts/run_final_optimization.py for full training")


if __name__ == "__main__":
    main()
