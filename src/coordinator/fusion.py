"""
TitanCoordinator: Ridge Regression with Momentum + Calendar Features

Phase 12: Structural Optimization - Target 23%+ R²

Key Optimizations (from audit):
- Model: Ridge(alpha=0.1) outperforms ElasticNet
- Calendar: is_friday, is_monday, is_q4 add ~7% R²
- Momentum: vol_ma5 (5-day rolling mean) adds ~3% R²
- Simple features beat complex ones

Architecture:
- Inputs: tech_pred + news_risk_score + momentum + calendar
- Model: Ridge with light regularization (alpha=0.1)
- Validation: Purged Walk-Forward (Train < 2023, Test >= 2023)

Expected: 23%+ R² (validated in audit)

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
    Ridge Coordinator with Momentum + Calendar Features (Phase 12).
    
    Key Optimizations:
    - Ridge(alpha=0.1) instead of ElasticNet
    - Added vol_ma5 (5-day volatility momentum)
    - Calendar features: is_friday, is_monday, is_q4
    - Direct shock_index feature for news impact
    
    Full Feature List:
    - tech_pred: HAR-RV prediction
    - news_risk_score: Probability of extreme event (0-1)
    - VIX_close: Market fear level
    - is_friday, is_monday, is_q4: Calendar effects
    - vol_ma5: 5-day rolling mean of volatility (momentum)
    - shock_index: News shock intensity
    """
    
    def __init__(self, experiment_name: str = "titan_v8_coordinator_phase12"):
        """Initialize TitanCoordinator with Ridge."""
        # Ridge with light regularization (from audit: α=0.1 is optimal)
        self.model = Ridge(
            alpha=0.1,
            fit_intercept=True
        )
        
        self.tracker = MLTracker(experiment_name)
        self.feature_cols = None
        self.target_col = "target_log_var"
        self.train_metrics = None
        self.test_metrics = None
        self.baseline_metrics = None
        self.sector_metrics = None
        
    def get_feature_columns(self) -> list:
        """
        Define coordinator input features.
        
        Phase 12 optimized feature set based on audit:
        - Agent predictions
        - Calendar effects (critical: +7% R²)
        - Momentum (important: +3% R²)
        - VIX context
        - News shock
        """
        return [
            # Agent predictions
            "tech_pred",           # HAR-RV baseline
            "news_risk_score",     # Probability of extreme event
            
            # Calendar features (KEY: +7% R²)
            "is_friday",           # Pre-weekend effect
            "is_monday",           # Start-of-week
            "is_q4",               # Year-end seasonality
            
            # Momentum (KEY: +3% R²)
            "vol_ma5",             # 5-day rolling volatility
            
            # Context
            "VIX_close",           # Market fear
            "shock_index",         # News shock intensity
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
        Add momentum features.
        
        vol_ma5: 5-day rolling mean of target_log_var (shifted by 1 to avoid leakage)
        Logic: Volatility clusters - if recent days were volatile, today likely is too.
        """
        df = df.copy()
        
        print("   📈 Adding momentum features...")
        
        # Need target_log_var for momentum calculation
        if "target_log_var" not in df.columns and "realized_vol" in df.columns:
            # Compute from realized_vol if needed
            df["target_log_var"] = np.log(df["realized_vol"].clip(lower=1e-10))
        
        if "target_log_var" in df.columns:
            # Calculate per-ticker momentum
            for ticker in df["ticker"].unique():
                mask = df["ticker"] == ticker
                # 5-day rolling mean, shifted by 1 to avoid leakage
                df.loc[mask, "vol_ma5"] = (
                    df.loc[mask, "target_log_var"]
                    .rolling(5, min_periods=1)
                    .mean()
                    .shift(1)
                )
            
            df["vol_ma5"] = df["vol_ma5"].fillna(df["target_log_var"].mean())
            print(f"      ✓ vol_ma5: 5-day rolling mean volatility")
        else:
            df["vol_ma5"] = 0
            print(f"      ⚠️ vol_ma5: Could not compute (no target_log_var)")
        
        return df
    
    def prepare_predictions_dataset(
        self,
        tech_agent=None,
        news_agent=None,
        targets_df: pd.DataFrame = None,
        news_features_df: pd.DataFrame = None,
        residuals_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Prepare a unified dataset with all features.
        
        Phase 12: Simplified to use pre-computed predictions.
        """
        print("\n📊 Preparing unified predictions dataset...")
        
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
        
        # ================================================
        # Add Calendar Features (KEY: +7% R²)
        # ================================================
        print("   🗓️ Adding calendar features...")
        df = self.add_calendar_features(df)
        print(f"      ✓ is_friday: {df['is_friday'].sum()} days")
        print(f"      ✓ is_monday: {df['is_monday'].sum()} days")
        print(f"      ✓ is_q4: {df['is_q4'].sum()} days")
        
        # ================================================
        # Add Momentum Features (KEY: +3% R²)
        # ================================================
        df = self.add_momentum_features(df)
        
        # ================================================
        # Add tech_pred from residuals
        # ================================================
        if residuals_df is not None:
            res_df = residuals_df.copy()
            res_df["date"] = pd.to_datetime(res_df["date"]).dt.tz_localize(None)
            if res_df["ticker"].dtype.name == "category":
                res_df["ticker"] = res_df["ticker"].astype(str)
            
            # Get tech prediction
            if "pred_tech_excess" in res_df.columns:
                df = pd.merge(
                    df, 
                    res_df[["date", "ticker", "pred_tech_excess"]], 
                    on=["date", "ticker"], 
                    how="left"
                )
                # Convert excess prediction to total prediction
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
        
        # ================================================
        # Add news_risk_score (from news classifier)
        # ================================================
        if news_agent is not None and hasattr(news_agent, 'df') and news_agent.df is not None:
            news_df = news_agent.df.copy()
            news_df["date"] = pd.to_datetime(news_df["date"]).dt.tz_localize(None)
            
            if "news_risk_score" in news_df.columns:
                df = pd.merge(
                    df,
                    news_df[["date", "ticker", "news_risk_score"]],
                    on=["date", "ticker"],
                    how="left"
                )
                df["news_risk_score"] = df["news_risk_score"].fillna(0.2)
                print(f"   ✓ Added news_risk_score from classifier")
            elif hasattr(news_agent, 'predict_proba'):
                # Generate risk scores
                risk_scores = news_agent.predict_proba(news_df)
                news_df["news_risk_score"] = risk_scores
                df = pd.merge(
                    df,
                    news_df[["date", "ticker", "news_risk_score"]],
                    on=["date", "ticker"],
                    how="left"
                )
                df["news_risk_score"] = df["news_risk_score"].fillna(0.2)
                print(f"   ✓ Added news_risk_score (generated)")
            else:
                df["news_risk_score"] = 0.2
                print(f"   ⚠️ No news_risk_score, using default 0.2")
        else:
            df["news_risk_score"] = 0.2
            print(f"   ⚠️ NewsAgent not provided, using default 0.2")
        
        # ================================================
        # Add shock_index from news features
        # ================================================
        if news_features_df is not None:
            nf_df = news_features_df.copy()
            nf_df["date"] = pd.to_datetime(nf_df["date"]).dt.tz_localize(None)
            if nf_df["ticker"].dtype.name == "category":
                nf_df["ticker"] = nf_df["ticker"].astype(str)
            
            if "shock_index" in nf_df.columns:
                df = pd.merge(
                    df,
                    nf_df[["date", "ticker", "shock_index"]],
                    on=["date", "ticker"],
                    how="left"
                )
                df["shock_index"] = df["shock_index"].fillna(0)
                print(f"   ✓ Added shock_index from news features")
            else:
                df["shock_index"] = 0
                print(f"   ⚠️ No shock_index in news features")
        else:
            df["shock_index"] = 0
            print(f"   ⚠️ No news_features provided, shock_index=0")
        
        # ================================================
        # Ensure VIX is present
        # ================================================
        if "VIX_close" not in df.columns:
            try:
                vix = pd.read_parquet("data/processed/vix.parquet")
                vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None)
                df = pd.merge(df, vix[["date", "VIX_close"]], on="date", how="left")
                df["VIX_close"] = df["VIX_close"].fillna(20)
            except:
                df["VIX_close"] = 20
        df["VIX_close"] = df["VIX_close"].ffill().fillna(20)
        
        # ================================================
        # Add sector for analysis
        # ================================================
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
        
        print(f"\n   📊 Final dataset: {len(df):,} rows")
        print(f"   Features available: {[f for f in self.get_feature_columns() if f in df.columns]}")
        
        return df
    
    def train(self, df: pd.DataFrame) -> dict:
        """
        Train the Ridge coordinator.
        
        Uses Purged Walk-Forward validation:
        - Train: data < 2023-01-01
        - Test: data >= 2023-01-01
        """
        print("\n" + "=" * 70)
        print("🎯 TRAINING TITAN COORDINATOR (Ridge + Momentum + Calendar)")
        print("   Phase 12: Structural Optimization")
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
        
        # Calculate baseline (tech_pred only)
        if "tech_pred" in df.columns:
            baseline_r2 = r2_score(y_test, test["tech_pred"])
            baseline_rmse = np.sqrt(mean_squared_error(y_test, test["tech_pred"]))
            self.baseline_metrics = {"R2": baseline_r2, "RMSE": baseline_rmse}
            print(f"\n   📈 Baseline (HAR-RV only):")
            print(f"      Test R²:  {baseline_r2:.4f} ({baseline_r2*100:.2f}%)")
            print(f"      Test RMSE: {baseline_rmse:.4f}")
        
        # Train Ridge
        print("\n   🔧 Training Ridge Coordinator...")
        print(f"      Alpha: 0.1")
        
        with self.tracker.start_run(run_name="titan_coordinator_ridge_phase12"):
            self.tracker.log_params({
                "model": "Ridge",
                "alpha": 0.1,
                "n_features": len(self.feature_cols),
                "features": self.feature_cols,
                "n_train": len(train),
                "n_test": len(test),
                "phase": 12
            })
            
            # Fit Ridge
            self.model.fit(X_train, y_train)
            
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            self.train_metrics = self.tracker.log_metrics(
                y_train.values, y_train_pred, step=0
            )
            self.test_metrics = self.tracker.log_metrics(
                y_test.values, y_test_pred, step=1
            )
            
            self.tracker.log_model(self.model, "titan_coordinator_ridge")
        
        # Print results
        print(f"\n   📈 Coordinator Results:")
        print(f"   {'Metric':<25} {'Train':>10} {'Test':>10}")
        print("   " + "-" * 47)
        print(f"   {'RMSE':<25} {self.train_metrics['RMSE']:>10.4f} {self.test_metrics['RMSE']:>10.4f}")
        print(f"   {'MAE':<25} {self.train_metrics['MAE']:>10.4f} {self.test_metrics['MAE']:>10.4f}")
        print(f"   {'R²':<25} {self.train_metrics['R2']:>10.4f} {self.test_metrics['R2']:>10.4f}")
        print(f"   {'Directional Accuracy':<25} {self.train_metrics['Directional_Accuracy']:>9.1f}% {self.test_metrics['Directional_Accuracy']:>9.1f}%")
        
        # Print coefficients
        print(f"\n   📊 Ridge Coefficients:")
        print("   " + "-" * 45)
        coef_df = pd.DataFrame({
            "feature": self.feature_cols,
            "coefficient": self.model.coef_
        })
        coef_df["abs_coef"] = coef_df["coefficient"].abs()
        coef_df = coef_df.sort_values("abs_coef", ascending=False)
        
        for _, row in coef_df.iterrows():
            print(f"      {row['feature']:20s}: {row['coefficient']:+.4f}")
        print(f"      {'Intercept':20s}: {self.model.intercept_:+.4f}")
        
        # Calculate sector-specific R²
        print(f"\n   📊 Sector Breakdown:")
        print("   " + "-" * 45)
        
        self.sector_metrics = {}
        for sector in test["sector"].dropna().unique():
            sector_mask = test["sector"] == sector
            if sector_mask.sum() > 50:
                sector_r2 = r2_score(y_test[sector_mask], y_test_pred[sector_mask])
                self.sector_metrics[sector] = sector_r2
                print(f"      {sector:15s}: R² = {sector_r2:.4f} ({sector_r2*100:.2f}%)")
        
        return {
            "train": self.train_metrics,
            "test": self.test_metrics,
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
        print("💰 PHASE 12 RESULTS: STRUCTURAL OPTIMIZATION")
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
   │ Titan V8 (Phase 12)     │ {titan_r2:>12.4f}   │ {titan_rmse:>12.4f}   │
   ├─────────────────────────┼────────────────┼────────────────┤
   │ Improvement             │ {improvement_r2_pct:>+11.2f}%   │                │
   └─────────────────────────┴────────────────┴────────────────┘
        """)
        
        if titan_r2 >= 0.23:
            print("   🏆 TARGET ACHIEVED: 23%+ R²!")
        elif titan_r2 >= 0.20:
            print("   ✅ STRONG: 20%+ R²!")
        elif titan_r2 > baseline_r2:
            print("   ✅ Titan V8 outperforms the baseline!")
        else:
            print("   ⚠️ Baseline is still better")
        
        # Sector analysis
        if self.sector_metrics:
            print("\n   📊 SECTOR ANALYSIS:")
            print("   " + "-" * 40)
            sorted_sectors = sorted(self.sector_metrics.items(), key=lambda x: x[1], reverse=True)
            for sector, r2 in sorted_sectors:
                marker = " ⭐" if r2 > 0.25 else ""
                print(f"      {sector:15s}: {r2*100:.2f}%{marker}")
        
        print("=" * 70)


def main():
    """Test the TitanCoordinator."""
    print("\n" + "=" * 70)
    print("🚀 TITAN COORDINATOR (Phase 12: Ridge + Momentum + Calendar)")
    print("=" * 70)
    
    coordinator = TitanCoordinator()
    print(f"   Model: Ridge(alpha=0.1)")
    print(f"   Features: {coordinator.get_feature_columns()}")
    print("   ✅ Coordinator initialized successfully")
    print("\n   Run scripts/run_final_optimization.py for full training")


if __name__ == "__main__":
    main()
