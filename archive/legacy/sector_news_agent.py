"""
SectorNewsAgent: Sector-Specific News Models (Phase 8)

Trains separate LightGBM models for each sector to handle the
heterogeneity problem where news impacts sectors differently.

Key Insight:
- Tech news affects AAPL/MSFT/NVDA similarly
- Energy news affects XOM/CVX/SLB similarly
- A single model conflates these different dynamics

Architecture:
- 6 separate LightGBM models (one per sector)
- Each model trained on sector-specific data
- Predictions merged by ticker sector assignment

Expected Benefits:
- Better capture of sector-specific news dynamics
- Reduced overfitting from heterogeneous training data
- Improved NewsAgent R² (currently negative)

Usage:
    from src.agents.sector_news_agent import SectorNewsAgent
    agent = SectorNewsAgent()
    agent.train()
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.tracker import MLTracker


# Sector mapping for 18-ticker universe
SECTOR_MAP = {
    # Tech (Momentum/Growth)
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech',
    # Finance (Rate Sensitivity)
    'JPM': 'Finance', 'BAC': 'Finance', 'V': 'Finance',
    # Industrial (Economic Health)
    'CAT': 'Industrial', 'GE': 'Industrial', 'BA': 'Industrial',
    # Consumer (Inflation Proxy)
    'WMT': 'Consumer', 'MCD': 'Consumer', 'COST': 'Consumer',
    # Energy (Commodity Risk)
    'XOM': 'Energy', 'CVX': 'Energy', 'SLB': 'Energy',
    # Healthcare (Defensive/Policy)
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare'
}

SECTORS = ['Tech', 'Finance', 'Industrial', 'Consumer', 'Energy', 'Healthcare']


def calculate_decay_kernel(series: pd.Series, group_col: pd.Series = None) -> pd.Series:
    """
    Calculate exponential decay kernel for a time series.
    
    Formula: 0.50 * shift(1) + 0.25 * shift(2) + 0.15 * shift(3) + 0.10 * shift(4)
    Stops at shift(4) to avoid weekly seasonality contamination.
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


class SectorNewsAgent:
    """
    Sector-Specific News Agent (Phase 8).
    
    Trains 6 separate LightGBM models, one for each sector:
    - Tech, Finance, Industrial, Consumer, Energy, Healthcare
    
    Each model learns sector-specific news dynamics:
    - Tech: AI hype, earnings sensitivity
    - Finance: Rate sensitivity, regulatory news
    - Energy: Oil price shocks, geopolitical events
    - Healthcare: Drug approvals, policy changes
    - Industrial: Supply chain, economic indicators
    - Consumer: Inflation, consumer sentiment
    
    Features (per sector):
    - news_memory: Decay kernel of news_count
    - shock_memory: Decay kernel of shock_index
    - sentiment_memory: Decay kernel of sentiment
    - PCA features: Topic embeddings
    - VIX_close: Market fear context
    """
    
    def __init__(self, experiment_name: str = "titan_v8_sector_news",
                 use_deseasonalized: bool = True):
        """
        Initialize SectorNewsAgent.
        
        Args:
            experiment_name: MLflow experiment name
            use_deseasonalized: If True, de-seasonalize news features
        """
        # Store models for each sector
        self.models: Dict[str, LGBMRegressor] = {}
        self.sector_metrics: Dict[str, dict] = {}
        
        # Default LightGBM params (robust)
        self.lgbm_params = {
            'n_estimators': 200,
            'learning_rate': 0.02,
            'max_depth': 3,
            'num_leaves': 8,
            'min_child_samples': 15,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        
        self.tracker = MLTracker(experiment_name)
        self.feature_cols = None
        self.target_col = "resid_tech"
        self.train_metrics = None
        self.test_metrics = None
        self.df = None
        
        self.use_deseasonalized = use_deseasonalized
        self.news_seasonal_maps = {}  # Per-sector seasonal maps
        
    def load_and_merge_data(self) -> pd.DataFrame:
        """
        Load residuals and news features, add sector column.
        """
        print("\n📂 Loading data for SectorNewsAgent...")
        
        # Load residuals
        residuals_path = Path("data/processed/residuals.parquet")
        if not residuals_path.exists():
            raise FileNotFoundError("Residuals not found! Run TechnicalAgent first.")
        
        residuals_df = pd.read_parquet(residuals_path)
        print(f"   ✓ Residuals: {len(residuals_df):,} rows")
        
        # Load news features
        news_path = Path("data/processed/news_features.parquet")
        news_df = pd.read_parquet(news_path)
        print(f"   ✓ News features: {len(news_df):,} rows")
        
        # Load targets for VIX
        targets_path = Path("data/processed/targets.parquet")
        targets_df = pd.read_parquet(targets_path)
        vix_df = targets_df[["date", "ticker", "VIX_close"]].copy()
        
        # Normalize dates
        for df in [residuals_df, news_df, vix_df]:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            if df["ticker"].dtype.name == "category":
                df["ticker"] = df["ticker"].astype(str)
        
        # Merge
        merged = pd.merge(residuals_df, news_df, on=["date", "ticker"], how="inner")
        merged = pd.merge(merged, vix_df, on=["date", "ticker"], how="left")
        merged["VIX_close"] = merged["VIX_close"].ffill().fillna(15)
        
        # Add sector
        merged["sector"] = merged["ticker"].map(SECTOR_MAP)
        merged = merged.dropna(subset=["sector"])
        
        # Sort
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        print(f"   ✓ Merged: {len(merged):,} rows")
        print(f"   ✓ Sectors: {merged['sector'].unique().tolist()}")
        
        # =============================================
        # FEATURE ENGINEERING (Per Sector)
        # =============================================
        print("\n   🔧 Engineering features...")
        
        # Add day of week for de-seasonalization
        merged["day_of_week"] = merged["date"].dt.dayofweek
        
        # De-seasonalize news_count per sector
        if self.use_deseasonalized:
            print("      De-seasonalizing news_count per sector...")
            
            for sector in SECTORS:
                sector_mask = merged["sector"] == sector
                if sector_mask.sum() == 0:
                    continue
                
                # Calculate median per (ticker, day_of_week) within sector
                sector_data = merged.loc[sector_mask]
                seasonal = sector_data.groupby(["ticker", "day_of_week"])["news_count"].median()
                self.news_seasonal_maps[sector] = seasonal.to_dict()
            
            # Apply de-seasonalization
            def get_news_seasonal(row):
                sector = row["sector"]
                key = (row["ticker"], row["day_of_week"])
                seasonal_map = self.news_seasonal_maps.get(sector, {})
                return seasonal_map.get(key, row["news_count"])
            
            merged["news_count_seasonal"] = merged.apply(get_news_seasonal, axis=1)
            merged["news_count_excess"] = merged["news_count"] - merged["news_count_seasonal"]
            news_col = "news_count_excess"
            print("      ✓ Created news_count_excess")
        else:
            news_col = "news_count"
        
        # Apply decay kernel
        print("      Applying decay kernels...")
        merged["news_memory"] = calculate_decay_kernel(
            merged[news_col],
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
        merged["shock_vix_memory"] = merged["shock_memory"] * merged["VIX_close"]
        
        print("      ✓ Created: news_memory, shock_memory, sentiment_memory, shock_vix_memory")
        
        # Drop NaN
        merged = merged.dropna(subset=["news_memory", "shock_memory", self.target_col])
        
        print(f"\n   📊 Final dataset: {len(merged):,} rows")
        
        # Sector breakdown
        sector_counts = merged.groupby("sector").size()
        print("\n   📊 Sector breakdown:")
        for sector in SECTORS:
            if sector in sector_counts.index:
                print(f"      {sector:12}: {sector_counts[sector]:,} rows")
        
        self.df = merged
        return merged
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get feature columns for training."""
        features = []
        
        # Decay kernel features
        kernel_features = ["news_memory", "shock_memory", "sentiment_memory", "shock_vix_memory"]
        features.extend([f for f in kernel_features if f in df.columns])
        
        # Current values
        current_features = ["sentiment_avg", "novelty_score"]
        features.extend([f for f in current_features if f in df.columns])
        
        # PCA columns
        pca_cols = [c for c in df.columns if c.startswith("news_pca_")]
        features.extend(sorted(pca_cols))
        
        # VIX
        if "VIX_close" in df.columns:
            features.append("VIX_close")
        
        return list(dict.fromkeys(features))
    
    def train_sector_model(self, sector: str, df: pd.DataFrame) -> dict:
        """
        Train a single sector model.
        
        Args:
            sector: Sector name (e.g., 'Tech')
            df: Full dataset (will be filtered to sector)
            
        Returns:
            Dictionary with train/test metrics
        """
        # Filter to sector
        sector_df = df[df["sector"] == sector].copy()
        
        if len(sector_df) < 100:
            print(f"      ⚠️ {sector}: Only {len(sector_df)} rows, skipping")
            return None
        
        # Get features
        feature_cols = self.get_feature_columns(sector_df)
        
        # Time-series split
        cutoff = pd.to_datetime("2023-01-01")
        train_mask = sector_df["date"] < cutoff
        test_mask = sector_df["date"] >= cutoff
        
        if train_mask.sum() < 50 or test_mask.sum() < 20:
            print(f"      ⚠️ {sector}: Insufficient train/test data, skipping")
            return None
        
        X_train = sector_df.loc[train_mask, feature_cols].fillna(0)
        y_train = sector_df.loc[train_mask, self.target_col]
        X_test = sector_df.loc[test_mask, feature_cols].fillna(0)
        y_test = sector_df.loc[test_mask, self.target_col]
        
        # Initialize model
        model = LGBMRegressor(**self.lgbm_params)
        
        # Train
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="rmse")
        
        # Evaluate
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Store model
        self.models[sector] = model
        
        # Return metrics
        metrics = {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        self.sector_metrics[sector] = metrics
        
        return metrics
    
    def train(self, df: pd.DataFrame = None) -> dict:
        """
        Train all sector models.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if df is None:
            df = self.df
        
        if df is None:
            raise ValueError("No data! Call load_and_merge_data first.")
        
        print("\n🎯 Training SectorNewsAgent...")
        print("   Training 6 sector-specific LightGBM models")
        
        self.feature_cols = self.get_feature_columns(df)
        print(f"\n   Features: {len(self.feature_cols)}")
        
        # Train each sector
        print(f"\n   {'Sector':<12} {'Train R²':>10} {'Test R²':>10} {'RMSE':>10} {'N_Test':>10}")
        print("   " + "-" * 54)
        
        all_test_preds = []
        all_test_true = []
        
        for sector in SECTORS:
            metrics = self.train_sector_model(sector, df)
            
            if metrics is None:
                print(f"   {sector:<12} {'SKIPPED':>10}")
                continue
            
            status = "✅" if metrics["test_r2"] > 0 else "❌"
            print(f"   {sector:<12} {metrics['train_r2']:>10.4f} {metrics['test_r2']:>10.4f} {metrics['test_rmse']:>10.4f} {metrics['test_samples']:>10} {status}")
            
            # Collect predictions for aggregate metrics
            sector_df = df[df["sector"] == sector]
            test_mask = sector_df["date"] >= pd.to_datetime("2023-01-01")
            X_test = sector_df.loc[test_mask, self.feature_cols].fillna(0)
            y_test = sector_df.loc[test_mask, self.target_col]
            
            if len(X_test) > 0:
                preds = self.models[sector].predict(X_test)
                all_test_preds.extend(preds)
                all_test_true.extend(y_test.values)
        
        # Calculate aggregate metrics
        from sklearn.metrics import r2_score, mean_squared_error
        
        if len(all_test_preds) > 0:
            agg_r2 = r2_score(all_test_true, all_test_preds)
            agg_rmse = np.sqrt(mean_squared_error(all_test_true, all_test_preds))
            
            self.test_metrics = {"R2": agg_r2, "RMSE": agg_rmse}
            
            print("   " + "-" * 54)
            print(f"   {'AGGREGATE':<12} {'':>10} {agg_r2:>10.4f} {agg_rmse:>10.4f} {len(all_test_preds):>10}")
        else:
            self.test_metrics = {"R2": 0, "RMSE": 0}
        
        # Calculate train metrics similarly
        all_train_preds = []
        all_train_true = []
        
        for sector in SECTORS:
            if sector not in self.models:
                continue
            
            sector_df = df[df["sector"] == sector]
            train_mask = sector_df["date"] < pd.to_datetime("2023-01-01")
            X_train = sector_df.loc[train_mask, self.feature_cols].fillna(0)
            y_train = sector_df.loc[train_mask, self.target_col]
            
            if len(X_train) > 0:
                preds = self.models[sector].predict(X_train)
                all_train_preds.extend(preds)
                all_train_true.extend(y_train.values)
        
        if len(all_train_preds) > 0:
            train_r2 = r2_score(all_train_true, all_train_preds)
            self.train_metrics = {"R2": train_r2}
        else:
            self.train_metrics = {"R2": 0}
        
        return {
            "train": self.train_metrics,
            "test": self.test_metrics,
            "sector_metrics": self.sector_metrics
        }
    
    def predict(self, df: pd.DataFrame = None) -> pd.Series:
        """
        Generate predictions for all rows using sector-specific models.
        """
        if df is None:
            df = self.df
        
        predictions = pd.Series(0.0, index=df.index)
        
        for sector in SECTORS:
            if sector not in self.models:
                continue
            
            sector_mask = df["sector"] == sector
            if sector_mask.sum() == 0:
                continue
            
            X = df.loc[sector_mask, self.feature_cols].fillna(0)
            preds = self.models[sector].predict(X)
            predictions.loc[sector_mask] = preds
        
        return predictions
    
    def get_feature_importance(self, sector: str = None) -> pd.DataFrame:
        """
        Get feature importance for a sector or aggregated.
        """
        if sector and sector in self.models:
            model = self.models[sector]
            importance_df = pd.DataFrame({
                "feature": self.feature_cols,
                "importance": model.feature_importances_
            })
        else:
            # Aggregate across sectors
            importance_dict = {f: 0 for f in self.feature_cols}
            for s, model in self.models.items():
                for feat, imp in zip(self.feature_cols, model.feature_importances_):
                    importance_dict[feat] += imp
            
            importance_df = pd.DataFrame({
                "feature": list(importance_dict.keys()),
                "importance": list(importance_dict.values())
            })
        
        importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
        
        total = importance_df["importance"].sum()
        if total > 0:
            importance_df["pct"] = (importance_df["importance"] / total * 100).round(1)
        else:
            importance_df["pct"] = 0.0
        
        return importance_df


def main():
    """Run SectorNewsAgent training."""
    import mlflow
    
    print("\n" + "=" * 70)
    print("🚀 PHASE 8: SECTOR-SPECIFIC NEWS AGENTS")
    print("   Training 6 separate models for heterogeneous news dynamics")
    print("=" * 70)
    
    mlflow.end_run()
    mlflow.set_experiment("titan_v8_phase8")
    
    # Initialize agent
    agent = SectorNewsAgent(use_deseasonalized=True)
    
    # Load data
    df = agent.load_and_merge_data()
    
    # Train all sector models
    metrics = agent.train(df)
    
    # Print feature importance
    print("\n📊 AGGREGATE FEATURE IMPORTANCE")
    print("-" * 50)
    importance = agent.get_feature_importance()
    print(importance.head(15).to_string(index=False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SECTOR NEWS AGENT SUMMARY")
    print("=" * 70)
    
    print(f"\n   Aggregate Test R²:  {metrics['test']['R2']:.4f} ({metrics['test']['R2']*100:.2f}%)")
    print(f"   Aggregate Test RMSE: {metrics['test']['RMSE']:.4f}")
    
    # Per-sector summary
    print(f"\n   Per-Sector Performance:")
    positive_sectors = sum(1 for s, m in metrics['sector_metrics'].items() if m['test_r2'] > 0)
    total_sectors = len(metrics['sector_metrics'])
    print(f"      Positive R² sectors: {positive_sectors}/{total_sectors}")
    
    best_sector = max(metrics['sector_metrics'].items(), key=lambda x: x[1]['test_r2'])
    worst_sector = min(metrics['sector_metrics'].items(), key=lambda x: x[1]['test_r2'])
    print(f"      Best:  {best_sector[0]} ({best_sector[1]['test_r2']:.4f})")
    print(f"      Worst: {worst_sector[0]} ({worst_sector[1]['test_r2']:.4f})")
    
    print("\n" + "=" * 70)
    print("✅ SectorNewsAgent training complete!")
    print("=" * 70)
    
    return agent, metrics


if __name__ == "__main__":
    main()

