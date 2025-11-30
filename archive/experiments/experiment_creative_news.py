"""
🧪 CREATIVE NEWS EXPERIMENTS: Manifesting 10%+ R² for Next-Day Volatility

This script explores unconventional approaches to extract predictive signal
from news for NEXT-DAY volatility. We believe it's possible!

Creative Hypotheses:
1. News Momentum - Shock persistence over multiple days
2. Sentiment Extremes - Min/Max sentiment, not average
3. Asymmetric Impact - Negative news amplified
4. Cross-Ticker Contagion - Sector-wide news spillover
5. News Surprise - Deviation from expected news volume
6. Overnight vs Pre-Market - Timing precision
7. Attention Decay - Exponentially weighted news history
8. Volatility State Conditioning - News impact × current vol regime
9. Ticker Sensitivity - Learn per-ticker news beta
10. Topic Concentration - Single-topic vs mixed news days

NO PERMANENT CHANGES - EXPERIMENTS ONLY

Usage:
    python scripts/experiment_creative_news.py
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
import pytz


def load_all_data():
    """Load all available data sources."""
    print("\n📂 Loading data...")
    
    # News base (raw with timestamps)
    news_base = pd.read_parquet("data/processed/news_base.parquet")
    news_base['date'] = pd.to_datetime(news_base['date'])
    
    # News features (aggregated)
    news_features = pd.read_parquet("data/processed/news_features.parquet")
    news_features['date'] = pd.to_datetime(news_features['date']).dt.tz_localize(None)
    
    # Targets
    targets = pd.read_parquet("data/processed/targets_deseasonalized.parquet")
    targets['date'] = pd.to_datetime(targets['date']).dt.tz_localize(None)
    
    # Residuals
    residuals = pd.read_parquet("data/processed/residuals.parquet")
    residuals['date'] = pd.to_datetime(residuals['date']).dt.tz_localize(None)
    
    # Normalize ticker types
    for df in [news_base, news_features, targets, residuals]:
        if df['ticker'].dtype.name == 'category':
            df['ticker'] = df['ticker'].astype(str)
    
    print(f"   ✓ news_base: {len(news_base):,} rows")
    print(f"   ✓ news_features: {len(news_features):,} rows")
    print(f"   ✓ targets: {len(targets):,} rows")
    print(f"   ✓ residuals: {len(residuals):,} rows")
    
    return news_base, news_features, targets, residuals


def train_and_evaluate(X_train, y_train, X_test, y_test, model=None):
    """Train model and return R²."""
    if model is None:
        model = LGBMRegressor(n_estimators=100, max_depth=3, random_state=42, verbose=-1)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, y_pred)
    
    return train_r2, test_r2, model


def main():
    """Run creative experiments."""
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🧪 CREATIVE NEWS EXPERIMENTS")
    print("   Manifesting 10%+ R² for Next-Day Volatility")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load data
    news_base, news_features, targets, residuals = load_all_data()
    
    # Sector map
    SECTOR_MAP = {
        'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech',
        'JPM': 'Finance', 'BAC': 'Finance', 'V': 'Finance',
        'CAT': 'Industrial', 'GE': 'Industrial', 'BA': 'Industrial',
        'WMT': 'Consumer', 'MCD': 'Consumer', 'COST': 'Consumer',
        'XOM': 'Energy', 'CVX': 'Energy', 'SLB': 'Energy',
        'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare'
    }
    
    # Base merge
    df = pd.merge(news_features, targets[['date', 'ticker', 'target_excess', 'realized_vol', 'target_log_var']], 
                  on=['date', 'ticker'], how='inner')
    df = pd.merge(df, residuals[['date', 'ticker', 'resid_tech', 'pred_tech_excess']], 
                  on=['date', 'ticker'], how='inner')
    df['sector'] = df['ticker'].map(SECTOR_MAP)
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Time split
    cutoff = pd.to_datetime("2023-01-01")
    
    all_results = []
    
    # ==========================================================================
    # EXPERIMENT 1: NEWS MOMENTUM (Shock Persistence)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 1: NEWS MOMENTUM")
    print("   Hypothesis: Shock effects persist for 2-5 days")
    print("=" * 70)
    
    df_exp1 = df.copy()
    
    # Create lagged shock features
    for ticker in df_exp1['ticker'].unique():
        mask = df_exp1['ticker'] == ticker
        for lag in [1, 2, 3, 4, 5]:
            df_exp1.loc[mask, f'shock_lag{lag}'] = df_exp1.loc[mask, 'shock_index'].shift(lag)
            df_exp1.loc[mask, f'news_lag{lag}'] = df_exp1.loc[mask, 'news_count'].shift(lag)
        
        # Exponential weighted average
        df_exp1.loc[mask, 'shock_ema3'] = df_exp1.loc[mask, 'shock_index'].ewm(span=3).mean()
        df_exp1.loc[mask, 'shock_ema5'] = df_exp1.loc[mask, 'shock_index'].ewm(span=5).mean()
    
    df_exp1 = df_exp1.dropna()
    
    momentum_features = ['shock_index', 'shock_lag1', 'shock_lag2', 'shock_lag3',
                         'news_count', 'news_lag1', 'news_lag2',
                         'shock_ema3', 'shock_ema5']
    
    train_mask = df_exp1['date'] < cutoff
    test_mask = df_exp1['date'] >= cutoff
    
    X_train = df_exp1.loc[train_mask, momentum_features].fillna(0)
    X_test = df_exp1.loc[test_mask, momentum_features].fillna(0)
    y_train = df_exp1.loc[train_mask, 'target_excess']
    y_test = df_exp1.loc[test_mask, 'target_excess']
    
    train_r2, test_r2, _ = train_and_evaluate(X_train, y_train, X_test, y_test)
    print(f"\n   Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    all_results.append({'exp': 'News Momentum', 'test_r2': test_r2})
    
    # ==========================================================================
    # EXPERIMENT 2: SENTIMENT EXTREMES
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 2: SENTIMENT EXTREMES")
    print("   Hypothesis: Extreme sentiment (min/max) matters more than average")
    print("=" * 70)
    
    # Need to re-aggregate from raw news with sentiment extremes
    ET = pytz.timezone('US/Eastern')
    news_base['datetime_et'] = news_base['date'].dt.tz_convert(ET)
    news_base['date_only'] = news_base['datetime_et'].dt.date
    
    # Aggregate with min/max sentiment
    sent_agg = news_base.groupby(['date_only', 'ticker']).agg({
        'sentiment_score': ['mean', 'min', 'max', 'std', lambda x: (x < 0).sum()],
        'raw_text': 'count'
    }).reset_index()
    
    sent_agg.columns = ['date', 'ticker', 'sentiment_avg', 'sentiment_min', 
                        'sentiment_max', 'sentiment_std', 'negative_count', 'news_count']
    sent_agg['date'] = pd.to_datetime(sent_agg['date'])
    sent_agg['sentiment_range'] = sent_agg['sentiment_max'] - sent_agg['sentiment_min']
    sent_agg['negative_ratio'] = sent_agg['negative_count'] / (sent_agg['news_count'] + 1)
    
    df_exp2 = pd.merge(sent_agg, targets[['date', 'ticker', 'target_excess']], 
                       on=['date', 'ticker'], how='inner')
    df_exp2 = df_exp2.dropna()
    
    sentiment_features = ['sentiment_avg', 'sentiment_min', 'sentiment_max', 
                          'sentiment_std', 'sentiment_range', 'negative_ratio', 'news_count']
    
    train_mask = df_exp2['date'] < cutoff
    test_mask = df_exp2['date'] >= cutoff
    
    if test_mask.sum() > 50:
        X_train = df_exp2.loc[train_mask, sentiment_features].fillna(0)
        X_test = df_exp2.loc[test_mask, sentiment_features].fillna(0)
        y_train = df_exp2.loc[train_mask, 'target_excess']
        y_test = df_exp2.loc[test_mask, 'target_excess']
        
        train_r2, test_r2, _ = train_and_evaluate(X_train, y_train, X_test, y_test)
        print(f"\n   Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
        all_results.append({'exp': 'Sentiment Extremes', 'test_r2': test_r2})
    
    # ==========================================================================
    # EXPERIMENT 3: ASYMMETRIC IMPACT
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 3: ASYMMETRIC IMPACT")
    print("   Hypothesis: Negative news has 2-3x stronger impact")
    print("=" * 70)
    
    df_exp3 = df.copy()
    
    # Separate positive and negative shock
    # Assuming shock_index is always positive (sum of keyword hits)
    # We use sentiment to determine direction
    df_exp3['positive_shock'] = df_exp3['shock_index'] * (df_exp3['sentiment_avg'] > 0).astype(float)
    df_exp3['negative_shock'] = df_exp3['shock_index'] * (df_exp3['sentiment_avg'] <= 0).astype(float)
    df_exp3['sentiment_x_shock'] = df_exp3['sentiment_avg'] * df_exp3['shock_index']
    
    # Squared negative sentiment (asymmetric)
    df_exp3['neg_sentiment_sq'] = np.where(df_exp3['sentiment_avg'] < 0, 
                                            df_exp3['sentiment_avg']**2, 0)
    
    asymmetric_features = ['shock_index', 'positive_shock', 'negative_shock', 
                           'sentiment_x_shock', 'neg_sentiment_sq', 'news_count']
    
    train_mask = df_exp3['date'] < cutoff
    test_mask = df_exp3['date'] >= cutoff
    
    X_train = df_exp3.loc[train_mask, asymmetric_features].fillna(0)
    X_test = df_exp3.loc[test_mask, asymmetric_features].fillna(0)
    y_train = df_exp3.loc[train_mask, 'target_excess']
    y_test = df_exp3.loc[test_mask, 'target_excess']
    
    train_r2, test_r2, _ = train_and_evaluate(X_train, y_train, X_test, y_test)
    print(f"\n   Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    all_results.append({'exp': 'Asymmetric Impact', 'test_r2': test_r2})
    
    # ==========================================================================
    # EXPERIMENT 4: CROSS-TICKER CONTAGION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 4: CROSS-TICKER CONTAGION")
    print("   Hypothesis: Sector-wide news spillover affects all tickers")
    print("=" * 70)
    
    df_exp4 = df.copy()
    
    # Calculate sector-level news aggregates
    sector_news = df_exp4.groupby(['date', 'sector']).agg({
        'shock_index': 'sum',
        'news_count': 'sum',
        'sentiment_avg': 'mean'
    }).reset_index()
    sector_news.columns = ['date', 'sector', 'sector_shock', 'sector_news', 'sector_sentiment']
    
    # Merge back
    df_exp4 = pd.merge(df_exp4, sector_news, on=['date', 'sector'], how='left')
    
    # Own shock vs sector shock
    df_exp4['own_shock_ratio'] = df_exp4['shock_index'] / (df_exp4['sector_shock'] + 1)
    df_exp4['sector_shock_others'] = df_exp4['sector_shock'] - df_exp4['shock_index']
    
    # Market-wide news (all sectors)
    market_news = df_exp4.groupby('date').agg({
        'shock_index': 'sum',
        'news_count': 'sum'
    }).reset_index()
    market_news.columns = ['date', 'market_shock', 'market_news']
    
    df_exp4 = pd.merge(df_exp4, market_news, on='date', how='left')
    df_exp4['own_vs_market'] = df_exp4['shock_index'] / (df_exp4['market_shock'] + 1)
    
    contagion_features = ['shock_index', 'sector_shock', 'sector_shock_others', 
                          'market_shock', 'own_vs_market', 'sector_sentiment']
    
    train_mask = df_exp4['date'] < cutoff
    test_mask = df_exp4['date'] >= cutoff
    
    X_train = df_exp4.loc[train_mask, contagion_features].fillna(0)
    X_test = df_exp4.loc[test_mask, contagion_features].fillna(0)
    y_train = df_exp4.loc[train_mask, 'target_excess']
    y_test = df_exp4.loc[test_mask, 'target_excess']
    
    train_r2, test_r2, _ = train_and_evaluate(X_train, y_train, X_test, y_test)
    print(f"\n   Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    all_results.append({'exp': 'Cross-Ticker Contagion', 'test_r2': test_r2})
    
    # ==========================================================================
    # EXPERIMENT 5: NEWS SURPRISE (vs Expected)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 5: NEWS SURPRISE")
    print("   Hypothesis: Unexpected news volume is the signal")
    print("=" * 70)
    
    df_exp5 = df.copy()
    
    # Expected news = rolling mean
    for ticker in df_exp5['ticker'].unique():
        mask = df_exp5['ticker'] == ticker
        df_exp5.loc[mask, 'expected_news'] = df_exp5.loc[mask, 'news_count'].rolling(20, min_periods=5).mean()
        df_exp5.loc[mask, 'expected_shock'] = df_exp5.loc[mask, 'shock_index'].rolling(20, min_periods=5).mean()
    
    df_exp5['news_surprise'] = df_exp5['news_count'] - df_exp5['expected_news']
    df_exp5['shock_surprise'] = df_exp5['shock_index'] - df_exp5['expected_shock']
    df_exp5['news_surprise_pct'] = df_exp5['news_surprise'] / (df_exp5['expected_news'] + 1)
    
    # Z-score of news count
    for ticker in df_exp5['ticker'].unique():
        mask = df_exp5['ticker'] == ticker
        rolling_std = df_exp5.loc[mask, 'news_count'].rolling(20, min_periods=5).std()
        df_exp5.loc[mask, 'news_zscore'] = df_exp5.loc[mask, 'news_surprise'] / (rolling_std + 0.1)
    
    df_exp5 = df_exp5.dropna()
    
    surprise_features = ['news_surprise', 'shock_surprise', 'news_surprise_pct', 'news_zscore']
    
    train_mask = df_exp5['date'] < cutoff
    test_mask = df_exp5['date'] >= cutoff
    
    X_train = df_exp5.loc[train_mask, surprise_features].fillna(0)
    X_test = df_exp5.loc[test_mask, surprise_features].fillna(0)
    y_train = df_exp5.loc[train_mask, 'target_excess']
    y_test = df_exp5.loc[test_mask, 'target_excess']
    
    train_r2, test_r2, _ = train_and_evaluate(X_train, y_train, X_test, y_test)
    print(f"\n   Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    all_results.append({'exp': 'News Surprise', 'test_r2': test_r2})
    
    # ==========================================================================
    # EXPERIMENT 6: VOLATILITY STATE CONDITIONING
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 6: VOLATILITY STATE CONDITIONING")
    print("   Hypothesis: News impact depends on current vol regime")
    print("=" * 70)
    
    df_exp6 = df.copy()
    
    # Get VIX
    try:
        vix = pd.read_parquet("data/processed/vix.parquet")
        vix['date'] = pd.to_datetime(vix['date']).dt.tz_localize(None)
        df_exp6 = pd.merge(df_exp6, vix[['date', 'VIX_close']], on='date', how='left')
        df_exp6['VIX_close'] = df_exp6['VIX_close'].fillna(20)
    except:
        df_exp6['VIX_close'] = 20
    
    # Lagged realized vol (yesterday's vol)
    for ticker in df_exp6['ticker'].unique():
        mask = df_exp6['ticker'] == ticker
        df_exp6.loc[mask, 'vol_lag1'] = df_exp6.loc[mask, 'realized_vol'].shift(1)
        df_exp6.loc[mask, 'vol_ema5'] = df_exp6.loc[mask, 'realized_vol'].ewm(span=5).mean()
    
    # Interaction: news × vol state
    df_exp6['shock_x_vix'] = df_exp6['shock_index'] * df_exp6['VIX_close'] / 20
    df_exp6['shock_x_vol_lag'] = df_exp6['shock_index'] * df_exp6['vol_lag1']
    df_exp6['news_x_vix'] = df_exp6['news_count'] * df_exp6['VIX_close'] / 20
    
    # High vol regime flag
    df_exp6['high_vol_regime'] = (df_exp6['VIX_close'] > df_exp6['VIX_close'].quantile(0.7)).astype(int)
    df_exp6['shock_high_vol'] = df_exp6['shock_index'] * df_exp6['high_vol_regime']
    
    df_exp6 = df_exp6.dropna()
    
    state_features = ['shock_index', 'VIX_close', 'vol_lag1', 'shock_x_vix', 
                      'shock_x_vol_lag', 'shock_high_vol']
    
    train_mask = df_exp6['date'] < cutoff
    test_mask = df_exp6['date'] >= cutoff
    
    X_train = df_exp6.loc[train_mask, state_features].fillna(0)
    X_test = df_exp6.loc[test_mask, state_features].fillna(0)
    y_train = df_exp6.loc[train_mask, 'target_excess']
    y_test = df_exp6.loc[test_mask, 'target_excess']
    
    train_r2, test_r2, _ = train_and_evaluate(X_train, y_train, X_test, y_test)
    print(f"\n   Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    all_results.append({'exp': 'Vol State Conditioning', 'test_r2': test_r2})
    
    # ==========================================================================
    # EXPERIMENT 7: TICKER-SPECIFIC NEWS BETA
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 7: TICKER-SPECIFIC NEWS BETA")
    print("   Hypothesis: Each ticker has different news sensitivity")
    print("=" * 70)
    
    df_exp7 = df.copy()
    
    # Train separate models per ticker, then combine
    ticker_betas = {}
    ticker_results = []
    
    for ticker in df_exp7['ticker'].unique():
        df_ticker = df_exp7[df_exp7['ticker'] == ticker].copy()
        
        train_t = df_ticker[df_ticker['date'] < cutoff]
        test_t = df_ticker[df_ticker['date'] >= cutoff]
        
        if len(train_t) < 30 or len(test_t) < 30:
            continue
        
        features = ['shock_index', 'news_count', 'sentiment_avg', 'novelty_score']
        
        X_train = train_t[features].fillna(0)
        X_test = test_t[features].fillna(0)
        y_train = train_t['target_excess']
        y_test = test_t['target_excess']
        
        model = Ridge(alpha=10.0)
        model.fit(X_train, y_train)
        
        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))
        
        ticker_betas[ticker] = model.coef_[0]  # shock_index coefficient
        ticker_results.append({'ticker': ticker, 'train_r2': train_r2, 'test_r2': test_r2, 
                               'shock_beta': model.coef_[0]})
    
    ticker_df = pd.DataFrame(ticker_results)
    if len(ticker_df) > 0:
        print(f"\n   {'Ticker':<8} {'Shock Beta':>12} {'Test R²':>12}")
        print("   " + "-" * 34)
        for _, row in ticker_df.sort_values('test_r2', ascending=False).head(5).iterrows():
            print(f"   {row['ticker']:<8} {row['shock_beta']:>12.4f} {row['test_r2']:>12.4f}")
        
        avg_r2 = ticker_df['test_r2'].mean()
        print(f"\n   Average per-ticker R²: {avg_r2:.4f}")
        all_results.append({'exp': 'Ticker-Specific Beta', 'test_r2': avg_r2})
    
    # ==========================================================================
    # EXPERIMENT 8: MULTI-DAY TARGET (Smoothed Volatility)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 8: MULTI-DAY TARGET")
    print("   Hypothesis: Predict 3-day avg vol instead of 1-day")
    print("=" * 70)
    
    df_exp8 = df.copy()
    
    # Create smoothed targets
    for ticker in df_exp8['ticker'].unique():
        mask = df_exp8['ticker'] == ticker
        df_exp8.loc[mask, 'target_3d'] = df_exp8.loc[mask, 'target_excess'].rolling(3).mean().shift(-2)
        df_exp8.loc[mask, 'target_5d'] = df_exp8.loc[mask, 'target_excess'].rolling(5).mean().shift(-4)
    
    df_exp8 = df_exp8.dropna(subset=['target_3d'])
    
    features = ['shock_index', 'news_count', 'sentiment_avg', 'novelty_score']
    pca_cols = [c for c in df_exp8.columns if c.startswith('news_pca_')][:5]
    features += pca_cols
    
    for target_name, target_col in [('1-day', 'target_excess'), ('3-day avg', 'target_3d')]:
        df_t = df_exp8.dropna(subset=[target_col])
        
        train_mask = df_t['date'] < cutoff
        test_mask = df_t['date'] >= cutoff
        
        if test_mask.sum() < 50:
            continue
        
        X_train = df_t.loc[train_mask, features].fillna(0)
        X_test = df_t.loc[test_mask, features].fillna(0)
        y_train = df_t.loc[train_mask, target_col]
        y_test = df_t.loc[test_mask, target_col]
        
        train_r2, test_r2, _ = train_and_evaluate(X_train, y_train, X_test, y_test)
        print(f"\n   {target_name}: Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
        all_results.append({'exp': f'Multi-Day ({target_name})', 'test_r2': test_r2})
    
    # ==========================================================================
    # EXPERIMENT 9: CHANGE PREDICTION (Vol Change, not Level)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 9: VOLATILITY CHANGE PREDICTION")
    print("   Hypothesis: Predict tomorrow's vol CHANGE, not level")
    print("=" * 70)
    
    df_exp9 = df.copy()
    
    # Create change targets
    for ticker in df_exp9['ticker'].unique():
        mask = df_exp9['ticker'] == ticker
        df_exp9.loc[mask, 'vol_change'] = df_exp9.loc[mask, 'target_excess'].diff()
        df_exp9.loc[mask, 'vol_pct_change'] = df_exp9.loc[mask, 'realized_vol'].pct_change()
    
    df_exp9 = df_exp9.dropna(subset=['vol_change'])
    
    features = ['shock_index', 'news_count', 'sentiment_avg', 'novelty_score']
    
    train_mask = df_exp9['date'] < cutoff
    test_mask = df_exp9['date'] >= cutoff
    
    X_train = df_exp9.loc[train_mask, features].fillna(0)
    X_test = df_exp9.loc[test_mask, features].fillna(0)
    y_train = df_exp9.loc[train_mask, 'vol_change']
    y_test = df_exp9.loc[test_mask, 'vol_change']
    
    train_r2, test_r2, _ = train_and_evaluate(X_train, y_train, X_test, y_test)
    print(f"\n   Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    all_results.append({'exp': 'Vol Change Prediction', 'test_r2': test_r2})
    
    # ==========================================================================
    # EXPERIMENT 10: ENSEMBLE ALL SIGNALS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT 10: ENSEMBLE ALL SIGNALS")
    print("   Hypothesis: Combine all creative features")
    print("=" * 70)
    
    # Merge all engineered features
    df_all = df.copy()
    
    # Add momentum features
    for ticker in df_all['ticker'].unique():
        mask = df_all['ticker'] == ticker
        df_all.loc[mask, 'shock_lag1'] = df_all.loc[mask, 'shock_index'].shift(1)
        df_all.loc[mask, 'shock_ema3'] = df_all.loc[mask, 'shock_index'].ewm(span=3).mean()
        df_all.loc[mask, 'expected_news'] = df_all.loc[mask, 'news_count'].rolling(20, min_periods=5).mean()
        df_all.loc[mask, 'vol_lag1'] = df_all.loc[mask, 'realized_vol'].shift(1)
    
    df_all['news_surprise'] = df_all['news_count'] - df_all['expected_news']
    df_all['negative_shock'] = df_all['shock_index'] * (df_all['sentiment_avg'] <= 0).astype(float)
    
    # Add VIX
    try:
        vix = pd.read_parquet("data/processed/vix.parquet")
        vix['date'] = pd.to_datetime(vix['date']).dt.tz_localize(None)
        df_all = pd.merge(df_all, vix[['date', 'VIX_close']], on='date', how='left')
        df_all['VIX_close'] = df_all['VIX_close'].fillna(20)
    except:
        df_all['VIX_close'] = 20
    
    df_all['shock_x_vix'] = df_all['shock_index'] * df_all['VIX_close'] / 20
    df_all['shock_x_vol'] = df_all['shock_index'] * df_all['vol_lag1']
    
    df_all = df_all.dropna()
    
    all_features = ['shock_index', 'news_count', 'sentiment_avg', 'novelty_score',
                    'shock_lag1', 'shock_ema3', 'news_surprise', 'negative_shock',
                    'VIX_close', 'vol_lag1', 'shock_x_vix', 'shock_x_vol']
    
    train_mask = df_all['date'] < cutoff
    test_mask = df_all['date'] >= cutoff
    
    X_train = df_all.loc[train_mask, all_features].fillna(0)
    X_test = df_all.loc[test_mask, all_features].fillna(0)
    y_train = df_all.loc[train_mask, 'target_excess']
    y_test = df_all.loc[test_mask, 'target_excess']
    
    # Try multiple models
    models = {
        'LightGBM': LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.03, random_state=42, verbose=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    print(f"\n   Features: {len(all_features)}")
    print(f"\n   {'Model':<20} {'Train R²':>12} {'Test R²':>12}")
    print("   " + "-" * 46)
    
    best_test_r2 = -999
    for name, model in models.items():
        train_r2, test_r2, _ = train_and_evaluate(X_train, y_train, X_test, y_test, model)
        print(f"   {name:<20} {train_r2:>12.4f} {test_r2:>12.4f}")
        all_results.append({'exp': f'Ensemble ({name})', 'test_r2': test_r2})
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("🏆 EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('test_r2', ascending=False)
    
    print(f"\n   {'Rank':<6} {'Experiment':<35} {'Test R²':>12}")
    print("   " + "-" * 55)
    
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        marker = " ⭐" if row['test_r2'] > 0 else ""
        print(f"   {i:<6} {row['exp']:<35} {row['test_r2']:>12.4f}{marker}")
    
    best = results_df.iloc[0]
    
    print("\n" + "=" * 70)
    print("💡 CONCLUSIONS")
    print("=" * 70)
    
    positive_results = results_df[results_df['test_r2'] > 0]
    
    if len(positive_results) > 0:
        print(f"""
   ✅ POSITIVE SIGNAL FOUND!
   
   Best Experiment: {best['exp']}
   Best Test R²: {best['test_r2']:.4f} ({best['test_r2']*100:.2f}%)
   
   Positive experiments:
""")
        for _, row in positive_results.iterrows():
            print(f"      - {row['exp']}: {row['test_r2']*100:.2f}%")
    else:
        print(f"""
   ❌ No positive R² achieved with these approaches.
   
   Best (least negative): {best['exp']}
   Best Test R²: {best['test_r2']:.4f} ({best['test_r2']*100:.2f}%)
   
   INSIGHT: News truly has limited predictive power for 
   next-day volatility LEVELS. The information is priced 
   in quickly by efficient markets.
   
   ALTERNATIVES TO CONSIDER:
   1. Use news for CLASSIFICATION (extreme events)
   2. Use news for SAME-DAY volatility nowcasting
   3. Focus on REGIME DETECTION, not point predictions
   4. Combine news with options-implied volatility
""")
    
    # Timing
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n   Duration: {duration}")
    print(f"   Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return results_df


if __name__ == "__main__":
    main()

