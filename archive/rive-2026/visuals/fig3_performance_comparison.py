"""
Figure 3: Performance Comparison of RIVE and Baseline Models

Clear, intuitive comparison showing:
- Panel A: Scatter plot of HAR vs RIVE R² (points above diagonal = RIVE wins)
- Panel B: R² improvement by sector (RIVE - HAR)

Out-of-sample period: 2020-2025 (test: 2023+)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

# Configuration
OUTPUT_DIR = Path("visuals/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# GICS sector mapping
SECTOR_MAP = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
    'GOOGL': 'Communication Services', 'AMZN': 'Consumer Discretionary',
    'TSLA': 'Consumer Discretionary', 'META': 'Communication Services',
    'BRK.B': 'Financials', 'JPM': 'Financials', 'V': 'Financials',
    'JNJ': 'Health Care', 'UNH': 'Health Care', 'XOM': 'Energy',
    'PG': 'Consumer Staples', 'MA': 'Financials', 'HD': 'Consumer Discretionary',
    'CVX': 'Energy', 'ABBV': 'Health Care', 'MRK': 'Health Care',
    'KO': 'Consumer Staples', 'PEP': 'Consumer Staples', 'COST': 'Consumer Staples',
    'AVGO': 'Technology', 'WMT': 'Consumer Staples', 'BAC': 'Financials',
    'DIS': 'Communication Services', 'ORCL': 'Technology', 'CSCO': 'Technology',
    'PFE': 'Health Care', 'ABT': 'Health Care', 'CRM': 'Technology',
    'TMO': 'Health Care', 'NFLX': 'Communication Services', 'ADBE': 'Technology',
    'INTC': 'Technology', 'AMD': 'Technology', 'NKE': 'Consumer Discretionary',
    'QCOM': 'Technology', 'TXN': 'Technology', 'UNP': 'Industrials',
    'UPS': 'Industrials', 'RTX': 'Industrials', 'HON': 'Industrials',
    'LMT': 'Industrials', 'CAT': 'Industrials', 'BA': 'Industrials',
    'GE': 'Industrials', 'DE': 'Industrials', 'MMM': 'Industrials',
    'AXP': 'Financials', 'GS': 'Financials', 'MS': 'Financials',
    'BLK': 'Financials', 'SPGI': 'Financials', 'AEP': 'Utilities',
    'NEE': 'Utilities', 'DUK': 'Utilities', 'AMT': 'Real Estate',
    'BKNG': 'Consumer Discretionary',
}

def load_and_calculate_stock_r2():
    """Calculate per-stock R² for HAR and RIVE models."""
    print("Loading data and calculating per-stock R²...")

    # Load data
    targets = pd.read_parquet("data/processed/targets.parquet")
    targets['date'] = pd.to_datetime(targets['date'])

    # Create HAR features
    df = targets.copy().sort_values(['ticker', 'date']).reset_index(drop=True)

    df["rv_lag_1"] = df.groupby("ticker", observed=True)["realized_vol"].shift(1)
    df["rv_lag_5"] = df.groupby("ticker", observed=True)["realized_vol"].transform(
        lambda x: x.rolling(5).mean()
    ).shift(1)
    df["rv_lag_22"] = df.groupby("ticker", observed=True)["realized_vol"].transform(
        lambda x: x.rolling(22).mean()
    ).shift(1)
    df["returns_sq_lag_1"] = (df["close"].pct_change() ** 2).shift(1)
    df["VIX_close"] = df["VIX_close"].ffill().fillna(15)
    df["rsi_14"] = df["rsi_14"].ffill().fillna(50)

    # Load predictions
    try:
        news_pred = pd.read_parquet("data/processed/news_predictions.parquet")
        news_pred['date'] = pd.to_datetime(news_pred['date'])
        df = pd.merge(df, news_pred[['date', 'ticker', 'news_pred']],
                     on=['date', 'ticker'], how='left')
        df['news_pred'] = df['news_pred'].fillna(0)
    except:
        df['news_pred'] = 0

    try:
        retail_pred = pd.read_parquet("data/processed/retail_predictions.parquet")
        retail_pred['date'] = pd.to_datetime(retail_pred['date'])
        retail_col = 'retail_pred' if 'retail_pred' in retail_pred.columns else 'retail_risk_score'
        df = pd.merge(df, retail_pred[['date', 'ticker', retail_col]],
                     on=['date', 'ticker'], how='left')
        df['retail_pred'] = df[retail_col].fillna(0)
    except:
        df['retail_pred'] = 0

    # Split train/test
    train_cutoff = pd.to_datetime("2023-01-01")
    train_df = df[df["date"] < train_cutoff].copy()
    test_df = df[df["date"] >= train_cutoff].copy()

    # Train HAR model
    tech_features = ["rv_lag_1", "rv_lag_5", "rv_lag_22", "returns_sq_lag_1", "VIX_close", "rsi_14"]
    train_clean = train_df.dropna(subset=tech_features + ["target_log_var"])
    train_clean = train_clean[np.isfinite(train_clean["target_log_var"])]

    har_model = Ridge(alpha=1.0)
    har_model.fit(train_clean[tech_features], train_clean["target_log_var"])

    # Generate predictions
    test_df["har_pred"] = har_model.predict(test_df[tech_features].fillna(0))
    test_df['rive_pred'] = (
        test_df['har_pred'] +
        0.2 * test_df['news_pred'] +
        0.1 * test_df['retail_pred']
    )

    # Clean test data
    test_df = test_df.dropna(subset=['target_log_var', 'har_pred', 'rive_pred'])
    test_df = test_df[
        np.isfinite(test_df['target_log_var']) &
        np.isfinite(test_df['har_pred']) &
        np.isfinite(test_df['rive_pred'])
    ]

    # Calculate per-stock R²
    results = []
    for ticker in test_df['ticker'].unique():
        ticker_data = test_df[test_df['ticker'] == ticker]

        if len(ticker_data) < 50:
            continue

        har_r2 = r2_score(ticker_data['target_log_var'], ticker_data['har_pred'])
        rive_r2 = r2_score(ticker_data['target_log_var'], ticker_data['rive_pred'])

        sector = SECTOR_MAP.get(ticker, 'Other')

        results.append({
            'ticker': ticker,
            'sector': sector,
            'har_r2': har_r2,
            'rive_r2': rive_r2,
            'improvement': rive_r2 - har_r2,
            'n_samples': len(ticker_data)
        })

    results_df = pd.DataFrame(results)

    print(f"  Calculated R² for {len(results_df)} stocks")
    print(f"  HAR mean R²: {results_df['har_r2'].mean():.4f}")
    print(f"  RIVE mean R²: {results_df['rive_r2'].mean():.4f}")

    return results_df

def create_figure(results_df):
    """Create clear two-panel comparison figure."""
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })

    # Professional colors
    color_har = '#E69F00'      # Muted orange
    color_rive = '#0072B2'     # Muted blue
    color_better = '#009E73'   # Muted green (RIVE wins)
    color_worse = '#D55E00'    # Muted red (HAR wins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ================================================================
    # Panel A: Scatter plot - HAR R² vs RIVE R²
    # ================================================================

    # Separate points where RIVE wins vs HAR wins
    rive_wins = results_df[results_df['improvement'] > 0]
    har_wins = results_df[results_df['improvement'] <= 0]

    # Plot diagonal line (parity)
    min_val = min(results_df['har_r2'].min(), results_df['rive_r2'].min()) - 0.05
    max_val = max(results_df['har_r2'].max(), results_df['rive_r2'].max()) + 0.05
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5,
             alpha=0.5, label='Parity (HAR = RIVE)', zorder=1)

    # Plot points
    ax1.scatter(rive_wins['har_r2'], rive_wins['rive_r2'],
               alpha=0.6, s=60, color=color_rive, edgecolor='black', linewidth=0.5,
               label=f'RIVE better (n={len(rive_wins)})', zorder=3)
    ax1.scatter(har_wins['har_r2'], har_wins['rive_r2'],
               alpha=0.6, s=60, color=color_har, edgecolor='black', linewidth=0.5,
               label=f'HAR better (n={len(har_wins)})', zorder=2)

    ax1.set_xlabel('HAR-RV $R^2$')
    ax1.set_ylabel('RIVE $R^2$')
    ax1.set_title('Panel A: Individual Stock Performance')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Set equal aspect and same limits
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax1.set_aspect('equal')

    # Add annotation
    ax1.text(0.98, 0.02,
             'Points above line:\nRIVE outperforms HAR',
             transform=ax1.transAxes, ha='right', va='bottom',
             fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ================================================================
    # Panel B: Sector-level R² improvement (RIVE - HAR)
    # ================================================================

    sector_stats = results_df.groupby('sector').agg({
        'improvement': 'mean',
        'ticker': 'count'
    }).sort_values('improvement', ascending=True)

    # Filter sectors with at least 2 stocks
    sector_stats = sector_stats[sector_stats['ticker'] >= 2]

    y_pos = np.arange(len(sector_stats))
    improvements = sector_stats['improvement'].values

    # Color bars: green if positive, orange if negative
    colors = [color_better if imp > 0 else color_har for imp in improvements]

    ax2.barh(y_pos, improvements, color=colors, edgecolor='black', linewidth=0.5)

    # Add zero line
    ax2.axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.3)

    ax2.set_yticks(y_pos)
    labels = [f"{sector} (n={int(sector_stats.loc[sector, 'ticker'])})"
              for sector in sector_stats.index]
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel('$R^2$ Improvement (RIVE - HAR)')
    ax2.set_title('Panel B: Average Improvement by Sector')
    ax2.grid(True, alpha=0.3, linewidth=0.5, axis='x')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Add annotation
    positive_sectors = (improvements > 0).sum()
    ax2.text(0.98, 0.02,
             f'{positive_sectors}/{len(improvements)} sectors\nshow improvement',
             transform=ax2.transAxes, ha='right', va='bottom',
             fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    return fig

def print_summary_statistics(results_df):
    """Print summary statistics."""
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON SUMMARY")
    print(f"{'='*70}\n")

    print(f"Total stocks: {len(results_df)}")
    print(f"Out-of-sample period: 2023-2025")
    print()

    print("Overall Results:")
    print(f"  HAR-RV mean R²:      {results_df['har_r2'].mean():.4f}")
    print(f"  RIVE mean R²:        {results_df['rive_r2'].mean():.4f}")
    print(f"  Mean improvement:    {results_df['improvement'].mean():.4f}")
    print()

    print("Win Rate:")
    rive_better = (results_df['improvement'] > 0).sum()
    print(f"  RIVE outperforms:    {rive_better}/{len(results_df)} stocks ({rive_better/len(results_df)*100:.1f}%)")
    print()

    print("Sector Performance:")
    sector_stats = results_df.groupby('sector').agg({
        'improvement': ['mean', 'count']
    }).round(4)
    sector_stats.columns = ['Avg Improvement', 'N Stocks']
    sector_stats = sector_stats[sector_stats['N Stocks'] >= 2].sort_values('Avg Improvement', ascending=False)
    print(sector_stats.to_string())

    print(f"\n{'='*70}\n")

def main():
    """Generate Figure 3."""
    print(f"\n{'='*70}")
    print("Figure 3: Performance Comparison (RIVE vs HAR)")
    print(f"{'='*70}\n")

    # Calculate results
    results_df = load_and_calculate_stock_r2()

    # Print statistics
    print_summary_statistics(results_df)

    # Create figure
    print("Generating figure...")
    fig = create_figure(results_df)

    # Save
    pdf_path = OUTPUT_DIR / "fig3_performance_comparison.pdf"
    png_path = OUTPUT_DIR / "fig3_performance_comparison.png"

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\nSaved:")
    print(f"  {pdf_path}")
    print(f"  {png_path}")

    print(f"\n{'='*70}")
    print("Complete.")
    print(f"{'='*70}\n")

    plt.close()

if __name__ == "__main__":
    main()
