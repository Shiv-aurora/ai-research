"""
Figure: R² Performance Across Stocks and Sectors

Simple empirical comparison showing:
- Panel A: Distribution of R² across individual stocks
- Panel B: Average R² by GICS sector

Clean, grayscale-friendly, publication-ready.
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

# GICS sector mapping for top stocks
SECTOR_MAP = {
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'NVDA': 'Technology',
    'GOOGL': 'Communication',
    'AMZN': 'Consumer Discretionary',
    'TSLA': 'Consumer Discretionary',
    'META': 'Communication',
    'BRK.B': 'Financials',
    'JPM': 'Financials',
    'V': 'Financials',
    'JNJ': 'Health Care',
    'UNH': 'Health Care',
    'XOM': 'Energy',
    'PG': 'Consumer Staples',
    'MA': 'Financials',
    'HD': 'Consumer Discretionary',
    'CVX': 'Energy',
    'ABBV': 'Health Care',
    'MRK': 'Health Care',
    'KO': 'Consumer Staples',
    'PEP': 'Consumer Staples',
    'COST': 'Consumer Staples',
    'AVGO': 'Technology',
    'WMT': 'Consumer Staples',
    'BAC': 'Financials',
    'DIS': 'Communication',
    'ORCL': 'Technology',
    'CSCO': 'Technology',
    'PFE': 'Health Care',
    'ABT': 'Health Care',
    'CRM': 'Technology',
    'TMO': 'Health Care',
    'NFLX': 'Communication',
    'ADBE': 'Technology',
    'INTC': 'Technology',
    'AMD': 'Technology',
    'NKE': 'Consumer Discretionary',
    'QCOM': 'Technology',
    'TXN': 'Technology',
    'UNP': 'Industrials',
    'UPS': 'Industrials',
    'RTX': 'Industrials',
    'HON': 'Industrials',
    'LMT': 'Industrials',
    'CAT': 'Industrials',
    'BA': 'Industrials',
    'GE': 'Industrials',
    'DE': 'Industrials',
    'MMM': 'Industrials',
    'AXP': 'Financials',
    'GS': 'Financials',
    'MS': 'Financials',
    'BLK': 'Financials',
    'SPGI': 'Financials',
    'AEP': 'Utilities',
    'NEE': 'Utilities',
    'DUK': 'Utilities',
    'AMT': 'Real Estate',
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

        if len(ticker_data) < 50:  # Skip stocks with insufficient data
            continue

        har_r2 = r2_score(ticker_data['target_log_var'], ticker_data['har_pred'])
        rive_r2 = r2_score(ticker_data['target_log_var'], ticker_data['rive_pred'])

        # Get sector
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
    print(f"  Mean improvement: {results_df['improvement'].mean():.4f}")

    return results_df

def create_figure(results_df):
    """Create two-panel R² comparison figure."""
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })

    # Professional, muted colors (colorblind-friendly)
    color_har = '#E69F00'   # Muted orange
    color_rive = '#0072B2'  # Muted blue

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ================================================================
    # Panel A: Distribution of R² across stocks
    # ================================================================
    bins = np.linspace(-0.2, 0.8, 30)

    ax1.hist(results_df['har_r2'], bins=bins, alpha=0.6, color=color_har,
             label='HAR-RV', edgecolor='black', linewidth=0.5)
    ax1.hist(results_df['rive_r2'], bins=bins, alpha=0.6, color=color_rive,
             label='RIVE', edgecolor='black', linewidth=0.5)

    ax1.axvline(results_df['har_r2'].mean(), color=color_har,
                linestyle='--', linewidth=2, label=f'HAR mean: {results_df["har_r2"].mean():.3f}')
    ax1.axvline(results_df['rive_r2'].mean(), color=color_rive,
                linestyle='--', linewidth=2, label=f'RIVE mean: {results_df["rive_r2"].mean():.3f}')

    ax1.set_xlabel('$R^2$')
    ax1.set_ylabel('Number of Stocks')
    ax1.set_title('Panel A: Distribution of $R^2$ Across Stocks')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linewidth=0.5, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ================================================================
    # Panel B: Average R² by sector
    # ================================================================
    sector_stats = results_df.groupby('sector').agg({
        'har_r2': 'mean',
        'rive_r2': 'mean',
        'ticker': 'count'
    }).sort_values('rive_r2', ascending=False)

    # Filter sectors with at least 2 stocks
    sector_stats = sector_stats[sector_stats['ticker'] >= 2]

    x = np.arange(len(sector_stats))
    width = 0.35

    ax2.barh(x - width/2, sector_stats['har_r2'], width,
             label='HAR-RV', color=color_har, edgecolor='black', linewidth=0.5)
    ax2.barh(x + width/2, sector_stats['rive_r2'], width,
             label='RIVE', color=color_rive, edgecolor='black', linewidth=0.5)

    ax2.set_yticks(x)
    ax2.set_yticklabels(sector_stats.index, fontsize=9)
    ax2.set_xlabel('Average $R^2$')
    ax2.set_title('Panel B: Average $R^2$ by Sector')
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linewidth=0.5, axis='x')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Add n counts to sector labels
    labels = [f"{sector} (n={int(sector_stats.loc[sector, 'ticker'])})"
              for sector in sector_stats.index]
    ax2.set_yticklabels(labels, fontsize=8)

    plt.tight_layout()

    return fig

def print_summary_statistics(results_df):
    """Print summary statistics."""
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}\n")

    print(f"Total stocks analyzed: {len(results_df)}")
    print(f"Test period: 2023-01-01 onwards")
    print()

    print("Overall Performance:")
    print(f"  HAR-RV mean R²:   {results_df['har_r2'].mean():.4f} (σ = {results_df['har_r2'].std():.4f})")
    print(f"  RIVE mean R²:     {results_df['rive_r2'].mean():.4f} (σ = {results_df['rive_r2'].std():.4f})")
    print(f"  Mean improvement: {results_df['improvement'].mean():.4f}")
    print()

    print("Per-Stock Performance:")
    print(f"  RIVE better than HAR: {(results_df['improvement'] > 0).sum()} stocks ({(results_df['improvement'] > 0).mean()*100:.1f}%)")
    print(f"  HAR better than RIVE: {(results_df['improvement'] < 0).sum()} stocks ({(results_df['improvement'] < 0).mean()*100:.1f}%)")
    print()

    print("Sector-Level Performance:")
    sector_stats = results_df.groupby('sector').agg({
        'har_r2': 'mean',
        'rive_r2': 'mean',
        'improvement': 'mean',
        'ticker': 'count'
    }).sort_values('rive_r2', ascending=False)

    sector_stats = sector_stats[sector_stats['ticker'] >= 2]

    print(f"\n{'Sector':<25} {'N':<5} {'HAR R²':<10} {'RIVE R²':<10} {'Δ R²':<10}")
    print("-" * 70)
    for sector in sector_stats.index:
        n = int(sector_stats.loc[sector, 'ticker'])
        har = sector_stats.loc[sector, 'har_r2']
        rive = sector_stats.loc[sector, 'rive_r2']
        imp = sector_stats.loc[sector, 'improvement']
        print(f"{sector:<25} {n:<5} {har:<10.4f} {rive:<10.4f} {imp:+<10.4f}")

    print(f"\n{'='*70}\n")

def main():
    """Generate stock and sector R² comparison figure."""
    print(f"\n{'='*70}")
    print("Figure: R² Performance Across Stocks and Sectors")
    print(f"{'='*70}\n")

    # Calculate per-stock R²
    results_df = load_and_calculate_stock_r2()

    # Print statistics
    print_summary_statistics(results_df)

    # Create figure
    print("Generating figure...")
    fig = create_figure(results_df)

    # Save outputs
    pdf_path = OUTPUT_DIR / "fig_stock_sector_r2.pdf"
    png_path = OUTPUT_DIR / "fig_stock_sector_r2.png"

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\nSaved:")
    print(f"  {pdf_path}")
    print(f"  {png_path}")

    print(f"\n{'='*70}")
    print("Figure generation complete.")
    print(f"{'='*70}\n")

    plt.close()

if __name__ == "__main__":
    main()
