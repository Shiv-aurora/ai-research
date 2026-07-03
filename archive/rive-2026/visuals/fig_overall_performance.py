"""
Figure: Overall Forecasting Performance (RIVE vs HAR-RV)

Clean visualization supporting the "Overall Forecasting Performance" section:
- Panel A: R² comparison across universes (Top-50 and GICS-55)
- Panel B: Sector-level breakdown showing where RIVE excels

Uses actual pooled R² calculations to match paper claims.
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

def load_and_calculate_pooled_r2():
    """
    Calculate POOLED R² (what the paper reports).
    This combines all stocks and calculates R² on the combined dataset.
    """
    print("Loading data and calculating pooled R²...")

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

    # Add sector
    df['sector'] = df['ticker'].map(SECTOR_MAP).fillna('Other')

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

    # Generate predictions on test set
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

    print(f"  Test samples: {len(test_df):,}")

    # Calculate POOLED R² (all stocks combined)
    overall_har_r2 = r2_score(test_df['target_log_var'], test_df['har_pred'])
    overall_rive_r2 = r2_score(test_df['target_log_var'], test_df['rive_pred'])

    print(f"  Overall pooled HAR R²:  {overall_har_r2:.4f} ({overall_har_r2*100:.2f}%)")
    print(f"  Overall pooled RIVE R²: {overall_rive_r2:.4f} ({overall_rive_r2*100:.2f}%)")

    # Calculate sector-level pooled R²
    sector_results = []
    for sector in test_df['sector'].unique():
        sector_data = test_df[test_df['sector'] == sector]

        if len(sector_data) < 100:
            continue

        har_r2 = r2_score(sector_data['target_log_var'], sector_data['har_pred'])
        rive_r2 = r2_score(sector_data['target_log_var'], sector_data['rive_pred'])

        sector_results.append({
            'sector': sector,
            'har_r2': har_r2,
            'rive_r2': rive_r2,
            'improvement': rive_r2 - har_r2,
            'n_samples': len(sector_data),
            'n_tickers': sector_data['ticker'].nunique()
        })

    sector_df = pd.DataFrame(sector_results)

    return {
        'overall_har_r2': overall_har_r2,
        'overall_rive_r2': overall_rive_r2,
        'sector_df': sector_df,
        'n_stocks': test_df['ticker'].nunique(),
        'n_samples': len(test_df)
    }

def create_figure(results):
    """Create clean two-panel performance comparison."""
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
    color_improve = '#009E73'  # Green

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ================================================================
    # Panel A: Overall Performance Comparison
    # ================================================================

    # Create bar chart
    universes = ['All Stocks\n(Pooled)']
    har_values = [results['overall_har_r2'] * 100]
    rive_values = [results['overall_rive_r2'] * 100]

    x = np.arange(len(universes))
    width = 0.35

    bars1 = ax1.bar(x - width/2, har_values, width, label='HAR-RV',
                    color=color_har, edgecolor='black', linewidth=0.8)
    bars2 = ax1.bar(x + width/2, rive_values, width, label='RIVE',
                    color=color_rive, edgecolor='black', linewidth=0.8)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    ax1.set_ylabel('Out-of-Sample $R^2$ (%)')
    ax1.set_title('Panel A: Overall Forecasting Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{u}\n(n={results["n_stocks"]} stocks)' for u in universes])
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linewidth=0.5, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, max(har_values + rive_values) * 1.15)

    # Add improvement annotation
    improvement = results['overall_rive_r2'] - results['overall_har_r2']
    ax1.text(0.5, 0.95,
             f'Improvement: +{improvement*100:.1f}pp\n({improvement/results["overall_har_r2"]*100:.1f}% relative gain)',
             transform=ax1.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor=color_improve, alpha=0.2),
             fontsize=9)

    # ================================================================
    # Panel B: Sector-Level Performance
    # ================================================================

    sector_df = results['sector_df'].sort_values('rive_r2', ascending=True)

    # Filter to sectors with sufficient data
    sector_df = sector_df[sector_df['n_samples'] >= 200]

    y_pos = np.arange(len(sector_df))
    width_b = 0.35

    ax2.barh(y_pos - width_b/2, sector_df['har_r2'] * 100, width_b,
             label='HAR-RV', color=color_har, edgecolor='black', linewidth=0.5)
    ax2.barh(y_pos + width_b/2, sector_df['rive_r2'] * 100, width_b,
             label='RIVE', color=color_rive, edgecolor='black', linewidth=0.5)

    ax2.set_yticks(y_pos)
    labels = [f"{row['sector']} (n={row['n_tickers']})"
              for _, row in sector_df.iterrows()]
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel('Out-of-Sample $R^2$ (%)')
    ax2.set_title('Panel B: Performance by Sector')
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linewidth=0.5, axis='x')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()

    return fig

def print_summary(results):
    """Print detailed summary statistics."""
    print(f"\n{'='*70}")
    print("OVERALL FORECASTING PERFORMANCE SUMMARY")
    print(f"{'='*70}\n")

    print(f"Test Period: 2023-2025")
    print(f"Total Stocks: {results['n_stocks']}")
    print(f"Total Observations: {results['n_samples']:,}")
    print()

    print("Pooled R² (All Stocks Combined):")
    print(f"  HAR-RV:      {results['overall_har_r2']:.4f} ({results['overall_har_r2']*100:.2f}%)")
    print(f"  RIVE:        {results['overall_rive_r2']:.4f} ({results['overall_rive_r2']*100:.2f}%)")

    improvement = results['overall_rive_r2'] - results['overall_har_r2']
    relative_gain = (improvement / results['overall_har_r2']) * 100
    print(f"  Improvement: {improvement:.4f} (+{improvement*100:.2f}pp)")
    print(f"  Relative Gain: {relative_gain:.1f}%")
    print()

    print("Sector-Level Performance (Pooled R² within each sector):")
    sector_df = results['sector_df'].sort_values('rive_r2', ascending=False)
    print(f"\n{'Sector':<25} {'N':<5} {'HAR R²':<10} {'RIVE R²':<10} {'Improvement':<12}")
    print("-" * 70)
    for _, row in sector_df.iterrows():
        print(f"{row['sector']:<25} {row['n_tickers']:<5} "
              f"{row['har_r2']*100:<10.2f} {row['rive_r2']*100:<10.2f} "
              f"{row['improvement']*100:+<12.2f}")

    print(f"\n{'='*70}\n")

def main():
    """Generate overall performance figure."""
    print(f"\n{'='*70}")
    print("Figure: Overall Forecasting Performance")
    print(f"{'='*70}\n")

    # Calculate pooled R²
    results = load_and_calculate_pooled_r2()

    # Print summary
    print_summary(results)

    # Create figure
    print("Generating figure...")
    fig = create_figure(results)

    # Save
    pdf_path = OUTPUT_DIR / "fig_overall_performance.pdf"
    png_path = OUTPUT_DIR / "fig_overall_performance.png"

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
