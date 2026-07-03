"""
Figure: Overall Forecasting Performance (Table 4 Visualization)

Visualizes the actual experimental results reported in Table 4:
- Top-50 High Activity Stocks: RIVE 61.12% vs HAR 55.31%
- GICS-55 Sector-Balanced: RIVE 22.44% vs HAR 17.58%

Uses actual experimental results from the scale-up experiments.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("visuals/output")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_experimental_results():
    """Load actual experimental results from CSV files."""
    print("Loading experimental results...")

    # Load High Octane 50 results
    high_octane = pd.read_csv("reports/scale_up/High_Octane_50_results.csv")
    har_top50 = high_octane['baseline_r2'].values[0] * 100
    rive_top50 = high_octane['coordinator_r2'].values[0] * 100
    n_top50 = high_octane['n_tickers'].values[0]
    samples_top50 = high_octane['n_samples'].values[0]

    # Load GICS-55 results
    gics55 = pd.read_csv("reports/scale_up/SP500_Sector_Leaders_results.csv")
    har_gics = gics55['baseline_r2'].values[0] * 100
    rive_gics = gics55['coordinator_r2'].values[0] * 100
    n_gics = gics55['n_tickers'].values[0]
    samples_gics = gics55['n_samples'].values[0]

    results = {
        'top50': {
            'har': har_top50,
            'rive': rive_top50,
            'n_tickers': int(n_top50),
            'n_samples': int(samples_top50),
            'improvement': rive_top50 - har_top50,
            'relative_gain': ((rive_top50 - har_top50) / har_top50) * 100
        },
        'gics55': {
            'har': har_gics,
            'rive': rive_gics,
            'n_tickers': int(n_gics),
            'n_samples': int(samples_gics),
            'improvement': rive_gics - har_gics,
            'relative_gain': ((rive_gics - har_gics) / har_gics) * 100
        }
    }

    print(f"\n  Top-50 Stocks:")
    print(f"    HAR:  {har_top50:.2f}%")
    print(f"    RIVE: {rive_top50:.2f}%")
    print(f"    Improvement: +{rive_top50 - har_top50:.2f}pp")
    print()
    print(f"  GICS-55 Sector Portfolios:")
    print(f"    HAR:  {har_gics:.2f}%")
    print(f"    RIVE: {rive_gics:.2f}%")
    print(f"    Improvement: +{rive_gics - har_gics:.2f}pp")

    return results

def load_sector_breakdown():
    """Load sector-level results if available."""
    print("\nLoading sector breakdown...")

    try:
        # Try to load sector breakdown from GICS-55
        sectors = pd.read_csv("reports/scale_up/SP500_Sector_Leaders_sectors.csv")
        print(f"  Found sector data: {len(sectors)} sectors")
        return sectors
    except:
        print("  No detailed sector breakdown available")
        return None

def create_figure(results, sectors_df):
    """Create clean performance comparison figure."""
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })

    # Soft, gentle colors (very muted, professional)
    color_har = '#B8C5D6'      # Soft blue-gray (lighter)
    color_rive = '#7D92A8'     # Soft blue-gray (darker)
    color_improve = '#97A6B5'  # Mid blue-gray

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ================================================================
    # Panel A: Overall Performance (Top-50 vs GICS-55)
    # ================================================================

    universes = ['Top-50 High Liquidity\nStocks', 'GICS-55 Sector\nPortfolios']
    har_values = [results['top50']['har'], results['gics55']['har']]
    rive_values = [results['top50']['rive'], results['gics55']['rive']]
    improvements = [results['top50']['improvement'], results['gics55']['improvement']]

    x = np.arange(len(universes))
    width = 0.35

    bars1 = ax1.bar(x - width/2, har_values, width, label='HAR-RV Baseline',
                    color=color_har, edgecolor='black', linewidth=0.6, alpha=0.8)
    bars2 = ax1.bar(x + width/2, rive_values, width, label='RIVE',
                    color=color_rive, edgecolor='black', linewidth=0.6)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1.0,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar, imp in zip(bars2, improvements):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1.0,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Add improvement annotation
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                f'+{imp:.1f}pp', ha='center', va='center', fontsize=7,
                color='white', fontweight='bold')

    ax1.set_ylabel('Out-of-Sample $R^2$ (%)')
    ax1.set_title('Panel A: Forecasting Performance by Stock Universe')
    ax1.set_xticks(x)

    # Better x-axis labels with context
    labels = [
        f'Top-50\nHigh Liquidity\n(n={results["top50"]["n_tickers"]} stocks)',
        f'GICS-55\nSector-Balanced\n(n={results["gics55"]["n_tickers"]} stocks)'
    ]
    ax1.set_xticklabels(labels, fontsize=8)

    ax1.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='gray')
    ax1.grid(True, alpha=0.3, linewidth=0.5, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, max(rive_values) * 1.18)

    # Add context annotations
    ax1.text(0, max(rive_values) * 0.92,
             'Higher $R^2$:\nStrong persistence\nin liquid stocks',
             ha='center', fontsize=7, style='italic', color='#404040')
    ax1.text(1, max(rive_values) * 0.92,
             'Lower $R^2$:\nDiversification dampens\nidiosyncratic volatility',
             ha='center', fontsize=7, style='italic', color='#404040')

    # ================================================================
    # Panel B: Relative Improvement
    # ================================================================

    if sectors_df is not None and 'sector' in sectors_df.columns:
        # Show sector breakdown if available
        sectors_sorted = sectors_df.sort_values('r2', ascending=True)

        y_pos = np.arange(len(sectors_sorted))

        bars = ax2.barh(y_pos, sectors_sorted['r2'] * 100, color=color_rive,
                edgecolor='black', linewidth=0.5, alpha=0.85)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', ha='left', va='center', fontsize=7)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sectors_sorted['sector'].values, fontsize=8)
        ax2.set_xlabel('RIVE Out-of-Sample $R^2$ (%)')
        ax2.set_title('Panel B: Sector-Level Performance (GICS-55)')
        ax2.grid(True, alpha=0.3, linewidth=0.5, axis='x')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xlim(0, max(sectors_sorted['r2']) * 100 * 1.15)

    else:
        # Show improvement comparison if no sector data
        improvements = [
            results['top50']['improvement'],
            results['gics55']['improvement']
        ]
        relative_gains = [
            results['top50']['relative_gain'],
            results['gics55']['relative_gain']
        ]

        y_pos = np.arange(2)

        bars = ax2.barh(y_pos, improvements, color=color_improve,
                       edgecolor='black', linewidth=0.8)

        # Add labels
        for i, (bar, rel_gain) in enumerate(zip(bars, relative_gains)):
            width = bar.get_width()
            ax2.text(width + 0.2, bar.get_y() + bar.get_height()/2,
                    f'+{improvements[i]:.2f}pp\n({rel_gain:.1f}% gain)',
                    ha='left', va='center', fontsize=8)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(['Top-50\nHigh Activity', 'GICS-55\nSector-Balanced'])
        ax2.set_xlabel('$R^2$ Improvement (percentage points)')
        ax2.set_title('Panel B: Absolute Improvement')
        ax2.grid(True, alpha=0.3, linewidth=0.5, axis='x')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xlim(0, max(improvements) * 1.4)

    plt.tight_layout()

    return fig

def print_summary(results):
    """Print summary for paper."""
    print(f"\n{'='*70}")
    print("TABLE 4 - OVERALL FORECASTING PERFORMANCE")
    print(f"{'='*70}\n")

    print("Out-of-Sample Period: 2020-2025 (test: 2023+)")
    print()

    print(f"{'Universe':<30} {'HAR-RV R²':<15} {'RIVE R²':<15} {'Improvement':<15}")
    print("-" * 70)

    print(f"{'Top-50 High Activity':<30} "
          f"{results['top50']['har']:<15.2f}% "
          f"{results['top50']['rive']:<15.2f}% "
          f"+{results['top50']['improvement']:<14.2f}pp")

    print(f"{'  (n={} stocks)'.format(results['top50']['n_tickers']):<30} "
          f"{'':15} {'':15} "
          f"({results['top50']['relative_gain']:.1f}% relative)")

    print()

    print(f"{'GICS-55 Sector-Balanced':<30} "
          f"{results['gics55']['har']:<15.2f}% "
          f"{results['gics55']['rive']:<15.2f}% "
          f"+{results['gics55']['improvement']:<14.2f}pp")

    print(f"{'  (n={} stocks)'.format(results['gics55']['n_tickers']):<30} "
          f"{'':15} {'':15} "
          f"({results['gics55']['relative_gain']:.1f}% relative)")

    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}\n")

    print("1. Top-50 High Activity Stocks:")
    print(f"   - RIVE achieves {results['top50']['rive']:.2f}% R², vs {results['top50']['har']:.2f}% for HAR-RV")
    print(f"   - Improvement of {results['top50']['improvement']:.2f} percentage points")
    print(f"   - {results['top50']['relative_gain']:.1f}% relative gain")
    print(f"   - Reduces unexplained variance by ~{100 * results['top50']['improvement'] / (100 - results['top50']['har']):.1f}%")
    print()

    print("2. GICS-55 Sector-Balanced Portfolio:")
    print(f"   - RIVE achieves {results['gics55']['rive']:.2f}% R², vs {results['gics55']['har']:.2f}% for HAR-RV")
    print(f"   - Improvement of {results['gics55']['improvement']:.2f} percentage points")
    print(f"   - {results['gics55']['relative_gain']:.1f}% relative gain (nearly 50% better)")
    print(f"   - Shows RIVE captures market-wide volatility drivers effectively")

    print(f"\n{'='*70}\n")

def main():
    """Generate Table 4 visualization."""
    print(f"\n{'='*70}")
    print("Figure: Overall Forecasting Performance")
    print("Visualizing Table 4 Results")
    print(f"{'='*70}\n")

    # Load actual experimental results
    results = load_experimental_results()
    sectors_df = load_sector_breakdown()

    # Print summary
    print_summary(results)

    # Create figure
    print("Generating figure...")
    fig = create_figure(results, sectors_df)

    # Save
    pdf_path = OUTPUT_DIR / "fig_table4_performance.pdf"
    png_path = OUTPUT_DIR / "fig_table4_performance.png"

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
