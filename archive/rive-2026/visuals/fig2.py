"""
Figure 2: Attention-Driven Regime Identification

Publication-quality visualization showing:
- Stock price with attention-regime shading
- Abnormal trading volume (volume / MA20, z-scored)
- High-attention regimes defined by sustained volume shocks
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# ============================================================
# Configuration
# ============================================================
TICKER = "AAPL"                  # Representative stock
VOLUME_THRESHOLD = 1.5           # High-attention threshold
MIN_REGIME_LENGTH = 3            # Minimum consecutive days for regime
OUTPUT_DIR = Path("visuals/output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Data Loading
# ============================================================
def load_stock_data(ticker):
    targets = pd.read_parquet("data/processed/targets.parquet")
    targets['date'] = pd.to_datetime(targets['date'])

    df = targets[targets['ticker'] == ticker].copy()
    df = df.sort_values('date').reset_index(drop=True)

    return df[['date', 'close', 'volume']]


# ============================================================
# Attention Measure
# ============================================================
def compute_attention_measure(df, window=20, threshold=1.5):
    df['vol_ma20'] = df['volume'].rolling(window, min_periods=5).mean()
    df['volume_ratio'] = df['volume'] / df['vol_ma20']

    df['high_attention_raw'] = (df['volume_ratio'] > threshold).astype(int)

    # Debounce regimes: require persistence
    df['high_attention'] = (
        df['high_attention_raw']
        .rolling(MIN_REGIME_LENGTH)
        .sum()
        .fillna(0)
        .ge(MIN_REGIME_LENGTH)
        .astype(int)
    )

    # Z-score volume ratio for visualization
    mean = df['volume_ratio'].mean()
    std = df['volume_ratio'].std()
    df['volume_norm'] = (df['volume_ratio'] - mean) / (std if std > 0 else 1)

    return df


# ============================================================
# Regime Span Detection
# ============================================================
def extract_regime_spans(df):
    spans = []
    in_regime = False
    start_date = None

    for i, row in df.iterrows():
        if row['high_attention'] == 1 and not in_regime:
            in_regime = True
            start_date = row['date']
        elif row['high_attention'] == 0 and in_regime:
            end_date = df.loc[i - 1, 'date']
            spans.append((start_date, end_date))
            in_regime = False

    if in_regime:
        spans.append((start_date, df['date'].iloc[-1]))

    return spans


# ============================================================
# Figure Construction
# ============================================================
def create_figure(df):
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.titlesize': 11
    })

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05}
    )

    regime_spans = extract_regime_spans(df)

    # ------------------------------------------------------------
    # Panel 1: Price
    # ------------------------------------------------------------
    ax1.plot(df['date'], df['close'], color='black', linewidth=0.8)

    for start, end in regime_spans:
        ax1.axvspan(start, end, color='gray', alpha=0.30, zorder=0)

    ax1.set_ylabel('Price')
    ax1.set_title('Price and Attention-Driven Regimes (Representative Stock)')
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.spines[['top', 'right']].set_visible(False)

    # Add legend to clarify shading
    regime_patch = Patch(facecolor='gray', alpha=0.30, label='High-attention regime')
    ax1.legend(handles=[regime_patch], loc='upper left', frameon=False)

    # ------------------------------------------------------------
    # Panel 2: Abnormal Volume
    # ------------------------------------------------------------
    ax2.plot(df['date'], df['volume_norm'], color='#1f77b4', linewidth=0.7)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.axhline(1.5, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)

    for start, end in regime_spans:
        ax2.axvspan(start, end, color='gray', alpha=0.30, zorder=0)

    ax2.set_ylabel('Abnormal Trading Volume\n(z-score)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    ax2.spines[['top', 'right']].set_visible(False)

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.08)

    return fig


# ============================================================
# Main
# ============================================================
def main():
    df = load_stock_data(TICKER)
    df = compute_attention_measure(df, threshold=VOLUME_THRESHOLD)

    fig = create_figure(df)

    pdf_path = OUTPUT_DIR / "fig2_attention_regimes.pdf"
    png_path = OUTPUT_DIR / "fig2_attention_regimes.png"

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved:\n  {pdf_path}\n  {png_path}")


if __name__ == "__main__":
    main()
