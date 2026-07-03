# Figure 2: Attention-Driven Regime Identification

## Overview

Publication-quality visualization demonstrating attention-driven regimes based on volume anomalies.

## Quick Start

```bash
python visuals/fig2.py
```

Output files:
- `visuals/output/fig2_attention_regimes.pdf` (for LaTeX)
- `visuals/output/fig2_attention_regimes.png` (for presentations)

## Customization

Edit the configuration at the top of `fig2.py`:

```python
TICKER = "AAPL"              # Change to any ticker in your dataset
VOLUME_THRESHOLD = 1.5       # Adjust sensitivity (higher = fewer regimes)
```

Available tickers include: AAPL, AMZN, TSLA, MSFT, etc. (55 total in dataset)

## What the Figure Shows

1. **Top Panel**: Daily closing prices with regime shading
2. **Bottom Panel**: Normalized volume ratio (z-score of Volume/MA₂₀)
3. **Shaded Regions**: High-attention periods where Volume/MA₂₀ > threshold

## Design Principles

- Clean, minimalist design suitable for academic journals
- No annotations, labels, or storytelling elements
- Empirical regime definition (volume-based, reproducible)
- Professional typography and styling

## LaTeX Integration

Use the provided template:

```bash
cat visuals/fig2_latex.tex
```

Then include in your paper with `\input{visuals/fig2_latex.tex}`

## Interpretation

The figure demonstrates:
- **Sustained volume anomalies** (not just noise)
- **Behavioral regime shifts** during attention spikes
- **Why static models struggle** during these periods

This empirically justifies the Retail Regime Agent without overclaiming.
