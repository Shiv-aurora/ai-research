# Reproducibility Guide

This document describes how to reproduce the results reported in the RIVE paper.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`
- A [Polygon.io](https://polygon.io/) API key (for market data and news)
- Approximately 20 GB disk space for raw and processed data

## Setup

```bash
pip install -r requirements.txt
echo 'POLYGON_API_KEY="your_key_here"' > .env
```

## Data Acquisition

RIVE uses intraday price data and news articles from Polygon.io covering 2018-01-01 through 2024-12-31. Because this is proprietary financial data accessed via a paid API, we cannot redistribute the raw data. Exact replication requires:

1. A Polygon.io subscription with access to historical aggregates (5-minute bars) and news endpoints
2. Running the ingestion pipeline, which fetches and processes data for the configured ticker universes

The data ingestion pipeline is:

```bash
python scripts/run_full_pipeline.py
```

This fetches price data, computes realized variance targets, ingests news, builds features, and trains all agents.

## Reproducing Main Results

### GICS-55 Universe (Table 4 in paper)

The primary results are on 55 sector-balanced S&P 500 stocks:

```bash
python scripts/scale_up/run_scale_up_experiments.py
```

**Expected output:** RIVE R-squared = 23.04%, HAR-RV-X baseline = 9.62%

### Benchmark Comparisons (GARCH, EWMA, DM tests)

```bash
python audits/february_2026/01_garch_mle_benchmark.py
python audits/february_2026/02_diebold_mariano_test.py
```

**Expected output:** DM statistic = +29.03 (p < 0.001) against HAR-RV-X

### Ablation Studies

```bash
python audits/march_2026/03_gics55_ablation.py
```

**Expected output:** Architecture-level ablation showing linear integration (4.8% R-squared) vs full RIVE (23.0% R-squared)

### Validation Suite

```bash
python scripts/validation/audit_defense.py
python scripts/validation/institutional_audit.py
```

## Random Seeds

- Global seed: 42 (set in `conf/base/config.yaml`)
- Ridge regression and LightGBM use deterministic training given fixed data

## Known Sources of Variation

Results may differ slightly from reported values due to:

- **Data snapshot timing**: Polygon.io data may be revised or updated after initial fetch
- **Floating-point non-determinism**: Minor platform differences in linear algebra routines
- **News coverage gaps**: Historical backfill of news data may have limited sentiment labels for earlier periods

In our testing, R-squared values are stable within +/- 0.1 percentage points across runs on the same data snapshot.

## Audit Trail

The `audits/` directory contains all verification scripts and results from the paper's audit process. See `audits/README.md` for details.
