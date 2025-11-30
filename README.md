# Titan V15: Multi-Source Volatility Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade volatility forecasting system combining technical analysis, news sentiment, and retail behavior signals. Achieves **61% R²** on high-activity stocks and **22% R²** on S&P 500 blue chips.

## 🏆 Key Results

| Universe | Tickers | Test R² | Baseline | Improvement |
|----------|---------|---------|----------|-------------|
| **High Octane** | 50 | **61.12%** | 55.31% | +5.81% |
| **S&P 500 Leaders** | 55 | **22.44%** | 17.58% | +4.86% |

### Validation Status
- ✅ **Shuffle Test:** -0.46% R² (no leakage)
- ✅ **Walk-Forward:** 32% mean R² across years
- ✅ **Sector Hold-Out:** 34% R² on unseen sectors
- ✅ **Block Bootstrap:** 95% CI [32.6%, 36.7%]
- ✅ **Institutional Grade:** Passes AQR/Two Sigma protocols

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TITAN V15 ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Technical   │  │ News        │  │ Retail              │ │
│  │ Agent       │  │ Classifier  │  │ Regime Agent        │ │
│  │ (HAR-RV)    │  │ (LightGBM)  │  │ (LightGBM)          │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│         ▼                ▼                     ▼            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              TITAN COORDINATOR (Ridge)                  ││
│  │  Features: tech_pred + news_risk + retail_risk +       ││
│  │            calendar + momentum + interactions           ││
│  └─────────────────────────────────────────────────────────┘│
│                           │                                 │
│                           ▼                                 │
│                  [Volatility Forecast]                      │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Shiv-aurora/ai-research.git
cd ai-research

# Install dependencies
pip install -r requirements.txt

# Set up API key
echo 'POLYGON_API_KEY="your_api_key"' > .env
```

### 2. Run Full Pipeline

```bash
# Train on current 18-ticker universe
python scripts/run_full_pipeline.py

# Run optimized Phase 15 version
python scripts/run_final_optimization.py
```

### 3. Scale-Up Experiments

```bash
# Run on 50 high-volatility stocks + 55 S&P 500 leaders
python scripts/scale_up/run_scale_up_experiments.py
```

### 4. Validation Suite

```bash
# Defense audit (shuffle, walk-forward, sector hold-out)
python scripts/validation/audit_defense.py

# Institutional validation (AQR/Two Sigma protocol)
python scripts/validation/institutional_audit.py
```

## 📁 Project Structure

```
titan-volatility-forecasting/
├── config/
│   └── default.yaml              # Configuration
├── src/
│   ├── agents/
│   │   ├── technical_agent.py    # HAR-RV model
│   │   ├── news_agent.py         # News classifier
│   │   └── retail_agent.py       # Retail regime detector
│   ├── coordinator/
│   │   └── fusion.py             # Ridge ensemble
│   ├── pipeline/
│   │   ├── ingest.py             # Data ingestion
│   │   ├── process_news.py       # News processing
│   │   └── deseasonalize.py      # Target transformation
│   └── utils/
│       └── tracker.py            # MLflow wrapper
├── scripts/
│   ├── run_full_pipeline.py      # Main training
│   ├── run_final_optimization.py # Production run
│   ├── scale_up/                 # Multi-universe testing
│   └── validation/               # Validation suite
├── reports/
│   └── figures/                  # Plots and charts
├── archive/                      # Experimental code
└── data/                         # Data files (gitignored)
```

## 📈 Methodology

### Signal Sources

1. **Technical (HAR-RV):** Realized volatility at 1-day, 5-day, 22-day lags
2. **News:** TF-IDF embeddings → PCA → LightGBM classifier for extreme events
3. **Retail:** Volume shock, price acceleration, hype signals

### Key Innovations

- **Winsorization:** Clipping target outliers at 2%/98% percentiles
- **De-seasonalization:** Removing calendar effects from volatility
- **Regime Detection:** Binary classification for high-attention periods
- **Interaction Features:** News × Retail signal interaction

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Ridge α | 100.0 | Strong regularization |
| Winsorization | 2% | Outlier handling |
| News threshold | 80th pct | Extreme event classification |
| Block size | 10 days | Bootstrap robustness |

## 📊 Performance by Sector

### High Octane Universe
| Sector | R² |
|--------|-----|
| Real Estate | 69.48% |
| Health Care | 68.81% |
| Info Tech | 65.98% |
| Industrials | 48.97% |
| Materials | 47.88% |

### S&P 500 Leaders
| Sector | R² |
|--------|-----|
| Consumer Disc | 29.70% |
| Info Tech | 29.13% |
| Materials | 25.52% |
| Utilities | 21.64% |

## 📚 References

- Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility." Journal of Financial Econometrics.
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." Journal of Econometrics.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👤 Author

Shiv Arora - Research Project 2024

