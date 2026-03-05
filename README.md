# RIVE: Regime-Integrated Volatility Ensemble

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
<<<<<<< HEAD
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
=======
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

>>>>>>> d718fdcda4602385fcd0c40e9a5269f55127be40

A multi-agent expert system for next-day equity volatility forecasting. RIVE integrates HAR-RV technical dynamics, news-derived risk signals, and retail-attention regime indicators through a ridge-regularized coordinator.

**Paper:** *RIVE: Regime-Integrated Volatility Ensemble* — submitted to Expert Systems with Applications (2026).

---

## Key Results

Out-of-sample evaluation on U.S. equities (benchmark window: 2023--2024):

| Universe | Tickers | RIVE R² | HAR-RV-X Baseline | Improvement | DM Statistic |
|----------|---------|---------|-------------------|-------------|--------------|
| **GICS-55** (sector-balanced) | 55 | **23.04%** | 9.62% | +13.42 pp | +29.03*** |
| **Top-50** (high-liquidity) | 49 | **61.12%** | 55.31% | +5.81 pp | — |

All Diebold-Mariano tests significant at p < 0.001 with Newey-West HAC standard errors.

### Additional Benchmarks (GICS-55)

| Model | R² |
|-------|-----|
| RIVE | **23.04%** |
| HAR-RV-X | 9.62% |
| EWMA | 3.74% |
| GARCH(1,1)-t | -12.44% |
| GJR-GARCH | -48.42% |
| GARCH(1,1)-N | -49.68% |

GARCH models produce negative R² due to a well-documented measurement mismatch: they target conditional variance of returns, while RIVE targets log realized variance from 5-minute bars.

---

## Sector-Level Performance (GICS-55)

| Sector | R² |
|--------|-----|
| Consumer Discretionary | 30.02% |
| Information Technology | 28.39% |
| Materials | 26.07% |
| Utilities | 22.84% |
| Communication Services | 20.84% |
| Financials | 17.10% |
| Industrials | 16.71% |
| Health Care | 13.57% |
| Energy | 13.43% |
| Consumer Staples | 10.06% |
| Real Estate | 2.60% |

<<<<<<< HEAD
All 11 sectors achieve positive R². RIVE outperforms GARCH(1,1) on 53 of 55 tickers (96.4%).
=======
| Test | Result | Description |
|------|--------|-------------|
|  **No Leakage** | ✅ PASS | Shuffle test R² = -0.46% (random noise = no signal) |
|  **Horizon Decay** | ✅ PASS | IC drops from +0.636 (T) → +0.398 (T+10), confirming signal persistence not leakage |
|  **Regime-Conditional** | ✅ PASS | R² ≥ 29.6% in all market regimes (Bull, Bear, High Vol, Low Vol) |
|  **Bootstrap Significance** | ✅ PASS | 95% CI: [32.6%, 36.7%], 5th pct (32.6%) > HAR baseline (15.4%) |
|  **Sector Generalization** | ✅ PASS | 34% R² on held-out sector (Industrial/Consumer) |
|  **Walk-Forward Stability** | ✅ PASS | Mean R² = 32% across years 2021-2024, including 2022 Bear Market |
|  **Random Shuffle** | ✅ PASS | Shuffled target → R² = -0.46% (confirms no spurious correlation) |
|  **Noise Robustness** | ✅ PASS | +50% Gaussian noise on retail signal → only 3.2% R² drop |
|  **Out-of-Universe** | ✅ PASS | Model trained on 12 tickers achieves 18.7% R² on 6 unseen tickers |

### Institutional Validation Summary

```
╔═══════════════════════════════════════════════════════════════╗
║              INSTITUTIONAL VALIDATION REPORT                  ║
╠═══════════════════════════════════════════════════════════════╣
║ Stability Score (Variance of Yearly R²):     0.0089           ║
║ Generalization Score (OOS Ticker R²):        18.7%            ║
║ Fragility Score (Noise Impact):              -3.2%            ║
╠═══════════════════════════════════════════════════════════════╣
║ FINAL ASSESSMENT:  🏆 INSTITUTIONAL GRADE                     ║
║                                                               ║
║ The model meets AQR/Two Sigma validation standards:           ║
║ ✓ Signal persists across prediction horizons                  ║
║ ✓ Performs consistently across market regimes                 ║
║ ✓ Results are statistically significant                       ║
╚═══════════════════════════════════════════════════════════════╝
```
>>>>>>> d718fdcda4602385fcd0c40e9a5269f55127be40

---

## Validation

| Test | Result |
|------|--------|
| Permutation (shuffle) | R² = -9.78% (confirms signal is real) |
| Year-by-year stability | 23.01% (2023), 23.03% (2024) |
| Hold-out tickers (80/20 split) | 22.65% R² on unseen tickers |
| DM significance | p < 0.001 for all benchmark comparisons |
| Data leakage detection | PASS (4 checks) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       RIVE ARCHITECTURE                         │
│          Regime-Integrated Volatility Ensemble                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │ Technical    │  │ News         │  │ Retail                │ │
│  │ Agent        │  │ Classifier   │  │ Regime Agent          │ │
│  │ (HAR-RV)     │  │ (LightGBM)   │  │ (LightGBM)            │ │
│  │              │  │              │  │                       │ │
│  │ Realized Vol │  │ TF-IDF + PCA │  │ Volume/Price Shocks   │ │
│  │ 1d, 5d, 22d  │  │ Extreme Risk │  │ Attention Detection   │ │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬───────────┘ │
│         │                 │                       │             │
│         ▼                 ▼                       ▼             │
│  ┌──────────────────────────────────────────────────────────────┤
│  │              RIVE COORDINATOR (Ridge α=100)                  │
│  │                                                              │
│  │  Features: tech_pred + news_risk + retail_risk +             │
│  │            calendar (M/F/Q4) + momentum (MA5/10, STD5)       │
│  │            + interaction (news × retail)                     │
│  │                                                              │
│  │  Robustness: Winsorization (2%/98%) + Strong Regularization  │
│  └──────────────────────────────────────────────────────────────┤
│                              │                                  │
│                              ▼                                  │
│                    [Volatility Forecast]                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
git clone <repository-url>
cd research-final

pip install -r requirements.txt
echo 'POLYGON_API_KEY="your_key"' > .env
```

### Run Pipeline

```bash
# Train RIVE (full pipeline: ingest → agents → coordinator)
python scripts/run_full_pipeline.py

# Scale-up experiments (Top-50 + GICS-55 universes)
python scripts/scale_up/run_scale_up_experiments.py
```

### Validation

```bash
python scripts/validation/audit_defense.py
python scripts/validation/institutional_audit.py
```

---

## Project Structure

```
rive/
├── src/                            # Core source code
│   ├── agents/
│   │   ├── technical_agent.py      # HAR-RV model (Ridge regression)
│   │   ├── news_agent.py           # News classifier (LightGBM)
│   │   └── retail_agent.py         # Retail regime detector (LightGBM)
│   ├── coordinator/
│   │   └── fusion.py               # RiveCoordinator (Ridge α=100)
│   ├── pipeline/
│   │   ├── ingest.py               # Price data ingestion (Polygon.io)
│   │   ├── ingest_news.py          # News data ingestion
│   │   ├── process_news.py         # TF-IDF + PCA processing
│   │   ├── ingest_retail.py        # Retail signal construction
│   │   └── deseasonalize.py        # Calendar adjustment
│   └── utils/
│       └── tracker.py              # MLflow experiment tracking
│
├── scripts/
│   ├── run_full_pipeline.py        # Main training pipeline
│   ├── run_final_optimization.py   # Production run
│   ├── scale_up/                   # Multi-universe experiments
│   └── validation/                 # Validation and audit suite
│
├── audits/                         # Audit trail (Jan–Mar 2026)
│   ├── january_2026/               # Data leakage, R² verification
│   ├── february_2026/              # GARCH MLE benchmarks, DM tests
│   └── march_2026/                 # Sector recomputation, ablation
│
├── paper/
│   └── figures/                    # Publication figures
│
├── reports/
│   └── scale_up/                   # Scale-up experiment results
│
├── archive/                        # Deprecated experimental code
│   ├── experiments/
│   ├── audits/
│   ├── legacy/
│   └── results/
│
├── conf/base/config.yaml           # Configuration
├── requirements.txt                # Python dependencies
├── REPRODUCIBILITY.md              # Reproduction instructions
├── CITATION.cff                    # Citation metadata
└── LICENSE                         # MIT License
```

---

## Methodology

### Signal Sources

| Signal | Model | Features | Output |
|--------|-------|----------|--------|
| Technical | Ridge (HAR-RV) | RV at 1d, 5d, 22d lags + VIX + RSI | Continuous prediction |
| News | LightGBM Classifier | TF-IDF embeddings, PCA, decay-weighted sentiment | Risk score (0-1) |
| Retail | LightGBM Classifier | Volume shock, price acceleration, attention proxy | Risk score (0-1) |

### Key Design Choices

1. **Classification for sparse signals**: News and retail agents predict extreme events (top 20% volatility) rather than continuous values
2. **Regime-aware integration**: The coordinator's calendar, momentum, and interaction features enable regime-conditional weighting
3. **Strong regularization**: Ridge alpha=100 with 2%/98% winsorization prevents overfitting
4. **Transparent combination**: Linear coordinator enables direct forecast attribution to each agent

---

## Citation

```bibtex
@article{arora2026rive,
  title={RIVE: Regime-Integrated Volatility Ensemble},
  author={Arora, Shivam and Margapuri, Venkat},
  journal={Expert Systems with Applications},
  year={2026},
  note={Submitted}
}
```

---

## License

<<<<<<< HEAD
MIT License. See [LICENSE](LICENSE) for details.
=======
MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Shivam Arora**  2025

---

## 🙏 Acknowledgments

- [Polygon.io](https://polygon.io/) for market data API
- [MLflow](https://mlflow.org/) for experiment tracking
- Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of Financial Econometrics*.
>>>>>>> d718fdcda4602385fcd0c40e9a5269f55127be40
