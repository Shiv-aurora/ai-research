# RIVE: Regime-Integrated Volatility Ensemble

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modular machine learning system for next-day equity volatility forecasting. RIVE decomposes the forecasting problem into specialized learners — technical dynamics (HAR-RV), news-driven extreme events (LightGBM classifier), and retail-attention regimes (LightGBM classifier) — fused through a ridge-regularized coordinator. The modular design is compared against monolithic ML baselines (LightGBM regressor, Elastic Net) on identical feature sets, demonstrating competitive accuracy with superior temporal stability.

**Paper:** *RIVE: Regime-Integrated Volatility Ensemble* — submitted to Machine Learning with Applications (2026).

---

## Key Results

### Fixed-Split Evaluation (Train < 2023, Test 2023–2024)

| Model | Test R² | RMSE | DM vs RIVE (MSE) |
|-------|---------|------|-------------------|
| Elastic Net (linear monolithic) | 23.57% | 1.2025 | p = 0.0002 |
| LightGBM (nonlinear monolithic) | 23.29% | 1.2047 | p = 0.39 (n.s.) |
| **RIVE** (modular ensemble) | **22.81%** | **1.2085** | — |
| HAR-RV-X | 9.62% | — | p < 0.001 |
| EWMA | 3.74% | — | p < 0.001 |

Fixed-split accuracy differences between RIVE and the monolithic LightGBM are **not statistically significant** (Diebold-Mariano test, p = 0.39). All three ML models significantly outperform traditional volatility benchmarks.

### Walk-Forward Evaluation (12 Quarterly Folds, Expanding Window)

| Model | Mean Fold R² | Std | Min Fold | Max Fold | Folds > 0 |
|-------|-------------|-----|----------|----------|-----------|
| **RIVE** | **19.39%** | **0.099** | -9.1% | 31.5% | **11/12** |
| Elastic Net | 15.18% | 0.168 | -35.0% | 27.6% | 11/12 |
| LightGBM | 9.05% | 0.562 | -164.9% | 40.2% | 10/12 |

RIVE exhibits the **lowest fold-to-fold variance** (std = 0.099 vs 0.562 for LightGBM). During the 2022-Q2 rate-hiking shock, LightGBM collapses to R² = -165% while RIVE's worst fold is -9%.

### Tail-Event Detection (Top Decile)

| Model | ROC-AUC | Severe Underprediction Rate |
|-------|---------|---------------------------|
| **RIVE** | **0.8051** | **14.9%** |
| LightGBM | 0.8035 | 15.6% |
| Elastic Net | 0.8004 | 16.1% |

RIVE best identifies extreme volatility days — the core use case its architecture targets.

### Cross-Sectional Generalization (Leave-One-Sector-Out)

All three models achieve positive R² across all 6 held-out sectors. RIVE has the lowest cross-sector variance (std = 0.080), confirming that the modular design generalizes rather than memorizing ticker-specific patterns.

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

## Experimental Validation

| Test | Result |
|------|--------|
| Diebold-Mariano (RIVE vs HAR-RV-X) | DM = +29.03, p < 0.001 |
| Diebold-Mariano (RIVE vs LightGBM) | DM = -0.87, p = 0.39 (n.s.) |
| Permutation (shuffle) test | R² = -9.78% (confirms signal is real) |
| Walk-forward stability (12 folds) | RIVE std = 0.099 (lowest of all models) |
| Leave-one-sector-out | R² > 0 in all 6 held-out sectors |
| Mincer-Zarnowitz calibration | All models reject H0 (mild miscalibration) |
| Feature-budget matching | 100% Jaccard similarity (fair comparison) |
| Data leakage detection | PASS (4 checks) |

---

## Quick Start

### Installation

```bash
git clone https://github.com/Shiv-aurora/ai-research.git
cd ai-research

pip install -r requirements.txt
echo 'POLYGON_API_KEY="your_key"' > .env
```

### Run RIVE Pipeline

```bash
# Train RIVE (full pipeline: ingest → agents → coordinator)
python scripts/run_full_pipeline.py

# Scale-up experiments (Top-50 + GICS-55 universes)
python scripts/scale_up/run_scale_up_experiments.py
```

### Run ML Baseline Comparisons

```bash
# All MLWA experiments (LightGBM, Elastic Net, walk-forward, DM tests, etc.)
python scripts/mlwa_experiments/run_all.py

# Individual experiments
python scripts/mlwa_experiments/exp1_monolithic_lightgbm.py
python scripts/mlwa_experiments/exp2_elastic_net.py
python scripts/mlwa_experiments/exp3_rolling_walkforward.py
python scripts/mlwa_experiments/exp4_dm_tests_ml.py
python scripts/mlwa_experiments/exp5_tail_metrics.py
python scripts/mlwa_experiments/exp6_sector_generalization.py
python scripts/mlwa_experiments/exp7_walkforward_table.py
python scripts/mlwa_experiments/exp8_calibration.py
python scripts/mlwa_experiments/exp9_feature_budget_check.py
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
│   ├── validation/                 # Validation and audit suite
│   └── mlwa_experiments/           # ML baseline comparisons
│       ├── run_all.py              # Master runner
│       ├── assemble_features.py    # Shared feature assembly
│       ├── exp1_monolithic_lightgbm.py
│       ├── exp2_elastic_net.py
│       ├── exp3_rolling_walkforward.py
│       ├── exp4_dm_tests_ml.py
│       ├── exp5_tail_metrics.py
│       ├── exp6_sector_generalization.py
│       ├── exp7_walkforward_table.py
│       ├── exp8_calibration.py
│       └── exp9_feature_budget_check.py
│
├── audits/                         # Audit trail (Jan–Mar 2026)
│   ├── january_2026/               # Data leakage, R² verification
│   ├── february_2026/              # GARCH MLE benchmarks, DM tests
│   └── march_2026/                 # Sector recomputation, ablation
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
| News | LightGBM Classifier | TF-IDF embeddings, PCA, decay-weighted sentiment | Risk score (0–1) |
| Retail | LightGBM Classifier | Volume shock, price acceleration, attention proxy | Risk score (0–1) |

### Key Design Choices

1. **Modular decomposition**: Each agent specializes on a distinct signal source, enabling transparent forecast attribution
2. **Classification for sparse signals**: News and retail agents predict extreme events (top 20% volatility) rather than continuous values
3. **Regime-aware integration**: Calendar, momentum, and interaction features enable regime-conditional weighting
4. **Strong regularization**: Ridge α=100 with 2%/98% winsorization prevents overfitting
5. **Fair benchmarking**: Monolithic baselines use the exact same 47-feature information set (100% Jaccard match)

---

## Citation

```bibtex
@article{arora2026rive,
  title={RIVE: Regime-Integrated Volatility Ensemble},
  author={Arora, Shivam and Margapuri, Venkat},
  journal={Machine Learning with Applications},
  year={2026},
  note={Submitted}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
