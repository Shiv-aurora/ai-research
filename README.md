# Regime-Conditional Conformal Prediction for Realized Volatility

Distribution-free uncertainty quantification for next-day realized volatility forecasting on a large US equity panel, with coverage guarantees that hold **conditionally on market regime** — not just on average.

> **Status: active development.** This project supersedes RIVE (Regime-Integrated Volatility Ensemble), which is frozen in [`archive/rive-2026/`](archive/rive-2026/ARCHIVE_NOTICE.md) with all results reproducible.

## The idea

Online conformal methods (ACI and descendants) deliver *marginal* coverage: right on average over time, but systematically under-covering in stress regimes and over-covering in calm ones — precisely backwards for risk management. This project adds a regime-conditional online calibration layer (soft-Mondrian quantile tracking with filtered regime probabilities) on top of a heterogeneous forecaster pool, yielding per-regime coverage with finite-sample guarantees, one-sided (VaR-grade) variants, and hierarchical pooling across the cross-section.

## Layout

```
src/
  data/          # WRDS TAQ, Risk Lab, FRED, OHLC loaders; RV estimators; CRSP universe
  forecasters/   # HAR family, LightGBM, GRU, TSFM zero-shot, online pool
  regimes/       # state features, online HMM, quantile-bin regimes
  conformal/     # ACI/DtACI/SF-OGD baselines + soft-Mondrian method + panel layer
  eval/          # coverage, width, VaR backtests, DM-HAC, MCS, tables, figures
  experiments/   # config-driven walk-forward runner
  utils/         # MLflow tracker, config, seeding
conf/experiments/  # one YAML per experiment (E0–E12)
scripts/           # build_rv_panel, validate_vs_risklab, run_experiment
paper2/            # manuscript (elsarticle)
docs/              # WRDS query log, data dictionary
archive/rive-2026/ # frozen predecessor project (see ARCHIVE_NOTICE.md)
```

## Data

WRDS TAQ (primary; 5-min realized variance, top-100 point-in-time universe 2005–2025), cross-validated against Chicago Booth Risk Lab QMLE volatilities and VOLARE; macro state variables from FRED; OHLC-based Yang-Zhang estimates for the robustness appendix. No proprietary vendor lock-in; every WRDS query is logged verbatim in `docs/wrds_queries.md`.
