# Regime-Conditional Conformal Prediction for Realized Volatility

Distribution-free uncertainty quantification for next-day realized volatility forecasting on a large US equity panel, with coverage that holds **conditionally on market regime** — not just on average.

> **Status: experiments complete (E0–E12), paper phase pending.**
> Start here: [`NEXT_STEPS.md`](NEXT_STEPS.md) (project state, handoff, work queue) and [`docs/RESULTS.md`](docs/RESULTS.md) (every finding with numbers and artifacts).
> This project supersedes RIVE, which is frozen in [`archive/rive-2026/`](archive/rive-2026/ARCHIVE_NOTICE.md) with all results reproducible.

## The idea

Online conformal methods (ACI and descendants) deliver *marginal* coverage: right on average over time, but systematically under-covering in stress regimes (84% when 90% is promised; 89% upper tail when 95% is promised) and over-covering in calm ones — precisely backwards for risk management. We add a regime-conditional online calibration layer on top of any point forecaster: pooled per-regime thresholds over the cross-section (rare regimes get ~100× the updates a single stock provides), DtACI-style adaptive learning rates per regime with an issued-interval corrector (zero tuned parameters), and one-sided heads for VaR. The same layer repairs a zero-shot foundation model (Chronos), whose native bands point the wrong way in crises. One honest boundary: the second day of a volatility spike stays under-covered no matter what we condition on — and option prices don't anticipate it either.

Point-forecast accuracy is deliberately held fixed (the pool ties HAR/LightGBM); the contribution is that the uncertainty statements become trustworthy, evaluated over ~394k out-of-sample stock-days, 7 methods, MCS, crisis episodes, and a disjoint-name holdout.

## Reproduce everything

```bash
.venv/bin/python scripts/run_all.py     # every table, dependency order, ~1-2h
bash scripts/run_tests.sh               # 68 tests, OpenMP-isolated groups
```

Per-experiment scripts are `scripts/e*.py`; each is one command and writes CSVs to `reports/` (logs to `reports/logs/`).

## Layout

```
NEXT_STEPS.md      # project state + handoff + ordered work queue (READ FIRST)
docs/              # RESULTS.md (findings catalog), data dictionary, WRDS query log
src/
  data/            # Risk Lab / FRED / CBOE / OHLC loaders, RV estimators, universe,
                   # panel assembly (identity audits, dedup, staleness guards)
  forecasters/     # HAR, LightGBM, GRU, online Hedge pool, GARCH-t/CAViaR/HAR-QREG
  regimes/         # online (filtered) Gaussian HMM, quantile-bin regimes
  conformal/       # ACI / DtACI / SF-OGD / KNN-state baselines,
                   # soft-Mondrian + panel_hierarchical.py (THE method),
                   # adaptive rates + corrector
  eval/            # coverage-by-state, VaR backtests (Kupiec/Christoffersen/DQ),
                   # panel DM-HAC, Model Confidence Set + interval score
  experiments/     # walk-forward engine (expanding window, leakage assertion)
  utils/           # config, seeding, process-pool pmap
scripts/           # build_rv_panel.py, e0..e12 experiments, run_all.py, run_tests.sh
reports/           # results tables (CSV) + logs/ (run logs)
paper2/            # manuscript skeleton (IJF elsarticle) — pending
conf/base/         # config.yaml (universe, alpha, eval windows, seeds)
data/processed_v2/ # rv_panel.parquet, e0_predictions.parquet, tsfm_predictions.parquet
archive/rive-2026/ # frozen predecessor (rejected MLWA version + honest leakage notice)
mlruns/, data/processed/  # legacy RIVE artifacts, untouched by policy
```

## Data

Universe: **CRSP point-in-time top-100 by market cap**, rebalanced each January 2005–2025 including later-delisted names (223 companies, 519,843 stock-days; built by `scripts/build_pit_universe.py` via the WRDS API, queries logged in `docs/wrds_queries.md`) — survivorship-bias-free by construction, identity keyed by PERMNO end-to-end. Realized volatilities: Chicago Booth **Risk Lab** (TAQ-based, fetched by PERMNO). State variables: **FRED** macro series (with silent-truncation guards) and **CBOE** VIX9D/VVIX term-structure signals. Column definitions: `docs/data_dictionary.md`.

## Engineering notes

- **macOS OpenMP rule:** torch and lightgbm each bundle libomp and segfault if imported in one process — tests and scripts keep them isolated (`scripts/run_tests.sh`, `tests/torch_isolated/`).
- Per-stock/per-config loops run across cores via `src/utils/parallel.py`.
- Nothing is ever deleted; superseded work is archived via `git mv`.
