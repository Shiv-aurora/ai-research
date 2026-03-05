# Audit Trail

This directory contains all verification and validation scripts used to audit the claims in the RIVE paper prior to journal submission. Each subdirectory corresponds to a round of auditing.

## January 2026

Initial validation suite covering data leakage detection, R-squared verification, shuffle tests, reproducibility checks, and GARCH baseline comparison.

| Script | Purpose |
|--------|---------|
| `01_data_leakage_audit.py` | Feature timestamp integrity and partition isolation |
| `02_r2_verification.py` | Independent R-squared recomputation |
| `03_shuffle_test.py` | Permutation test (destroy predictive structure) |
| `04_reproducibility_test.py` | Cross-run consistency check |
| `05_additional_tests.py` | Supplementary validation diagnostics |
| `06_garch_comparison.py` | Initial GARCH baseline comparison |
| `07_rigorous_garch_test.py` | Extended GARCH testing with MLE |

## February 2026

Rigorous benchmark comparisons using proper MLE-fitted GARCH models and formal statistical significance testing.

| Script | Purpose |
|--------|---------|
| `01_garch_mle_benchmark.py` | GARCH(1,1)-N, GARCH-t, GJR-GARCH, EWMA via MLE |
| `02_diebold_mariano_test.py` | Diebold-Mariano tests with Newey-West HAC errors |
| `03_conditional_regime_test.py` | Stress-stratified performance by market regime |

**Key results:** RIVE 23.04% R-squared, HAR-RV-X 9.62%, DM = +29.03 (p < 0.001)

## March 2026

Final audit round addressing pipeline consistency, sector recomputation, and architecture-level ablation.

| Script | Purpose |
|--------|---------|
| `01_sector_r2_raw_targets.py` | Sector R-squared on raw targets (consistent with 23.04% headline) |
| `02_top50_ablation.py` | Top-50 ablation attempt (data availability check) |
| `03_gics55_ablation.py` | Architecture-level ablation on GICS-55 |
| `04_generate_figures.py` | Publication figure generation (time series, calibration, sector bar) |

**Key results:** Linear integration 4.8% vs RIVE 23.0%; all 11 sector R-squared values recomputed on raw-target pipeline.
