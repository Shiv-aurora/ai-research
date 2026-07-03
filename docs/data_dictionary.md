# Data Dictionary

Definitions for every column in the v2 panel (`data/processed_v2/`). Populated as Phase 1 builds each dataset.

## Conventions

- All timestamps are US/Eastern trading days; a feature dated `t` is observable by the close of day `t`.
- The forecast target for day `t+1` is `log_rv_{t+1}`; models see only columns dated ≤ `t`.
- Realized variance is 5-minute subsampled from TAQ trades, cleaned per Barndorff-Nielsen–Hansen–Lunde–Shephard rules; `bpv` is bipower variation on the same grid.
- Per-stock standardization windows (e.g., trailing MAD of residuals) are strictly trailing.

## Tables

_To be filled in Phase 1: `rv_panel` (stock-day RV/BPV), `universe` (point-in-time membership), `state_vars` (market-level daily state vector), `risklab_xval` (cross-validation vs Risk Lab)._
