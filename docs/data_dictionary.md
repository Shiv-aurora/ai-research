# Data Dictionary

Definitions for every column in the v2 panel (`data/processed_v2/`).

## Conventions

- All dates are US trading days; a feature dated `t` is observable by the close of day `t`.
- The forecast target for day `t+1` is `log_rv` at `t+1`; the walk-forward engine creates the shifted target internally (never stored, one source of truth for alignment).
- Trailing windows (percentiles, MADs) are strictly backward-looking; cross-sectional dispersion enters lagged one day.

## `rv_panel.parquet` (stock-day, ~510k rows, 100 tickers, 2005–2025)

| column | definition | source |
|---|---|---|
| ticker | universe symbol (provisional top-100; see `src/data/universe.py`) | — |
| permno | CRSP permno chosen by liveness+history disambiguation; META pinned via `PERMNO_OVERRIDES` | Risk Lab symbol.php |
| date | trading day (naive dates, US/Eastern trading calendar) | — |
| log_rv | ln(daily realized variance) from the configured estimator (`data.rv.estimator`, default `log_rv5` = 5-min subsampled trades RV; alternative `log_rv` = QMLE all-trades) | Risk Lab data.php |
| rv_daily | exp(log_rv), daily variance units | derived |
| stock_rv_pctl | trailing 750-day percentile of the stock's own log_rv | derived |
| VIXCLS | CBOE VIX close | FRED |
| DGS10, DGS3MO | 10y / 3m constant-maturity Treasury yields | FRED |
| DBAA, DAAA | Moody's Baa / Aaa corporate yields (daily series) | FRED |
| term_spread | DGS10 − DGS3MO | derived |
| credit_spread | DBAA − DAAA | derived |
| vix_pctl | trailing 750-day percentile of VIXCLS | derived |
| mkt_log_rv | cross-sectional mean of log_rv that day (market-level RV) | derived |
| mkt_rv_pctl | trailing 750-day percentile of mkt_log_rv | derived |
| xs_dispersion | cross-sectional std of log_rv, **lagged one day** | derived |

## `e0_predictions.parquet` (walk-forward out-of-sample predictions, 2010+)

| column | definition |
|---|---|
| ticker, date | as above; row (i, t) refers to the forecast MADE at close of t |
| target | realized log_rv at t+1 (the verification value) |
| har | HAR-RV per-stock OLS forecast of target |
| lgbm | pooled LightGBM forecast |
| pool | online Hedge (QLIKE) combination of the experts |

## Raw caches

- `data/raw/risklab/<TICKER>.parquet` — full Risk Lab history per ticker: QMLE/5-min/15-min annualized vols for trades and quotes + derived `log_rv`, `log_rv5`. Identity-audited via `scripts/audit_universe_identity.py` (report: `reports/universe_identity_audit.csv`).
- `data/raw/fred/<SERIES>.csv` — raw FRED downloads (truncation-guarded on load).

## Known limitations (to fix before submission)

- Provisional universe carries survivorship bias → replace with CRSP point-in-time top-100 when WRDS access renews.
- No bipower variation column yet (Risk Lab doesn't publish it) → HAR-J/CHAR variants and jump analysis need the WRDS TAQ pipeline (`src/data/rv_estimators.py` is ready).
- Risk Lab vs VOLARE cross-validation report still to be produced.
