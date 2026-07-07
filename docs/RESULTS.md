# Results Catalog

Every finding in the project, with headline numbers, the artifact that holds
them, and the script that regenerates them. All artifacts live in `reports/`
(tables as CSV, console logs in `reports/logs/`). Everything regenerates via
`.venv/bin/python scripts/run_all.py` (fixed seeds; ~1-2h on an M1 Max).

Evaluation conventions: alpha=0.10 two-sided unless stated; regimes sliced
by trailing-750d VIX percentile bins calm(<=.5)/normal(.5-.8)/
elevated(.8-.95)/stress(>.95); eval sample 2010+ after 100-day calibration
warmup; ~394k out-of-sample stock-days, 100 tickers, panel 2005-2025.

---

## E0 — Point-forecast sanity (Table 1)

The pool only needs to be non-degraded; point accuracy is NOT the paper's
claim (this must be said explicitly in the intro — it is what killed RIVE).

| model | QLIKE | RMSE(log) |
|---|---|---|
| HAR | 0.1923 | 0.5317 |
| LightGBM | 0.1898 | 0.5292 |
| Hedge pool | 0.1918 | 0.5313 |

DM-HAC pool vs HAR p=0.055 (pool marginally better, not significant).
Artifacts: `reports/e0_point_sanity.csv`, preds in
`data/processed_v2/e0_predictions.parquet`. Script: `scripts/e0_point_sanity.py`.

## E1 — The phenomenon (Figure 1 / motivating exhibit)

Per-stock ACI is marginally valid (89.90% vs 90 target) but conditionally
broken: stress-regime coverage **84.0%**, stress upper-tail **88.6%**
(nominal 95). Calm/normal are over-covered — validity on average is bought
by failing exactly when it matters.
Artifact: `reports/e1_prototype_coverage_by_state.csv`.
Script: `scripts/e1_prototype.py`.

## E2 — Main interval comparison + MCS (Table 2)

Seven methods, common 318,892 stock-day sample, alpha=0.10:

| method | marginal | stress | stress upper | width | width stress |
|---|---|---|---|---|---|
| aci | .8985 | .8397 | .8912 | 1.709 | 1.798 |
| dtaci | .8993 | .8524 | .9041 | 1.708 | 1.866 |
| sfogd | .8989 | .8504 | .9036 | 1.739 | 1.878 |
| har_qreg | .8907 | .8206 | .8562 | 1.651 | 1.717 |
| knn_state | .8936 | .8405 | .8947 | 1.662 | 1.811 |
| rc_hand | .8975 | **.8829** | **.9378** | 1.692 | 2.172 |
| rc_adaptive | .8990 | **.8829** | **.9397** | 1.721 | 2.251 |

Only the regime-conditional methods hold stress coverage; they do it by
reallocating width INTO stress (+25%) while staying flat elsewhere.
KNN-state (similarity weighting, HopCPT/NexCP spirit) barely improves on
ACI in stress — similarity without coverage feedback does not work here.
MCS over daily interval scores keeps {rc_hand, har_qreg, rc_adaptive,
dtaci, knn_state} and ELIMINATES aci (p=.055) and sfogd (p=.000):
conditional validity costs nothing in average interval score.
Artifacts: `reports/e2_full_summary.csv`, `reports/e2_full_mcs.csv`.
Script: `scripts/e2_full.py`.

## E3 — VaR application (Table 3)

Next-day VaR from calibrated vol; z = -ret/sigma_pred; 368,892 stock-days.
Stress-day breach rates (nominal 5 / 1):

| method | 95% stress breach | 99% stress breach | 99% DQ pass |
|---|---|---|---|
| normal | 12.77% | 6.88% | .00 |
| fhs | 9.22% | 3.05% | .52 |
| garch_t | 11.46% | 3.72% | .36 |
| caviar | 11.41% | 4.11% | .27 |
| aci | 8.88% | 3.44% | .38 |
| rc_panel (hand) | 5.63% | 1.71% | .65 |
| rc_adaptive | 5.70% | **1.86%** | **.77** |

rc_adaptive also has the best marginals (5.01 / 1.08) and the best
Kupiec/Christoffersen pass rates. The two most-cited classical VaR models
(GARCH-t, CAViaR) fail on stress days like everything else.
Artifact: `reports/e3_var_summary.csv`. Script: `scripts/e3_var.py`.

## E4 — Foundation models have the same disease

Chronos-Bolt zero-shot: QLIKE 0.205 (competitive, no training) but its
native 80% band is DIRECTIONALLY miscalibrated in stress: upside misses
14.6% vs 10 nominal against 5.7% downside — the band points the wrong way
in crises. (Bolt structurally cannot emit quantiles beyond [.1,.9].) Our
layer flattens coverage to 79.8+-0.1% in every regime for +7-10% width.
Artifacts: `reports/e4_tsfm_raw_coverage.csv`,
`reports/e4_tsfm_repaired_coverage.csv`; directional table in
`reports/logs/e4_tsfm_regen.log`. Scripts: `scripts/tsfm_predict.py` (torch
process — OpenMP isolation), `scripts/e4_tsfm.py`.

## E5 — Adaptive rates: zero tuned parameters

DtACI-style expert bank per regime + issued-interval corrector (conformal
PID structure). With NO tuning: exact tie with hand-tuned stress coverage
(88.28%), better calm/normal/elevated (90.0), +2% width; VaR marginals
better than hand-tuned. The aggregator REDISCOVERS both hand-tuned findings
online: stress rates ~2x calm (eff eta .0094 vs tuned .01) and 1%-VaR rates
~2x the 5% ones (.020 vs tuned .02).
Artifact: `reports/e5_adaptive_rates.csv`. Script: `scripts/e5_adaptive_rates.py`.
Key implementation lessons in `src/conformal/panel_hierarchical.py` docstring.

## E6 — Ablations + mechanism decomposition (KEY for framing)

| config | stress cov | note |
|---|---|---|
| per-stock (no pooling) | .8024 | starvation — pooling is the big lever |
| K=1 pooled adaptive | .8793 | most of avg coverage from pooling+adaptivity |
| K=4 (canonical) | .8828 | regimes add +0.4pp on this slice |
| pooled, no offsets | .8830 | offsets negligible |
| forecaster=har/lgbm | .8849/.8833 | layer is forecaster-agnostic |

Where regimes actually matter (decomposition run, see
`reports/e6_k_decomposition_notes.md`):
1. Day-2-of-stress-entry coverage: K=1 75.4% vs K=4 **80.2%**.
2. 5% VaR conditional balance: K=1 inverts (calm 5.26/stress 3.62);
   K=4 balanced (5.00/5.71).
3. Per-regime guarantees only exist for K>1; the forward-looking acute
   group is only expressible with regimes.
Honest nuance to report: 1% VaR K=1 beats K=4 in stress (1.18 vs 1.86).
Paper framing: pooling repairs coverage, adaptivity removes tuning, regimes
buy transitions + conditional balance + guarantees.
Artifact: `reports/e6_ablations.csv`. Script: `scripts/e6_ablations.py`.

## Onset / irreducibility (E2 rounds 3-4)

Day-2 of a vol spike is under-covered (~80-83%) by every backward-looking
conditioning scheme tried (granularity, freshness subgroups, transition
groups). Adding the forward-looking VIX9D/VIX>1 acute group lifts stress
upper-tail to **94.8%** (target 95) but day-2 does NOT move — even option
prices don't anticipate day-2 continuations. Frame as an honest negative
result with market corroboration.
Artifacts: `reports/e2onset_*.csv`, `reports/e2_panel_summary.csv` (dse
profiles in `reports/logs/e2_onset_regen.log`). Scripts: `scripts/e2_onset.py`,
`scripts/e2_transition.py`.

## E9 — Crisis episodes

| episode | aci cov/upper | ours cov/upper |
|---|---|---|
| 2011 US downgrade | .869/.927 | .889/.939 |
| 2015 CNY deval | .887/.920 | .878/.922 |
| 2018 Volmageddon | .851/.884 | .860/.915 |
| 2018 Q4 selloff | .909/.925 | .942/.966 |
| **2020 COVID** | .801/.859 | **.886/.951** |
| 2022 bear market | .916/.953 | .899/.946 |
| 2024 yen unwind | .865/.940 | .874/.957 |
| 2025 tariff shock | .744/.876 | .815/.909 |

COVID is the showcase. Flash-onset episodes (2015, 2025) are hardest —
consistent with onset irreducibility.
Artifact: `reports/e9_stress_windows.csv`. Script: `scripts/e9_stress_windows.py`.

## E10 — Held-out generalization

Frozen zero-tuning config on disjoint alternating-alphabetical 50-name
halves: ACI stress .838/.842, ours .873/.877 (upper .930/.932) — the
failure and repair reproduce on both halves; slightly smaller than
full-panel gains, consistent with halved pooling cross-section.
Artifact: `reports/e10_holdout.csv`. Script: `scripts/e10_holdout.py`.

## E12 — Alpha robustness

Stress gap and repair hold at alpha 0.05 / 0.10 / 0.20 (e.g. at 0.05:
ACI stress .904 vs ours .931 against .95 target).
Artifact: `reports/e12_alpha_sweep.csv`. Script: `scripts/e12_alpha_sweep.py`.

## Regime-estimator robustness

Online HMM (K=3, soft, filtered) vs aligned VIX bins (K=4 hard): stress
.875 vs .883 — conclusions do not depend on the regime estimator;
transparent bins suffice.
Artifact: `reports/e2_panel_summary.csv`. Script: `scripts/e2_panel.py`.

## Data-quality audits

- Recycled-ticker PERMNO audit (BAC/META/PM/V/JPM traps):
  `reports/universe_identity_audit.csv`, `scripts/audit_universe_identity.py`.
- Corporate-action duplicate rows (DHR/JNJ) deduped in `src/data/panel.py`
  with uniqueness assert; staleness guard raises on dead histories.
- FRED silent-truncation guard in `src/data/fred.py`.
