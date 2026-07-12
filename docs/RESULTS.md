# Results Catalog

Every finding in the project, with headline numbers, the artifact that holds
them, and the script that regenerates them. All artifacts live in `reports/`
(tables as CSV, console logs in `reports/logs/`). Everything regenerates via
`.venv/bin/python scripts/run_all.py` (fixed seeds; ~1-2h on an M1 Max).

Evaluation conventions: alpha=0.10 two-sided unless stated; regimes sliced
by trailing-750d VIX percentile bins calm(<=.5)/normal(.5-.8)/
elevated(.8-.95)/stress(>.95); eval sample 2010+ after 100-day calibration
warmup; ~375k out-of-sample stock-days (E2 seven-method common sample:
269,705), panel 2005-2025: 519,843 stock-days.

Universe: CRSP point-in-time top-100 by market cap, rebalanced each January
(scripts/build_pit_universe.py via the WRDS API; formation = prior December
month-end; shrcd 10/11, exchcd 1/2/3, share classes aggregated by permco).
223 unique companies over 2005-2025, mean turnover 10.6 names/yr, 42 members
later delisted and RETAINED through death (Lehman's last panel row is
2008-09-17). Identity is the CRSP permno end-to-end; Risk Lab RVs fetched by
permno (no ticker matching anywhere). Survivorship-bias-free by construction.

---

## E0 — Point-forecast sanity (Table 1)

The pool only needs to be non-degraded; point accuracy is NOT the paper's
claim (this must be said explicitly in the intro — it is what killed RIVE).

| model | QLIKE | RMSE(log) |
|---|---|---|
| HAR | 0.1937 | 0.5320 |
| LightGBM | 0.1883 | 0.5286 |
| Hedge pool | 0.1906 | 0.5302 |

DM-HAC pool vs HAR p=0.054 (pool marginally better, not significant).
Artifacts: `reports/e0_point_sanity.csv`, preds in
`data/processed_v2/e0_predictions.parquet`. Script: `scripts/e0_point_sanity.py`.

## E1 — The phenomenon (Figure 1 / motivating exhibit)

Per-stock ACI is marginally valid (~89.9% vs 90 target) but conditionally
broken: stress-regime coverage **84.2%**, stress upper-tail **88.1%**
(nominal 95). Calm/normal are over-covered — validity on average is bought
by failing exactly when it matters.
Artifact: `reports/e1_prototype_coverage_by_state.csv`.
Script: `scripts/e1_prototype.py`.

## E2 — Main interval comparison + MCS (Table 2)

Seven methods, common 269,705 stock-day sample, alpha=0.10:

| method | marginal | stress | stress upper | width | width stress |
|---|---|---|---|---|---|
| aci | .8986 | .8292 | .9006 | 1.703 | 1.803 |
| dtaci | .8986 | .8360 | .9058 | 1.699 | 1.870 |
| sfogd | .8992 | .8403 | .9102 | 1.734 | 1.909 |
| har_qreg | .8899 | .8147 | .8484 | 1.645 | 1.708 |
| knn_state | .8941 | .8294 | .9081 | 1.667 | 1.837 |
| rc_hand | .8976 | **.8818** | **.9405** | 1.695 | 2.256 |
| rc_adaptive | .8992 | **.8812** | **.9399** | 1.726 | 2.336 |

Only the regime-conditional methods hold stress coverage; they do it by
reallocating width INTO stress (+25%) while staying flat elsewhere.
KNN-state (similarity weighting, HopCPT/NexCP spirit) barely improves on
ACI in stress — similarity without coverage feedback does not work here.
The stress gap vs the best marginal baseline is +4.1pp (vs sfogd) to
+5.2pp (vs aci) — LARGER than on any development sample. MCS over daily
interval scores eliminates sfogd (p=.000) and keeps the rest, rc_hand with
the best mean loss: conditional validity costs nothing in average interval
score.
Date-clustered inference (daily cross-sectional means, Newey-West over the
daily series): stress coverage rc_adaptive .880 (se .019) vs aci .830
(se .026); paired stress-coverage difference vs rc_adaptive is significant
at 1% against every baseline (aci +5.05pp t=3.05 p=.002; dtaci t=2.78;
sfogd t=2.63; har_qreg t=2.71; knn t=2.77). Marginal differences among
conformal methods are ~0 (|t|<0.5), as marginal validity predicts.
rc_hand vs rc_adaptive stress difference is noise (p=.83).

Artifacts: `reports/e2_full_summary.csv`, `reports/e2_full_mcs.csv`,
`reports/e2_clustered_se.csv`.
Script: `scripts/e2_full.py`.

## E3 — VaR application (Table 3)

Next-day VaR from calibrated vol; z = -ret/sigma_pred.
Stress-day breach rates (nominal 5 / 1):

| method | 95% stress breach | 99% stress breach | 99% DQ pass |
|---|---|---|---|
| normal | 12.47% | 6.68% | .03 |
| fhs | 9.03% | 2.74% | .61 |
| garch_t | 11.88% | 4.06% | .42 |
| caviar | 12.63% | 5.38% | .35 |
| aci | 8.62% | 3.06% | .48 |
| rc_panel (hand) | 5.56% | 1.62% | .70 |
| rc_adaptive | **5.74%** | **1.77%** | **.72** |

rc_adaptive also has the best marginals (5.02 / 1.07) and the best
Kupiec/Christoffersen pass rates. The two most-cited classical VaR models
(GARCH-t, CAViaR) fail on stress days like everything else.
Artifact: `reports/e3_var_summary.csv`. Script: `scripts/e3_var.py`.

## E4 — Foundation models have the same disease

Chronos-Bolt zero-shot: competitive accuracy with no training, but its
native 80% band is marginally low (77.9%) and DIRECTIONALLY miscalibrated
in stress: upside misses 14.6% vs 10 nominal against 5.7% downside — the
band points the wrong way in crises. (Bolt structurally cannot emit
quantiles beyond [.1,.9].) Our layer flattens coverage to 79.8+-0.1% in
every regime.
Artifacts: `reports/e4_tsfm_raw_coverage.csv`,
`reports/e4_tsfm_repaired_coverage.csv`; directional table in
`reports/logs/e4_tsfm_run.log`. Scripts: `scripts/tsfm_predict.py` (torch
process — OpenMP isolation), `scripts/e4_tsfm.py`.

## E5 — Adaptive rates: zero tuned parameters

DtACI-style expert bank per regime + issued-interval corrector (conformal
PID structure). With NO tuning: ties hand-tuned stress coverage (88.1 vs
88.2 upper .939 both), better calm/normal (90.0), +2% width; VaR marginals
better than hand-tuned (5.02/1.07 vs 5.15/1.19). The aggregator REDISCOVERS
the hand-tuned findings online (stress rates fast, extreme quantiles large).
Artifact: `reports/e5_adaptive_rates.csv`. Script: `scripts/e5_adaptive_rates.py`.
Key implementation lessons in `src/conformal/panel_hierarchical.py` docstring.

## E6 — Ablations + mechanism decomposition (KEY for framing)

| config | stress cov | note |
|---|---|---|
| per-stock (no pooling) | .7998 | starvation — pooling is the big lever |
| K=1 pooled adaptive | .8738 | most of avg coverage from pooling+adaptivity |
| K=4 (canonical) | .8828 | regimes add +0.9pp on this slice |
| pooled, no offsets | .8828 | offsets negligible |
| forecaster=har/lgbm | .8835/.8831 | layer is forecaster-agnostic |

(K=2..5 within 0.01pp of each other — two regimes already capture the
transition mechanism; the K choice is not load-bearing.)

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

## E6b — Oracle vs estimated regimes (the honesty-remark ablation)

Identical pooled adaptive calibrator under three membership sources:

| membership | stress cov | stress upper | day-2 cov |
|---|---|---|---|
| VIX bins K=4 (canonical, causal) | .8828 | .9390 | .797 |
| online HMM, filtered (causal) | .8763 | .9291 | .755 |
| full-sample HMM, SMOOTHED (leaky oracle) | .8793 | .9390 | .761 |

Even though filtered and oracle memberships differ substantially where it
matters (stress-day mean abs. difference in stress-prob 0.38), coverage is
nearly identical: perfect hindsight regime knowledge buys +0.3pp stress
coverage over the causal filter. Two paper-level implications: (1) the
guarantee conditioning on the algorithm's own filtered state costs almost
nothing vs latent-truth conditioning — regime-estimation error is NOT the
binding constraint; (2) the day-2 transition pit survives even ORACLE
regimes (.76 vs .80 for bins) — the strongest corroboration yet that onset
under-coverage is score distribution shift, not regime misdetection.
Transparent VIX bins remain as good as or better than both HMM variants.
Artifacts: `reports/e6b_oracle_regimes.csv`, `reports/e6b_oracle_dse.csv`.
Script: `scripts/e6b_oracle_regimes.py`.

## Onset / irreducibility (E2 rounds 3-4)

Day-2 of a vol spike is under-covered (72-76%) by every backward-looking
conditioning scheme tried (granularity, freshness subgroups, transition
groups). Adding the forward-looking VIX9D/VIX>1 acute group lifts stress
upper-tail to **94.8%** (target 95; post-2012 sample where VIX9D exists)
but day-2 does NOT move — even option prices don't anticipate day-2
continuations. Frame as an honest negative result with market corroboration.
Artifacts: `reports/e2onset_*.csv`, `reports/e2_panel_summary.csv` (dse
profiles in `reports/logs/e2_onset_run.log`). Scripts: `scripts/e2_onset.py`,
`scripts/e2_transition.py`.

## E9 — Crisis episodes

| episode | aci cov/upper | ours cov/upper |
|---|---|---|
| 2011 US downgrade | .862/.919 | .888/.936 |
| 2015 CNY deval | .881/.912 | .878/.918 |
| 2018 Volmageddon | .847/.877 | .856/.907 |
| 2018 Q4 selloff | .907/.918 | .940/.965 |
| **2020 COVID** | .786/.879 | **.879/.949** |
| 2022 bear market | .919/.954 | .901/.944 |
| 2024 yen unwind | .847/.950 | .848/.958 |
| **2025 tariff shock** | .765/.897 | **.864/.912** |

COVID is the showcase. Flash-onset episodes (2015, 2025) are hardest —
consistent with onset irreducibility.
Artifact: `reports/e9_stress_windows.csv`. Script: `scripts/e9_stress_windows.py`.

## E10 — Held-out generalization

Frozen zero-tuning config on disjoint alternating-alphabetical halves of
the PIT universe: ACI stress .832/.829, ours .875/.876 (upper .930/.932) —
the failure and repair reproduce on both halves; slightly smaller than
full-panel gains, consistent with halved pooling cross-section.
Artifact: `reports/e10_holdout.csv`. Script: `scripts/e10_holdout.py`.

## E12 — Alpha robustness

Stress gap and repair hold at alpha 0.05 / 0.10 / 0.20 (at 0.05: ACI
stress .891 vs ours .932 against .95 target; at 0.20: .730 vs .790
against .80).
Artifact: `reports/e12_alpha_sweep.csv`. Script: `scripts/e12_alpha_sweep.py`.

## Regime-estimator robustness

Online HMM (K=3, soft, filtered) stress coverage .868 vs aligned VIX bins
(K=4 hard) .884, both far above ACI .831 — conclusions do not depend on
the regime estimator; transparent bins suffice.
Artifact: `reports/e2_panel_summary.csv`. Script: `scripts/e2_panel.py`.

## Data-quality audits

- Recycled-ticker PERMNO audit (BAC/META/PM/V/JPM traps):
  `reports/universe_identity_audit.csv`, `scripts/audit_universe_identity.py`.
- Corporate-action duplicate rows (DHR/JNJ) deduped in `src/data/panel.py`
  with uniqueness assert; staleness guard raises on dead histories.
- FRED silent-truncation guard in `src/data/fred.py`.
