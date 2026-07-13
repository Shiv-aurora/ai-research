# Results Catalog

> Headline configuration (2026-07-12, R1): hard VIX-bin regimes,
> adaptive rates, NO per-stock offsets (theory-matching config;
> offsets are an e6 ablation arm). All numbers below regenerated
> under this configuration.

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

Ten methods, common 269,705 stock-day sample, alpha=0.10 (R4 run: adds
tcp_rm = Aich et al. rolling conformal + Robbins-Monro offset, ported to
our score scale, and xs_panel = Tu-Giesecke-spirit cross-sectional
split-conformal with adaptive level):

| method | marginal | stress | stress upper | width | width stress |
|---|---|---|---|---|---|
| aci | .8986 | .8292 | .9006 | 1.703 | 1.803 |
| dtaci | .8986 | .8360 | .9058 | 1.699 | 1.870 |
| sfogd | .8992 | .8403 | .9102 | 1.734 | 1.909 |
| tcp_rm | .8995 | .8488 | .9190 | 1.790 | 2.069 |
| har_qreg | .8899 | .8147 | .8484 | 1.645 | 1.708 |
| knn_state | .8941 | .8294 | .9081 | 1.667 | 1.837 |
| xs_panel | .9003 | .8272 | .9326 | 1.800 | 1.985 |
| rc_hand | .8976 | **.8818** | **.9405** | 1.696 | 2.251 |
| rc_adaptive | .8992 | **.8812** | **.9398** | 1.726 | 2.337 |
| pooled_k1 | .9002 | .8731 | .9414 | 1.728 | 2.268 |

tcp_rm is the best per-stock baseline in stress (.849, still 3.1pp below
rc_adaptive, t=2.21 p=.027 — the only pairwise gap not significant at 1%)
but has the widest per-stock intervals (rolling 60-day window forgets calm
scores wholesale). xs_panel shows cross-sectional information ALONE does
not fix stress (.827 ~= aci) — pooling must sit inside regime-conditional
tracking. MCS eliminates sfogd, tcp_rm, AND xs_panel (p=.000).

pooled_k1 (K=1, no regimes) now sits IN the main table: regimes' further
average stress contribution is +0.8pp (t=0.63, p=.53 clustered — not
significant); regimes' real contributions are conditional (E6/E6c).

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
at 1% against every baseline except tcp_rm (+3.1pp t=2.21 p=.027, 5%
level) — aci +5.06pp t=3.05 p=.002; dtaci t=2.78; sfogd t=2.63;
har_qreg t=2.71; knn t=2.78; xs_panel +5.3pp t=3.22 p=.001.
Marginal differences among
conformal methods are ~0 (|t|<0.5), as marginal validity predicts.
rc_hand vs rc_adaptive stress difference is noise (p=.83).

Sample flow (519,843 -> 269,705): 394,213 with forecast+target; 375,072
non-warm conformal; binding constraint is HAR-QREG's 800-day training
minimum (its usable set IS the common sample); rc methods usable on
365,569. Per-stock coverage distribution (>=25 stress obs, 107 stocks):
median stock stress coverage aci .839 -> rc .880; q10 .780 -> .846;
share of stocks below 85% stress coverage .70 -> .14 — pooling lifts the
whole cross-section, not just the average.

Artifacts: `reports/e2_full_summary.csv`, `reports/e2_full_mcs.csv`,
`reports/e2_clustered_se.csv`, `reports/e2_sample_flow.csv`,
`reports/e2_per_stock_coverage.csv`.
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
| rc_adaptive | **5.73%** | **1.76%** | **.73** |

rc_adaptive also has the best marginals (5.00 / 1.07) and the best
Kupiec/Christoffersen pass rates. The two most-cited classical VaR models
(GARCH-t, CAViaR) fail on stress days like everything else.
Artifact: `reports/e3_var_summary.csv`. Script: `scripts/e3_var.py`.

## E4 — Foundation models have the same disease

Chronos-Bolt zero-shot: competitive accuracy with no training, but its
native 80% band is marginally low (77.9%) and DIRECTIONALLY miscalibrated
in stress: upside misses 14.6% vs 10 nominal against 5.7% downside — the
band points the wrong way in crises. (Bolt structurally cannot emit
quantiles beyond [.1,.9].) Our layer flattens coverage to 80.0-80.2% in
every regime.
Artifacts: `reports/e4_tsfm_raw_coverage.csv`,
`reports/e4_tsfm_repaired_coverage.csv`; directional table in
`reports/logs/e4_tsfm_run.log`. Scripts: `scripts/tsfm_predict.py` (torch
process — OpenMP isolation), `scripts/e4_tsfm.py`.

## E5 — Adaptive rates: zero tuned parameters

DtACI-style expert bank per regime + issued-interval corrector (conformal
PID structure). With NO tuning: ties hand-tuned stress coverage (88.1 vs
88.2 upper .939 both), better calm/normal (90.0), +2% width; VaR marginals
better than hand-tuned (5.00/1.07 vs hand's 5.07/1.18). The aggregator REDISCOVERS
the hand-tuned findings online (stress rates fast, extreme quantiles large).
Artifact: `reports/e5_adaptive_rates.csv`. Script: `scripts/e5_adaptive_rates.py`.
Key implementation lessons in `src/conformal/panel_hierarchical.py` docstring.

## E6 — Ablations + mechanism decomposition (KEY for framing)

| config | stress cov | note |
|---|---|---|
| per-stock (no pooling) | .7998 | starvation — pooling is the big lever |
| K=1 pooled adaptive | .8747 | most of avg coverage from pooling+adaptivity |
| K=4 (canonical) | .8828 | regimes add +0.9pp on this slice |
| pooled + per-stock offsets | .8828 | offsets negligible (headline has none) |
| forecaster=har/lgbm | .8840/.8831 | layer is forecaster-agnostic |

(K=2..5 within 0.01pp of each other — two regimes already capture the
transition mechanism; the K choice is not load-bearing.)

Where regimes actually matter (decomposition run, see
`reports/e6_k_decomposition_notes.md`):
1. Day-2-of-stress-entry coverage: K=1 73.4% vs K=4 **79.4%** (reports/e6_dse_by_k.csv, scripts/e6c_dse_by_k.py).
2. 5% VaR conditional balance: K=1 inverts (calm 5.26/stress 3.62);
   K=4 balanced (5.00/5.73). (K-sweep VaR numbers pending fold-in to a
   regenerable script — R3 queue.)
3. Per-regime guarantees only exist for K>1; the forward-looking acute
   group is only expressible with regimes.
Honest nuance to report: 1% VaR K=1 beats K=4 in stress (1.18 vs 1.86).
Paper framing: pooling repairs coverage, adaptivity removes tuning, regimes
buy transitions + conditional balance + guarantees.
Artifact: `reports/e6_ablations.csv`. Script: `scripts/e6_ablations.py`.

## E13 — Pooling-mechanism identification (sum vs average vs rate)

Fixed-rate 2x2 {pooled, per-stock} x {fast ~0.2/day, slow ~0.002/day}
plus averaged-error arm (all K=4 bins, no adaptivity, no offsets):

| arm | stress cov | marginal |
|---|---|---|
| pooled summed (fast) | .8405 | .8955 |
| pooled averaged (fast) | .8431 | .8956 |
| pooled rate/100 (slow) | .7704 | .8655 |
| per-stock standard (slow) | .7699 | .8582 |
| per-stock rate x100 (fast) | .8396 | .8951 |

Findings: (1) sum-vs-average is pure rescaling; (2) the average-coverage
channel is EFFECTIVE DAILY STEP, not cross-sectional information (slow
pooled == slow per-stock; rate-matched per-stock == pooled) — consistent
with heavy cross-sectional dependence; (3) that speed needs eta=0.2/obs
per stock (100x off-grid; the per-stock ADAPTIVE arm free to choose
within the standard grid lands at .800). Pooling = automatic panel-scaled
rate inside a standard spec + tighter per-stock coverage distribution.
Paper: "What pooling actually is" + tab:mechanism in ablations.
Artifact: `reports/e13_pooling_mechanism.csv`.
Script: `scripts/e13_pooling_mechanism.py` (average_errors flag in
panel_hierarchical.py, fixed-rate path only).

## E6b — Oracle vs estimated regimes (the honesty-remark ablation)

Identical pooled adaptive calibrator under three membership sources:

| membership | stress cov | stress upper | day-2 cov |
|---|---|---|---|
| VIX bins K=4 (canonical, causal) | .8828 | .9390 | .794 |
| online HMM, filtered (causal) | .8767 | .9293 | .756 |
| full-sample HMM, SMOOTHED (noncausal hindsight) | .8789 | .9388 | .761 |

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

## E6d — Fixed intervals, reclassified evaluation (conditioning gap)

Holding the canonical causal run's ISSUED intervals fixed, re-slice
coverage by hindsight stress definitions (full-sample smoothed HMM):
causal VIX tail .8828/.9390 (14,093 sd); hindsight HMM state
.8972/.9485 (119,058 sd — much broader state); size-matched hindsight
tail .8792/.9518 (9,863 sd); days in hindsight-stress but not VIX-tail
covered at .8992. Conclusion: neither the algorithm (E6b) nor the
evaluation slicing (E6d) is materially helped by hindsight — the
conditioning gap is ~0.4pp two-sided, zero in the upper tail.
Artifact: `reports/e6d_reclassified_eval.csv`.
Script: `scripts/e6d_reclassified_eval.py`.

## Onset / irreducibility (E2 rounds 3-4)

Day-2 of a vol spike is under-covered (73-79%) by every backward-looking
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
| **2025 tariff shock** | .765/.897 | **.862/.911** |

COVID is the showcase. Flash-onset episodes (2015, 2025) are hardest —
consistent with onset irreducibility.
Artifact: `reports/e9_stress_windows.csv`. Script: `scripts/e9_stress_windows.py`.

## E10 — Held-out generalization

Frozen zero-tuning config on disjoint alternating-alphabetical halves of
the PIT universe: ACI stress .832/.829, ours .875/.876 (upper .930/.932) —
the failure and repair reproduce on both halves; slightly smaller than
full-panel gains, consistent with halved pooling cross-section.
Artifact: `reports/e10_holdout.csv`. Script: `scripts/e10_holdout.py`.

## E14 — Temporal robustness: subperiod split + leave-one-crisis-out (R4)

Same issued intervals (full panel, online, constants a priori) evaluated
by subperiod, split 2018-01-01:

| period | aci stress | rc stress | paired gap (clustered) |
|---|---|---|---|
| 2010-2017 | .8957 | .8826 | -1.3pp (t=-1.89, p=.06 ns) |
| 2018-2025 | .7877 | .8829 | +9.4pp (t=4.34, p<.0001) |

The stress deficit is a phenomenon of the recent fast-shift era
(Volmageddon, COVID, 2022, 2024-25); in the slow-building 2010-2017
episodes ACI kept up. RC delivers ~.883 in BOTH halves — insurance
framing: large payoff when the phenomenon is present, no significant cost
when absent. Marginals ~.90 everywhere.

Leave-one-crisis-out stress coverage: RC range .8813-.8865 whichever
episode year is dropped (max move 0.4pp); ACI never above .8536 (best
case = COVID dropped). No single crisis drives the result.
Artifacts: `reports/e14_subperiod.csv`, `reports/e14_subperiod_gap.csv`,
`reports/e14_loco.csv`. Script: `scripts/e14_temporal_holdout.py`.

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
