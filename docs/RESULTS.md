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
warmup; ~375k out-of-sample stock-days (E2 thirteen-method common sample:
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

Thirteen methods, common 269,705 stock-day sample, alpha=0.10 (R4 added
tcp_rm and xs_panel; R6 added pogo; R8 added cpid = conformal PID
(Angelopoulos et al. 2023: tracking + saturated integrator +
trailing-quantile scorecaster, per stock) and rkr = Ramalingam-Kiyani-
Roth no-regret FTRL multigroup ACI (per stock, overlapping
marginal+bin groups)):

| method | marginal | stress | stress upper | width | width stress |
|---|---|---|---|---|---|
| aci | .8986 | .8292 | .9006 | 1.703 | 1.803 |
| dtaci | .8986 | .8360 | .9058 | 1.699 | 1.870 |
| sfogd | .8992 | .8403 | .9102 | 1.734 | 1.909 |
| tcp_rm | .8995 | .8488 | .9190 | 1.790 | 2.069 |
| cpid | .8993 | .8363 | .9048 | 1.724 | 1.855 |
| rkr | .8984 | .8457 | .9175 | 1.720 | 1.921 |
| har_qreg | .8899 | .8147 | .8484 | 1.645 | 1.708 |
| knn_state | .8941 | .8294 | .9081 | 1.667 | 1.837 |
| xs_panel | .9003 | .8272 | .9326 | 1.800 | 1.985 |
| rc_hand | .8976 | **.8818** | **.9405** | 1.696 | 2.251 |
| rc_adaptive | .8992 | **.8812** | **.9398** | 1.726 | 2.337 |
| pooled_k1 | .9002 | .8731 | .9414 | 1.728 | 2.268 |
| pogo | .8577 | .7706 | .8854 | 1.743 | 2.140 |

tcp_rm is the best per-stock baseline in stress (.849, still 3.1pp below
rc_adaptive, t=2.21 p=.027 — the only pairwise gap not significant at 1%)
but has the widest per-stock intervals (rolling 60-day window forgets calm
scores wholesale). xs_panel shows cross-sectional information ALONE does
not fix stress (.827 ~= aci) — pooling must sit inside regime-conditional
tracking. MCS eliminates sfogd, tcp_rm, cpid, rkr, xs_panel, AND pogo (p<=.0005).

R8 baselines: cpid stress .8363 (-4.3pp vs rc, p=.008) — RC is NOT
'PID + faster rate'; the corrector's integrator without the pooled
panel signal leaves the deficit. rkr stress .8457 (-3.5pp, p=.007) —
second-best per-stock method; group-conditional FTRL helps a single
stream but per-stock stress history is too short. Both significant at
1%; tcp_rm remains the only 5%-level exception.

pogo (POGO): undercovers everywhere (marginal .858, stress .771,
-11.0pp vs rc, t=8.4) — CONSISTENT with its own parameter-free bound,
which permits ~2.5pp/side slack at T~3,800 rounds and is vacuous for a
~150-round per-stock stress group. Reading: the group-conditional
guarantee interface alone does not bite at equity horizons; coverage
comes from the tracking loop. (A pooled sequential-round port of POGO
does worse, .832 marginal — not reported in the paper.)

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
| pooled_k1 | 3.71% | **1.17%** | .68 |
| rc_panel (hand) | 5.49% | 1.61% | .64 |
| rc_adaptive | **5.73%** | 1.76% | .66 |

R6 upgrade: columns now Kupiec / Christoffersen INDEPENDENCE (honest
relabel — the old 'christoffersen' column WAS the independence test) /
combined CC (2 df) / DQ / DQ-stress (stress-indicator regressor), plus
a date-clustered calm-vs-stress balance z-test and a pooled K=1 arm.

KEY FINDING (the statistically unambiguous regime benefit): at 5% VaR,
every baseline overshoots stress (balance z = 2.1-3.9, all p<.05);
pooled_k1 INVERTS (calm 5.3 vs stress 3.7, z=-1.8); only rc heads are
balanced (5.0/5.7, z=0.3-0.5). At 99%, pooled_k1 posts the best CC/DQ-S
pass shares (.91/.73 vs rc .87/.64) — deep-tail data-splitting cost,
reported honestly at both levels. rc_adaptive marginals 5.00 / 1.07.
The DQ pass column above is DQ-stress.
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

## E15 — Expanded rate grid: the fast-rate objection closed (R6)

The referee objection: E13 showed the pooled tracker's effective stress
rate (~0.2/obs) exceeds the adaptive expert grid's max (0.064), so the
headline gap might exist only because per-stock baselines were denied
the rates they needed. Answer: extend the grid to 0.512
(+0.128/0.256/0.512) and rerun everything.

| arm | grid | stress | diff vs RC (clustered) | eff. stress rate |
|---|---|---|---|---|
| RC K=4 pooled summed | std | .8828 | — | .010 |
| RC K=4 pooled summed | exp | .8827 | -0.0pp, p=.99 | .021 |
| pooled K=1 | exp | .8721 | -0.9, p=.43 | .033 |
| pooled K=4 averaged errors | exp | .8497 | -3.3, p=.0002 | .244 |
| per-stock K=4 | std | .7996 | -8.3, p<.0001 | .029 |
| per-stock K=4 | exp | .8366 | -4.6, p<.0001 | .175 |
| per-stock K=1 | exp | .8429 | -3.9, p=.005 | .136 |
| DtACI raw-score, etas to 0.8 | exp | .8548 | -2.7, p=.039 | — |

Readout: the per-stock aggregator FINDS and USES the fast rates (issued
eff. rate .175 in stress) and recovers about half the deficit
(.800 → .837) — but every fast-rate route still trails RC by 2.7–4.6pp,
all significant. RC itself is insensitive to the grid ceiling.
Mechanism refinement over E13: effective daily speed is necessary but
not sufficient; the pooled summed update takes n_t small steps per day
(same daily speed, ~1/n̄ gradient variance), and a single stream at rate
0.2 tracks its own noise. Paper: tab:expgrid + "Fast rates are necessary
but not sufficient" paragraph in ablations.
Artifacts: `reports/e15_expanded_grid.csv`,
`reports/e15_expanded_grid_tests.csv`, `reports/e15_eff_eta.csv`.
Script: `scripts/e15_expanded_grid.py`.

## E16 — Numeric theorem bounds + inference robustness (R6)

A-priori per-regime deviation bounds at the paper's constants (B=12,
eta_max=.064, eta_corr=.002, n_bar=100):

| regime | N_k | Prop 1 | Prop 2 |
|---|---|---|---|
| calm | 230,278 | .105 | .066 |
| normal | 99,752 | .161 | .153 |
| elevated | 49,590 | .163 | .309 |
| stress | 14,593 | .171 | **1.048 (vacuous)** |

Stated honestly in Remark rem:numeric + limitations item five: the
theorems are worst-case pathwise control; empirical tables carry the
finite-sample evidence (realized deviations are observable exactly via
the telescoped identity and are ~100x inside the bounds).

Episode-block bootstrap (18 stress episodes, gap>20 trading days,
10,000 resamples) for the stress gap rc_adaptive − baseline:
ACI +5.2pp CI [−0.1, +9.7] p=.057; DtACI +4.5 p=.063; SF-OGD +4.1
p=.077; TCP-RM +3.2 p=.108; XS-panel +5.4 CI [+2.8, +8.9] p=.0004;
pooled_k1 +0.8 p=.48. Interpretation: the gap vs per-stock trackers is
era-heterogeneous (huge post-2018, ~0 before), so 18 episodes give a
borderline CI; vs XS-panel the deficit is uniform → significant. Paper
reports BOTH treatments with the insurance framing.

HAC lag sensitivity (RC−ACI stress): rule lag 4 t=3.05 p=.002; lags
0/5/10/22 all p<.02; lag 66 p=.072.

Artifacts: `reports/e16_bound_values.csv`,
`reports/e16_episode_bootstrap.csv`, `reports/e16_hac_sensitivity.csv`.
Script: `scripts/e16_bounds_inference.py`.

## E17 — K=1 vs K=4: the distinctive-benefit inference package (R6)

Identical adaptive calibrator, only K varied. Honest verdict: on the
interval side point estimates consistently favor K=4 but are
underpowered; the statistically unambiguous regime benefit is VaR
balance (see E3).

- Per-regime coverage diffs: all ns (stress +0.67pp, p=.53). Interval
  scores by regime: all ns (stress −0.038, p=.62 — favoring K=4).
- Day-2 transition coverage: .794 vs .734, diff +5.3pp, HAC CI
  [−3.8, +14.5], p=.25 — only 15 day-2 dates.
- Regime-calibration dispersion (weighted RMS dev from .90): .0034 (K4)
  vs .0054 (K1), diff ns (block bootstrap p=.44); max-abs .017 vs .025,
  p=.52.
- 57 market-level stress entries.
Paper: "What regimes add, with uncertainty attached" paragraph in
ablations. Artifacts: `reports/e17_regime_slices.csv`,
`reports/e17_dse_diff.csv`, `reports/e17_dispersion.csv`.

## E18 — Onset uncertainty + widening Pareto (R6)

- Sample honesty: 57 entries, 25 episodes; day-2 = 1,320 stock-days on
  14 DATES. Day-2 coverage .797, HAC se .072 → CI [.65, .94] does not
  exclude nominal. Support = stability: leave-one-episode-out range
  [.764, .848]; day-1 .851 ± .041; pit present in every method/era.
- Trajectory: issued upper threshold 2.30 (day1) → 2.37 (day2) → 3.78
  (day3); day-2 realized score mean 1.15 (2x day 1), upper-miss 16.7%.
  The tracker makes the full move one day late.
- Pareto overlay (causal ×m widening on prev-day dse 1–3, no feedback):
  m=1.25 → day-2 .858 at +0.9% stress-interior IS; m=1.5 → .895
  (nominal) at +2.5% IS and +50% onset width; m=2 → .930. The window is
  unpurchased, not uncoverable; aggregator declines because day-2 = 14
  of 4,002 dates. Open problem reframed: selective onset widening.
Paper: new subsection sec:onset-pareto. Artifacts: `reports/e18_*.csv`.

## E19 — Calendar-time crisis inference + multiplicity (R8)

Full-calendar regression d_t = b0 + b1*1{stress} of daily paired
coverage diffs (no event-time compression). PRIMARY ENDPOINT
(prespecified): stress UPPER-tail RC-ACI. Result: b0 = -0.1pp (calm gap
~0), b1 = +4.0pp — significant under calendar HAC (p=.009),
episode-clustered SEs (p=.016), wild cluster bootstrap over 18 episodes
(p=.003); leave-one-episode-out spans [+2.3,+4.8], always positive.
THE PRIMARY ENDPOINT IS CERTIFIED UNDER THE MOST CONSERVATIVE
TREATMENT.

Secondary (two-sided): b1 positive vs every per-stock baseline
(+3.3..+5.8, calm gaps ~0) but wild-cluster p=.13-.34; Romano-Wolf
stepdown leaves only pogo (adj p=.017) and xs_panel (.041); aci adj
p=.099. Stated as an explicit evidential hierarchy in the paper.

Non-inferiority: RC-ACI daily interval score -0.003, CI [-.022,+.015],
margin=2% of ACI mean (.047) → formally NON-INFERIOR (point estimate
favors RC).
Artifacts: `reports/e19_*.csv`. Script: `scripts/e19_crisis_inference.py`.

## E20 — Effective panel size + mechanism simulations (R8)

Empirical intra-date correlation of upper-miss indicators (moment
identity): rho=.158 all days → n_eff ~ 6; rho=.898 STRESS days →
n_eff ~ 1.1. On crisis days the panel is informationally ~one stream:
within stress, pooling's residual edge over rate-matched per-stock arms
comes from the graded batch signal (miss FRACTION vs binary miss) and
the shared threshold, not variance averaging. Fixes the wrong 1/n
claim from R6 (reviewer was right).

Simulations (factor panels, controlled fixed rate): pooled K=2 >
pooled K=1 > per-stock in every cell of rho{0,.3,.6,.9} x n{10,100} x
sigma_h{0,.5}; pooled advantage largest at rho=0/n=100 (.83 vs .51),
shrinks monotonically in rho (.67 at rho=.9), eroded by heterogeneity
at low rho — exactly the Var(mean)=sigma^2[rho+(1-rho)/n] prediction.
Artifacts: `reports/e20_rho_neff.csv`, `reports/e20_sim_grid.csv`.

## E21 — Stress definitions external to the algorithm (R8)

Same issued intervals re-sliced: canonical +5.2pp (p=.002); VIX>=30
+2.7 (p=.015, 230 dates); mkt-RV tail +4.1 (p=.003); named windows
+1.3 (p=.06); VIX>=25 +0.5 ns (475 mild dates); credit-spread tail
+0.7 ns (different phenomenon). Repair = property of acute-vol states
however defined, not of the VIX-percentile encoding.

Interval validity: 365,569 issued intervals, min q_hi 1.28, min q_lo
0.97, ZERO negative, ZERO crossed.
Artifacts: `reports/e21_*.csv`. Script: `scripts/e21_stress_definitions.py`.

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
