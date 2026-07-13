# Replication manifest

Every exhibit in the paper maps to one experiment script and its CSV
artifacts. `scripts/run_all.py` regenerates all of them in order with
fixed seeds; `scripts/make_figures.py` renders the figures from the
artifacts. Internal experiment IDs (E0, E2, ...) index this manifest and
the per-experiment notes in `docs/RESULTS.md`.

| Paper exhibit | ID | Script | Artifacts (reports/) |
|---|---|---|---|
| Table: point-forecast sanity (`tab:e0`) | E0 | `scripts/e0_point_sanity.py` | `e0_point_sanity.csv` |
| Figure: coverage by regime (`fig:coverage`) | E2 | `scripts/e2_full.py` | `e2_full_summary.csv` |
| Table: main comparison (`tab:main`) | E2 | `scripts/e2_full.py` | `e2_full_summary.csv`, `e2_clustered_se.csv` |
| Table: per-stock distribution (`tab:perstock`) | E2 | `scripts/e2_full.py` | `e2_per_stock_coverage.csv` |
| Table: sample flow (`tab:flow`) | E2 | `scripts/e2_full.py` | `e2_sample_flow.csv` |
| MCS statements in text | E11 | `scripts/e2_full.py` | `e2_full_mcs.csv` |
| Table + figure: VaR backtests (`tab:var`, `fig:var`) | E3 | `scripts/e3_var.py` | `e3_var_summary.csv` |
| Figure: TSFM repair (`fig:tsfm`) | E4 | `scripts/tsfm_predict.py`, `scripts/e4_tsfm.py` | `e4_tsfm_raw_coverage.csv`, `e4_tsfm_repaired_coverage.csv` |
| Adaptive-rates statements (Section: self-tuning) | E5 | `scripts/e5_adaptive_rates.py` | `e5_adaptive_rates.csv` |
| Table: ablations (`tab:ablations`) | E6 | `scripts/e6_ablations.py` | `e6_ablations.csv` |
| Figure: mechanism decomposition (`fig:mechanism`) | E6/E6b | `scripts/e6_ablations.py`, `scripts/e6b_oracle_regimes.py` | `e6_ablations.csv`, `e6b_oracle_regimes.csv` |
| Figure: days-since-entry profile (`fig:dse`) | E6b | `scripts/e6b_oracle_regimes.py` | `e6b_oracle_dse.csv` |
| Day-2 by K statements | E6c | `scripts/e6c_dse_by_k.py` | `e6_dse_by_k.csv` |
| Hindsight reclassified evaluation statements | E6d | `scripts/e6d_reclassified_eval.py` | `e6d_reclassified_eval.csv` |
| Figure: crisis episodes (`fig:episodes`) | E9 | `scripts/e9_stress_windows.py` | `e9_stress_windows.csv` |
| Split-halves holdout statements | E10 | `scripts/e10_holdout.py` | `e10_holdout.csv` |
| Figure: alpha sweep (`fig:alpha`) | E12 | `scripts/e12_alpha_sweep.py` | `e12_alpha_sweep.csv` |
| Table: pooling mechanism (`tab:mechanism`) | E13 | `scripts/e13_pooling_mechanism.py` | `e13_pooling_mechanism.csv` |
| Table: temporal robustness (`tab:temporal`) | E14 | `scripts/e14_temporal_holdout.py` | `e14_subperiod.csv`, `e14_subperiod_gap.csv`, `e14_loco.csv` |
| Expanded rate-grid statements/table | E15 | `scripts/e15_expanded_grid.py` | `e15_expanded_grid.csv`, `e15_expanded_grid_tests.csv`, `e15_eff_eta.csv` |
| Numeric theorem bounds; episode bootstrap; HAC sensitivity | E16 | `scripts/e16_bounds_inference.py` | `e16_bound_values.csv`, `e16_episode_bootstrap.csv`, `e16_hac_sensitivity.csv` |
| K=1 vs K=4 conditional inference | E17 | `scripts/e17_k1_vs_k4.py` | `e17_regime_slices.csv`, `e17_dse_diff.csv`, `e17_dispersion.csv` |
| Onset uncertainty + widening Pareto | E18 | `scripts/e18_onset_uncertainty.py` | `e18_entries.csv`, `e18_dse_ci.csv`, `e18_loo_day2.csv`, `e18_trajectories.csv`, `e18_pareto.csv` |
| Regime-estimator robustness statements | — | `scripts/e2_panel.py` | `e2_panel_summary.csv` |
| Onset/transition statements | — | `scripts/e2_onset.py`, `scripts/e2_transition.py` | `e2onset_*.csv`, `e2_transition_*.csv` |
