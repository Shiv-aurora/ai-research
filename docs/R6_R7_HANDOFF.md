# R6/R7 Handoff — state as of 2026-07-13

Working doc for the current review round (the long referee-style review
received 2026-07-13, pasted in chat). If you are a fresh session: read
this file, then `docs/RESULTS.md` (E15–E18 sections) and
`~/.claude/projects/.../memory/project-rccv-flagship.md`. Repo:
`github.com/Shiv-aurora/ai-research`, branch `main`, last pushed commit
`50db373` (R5). **UPDATE: R6 committed as `eb5487e` and pushed. Steps 1-5 of the remaining list are DONE (RESULTS.md E2/E3 updated, PDF rebuilt clean 45pp/0 undefined/0 'Appendix Appendix'/0 visible 'oracle', figures regenerated with de-oracled legends, 76+2 tests green). Remaining = step 6 (R7 leftovers, chiefly the figure overhaul) and steps 7-8.**

## Context

R6 = empirical workstream answering the review's substantive demands.
R7 = the results-dependent rewrite/format pass (task #16, blocked on R6).
Standing rules: never delete; no arXiv; "hindsight" never "oracle";
wording de-escalations from R5 stay; WRDS token only via env var;
CRSP data never published.

## R6 — DONE (all experiments run, all results final)

### New experiments (scripts + artifacts all exist and ran clean)

| ID | Script | What it showed |
|---|---|---|
| E15 | `scripts/e15_expanded_grid.py` | **The decisive test.** Expanded expert grid (+0.128/0.256/0.512): per-stock adaptive FINDS the fast rates (eff. stress rate .175) and still trails RC by 4.6pp (p<.0001); widened DtACI −2.7pp (p=.039); averaged-error pooled −3.3pp (p=.0002); RC identical under either grid (.8827/.8828). Fast-rate objection refuted; mechanism = rate + variance reduction. |
| E16 | `scripts/e16_bounds_inference.py` | Numeric theorem bounds: Prop2 stress = **1.048 (vacuous)**, calm .066, normal .153, elevated .309. Episode-block bootstrap (18 episodes): RC−ACI stress +5.2pp CI [−0.1,+9.7] p=.057 (borderline, era-heterogeneous); vs XS-panel CI [+2.8,+8.9] p=.0004. HAC lag sensitivity: p=.002 at rule lag 4, stable lags 0–22, p=.07 at 66. |
| E17 | `scripts/e17_k1_vs_k4.py` | K=1 vs K=4 inference: per-regime coverage/IS diffs ALL ns (stress +0.7pp p=.53); day-2 diff +5.3pp CI [−3.8,+14.5] (15 dates); dispersion favors K=4 but ns (p=.44). 57 market stress entries. |
| E18 | `scripts/e18_onset_uncertainty.py` | Day-2: 1,320 obs on **14 dates**, cov .797 HAC CI [.65,.94]; LOO-episode range [.764,.848]; threshold trajectory 2.30→2.37→3.78 (tracker moves one day late); widening overlay Pareto: ×1.5 → day-2 .895 at +2.5% stress-interior IS, +50% onset width. Onset = "unpurchased, not uncoverable". |
| E2 rerun | `scripts/e2_full.py` | Now **11 methods** incl. POGO (Bharti et al. 2606.00419, implemented in `src/conformal/pogo.py`, run per stock with our VIX bins as groups): marginal .858, stress .771 (−11pp vs RC t=8.4) — consistent with its own loose parameter-free bound (~2.5pp/side at T≈3,800; vacuous for 150-round stress group). MCS eliminates sfogd/tcp_rm/xs_panel/pogo. Common sample UNCHANGED 269,705. New artifact `reports/e2_daily_coverage.parquet` feeds E16. |
| E3 rerun | `scripts/e3_var.py` | New columns: Christoffersen INDEPENDENCE (honest relabel), combined CC (2df), DQ-stress (stress-indicator regressor), balance z (date-clustered calm-vs-stress test), pooled_k1 arm. KEY: at 5% VaR only RC balances calm/stress (z=0.3–0.5); all baselines overshoot (z=2.1–3.9); pooled_k1 inverts (z=−1.8). At 99% pooled_k1 best CC/DQ-S (.91/.73 vs RC .87/.64) — reported honestly. |

### Code changes (src/)
- `src/conformal/pogo.py` NEW — POGO port (universal portfolio coin-betting, per-side, midpoint-grid Jeffreys integral). Tests: `tests/test_pogo.py` (3 pass).
- `src/conformal/panel_hierarchical.py` — `average_errors` now works in the adaptive path (denom applied to bank + corrector). Test: `tests/test_adaptive_avg_errors.py` (exact equivalence, passes).
- `src/conformal/dtaci.py` — `run_dtaci(..., etas=...)` param exposed.
- `src/eval/var_backtests.py` — `christoffersen_cc()` added; `dq_test(extra=...)`; `backtest_panel` renamed christoffersen_p→independence_p, added cc_p, dq_stress_p, stress_col param.
- `scripts/run_all.py` — STEPS extended with e15–e18.
- `docs/replication_manifest.md` — rows for E15–E18 added.

### Paper edits (paper2/) — ALL APPLIED, not yet rebuilt
- `main.tex`: abstract rewritten (149 words, post-2018 concentration added, "finite-sample" softened); highlight 5 → "persistent onset limitation"; AI-use declaration section added before references.
- `sections/results.tex`: tab:main 11 rows (+POGO); "Six readings" (new 6th = POGO/group-guarantee reading); MCS sentence +POGO; full HAC spec written out (Bartlett, lag rule, event time, ±=1 SE); episode bootstrap + lag sensitivity + insurance framing added; baseline intro paragraph now describes XS-panel as adaptation + POGO; pool-vs-LightGBM DM p=.387 reported; tab:var rebuilt (Balance z + Kupiec/CC/DQ-S columns, pooled_k1 row, honest 99% nuance); fig:var caption fixed; Chronos wording scoped to checkpoint + point-location-bias caveat.
- `sections/ablations.tex`: "Fast rates are necessary but not sufficient" paragraph + tab:expgrid (E15); "What regimes add, with uncertainty attached" paragraph rewritten with E17/E3 inference (transition CI, ns slices, dispersion, VaR balance z, certified-vs-uncertified reading).
- `sections/onset.tex`: title → "a persistent limitation"; causal claims bounded (objective-weighting caveat; "this signal, not option prices generally"); NEW subsection `sec:onset-pareto` (entry counts, day-2 CI, LOO, trajectories, Pareto, reframed open problem).
- `sections/theory.tex`: NEW `Remark rem:numeric` after Prop 2 proof — numeric bounds incl. vacuous stress value, realized-identity mitigation.
- `sections/conclusion.tex`: limitations 4→7 (bound vacuity, no untouched holdout, ns average K gain); "21 years" conclusion claim replaced with two-era statement; VaR claim rescoped to balance.
- `sections/intro.tex`: contribution 2 references rem:numeric; contribution 3 "removes manual regime-specific learning-rate selection"; contribution 5 → "A persistent onset limitation"; Chronos "points the wrong way" → "asymmetrically so".
- `sections/related.tex`: Bharti noted as implemented; three closest proposals paragraph (TCP-RM constants, XS-panel adaptation honesty, POGO); Zhong verb fix; "smallest version of the claim" sentence added to gap paragraph.
- `sections/data.tex`: dataset citations (\citep{crsp,risklab,fred,cboe}); survivor claim narrowed; 4,002 eval dates / 150 stress dates stated; data-vs-evaluation span split stated; flow table +POGO row (356,172), "eleven methods".
- `sections/method.tex`: acute group = five-state hard partition note; "Appendix Appendix" fix.
- `sections/appendix_algorithm.tex`: unified explicit timeline paragraph (close t−1 issue → close t observe); 1/K fallback honestly scoped vs theorem.
- `refs.bib`: Noguer i Alonso brace-protected (sorts under N); dataset @misc entries appended.

### docs/RESULTS.md — E15, E16, E17, E18 sections added.

## REMAINING — exact next steps, in order

1. **RESULTS.md E2/E3 sections** still describe 10 methods / old VaR
   columns. A python3 heredoc edit was blocked by a transient
   permission-classifier outage — redo it: update E2 header to eleven
   methods + add pogo row (.8577/.7706/.8854/1.743/2.140) + MCS
   sentence + POGO interpretation; update E3 table (pooled_k1 3.71/1.17
   row, rc_panel 5.49/1.61, rc_adaptive 5.73/1.76) + relabel note +
   balance-z finding + 99% pooled_k1 nuance.
2. **Rebuild PDFs**: `cd paper2 && latexmk -pdf -interaction=nonstopmode
   main.tex` (and titlepage.tex unchanged). Fix any LaTeX errors
   (check the new tables/labels: tab:expgrid, sec:onset-pareto,
   rem:numeric referenced from intro/conclusion/onset). Check 0
   undefined refs.
3. **Consistency sweep after rebuild** (grep the PDF text):
   - no remaining "ten methods", "Five readings", "honest limit",
     "oracle" in prose, "Appendix Appendix";
   - abstract ≤150 words (currently 149);
   - every number cited in new text matches reports/*.csv (E15 table
     rounds .8828→.883 style, day-2 .797, 14 dates, z values).
4. **Full test suite**: `bash scripts/run_tests.sh` from repo root
   (76 tests expected: 72 + 3 pogo + 1 avg-errors).
5. **Commit R6** (single commit, message summarizing: E15 expanded grid
   refutes fast-rate objection; POGO implemented as 11th method; E16
   numeric bounds + episode bootstrap; E17 K-inference honesty; E18
   onset uncertainty + Pareto; VaR upgrade; all paper edits). Push.
6. **R7 leftovers** (task #16) — mostly done inline already; remaining:
   - Highlights → separate file (`paper2/highlights.tex`, standalone)
     and drop from anonymized main.tex? (IJF wants separate upload;
     keep or move — decide, low risk either way.)
   - Figures (reviewer): truncated axes on figs 1/3/6 → point-and-
     whisker with CIs; fig:var and fig:mechanism partially redundant
     with tables; fig:dse (fig 7) needs event counts + CIs (data now in
     reports/e18_dse_ci.csv). `scripts/make_figures.py` regenerates.
     This is the largest remaining chunk.
   - arXiv capitalization consistency in refs.bib (minor).
   - Consider reviewer's "title should emphasize panel-pooled
     calibration" — USER DECISION, do not change unilaterally. Flag in
     final report.
   - Cover-letter items (user-facing, not in repo): prior-version
     disclosure (RIVE/MLWA rejection) required by IJF; 4–6 suggested
     reviewers; replication zip.
7. **Memory update**: rewrite the R6/R7 block in
   `~/.claude/projects/-Users-shivamarora-Documents-research-final/memory/project-rccv-flagship.md`
   when committing.
8. **Report to user**: lead with E15 (thesis survived), then the two
   honesty findings (episode bootstrap p=.057; K=1-vs-K=4 interval
   claims uncertified, VaR balance certified), onset reframe
   (unpurchased not uncoverable), POGO result. Note title question.

## Key numbers cheat-sheet (for consistency checks)

- Main: RC stress .8812 (adaptive) / .8818 (hand); pooled_k1 .8731;
  ACI .8292; TCP-RM .8488; XS-panel .8272; POGO .7706; common 269,705.
- E15: ps_k4_exp .8366 (−4.6, p<.0001); dtaci_exp .8548 (−2.7, .039);
  pooled_avg_exp .8497; eff rates: ps_k4_exp .175, rc_std .010.
- E16: bounds .066/.153/.309/1.048; boot ACI +5.2 [−.007,.097] p=.057;
  XS [+.028,+.089] p=.0004; HAC lag4 p=.0023, lag66 p=.072.
- E17: stress diff +0.67pp p=.53; day-2 +5.3 CI [−3.8,+14.5]; disp rms
  .0034 vs .0054 p=.44; 57 entries.
- E18: day-2 .797 se .072, 14 dates; LOO [.764,.848]; traj 2.30/2.37/
  3.78; Pareto m=1.5 → .895, IS 3.09 vs 3.02, width 3.07 vs 2.05.
- E3: balance z: normal 2.63, fhs 2.23, garch 3.53, caviar 3.92, aci
  2.08, k1 −1.79, rc .29, rca .52. 99% stress: k1 1.17, rc 1.61,
  rca 1.76. Pass 99% (kup/cc/dqS): k1 .83/.91/.73; rca .88/.87/.64.
- E0: pool vs LGBM DM +0.86 p=.387.
