# NEXT STEPS — Project State & Handoff

Written 2026-07-06. This document is self-contained: any collaborator or
agent should be able to continue the project from here without prior
context. Read this, then `docs/RESULTS.md` (all findings + numbers), then
the module docstrings referenced below.

---

## 1. What this project is

**Working codename RCCV** — regime-conditional conformal volatility.
A paper on distribution-free uncertainty quantification for next-day
realized-volatility forecasts on a 100-name US equity panel (2005–2025,
~510k stock-days, Chicago Booth Risk Lab 5-min RVs).

**Thesis (one sentence, every clause has a table):** volatility forecasts'
uncertainty estimates are only valid on average — they fail precisely in
crises — and a panel-pooled, regime-conditional, self-tuning conformal
layer repairs this for any forecaster (classical or foundation model),
except for a 1–2-day window at spike onset that even option prices cannot
anticipate.

**Critical framing rule:** point-forecast accuracy is held fixed BY DESIGN
(pool ≈ HAR ≈ LGBM, see E0). The contribution is calibration, not
accuracy. The predecessor paper (RIVE, frozen in `archive/rive-2026/`,
rejected from MLWA July 2026 for "insufficient scientific significance")
died because its claim was "our ensemble is more stable." Do not let any
reader expect accuracy gains.

**Honest mechanism decomposition (must be IN the paper, E6):**
panel pooling repairs average stress coverage (per-stock 80.2% → pooled
87.9%); adaptive rates remove all tuning; the regime layer buys regime
transitions (day-2: 75.4→80.2), conditional VaR balance, per-regime
guarantees, and the forward-looking acute-stress group.

**Venue plan:** arXiv (q-fin.ST + stat.ME) as soon as draft is solid →
International Journal of Forecasting. Backups: Quantitative Finance →
J. Financial Econometrics (only if theory lands crisply) → J. Forecasting.
ICAIF short version optional for ML-facing visibility. Do NOT target
NeurIPS/ICML main tracks (method is an assembly, not a new primitive).

## 2. What is DONE (do not redo)

All experiments final, reproducible via `.venv/bin/python scripts/run_all.py`
(fixed seeds, artifact checks, logs → `reports/logs/`). 68 tests green via
`bash scripts/run_tests.sh` (NOT plain pytest — see §6 OpenMP rule).

- E0 point sanity; E1 motivating exhibit; E2 seven-method main table + MCS;
  E3 VaR with GARCH-t/CAViaR/FHS + Kupiec/Christoffersen/DQ; E4 Chronos;
  E5 adaptive rates; E6 ablations + K-decomposition; onset/irreducibility
  (VIX9D acute group); E9 crisis episodes; E10 disjoint-halves holdout;
  E12 alpha sweep; regime-estimator robustness; identity/dedup/truncation
  audits. **All numbers and artifact paths: `docs/RESULTS.md`.**
- Method implementation: `src/conformal/panel_hierarchical.py` (READ ITS
  DOCSTRING — it records the design lessons). Baselines in
  `src/conformal/` and `src/forecasters/quantile_baselines.py`.
- Per-stock loops are parallelized (`src/utils/parallel.py::pmap`) — keep
  it that way; the machine is a 10-core M1 Max.

## 3. Waiting on WRDS access (user applied ~2026-07-06, ETA ~10 days)

When access arrives, in order:

1. **Point-in-time universe.** Build "top-100 by market cap as of each
   January, incl. later-dead names" from CRSP (msf/msenames + delist).
   Replace `PROVISIONAL_TOP100` in `src/data/universe.py` (function
   `get_universe` has a NotImplementedError stub for this path). Log every
   query verbatim in `docs/wrds_queries.md` (format defined there).
2. **Rebuild panel** (`scripts/build_rv_panel.py`) — Risk Lab fetch keyed
   by PERMNO already; the identity audit (`scripts/audit_universe_identity.py`)
   must pass; staleness guard in `src/data/panel.py` will catch mismatches.
3. **Regenerate everything:** `scripts/run_all.py` (~1-2h). Diff every
   table in `docs/RESULTS.md`; conclusions are expected to strengthen or
   hold (calibration claims have no survivorship mechanism).
4. **TAQ verification subset (optional, referee armor):** recompute 5-min
   RV from raw TAQ for ~25 names × a few years using
   `src/data/rv_estimators.py` (BNHLS cleaning implemented + unit-tested);
   report median per-stock log-RV correlation vs Risk Lab (target >0.95).
5. If WRDS does NOT come through: fallback documented in §5 item G.

## 4. Paper writing (LAST step, after WRDS regen; skeleton may start anytime)

`paper2/` has stubs (main.tex elsarticle, refs.bib, sections/, figures/).

- Structure: Intro (E1 exhibit as Figure 1, framing rule from §1) →
  Related work (explicit gap paragraph vs 2025-26 papers: POGO 2606.00419,
  switching-SSM+ACI 2512.03298, TCP 2507.05470, conformal VaR 2603.22569,
  HopCPT/NexCP for similarity weighting — our gap: online-estimated
  regimes + panel pooling + one-sided heads + conditional evaluation) →
  Method (three layers, algorithm box; state the POGO equivalence of the
  hard-regime hand-tuned variant explicitly) → Theory (§5A) → Data →
  Results (E2→E3→E4, decomposition E6, episodes E9) → onset/irreducibility
  → limitations (survivorship §5G, single market, day-2).
- **MLWA red-team checklist (all four complaints must be answered):**
  R1 leakage → no text data, leakage assert in walkforward, causality
  tests; R1 unexplained features → gone; R1+R2 stale refs → 2023-26-heavy
  bib; R2 format → descriptive title (candidates in the plan file),
  numbered equations, vector figures, FULL affiliation (ASK USER for
  Villanova department), co-author Venkat Margapuri.
- Figures to build (none exist; all data in `reports/*.csv`):
  coverage-by-regime bars (E1 vs E2), width dynamics around 2020-03,
  effective-eta paths per regime (attrs saved by adaptive runs),
  days-since-entry profile, episode panel, decomposition chart.
- Title candidates + full approved plan:
  `~/.claude/plans/what-do-you-think-validated-platypus.md` (user-side file).

## 5. Work that can be done NOW (no WRDS, no paper) — ordered queue

Each item is self-contained with file pointers.

**A. Theory (the careful one — Props 1-3).**
Write LaTeX + proofs in `paper2/sections/theory.tex`:
  P1: hard-regime per-group coverage for the pooled tracker (adapt
  Gibbs–Candès 2021 threshold-tracking bound per regime; per-observation
  sum-steps mean the effective step count is stock-days in regime).
  P2: soft memberships — pi-weighted coverage error → 0 at
  O(1/sqrt(sum_t pi_t(k))) via OGD regret; marginal validity of the
  mixture because sum_k pi_k = 1. Match the update in
  `src/conformal/panel_hierarchical.py` EXACTLY (issued-interval rule).
  P3: one-sided analogue.
  Honesty remark: guarantees condition on the algorithm's own filtered
  state (F_t-measurable), not the latent regime; degradation bounded by
  filter TV error; point to oracle ablation (item B).
  STRETCH (high-ceiling, separate paper if it cracks): (i) coverage bounds
  with ESTIMATED groups (existing literature assumes groups given);
  (ii) impossibility/lower bound for day-2 onset coverage. User wants
  strong-model effort reserved for exactly these two.

**B. Oracle-regime ablation (cheap, supports the honesty remark).**
Full-sample smoothed HMM memberships (oracle) vs our filtered ones in the
calibrator; add row to E6. Reuse `src/regimes/online_hmm.py` (fit once on
all data, predict smoothed probs — explicitly labeled leaky-by-design).

**C. Pool expansion (compute-only).** Add GRU (`src/forecasters/neural.py`,
torch — MUST run in its own process, §6) and Chronos
(`data/processed_v2/tsfm_predictions.parquet` already has q50) as pool
members in `scripts/e0_point_sanity.py` (EXPERTS list) → rerun run_all.
Expectation: nothing changes materially; purpose is making "heterogeneous
pool" literally true.

**D. Coverage-difference significance tests.** Date-clustered (HAC over
days) tests for stress-coverage differences rc vs each baseline; add to
`scripts/e2_full.py` output. Machinery: `src/eval/dm_hac.py` pattern.

**E. Fold the K-decomposition into scripts.** The day-2/dse and VaR K=1
vs K=4 numbers in `reports/e6_k_decomposition_notes.md` were produced by
session-log snippets; add them to `scripts/e6_ablations.py` so they are
artifact-reproducible.

**F. Survivorship robustness WITHOUT WRDS.** Expand to Risk Lab's full
available cross-section (loader: `src/data/risklab.py`; universe list is
the constraint — enumerate available tickers via the symbol endpoint) or
slice current results by 2005-incumbents vs later additions. Purpose:
show conclusions are not an artifact of the provisional top-100.

**G. If WRDS never arrives:** F becomes the primary defense + limitation
paragraph; VOLARE (volare.unime.it) spot-check for measurement validation.

**H. Optional stretch experiment (days, decide before scope-lock):**
crypto or FX replication (free HF data; BTC/ETH realized vol) — converts
the finding into a cross-asset stylized fact and is the single
highest-leverage addition left.

## 6. Conventions & gotchas (violating these has burned us)

- **NEVER delete anything.** Archive (`git mv`) instead. `archive/rive-2026/`
  and `mlruns/` stay untouched.
- **OpenMP rule:** torch and lightgbm each bundle their own libomp.dylib;
  importing both in one process SEGFAULTS on macOS. Run tests via
  `bash scripts/run_tests.sh` (two isolated groups); GRU/Chronos work goes
  in separate scripts/processes. KMP_DUPLICATE_LIB_OK does NOT fix it.
- **Parallelize** per-stock/per-config loops with `src/utils/parallel.py`
  pmap (module-level workers — macOS spawn can't pickle closures).
- **Threshold-tracking sign:** updates are `q += eta*(err - a_side)`
  (widen on miss). The alpha-tracking parameterization has the opposite
  sign; tests enforce ours.
- **Recycled tickers:** identity via PERMNO_OVERRIDES + liveness rule in
  `src/data/risklab.py`; never trust ticker-only matches (META trap).
- Background jobs: write full logs to `reports/logs/` (no `tail` pipes).
- Config: `conf/base/config.yaml`; seeds fixed via `src/utils/seeding.py`.
- User context: hasty typing, read for intent; full delegation; wants
  strong-model effort reserved for §5A stretch theory, everything else
  default effort.

## 7. Current status snapshot

Phases P0–P5 complete (tasks #1–#6). P6 (theory+paper) pending. Last
commits: adaptive rates 4afe71c, baselines+E10 fa3e48e, E9+MCS 319b967,
E2 final 523a980, parallelization 6cd0c81, E6 7f5c5dc. Memory files for
Claude sessions live in the user's `~/.claude` project memory; this file
is the repo-side source of truth.
