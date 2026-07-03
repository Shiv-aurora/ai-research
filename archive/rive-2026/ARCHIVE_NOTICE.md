# RIVE (2025–2026) — Frozen Archive

This directory is a complete, frozen snapshot of the **RIVE (Regime-Integrated Volatility Ensemble)** project as of July 2026, archived when the research pivoted to its successor (regime-conditional conformal prediction for realized volatility — see the repo root).

**Nothing here should be modified.** It exists so that every RIVE result remains reproducible and citable as prior work.

## Status at freeze

- Paper (`mlwa_paper.tex`) was submitted to *Machine Learning with Applications* and **rejected** (associate editor: insufficient scientific significance; Reviewer 1: suspected news-feature look-ahead, underexplained PCA features, stale references; Reviewer 2: title/formatting/affiliation issues).
- Headline results: fixed-split R² ≈ 23% for RIVE / LightGBM / Elastic Net (statistically indistinguishable from LightGBM, DM p = 0.39); RIVE had the lowest walk-forward fold variance (std 0.099 vs 0.562 for LightGBM) and best top-decile tail detection (ROC-AUC 0.8051).
- Known unresolved discrepancy: README table says RIVE fixed-split R² = 22.81% (commit 57a9bfc, "exact" recomputation); the paper says 23.04% with identical RMSE/MAE.

## Known defect found in post-rejection audit (July 2026)

**Confirmed data leakage in the news feature pipeline**: `src/pipeline/process_news.py` fits the TF-IDF vocabulary (line ~151) and TruncatedSVD embedding space (line ~158) on the **full sample** (train + test). The downstream PCA is train-only (line ~269), but operates on contaminated embeddings — so `news_pca_0..19` and `novelty_score` partially encode test-period news structure. The effective-date timing itself (4 PM ET cutoff, `ingest_news.py`) was verified correct. Reviewer 1's suspicion was directionally right, for a different mechanism than they guessed. The successor project uses no text features.

Also known: the walk-forward comparison in `scripts/mlwa_experiments/exp3_rolling_walkforward.py` does not retrain the real three-agent pipeline per fold (it proxies RIVE with a Ridge on raw features), and applies asymmetric guardrails across models (RIVE: winsorized target; Elastic Net: clipped predictions; LightGBM: none).

## Path map (old → archived)

| Old path | Archived path |
|---|---|
| `src/` | `archive/rive-2026/src/` |
| `scripts/` | `archive/rive-2026/scripts/` |
| `conf/` | `archive/rive-2026/conf/` |
| `audits/`, `models/`, `paper/`, `reports/`, `tests/`, `visuals/` | `archive/rive-2026/<same>` |
| `README.md`, `REPRODUCIBILITY.md`, `CITATION.cff`, `requirements.txt` | `archive/rive-2026/<same>` |
| `paper.tex`, `mlwa_paper.tex`, `final_report.tex` | `archive/rive-2026/<same>` (force-added; previously gitignored) |
| `data/` | **not moved** — shared at repo root; `archive/rive-2026/data` is a symlink to `../../data` so archived scripts run unmodified |
| `mlruns/` | **not moved** — shared at repo root |

## Reproducing RIVE results

```bash
cd archive/rive-2026
pip install -r requirements.txt
python scripts/mlwa_experiments/run_all.py          # MLWA experiment suite (uses ../../data/processed)
```

Regenerating `data/processed/*` from scratch requires a Polygon.io subscription (lapsed) — the processed parquets are preserved and sufficient for all experiments.
