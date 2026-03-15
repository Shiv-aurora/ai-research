"""
Experiment 7: Formalized Walk-Forward Comparison Table
=======================================================

Produces a publication-ready table from the Experiment 3 rolling
walk-forward results. If rolling_results.csv exists, uses it directly.
Otherwise, re-runs the walk-forward evaluation.

Output table:
    Model | Pooled OOS R² | Mean Fold R² | Std | Min | Max | Folds > 0

Usage:
    python scripts/mlwa_experiments/exp7_walkforward_table.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_PATH = Path(__file__).parent / "rolling_results.csv"


def main():
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: WALK-FORWARD COMPARISON TABLE")
    print("=" * 70)

    if RESULTS_PATH.exists():
        results_df = pd.read_csv(RESULTS_PATH)
        print(f"  Loaded {len(results_df)} folds from {RESULTS_PATH.name}")
    else:
        print("  rolling_results.csv not found. Running Experiment 3...")
        from scripts.mlwa_experiments.exp3_rolling_walkforward import main as run_exp3
        res = run_exp3()
        results_df = res["results_df"]

    # Compute aggregate stats
    models = {
        "RIVE": "r2_rive",
        "Monolithic LightGBM": "r2_lgbm",
        "Elastic Net": "r2_enet",
    }

    # Pooled OOS R² (not stored in CSV — approximate with mean weighted by n_test)
    # For exact pooled R², we'd need raw predictions. Use mean as proxy.
    # Note: if run from exp3 directly, pooled is available. From CSV, use mean.

    n_folds = len(results_df)

    print(f"\n  Number of folds: {n_folds}")
    print(f"  Evaluation: expanding-window, quarterly refitting")
    print(f"  Period: {results_df['test_start'].min()} to {results_df['test_end'].max()}")

    # ---- MAIN TABLE ----
    print(f"\n{'=' * 85}")
    print(f"  TABLE: Walk-Forward Model Comparison ({n_folds} Quarterly Folds)")
    print(f"{'=' * 85}")

    header = (f"  {'Model':<25} {'Mean R²':>9} {'Std':>8} "
              f"{'Min':>8} {'Max':>8} {'Folds>0':>8} {'RMSE_avg':>9}")
    print(header)
    print("  " + "-" * 80)

    table_rows = []
    for name, col in models.items():
        r2_vals = results_df[col]
        rmse_col = col.replace("r2_", "rmse_")
        rmse_vals = results_df[rmse_col] if rmse_col in results_df.columns else None

        row = {
            "Model": name,
            "Mean_R2": r2_vals.mean(),
            "Std_R2": r2_vals.std(),
            "Min_R2": r2_vals.min(),
            "Max_R2": r2_vals.max(),
            "Folds_positive": (r2_vals > 0).sum(),
            "RMSE_avg": rmse_vals.mean() if rmse_vals is not None else float("nan"),
        }
        table_rows.append(row)

        folds_str = f"{row['Folds_positive']}/{n_folds}"
        rmse_str = f"{row['RMSE_avg']:.4f}" if not np.isnan(row["RMSE_avg"]) else "---"
        print(f"  {name:<25} {row['Mean_R2']:>+9.4f} {row['Std_R2']:>8.4f} "
              f"{row['Min_R2']:>+8.4f} {row['Max_R2']:>+8.4f} {folds_str:>8} "
              f"{rmse_str:>9}")

    print("  " + "-" * 80)

    # ---- PAIRWISE WIN RATES ----
    print(f"\n  Pairwise Win Rates (fold-level R²):")
    print("  " + "-" * 50)

    rive_vs_lgbm = (results_df["r2_rive"] > results_df["r2_lgbm"]).sum()
    rive_vs_enet = (results_df["r2_rive"] > results_df["r2_enet"]).sum()
    lgbm_vs_enet = (results_df["r2_lgbm"] > results_df["r2_enet"]).sum()

    print(f"    RIVE > LightGBM:   {rive_vs_lgbm}/{n_folds} folds ({rive_vs_lgbm/n_folds*100:.0f}%)")
    print(f"    RIVE > Elastic Net: {rive_vs_enet}/{n_folds} folds ({rive_vs_enet/n_folds*100:.0f}%)")
    print(f"    LGBM > Elastic Net: {lgbm_vs_enet}/{n_folds} folds ({lgbm_vs_enet/n_folds*100:.0f}%)")

    # ---- STABILITY RANKING ----
    print(f"\n  Stability Ranking (lower std = more stable):")
    print("  " + "-" * 50)
    ranked = sorted(table_rows, key=lambda x: x["Std_R2"])
    for i, r in enumerate(ranked, 1):
        print(f"    {i}. {r['Model']:<25} std = {r['Std_R2']:.4f}")

    # ---- WORST-CASE ANALYSIS ----
    print(f"\n  Worst-Case Analysis:")
    print("  " + "-" * 50)

    worst_fold_idx = results_df["r2_rive"].idxmin()
    worst_fold = results_df.loc[worst_fold_idx]
    print(f"    Hardest quarter: {worst_fold.get('test_start', 'N/A')}")
    print(f"      RIVE:       {worst_fold['r2_rive']:+.4f}")
    print(f"      LightGBM:   {worst_fold['r2_lgbm']:+.4f}")
    print(f"      Elastic Net: {worst_fold['r2_enet']:+.4f}")

    # ---- PER-FOLD DETAIL ----
    print(f"\n  Per-Fold Detail:")
    print("  " + "-" * 75)
    print(f"  {'Fold':<8} {'Start':>12} {'End':>12} {'RIVE':>9} {'LGBM':>9} {'ENET':>9} {'Best':>10}")
    print("  " + "-" * 75)

    for _, r in results_df.iterrows():
        vals = {"RIVE": r["r2_rive"], "LGBM": r["r2_lgbm"], "ENET": r["r2_enet"]}
        best = max(vals, key=vals.get)
        start = str(r.get("test_start", ""))[:10]
        end = str(r.get("test_end", ""))[:10]
        print(f"  {r.get('fold', ''):>8} {start:>12} {end:>12} "
              f"{r['r2_rive']:>+9.4f} {r['r2_lgbm']:>+9.4f} {r['r2_enet']:>+9.4f} "
              f"{best:>10}")

    # ---- LATEX-READY TABLE ----
    print(f"\n{'=' * 70}")
    print("  LATEX TABLE (copy-paste ready)")
    print(f"{'=' * 70}")
    print(r"""
\begin{table}[htbp]
\centering
\caption{Walk-Forward Model Comparison (%d Quarterly Folds)}
\label{tab:walkforward}
\begin{tabular}{lcccccc}
\toprule
Model & Mean $R^2$ & Std & Min & Max & Folds $>0$ & Avg RMSE \\
\midrule""" % n_folds)

    for r in table_rows:
        folds_str = f"{r['Folds_positive']}/{n_folds}"
        rmse_str = f"{r['RMSE_avg']:.4f}" if not np.isnan(r["RMSE_avg"]) else "---"
        print(f"{r['Model']} & {r['Mean_R2']:+.4f} & {r['Std_R2']:.4f} & "
              f"{r['Min_R2']:+.4f} & {r['Max_R2']:+.4f} & {folds_str} & {rmse_str} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    # Save
    out_df = pd.DataFrame(table_rows)
    out_path = Path(__file__).parent / "walkforward_table.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Table saved to: {out_path}")

    return table_rows


if __name__ == "__main__":
    main()
