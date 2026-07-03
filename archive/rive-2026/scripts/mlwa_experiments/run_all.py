"""
MLWA Experiments: Master Runner
================================

Runs all three experiments for the MLWA paper repositioning:

  1. Monolithic LightGBM regressor on full feature set
  2. Elastic Net baseline on full feature set
  3. Rolling walk-forward evaluation with periodic refitting

Produces a unified comparison table at the end.

Usage:
    python scripts/mlwa_experiments/run_all.py
    python scripts/mlwa_experiments/run_all.py --skip-rolling   # skip exp 3 (slow)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlwa_experiments.assemble_features import assemble_full_features
from scripts.mlwa_experiments.exp1_monolithic_lightgbm import train_monolithic_lightgbm
from scripts.mlwa_experiments.exp2_elastic_net import train_elastic_net
from scripts.mlwa_experiments.exp3_rolling_walkforward import run_rolling_walkforward


def main():
    parser = argparse.ArgumentParser(description="Run all MLWA experiments")
    parser.add_argument("--skip-rolling", action="store_true",
                        help="Skip Experiment 3 (rolling walk-forward)")
    args = parser.parse_args()

    start = datetime.now()

    print("\n" + "=" * 70)
    print("  MLWA PAPER EXPERIMENTS")
    print("  Repositioning RIVE: Expert System -> ML System")
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Shared feature assembly (done once)
    df, feature_cols = assemble_full_features()

    # ---- Experiment 1 ----
    print("\n")
    res_lgbm = train_monolithic_lightgbm(df, feature_cols)

    # ---- Experiment 2 ----
    print("\n")
    res_enet = train_elastic_net(df, feature_cols)

    # ---- Experiment 3 ----
    res_rolling = None
    if not args.skip_rolling:
        print("\n")
        res_rolling = run_rolling_walkforward(df, feature_cols)

    # ==================================================================
    # UNIFIED COMPARISON TABLE
    # ==================================================================
    print("\n" + "=" * 70)
    print("  UNIFIED RESULTS: FIXED-SPLIT EVALUATION")
    print("=" * 70)
    print(f"""
  +-------------------------------+----------+----------+----------+
  | Model                         | Test R²  | RMSE     | MAE      |
  +-------------------------------+----------+----------+----------+
  | Elastic Net (linear mono.)    | {res_enet['test_r2']:>8.4f} | {res_enet['test_rmse']:>8.4f} | {res_enet['test_mae']:>8.4f} |
  | LightGBM (nonlinear mono.)    | {res_lgbm['test_r2']:>8.4f} | {res_lgbm['test_rmse']:>8.4f} | {res_lgbm['test_mae']:>8.4f} |
  | RIVE (modular ensemble)       |   0.2304 |      --- |      --- |
  +-------------------------------+----------+----------+----------+
""")

    # Sector comparison
    print("  Sector-Level R² Comparison:")
    print("  " + "-" * 65)
    print(f"  {'Sector':<15} {'Elastic Net':>12} {'LightGBM':>12} {'RIVE (ref)':>12}")
    print("  " + "-" * 65)

    all_sectors = sorted(set(list(res_lgbm["sector_r2"].keys())
                             + list(res_enet["sector_r2"].keys())))
    for s in all_sectors:
        enet_r2 = res_enet["sector_r2"].get(s, float("nan"))
        lgbm_r2 = res_lgbm["sector_r2"].get(s, float("nan"))
        print(f"  {s:<15} {enet_r2:>12.4f} {lgbm_r2:>12.4f} {'---':>12}")

    if res_rolling is not None:
        print(f"\n{'=' * 70}")
        print("  ROLLING WALK-FORWARD EVALUATION")
        print("=" * 70)

        pooled = res_rolling["pooled_r2"]
        avg = res_rolling["avg_r2"]
        n = res_rolling["n_folds"]
        wr_lgbm = res_rolling["rive_wins_lgbm"]
        wr_enet = res_rolling["rive_wins_enet"]

        print(f"""
  +-------------------------------+------------------+------------------+
  | Model                         | Pooled OOS R²    | Mean Fold R²     |
  +-------------------------------+------------------+------------------+
  | Elastic Net                   | {pooled['ElasticNet']:>16.4f} | {avg['ElasticNet']:>16.4f} |
  | Monolithic LightGBM           | {pooled['LightGBM']:>16.4f} | {avg['LightGBM']:>16.4f} |
  | RIVE                          | {pooled['RIVE']:>16.4f} | {avg['RIVE']:>16.4f} |
  +-------------------------------+------------------+------------------+

  RIVE Win Rate:
    vs LightGBM:  {wr_lgbm}/{n} folds ({wr_lgbm/n*100:.0f}%)
    vs ElasticNet: {wr_enet}/{n} folds ({wr_enet/n*100:.0f}%)
""")

    # Conclusion
    print("=" * 70)
    print("  INTERPRETATION GUIDE (for paper)")
    print("=" * 70)
    print("""
  The three-way comparison addresses the key ML reviewer concern:

  1. If RIVE > LightGBM > Elastic Net:
     -> Modularity AND nonlinearity both contribute.
     -> Strongest argument for the regime-aware design.

  2. If LightGBM ~ RIVE > Elastic Net:
     -> Benefit is mainly from nonlinear feature interactions.
     -> Argue RIVE adds interpretability + regime transparency.

  3. If LightGBM > RIVE:
     -> Monolithic model is sufficient on this feature set.
     -> Reframe paper toward interpretability / decomposability.

  The walk-forward results strengthen whichever case applies by
  showing consistency under changing market conditions.
""")

    end = datetime.now()
    print(f"  Total duration: {end - start}")
    print("=" * 70)


if __name__ == "__main__":
    main()
