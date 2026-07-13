"""Reproduce every table in the paper from scratch, in dependency order.

Runs each experiment script as a subprocess (fresh interpreter per script —
this also enforces the torch/lightgbm OpenMP isolation rule) and verifies
its expected artifacts exist afterwards. Total runtime is dominated by the
E0 walk-forward refits and the E3 GARCH/CAViaR fits.

Usage: .venv/bin/python scripts/run_all.py [--from STEP]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"

STEPS: list[tuple[str, list[str]]] = [
    # (script, expected artifacts under reports/ or data/processed_v2/)
    ("e0_point_sanity",   ["reports/e0_point_sanity.csv",
                           "data/processed_v2/e0_predictions.parquet"]),
    ("e1_prototype",      ["reports/e1_prototype_coverage_by_state.csv"]),
    ("e2_panel",          ["reports/e2_panel_summary.csv"]),
    ("e2_onset",          ["reports/e2onset_rc_onset_dse.csv"]),
    ("e2_full",           ["reports/e2_full_summary.csv",
                           "reports/e2_full_mcs.csv"]),
    ("e3_var",            ["reports/e3_var_summary.csv"]),
    ("e4_tsfm",           ["reports/e4_tsfm_raw_coverage.csv",
                           "reports/e4_tsfm_repaired_coverage.csv"]),
    ("e5_adaptive_rates", ["reports/e5_adaptive_rates.csv"]),
    ("e6_ablations",      ["reports/e6_ablations.csv"]),
    ("e6b_oracle_regimes", ["reports/e6b_oracle_regimes.csv"]),
    ("e6c_dse_by_k",      ["reports/e6_dse_by_k.csv"]),
    ("e6d_reclassified_eval", ["reports/e6d_reclassified_eval.csv"]),
    ("e9_stress_windows", ["reports/e9_stress_windows.csv"]),
    ("e10_holdout",       ["reports/e10_holdout.csv"]),
    ("e12_alpha_sweep",   ["reports/e12_alpha_sweep.csv"]),
    ("e13_pooling_mechanism", ["reports/e13_pooling_mechanism.csv"]),
    ("e14_temporal_holdout", ["reports/e14_subperiod.csv",
                              "reports/e14_loco.csv"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None,
                    help="resume from this step name")
    args = ap.parse_args()

    names = [s for s, _ in STEPS]
    start = names.index(args.start) if args.start else 0

    (ROOT / "reports" / "logs").mkdir(parents=True, exist_ok=True)
    for name, artifacts in STEPS[start:]:
        t0 = time.time()
        log = ROOT / "reports" / "logs" / f"{name}_run.log"
        print(f"[{name}] running (log: {log.relative_to(ROOT)}) ...",
              flush=True)
        with open(log, "w") as fh:
            rc = subprocess.run([str(PY), str(ROOT / "scripts" / f"{name}.py")],
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            print(f"[{name}] FAILED (exit {rc}) — see {log}")
            sys.exit(rc)
        missing = [a for a in artifacts if not (ROOT / a).exists()]
        if missing:
            print(f"[{name}] missing artifacts: {missing}")
            sys.exit(1)
        print(f"[{name}] OK ({time.time() - t0:.0f}s)")

    print("\nALL EXPERIMENTS REPRODUCED")


if __name__ == "__main__":
    main()
