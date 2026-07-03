"""E0: point-forecast sanity table on the real panel.

HAR vs pooled LightGBM vs online Hedge pool, walk-forward 2010+, with
panel-aware DM-HAC tests. The contribution of the paper is the conformal
layer; this table only needs to show the pool is non-degraded.

Usage: .venv/bin/python scripts/e0_point_sanity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.dm_hac import dm_test
from src.experiments.walkforward import run_walkforward
from src.forecasters.base import qlike
from src.forecasters.har import HARForecaster
from src.forecasters.lgbm import LGBMForecaster
from src.forecasters.pool import hedge_combine
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeding import seed_everything

EXPERTS = ["har", "lgbm"]


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    panel = pd.read_parquet(PROJECT_ROOT / cfg["data"]["processed_path"] / "rv_panel.parquet")

    res = run_walkforward(panel, [HARForecaster(), LGBMForecaster()],
                          eval_start=cfg["evaluation"]["online_eval_start"],
                          verbose=True)
    preds = res.predictions.dropna(subset=EXPERTS)
    preds = hedge_combine(preds, EXPERTS)

    models = [*EXPERTS, "pool"]
    print(f"\nE0 point-forecast sanity (n={len(preds):,} OOS stock-days)")
    print(f"{'model':<8} {'QLIKE':>8} {'RMSE(log)':>10}")
    rows = []
    for m in models:
        ql = qlike(preds["target"].values, preds[m].values).mean()
        rmse = np.sqrt(((preds[m] - preds["target"]) ** 2).mean())
        rows.append({"model": m, "qlike": ql, "rmse_log": rmse})
        print(f"{m:<8} {ql:>8.4f} {rmse:>10.4f}")

    print("\nDM-HAC (negative favors first model):")
    for a, b in [("pool", "har"), ("pool", "lgbm"), ("lgbm", "har")]:
        la = pd.Series(qlike(preds["target"].values, preds[a].values))
        lb = pd.Series(qlike(preds["target"].values, preds[b].values))
        r = dm_test(la, lb, preds["date"])
        print(f"  {a} vs {b}: DM={r['dm']:+.2f}  p={r['p']:.4f}")
        rows.append({"model": f"dm_{a}_vs_{b}", "qlike": r["dm"], "rmse_log": r["p"]})

    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "e0_point_sanity.csv", index=False)
    preds.to_parquet(PROJECT_ROOT / cfg["data"]["processed_path"] / "e0_predictions.parquet",
                     index=False)
    print(f"\nsaved -> {out / 'e0_point_sanity.csv'} and e0_predictions.parquet")


if __name__ == "__main__":
    main()
