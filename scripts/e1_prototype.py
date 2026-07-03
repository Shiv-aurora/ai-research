"""E1 prototype: does marginal ACI miscover conditionally on market state,
on REAL panel data with a REAL forecaster?

Pipeline: HAR walk-forward point forecasts (2010+) -> per-stock two-sided ACI
-> coverage sliced by trailing VIX percentile. This is the go/no-go gate for
the paper's premise (P3 in the plan); run here early on the provisional panel
for an advance signal.

Usage: .venv/bin/python scripts/e1_prototype.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal.aci import run_aci_panel
from src.eval.coverage import coverage_by_state, marginal_coverage
from src.experiments.walkforward import run_walkforward
from src.forecasters.base import qlike
from src.forecasters.har import HARForecaster
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeding import seed_everything


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    panel = pd.read_parquet(PROJECT_ROOT / cfg["data"]["processed_path"] / "rv_panel.parquet")

    print(f"panel: {len(panel):,} stock-days, {panel.ticker.nunique()} tickers")

    print("\n[1/3] HAR walk-forward (quarterly refits)...")
    res = run_walkforward(panel, [HARForecaster()],
                          eval_start=cfg["evaluation"]["online_eval_start"],
                          verbose=False)
    preds = res.predictions.dropna(subset=["har"])
    ql = qlike(preds["target"].values, preds["har"].values).mean()
    rmse = ((preds["har"] - preds["target"]) ** 2).mean() ** 0.5
    print(f"  HAR OOS: QLIKE={ql:.4f}  RMSE(log)={rmse:.4f}  n={len(preds):,}")

    print("\n[2/3] per-stock two-sided ACI (alpha=0.10)...")
    alpha = cfg["evaluation"]["alpha_two_sided"]
    aci = run_aci_panel(preds, "har", alpha=alpha, eta=0.05, warmup=100)
    aci["width"] = aci["q_lo"] + aci["q_hi"]

    # attach state variable
    state = panel[["ticker", "date", "vix_pctl"]]
    aci = aci.merge(state, on=["ticker", "date"], how="left")

    print(f"  marginal coverage: {marginal_coverage(aci):.4f} (target {1-alpha:.2f})")

    print("\n[3/3] coverage by VIX-percentile regime:")
    by = coverage_by_state(aci, "vix_pctl")
    print(by.round(4).to_string())

    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    by.to_csv(out / "e1_prototype_coverage_by_state.csv")
    print(f"\nsaved -> {out / 'e1_prototype_coverage_by_state.csv'}")


if __name__ == "__main__":
    main()
