"""Walk-forward evaluation engine.

Generalizes the expanding-window fold discipline of the archived RIVE
experiment (archive/rive-2026/scripts/mlwa_experiments/exp3_rolling_walkforward.py)
into a reusable engine:

  - expanding window, refits every `refit_frequency` (quarterly by default)
  - HARD leakage assertion per fold: max(train.date) < min(test.date)
  - forecasters emit next-day log_rv point forecasts; the engine aligns them
    with realized targets and returns one tidy frame of out-of-sample rows

The engine is deliberately forecaster-agnostic: anything implementing
src.forecasters.base.Forecaster can be evaluated, and conformal layers
consume the engine's output rather than talking to forecasters directly.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.forecasters.base import Forecaster


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame          # ticker,date,target,<one column per forecaster>
    fold_log: pd.DataFrame = field(default=None)


def quarterly_folds(dates: pd.Series, eval_start: str, eval_end: str | None = None):
    """Yield (train_end_exclusive, test_start, test_end_exclusive) per quarter."""
    start = pd.Timestamp(eval_start)
    end = pd.Timestamp(eval_end) if eval_end else dates.max() + pd.Timedelta(days=1)
    q_starts = pd.date_range(start, end, freq="QS")
    for i, ts in enumerate(q_starts):
        te = q_starts[i + 1] if i + 1 < len(q_starts) else end
        if ts >= end:
            break
        yield ts, ts, min(te, end)


def run_walkforward(
    panel: pd.DataFrame,
    forecasters: list[Forecaster],
    eval_start: str,
    eval_end: str | None = None,
    min_train_days: int = 500,
    refit_every: int = 1,
    verbose: bool = True,
) -> WalkForwardResult:
    """Expanding-window walk-forward with quarterly refits.

    panel: (ticker, date, log_rv, ...) sorted by ticker,date. The engine
    creates the t+1 target internally; rows without a next-day target
    (last day per ticker) are dropped from evaluation.
    """
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    target = panel.groupby("ticker", group_keys=False)["log_rv"].shift(-1)

    rows, fold_rows = [], []
    for fold_i, (train_end, test_start, test_end) in enumerate(
        quarterly_folds(panel["date"], eval_start, eval_end)
    ):
        train_mask = panel["date"] < train_end
        test_mask = (panel["date"] >= test_start) & (panel["date"] < test_end)
        if train_mask.sum() < min_train_days or test_mask.sum() == 0:
            continue

        train = panel[train_mask]
        test = panel[test_mask]

        # Leakage guard: nothing in train may be dated at/after the test start.
        assert train["date"].max() < test["date"].min(), (
            f"leakage: train end {train['date'].max()} >= test start {test['date'].min()}"
        )

        # predict() needs trailing history to build lag features: give each
        # forecaster the panel up to test_end, but keep only test-row outputs.
        context_mask = panel["date"] < test_end
        context = panel[context_mask]
        test_idx = panel.index[test_mask]

        fold_out = {"date_start": test_start, "date_end": test_end,
                    "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())}
        preds_this_fold = {}
        # Sparse refits for expensive models: refitting every `refit_every`
        # folds is still causal — the stale model saw only older data.
        do_fit = (len(fold_rows) % refit_every == 0)
        for fc in forecasters:
            if do_fit or not getattr(fc, "_fitted_once", False):
                fc.fit(train)
                fc._fitted_once = True
            full_pred = fc.predict(context)
            preds_this_fold[fc.name] = full_pred.loc[test_idx]
        fold_rows.append(fold_out)

        block = pd.DataFrame({
            "ticker": panel.loc[test_idx, "ticker"],
            "date": panel.loc[test_idx, "date"],
            "target": target.loc[test_idx],
            **preds_this_fold,
        })
        rows.append(block)
        if verbose:
            print(f"  fold {fold_i:2d} {test_start.date()} -> {test_end.date()}: "
                  f"train={fold_out['n_train']:,} test={fold_out['n_test']:,}")

    predictions = pd.concat(rows, ignore_index=True).dropna(subset=["target"])
    return WalkForwardResult(predictions=predictions,
                             fold_log=pd.DataFrame(fold_rows))
