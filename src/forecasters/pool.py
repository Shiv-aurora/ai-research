"""Online expert aggregation over the forecaster pool.

Exponentially weighted average forecaster (Hedge) on QLIKE loss, updated
daily per stock-day: weights at time t depend only on losses realized at
times < t, so the combined forecast is strictly causal. This is the simple,
theoretically grounded end of the aggregation literature (Cesa-Bianchi &
Lugosi 2006); BOA can be swapped in later if second-order refinements matter.

The pool operates on the walk-forward predictions frame (one column per
forecaster), not on forecaster objects — aggregation is a post-processing
layer, which keeps every expert's forecasts inspectable.
"""

import numpy as np
import pandas as pd

from src.forecasters.base import qlike


def hedge_combine(
    preds: pd.DataFrame,
    expert_cols: list[str],
    eta: float = 2.0,
    out_col: str = "pool",
) -> pd.DataFrame:
    """Add an online Hedge combination column to a predictions frame.

    preds: columns ticker, date, target, <expert_cols>. Weights are global
    (shared across stocks) and updated once per DATE using the mean QLIKE of
    each expert over that date's cross-section — a design choice that keeps
    the weight stream long (~4k updates) and stable rather than per-stock
    noisy. Rows where any expert is NaN fall back to the available experts
    (weights renormalized).
    """
    preds = preds.sort_values(["date", "ticker"]).reset_index(drop=True)
    dates = preds["date"].unique()
    log_w = pd.Series(0.0, index=expert_cols)

    combined = np.full(len(preds), np.nan)
    weight_log = []
    for d in dates:
        rows = preds.index[preds["date"] == d]
        block = preds.loc[rows]

        w = np.exp(log_w - log_w.max())
        w /= w.sum()
        weight_log.append((d, *w.values))

        # combine in log-RV space: weighted mean of expert forecasts
        P = block[expert_cols].values
        mask = ~np.isnan(P)
        wm = np.where(mask, w.values, 0.0)
        norm = wm.sum(axis=1, keepdims=True)
        ok = norm[:, 0] > 0
        combined[rows[ok]] = (np.nan_to_num(P[ok]) * wm[ok]).sum(axis=1) / norm[ok, 0]

        # update weights with this date's realized losses (used only for t+1 on)
        y = block["target"].values
        for j, c in enumerate(expert_cols):
            m = mask[:, j] & ~np.isnan(y)
            if m.any():
                loss = qlike(y[m], P[m, j]).mean()
                log_w[c] -= eta * min(loss, 10.0)  # clip pathological losses
        log_w -= log_w.max()  # numerical hygiene

    preds[out_col] = combined
    weights = pd.DataFrame(weight_log, columns=["date", *expert_cols])
    # attrs must stay JSON-serializable (pandas embeds them in parquet
    # metadata) — store weights as records, not a DataFrame.
    preds.attrs["hedge_weights"] = weights.assign(
        date=weights["date"].astype(str)).to_dict("list")
    return preds
