"""Cross-sectional panel conformal baseline (Tu & Giesecke 2026 spirit).

Tu & Giesecke (arXiv:2605.17705) calibrate each unit's prediction set from
a contemporaneous cross-section of related units plus an adaptive
miscoverage level updated on feedback. In our daily panel all targets for
day t+1 realize simultaneously, so the faithful analogue calibrates from
the most recent OBSERVED cross-sections: the threshold for every stock on
day t is the finite-sample quantile of the pooled standardized scores from
the previous L days' cross-sections (~100 stocks x L days), at a per-side
level that adapts ACI-style on the panel-average coverage error. This is
cross-sectional information pooling WITHOUT regime structure and WITHOUT
per-regime tracking: exactly the "panel conformal" alternative a reader of
that line of work would reach for. Scores are the same MAD-standardized
residuals our method uses, so widths are comparable in raw units.
"""

import numpy as np
import pandas as pd

from src.conformal.scores import standardized_scores

L_DAYS = 5           # calibration window: last L observed cross-sections
GAMMA = 0.005        # adaptive-level step (per day, panel-average error)


def run_panel_xs(
    preds: pd.DataFrame,
    forecast_col: str,
    alpha: float = 0.10,
    warmup_days: int = 100,
    scale_window: int = 250,
) -> pd.DataFrame:
    """Cross-sectional split-conformal over a walk-forward panel.

    preds: columns ticker, date, target, <forecast_col>. Returns per
    stock-day: q_lo, q_hi (standardized units), sigma_hat, coverage
    indicators, warmup flag. Interval in raw units is
    m +/- q_side * sigma_hat.
    """
    a_side = alpha / 2.0
    parts = [standardized_scores(g.sort_values("date"), forecast_col,
                                 window=scale_window)
             for _, g in preds.groupby("ticker")]
    df = pd.concat(parts).dropna(subset=["s_std"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    dates = df["date"].unique()
    warmup_end = dates[min(warmup_days, len(dates) - 1)]
    lvl_lo, lvl_hi = a_side, a_side       # adaptive per-side levels

    buf: list[np.ndarray] = []            # last L_DAYS cross-sections
    out_qlo = np.zeros(len(df))
    out_qhi = np.zeros(len(df))
    warm_flag = np.zeros(len(df), dtype=bool)
    for date, g in df.groupby("date", sort=True):
        idx = g.index.values
        s = g["s_std"].values
        warm = date <= warmup_end
        if buf:
            cal = np.concatenate(buf)
            m = len(cal)
            k_lo = min(m - 1, int(np.ceil((m + 1) * (1 - lvl_lo))) - 1)
            k_hi = min(m - 1, int(np.ceil((m + 1) * (1 - lvl_hi))) - 1)
            q_lo = float(np.sort(-cal)[k_lo])
            q_hi = float(np.sort(cal)[k_hi])
        else:
            q_lo = q_hi = np.inf
            warm = True
        out_qlo[idx] = q_lo
        out_qhi[idx] = q_hi
        warm_flag[idx] = warm
        if not warm:
            # adaptive level: shrink on excess misses (wider), grow on
            # over-coverage (narrower); clipped away from 0 and 1/2
            err_lo = float(((-s) > q_lo).mean())
            err_hi = float((s > q_hi).mean())
            lvl_lo = float(np.clip(lvl_lo + GAMMA * (a_side - err_lo),
                                   1e-4, 0.499))
            lvl_hi = float(np.clip(lvl_hi + GAMMA * (a_side - err_hi),
                                   1e-4, 0.499))
        buf.append(s)
        if len(buf) > L_DAYS:
            buf.pop(0)

    df["q_lo"] = out_qlo
    df["q_hi"] = out_qhi
    df["warmup"] = warm_flag
    df["covered_lo"] = -df["s_std"] <= df["q_lo"]
    df["covered_hi"] = df["s_std"] <= df["q_hi"]
    df["covered"] = df["covered_lo"] & df["covered_hi"]
    return df
