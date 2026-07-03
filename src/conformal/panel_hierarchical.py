"""Panel soft-Mondrian: pooled per-regime thresholds + per-stock offsets.

The per-stock calibrator (mondrian_soft.py) starves in rare regimes: a stock
visits 'stress' only ~150 times in 15 years, so its stress threshold never
converges. Here per-regime thresholds are POOLED across the cross-section of
standardized scores (a stock-day contributes one update; stress regimes get
~100x more updates), with a small per-stock additive offset (L2-shrunk to 0)
absorbing residual idiosyncrasy.

Daily protocol (strictly causal):
  1. At day t every stock i is issued the interval
         m_it +/- (q_side(pi_t) + delta_i) * sigma_hat_it
     using thresholds/offsets as of the end of day t-1.
  2. After all of day t's outcomes are observed, pooled thresholds take one
     OGD step PER STOCK-DAY (sum-based update: eta is a per-observation
     step). This is the point of pooling — convergence is governed by the
     count of stock-days in a regime (~15k for stress on a 100-name panel),
     not the count of days (~150 per stock). A mean-based daily step would
     reduce gradient variance but keep single-stream convergence speed.
     Each stock's offset then takes its own small step.

Output schema matches run_aci (q_lo/q_hi are in STANDARDIZED units; width in
raw log-RV units is (q_lo + q_hi + 2*delta) * sigma_hat — reported directly).
"""

import numpy as np
import pandas as pd

from src.conformal.scores import standardized_scores


def run_panel_mondrian(
    preds: pd.DataFrame,
    membership: pd.DataFrame,      # date-indexed (K columns), soft or hard
    forecast_col: str,
    alpha: float = 0.10,
    eta_by_regime: list[float] | float = 0.002,   # PER-OBSERVATION step
    eta_offset: float = 0.005,
    offset_l2: float = 0.001,
    warmup_days: int = 100,
    scale_window: int = 250,
    one_sided: bool = False,
    score_col: str | None = None,
) -> pd.DataFrame:
    """Pooled panel calibrator over a walk-forward predictions frame.

    one_sided=True tracks only the upper threshold at level alpha (the risk
    head: P(score > q) <= alpha). score_col overrides the default
    MAD-standardized forecast residual with a caller-supplied score column
    (e.g. standardized returns for the VaR application); it must already be
    on a cross-sectionally comparable scale.
    """
    K = membership.shape[1]
    if np.isscalar(eta_by_regime):
        eta_by_regime = [float(eta_by_regime)] * K
    eta = np.asarray(eta_by_regime)
    a_side = alpha if one_sided else alpha / 2.0

    # per-stock standardized scores (or caller-supplied score column)
    if score_col is None:
        parts = [standardized_scores(g.sort_values("date"), forecast_col,
                                     window=scale_window)
                 for _, g in preds.groupby("ticker")]
        df = pd.concat(parts)
    else:
        df = preds.copy()
        df["s_std"] = df[score_col]
        df["sigma_hat"] = 1.0
    df = df.dropna(subset=["s_std"]).sort_values(["date", "ticker"])
    df = df.reset_index(drop=True)

    dates = df["date"].unique()
    warmup_end = dates[min(warmup_days, len(dates) - 1)]
    warm_scores = df.loc[df["date"] <= warmup_end, "s_std"].values
    q_lo = np.full(K, np.nan if one_sided
                   else float(np.quantile(-warm_scores, 1 - a_side)))
    q_hi = np.full(K, float(np.quantile(warm_scores, 1 - a_side)))

    tickers = df["ticker"].unique()
    delta = pd.Series(0.0, index=tickers)

    date_groups = df.groupby("date", sort=True)
    out_qlo = np.zeros(len(df))
    out_qhi = np.zeros(len(df))
    out_cov = np.zeros((len(df), 3), dtype=bool)
    warm_flag = np.zeros(len(df), dtype=bool)

    for d, block in date_groups:
        idx = block.index.values
        pi = membership.reindex([d]).values
        if np.isnan(pi).any():
            pi = np.full((1, K), 1.0 / K)
        pi = pi[0] / pi[0].sum()

        ql = np.nan if one_sided else float(pi @ q_lo)
        qh = float(pi @ q_hi)
        dl = delta.reindex(block["ticker"]).values

        s = block["s_std"].values
        cov_lo = np.ones(len(s), dtype=bool) if one_sided else (-s) <= ql + dl
        cov_hi = s <= qh + dl
        covered = cov_lo & cov_hi

        out_qlo[idx] = ql + dl
        out_qhi[idx] = qh + dl
        out_cov[idx, 0] = covered
        out_cov[idx, 1] = cov_lo
        out_cov[idx, 2] = cov_hi

        if d <= warmup_end:
            warm_flag[idx] = True
            continue

        # pooled per-regime update: one per-observation step per stock-day
        # (sum over the cross-section; eta is a per-observation step size)
        n = len(block)
        if not one_sided:
            q_lo += eta * pi * ((~cov_lo).sum() - a_side * n)
        q_hi += eta * pi * ((~cov_hi).sum() - a_side * n)

        # per-stock offsets: own miss on either side, shrunk to zero
        own_err = ((~cov_lo).astype(float) + (~cov_hi).astype(float)) / 2.0
        step = eta_offset * (own_err - a_side) - offset_l2 * delta.reindex(
            block["ticker"]).values
        delta.loc[block["ticker"]] = delta.reindex(block["ticker"]).values + step

    df["q_lo"] = out_qlo
    df["q_hi"] = out_qhi
    df["covered"] = out_cov[:, 0]
    df["covered_lo"] = out_cov[:, 1]
    df["covered_hi"] = out_cov[:, 2]
    df["warmup"] = warm_flag
    if one_sided:
        df["width"] = df["q_hi"] * df["sigma_hat"]
    else:
        df["width"] = (df["q_lo"] + df["q_hi"]) * df["sigma_hat"]
    return df
