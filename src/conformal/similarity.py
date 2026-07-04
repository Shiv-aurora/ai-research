"""Similarity-weighted (KNN-state) conformal: the continuous-state rival.

The obvious alternative to discrete regimes is to weight calibration scores
by similarity of today's market state to past states (HopCPT and NexCP are
the learned/weighted variants of this idea). This baseline implements the
pooled panel version: at day t, find the k most similar past days in market-
state space (Euclidean on standardized state vars), pool ALL stocks'
MAD-standardized scores from those days, and issue the empirical
(1 - a_side) quantile as the day's threshold.

Strictly causal: only days < t are candidates; state distances use close-of-
day states. No online tracking — its weakness vs our method is exactly the
lack of a coverage feedback loop, which the comparison is designed to show.
"""

import numpy as np
import pandas as pd

from src.conformal.scores import standardized_scores


def run_knn_state_conformal(
    preds: pd.DataFrame,
    market_state: pd.DataFrame,      # date-indexed state vars (numeric)
    forecast_col: str,
    alpha: float = 0.10,
    k: int = 250,
    warmup_days: int = 100,
    scale_window: int = 250,
) -> pd.DataFrame:
    """Same output schema as run_panel_mondrian (q_lo/q_hi standardized)."""
    a_side = alpha / 2.0

    parts = [standardized_scores(g.sort_values("date"), forecast_col,
                                 window=scale_window)
             for _, g in preds.groupby("ticker")]
    df = pd.concat(parts).dropna(subset=["s_std"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # standardized market-state matrix aligned to score dates
    dates = pd.DatetimeIndex(df["date"].unique()).sort_values()
    X = market_state.reindex(dates).astype(float)
    X = (X - X.mean()) / X.std()
    X = X.fillna(0.0).values

    scores_by_day = df.groupby("date")["s_std"].apply(np.asarray)
    day_scores = [scores_by_day.loc[d] for d in dates]

    q_lo_by_day, q_hi_by_day = {}, {}
    for t, d in enumerate(dates):
        if t <= warmup_days:
            q_lo_by_day[d] = q_hi_by_day[d] = np.nan
            continue
        dist = np.linalg.norm(X[:t] - X[t], axis=1)
        sel = np.argsort(dist)[:min(k, t)]
        pool = np.concatenate([day_scores[j] for j in sel])
        q_lo_by_day[d] = float(np.quantile(-pool, 1 - a_side))
        q_hi_by_day[d] = float(np.quantile(pool, 1 - a_side))

    df["q_lo"] = df["date"].map(q_lo_by_day)
    df["q_hi"] = df["date"].map(q_hi_by_day)
    df["warmup"] = df["q_lo"].isna()
    df["covered_lo"] = (-df["s_std"]) <= df["q_lo"]
    df["covered_hi"] = df["s_std"] <= df["q_hi"]
    df["covered"] = df["covered_lo"] & df["covered_hi"]
    df["width"] = (df["q_lo"] + df["q_hi"]) * df["sigma_hat"]
    return df
