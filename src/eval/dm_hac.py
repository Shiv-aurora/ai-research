"""Diebold-Mariano test with HAC (Newey-West) variance, panel-aware.

For panel forecasts we first average the loss differential per date across
the cross-section (clustering by date), then run DM on the resulting time
series — the standard remedy for cross-sectional dependence.
"""

import numpy as np
import pandas as pd
from scipy import stats


def dm_test(
    loss_a: pd.Series,
    loss_b: pd.Series,
    dates: pd.Series,
    max_lag: int | None = None,
) -> dict:
    """H0: E[loss_a - loss_b] = 0. Negative statistic favors model a."""
    d = pd.DataFrame({"d": (loss_a - loss_b).values, "date": dates.values})
    daily = d.groupby("date")["d"].mean().dropna()
    n = len(daily)
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))

    x = daily.values - daily.values.mean()
    gamma0 = (x @ x) / n
    var = gamma0
    for lag in range(1, max_lag + 1):
        w = 1 - lag / (max_lag + 1)
        var += 2 * w * (x[lag:] @ x[:-lag]) / n

    dm_stat = daily.mean() / np.sqrt(var / n)
    p = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return {"dm": float(dm_stat), "p": float(p), "mean_diff": float(daily.mean()),
            "n_dates": n}


def hac_mean_se(values: pd.Series, dates: pd.Series,
                max_lag: int | None = None) -> dict:
    """Date-clustered mean and Newey-West SE of a panel indicator/loss.

    Averages per date (clustering by date), then computes a HAC variance
    over the daily series — same construction as dm_test with loss_b = 0.
    """
    d = pd.DataFrame({"v": values.values, "date": dates.values})
    daily = d.groupby("date")["v"].mean().dropna()
    n = len(daily)
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    x = daily.values - daily.values.mean()
    gamma0 = (x @ x) / n
    var = gamma0
    for lag in range(1, max_lag + 1):
        w = 1 - lag / (max_lag + 1)
        var += 2 * w * (x[lag:] @ x[:-lag]) / n
    return {"mean": float(daily.mean()), "se": float(np.sqrt(var / n)),
            "n_dates": n}
