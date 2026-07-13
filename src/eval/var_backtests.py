"""Regulatory VaR backtests: Kupiec POF, Christoffersen independence,
and the Engle-Manganelli dynamic quantile (DQ) test.

All take a boolean exceedance series (True = loss worse than VaR) and the
nominal level p, returning the test statistic and p-value.
"""

import numpy as np
import pandas as pd
from scipy import stats


def kupiec_pof(exceed: np.ndarray, p: float) -> dict:
    """Proportion-of-failures LR test: H0 E[exceed] = p."""
    n = len(exceed)
    x = int(exceed.sum())
    if n == 0:
        return {"stat": np.nan, "p": np.nan, "rate": np.nan, "n": 0}
    pi_hat = x / n
    if x in (0, n):
        ll_alt = 0.0
    else:
        ll_alt = x * np.log(pi_hat) + (n - x) * np.log(1 - pi_hat)
    ll_null = x * np.log(p) + (n - x) * np.log(1 - p)
    lr = -2 * (ll_null - ll_alt)
    return {"stat": float(lr), "p": float(1 - stats.chi2.cdf(lr, df=1)),
            "rate": pi_hat, "n": n}


def christoffersen_independence(exceed: np.ndarray) -> dict:
    """LR test of first-order independence of exceedances (clustering)."""
    e = exceed.astype(int)
    if len(e) < 3:
        return {"stat": np.nan, "p": np.nan}
    pairs = np.stack([e[:-1], e[1:]], axis=1)
    n00 = int(((pairs[:, 0] == 0) & (pairs[:, 1] == 0)).sum())
    n01 = int(((pairs[:, 0] == 0) & (pairs[:, 1] == 1)).sum())
    n10 = int(((pairs[:, 0] == 1) & (pairs[:, 1] == 0)).sum())
    n11 = int(((pairs[:, 0] == 1) & (pairs[:, 1] == 1)).sum())
    if n01 + n11 == 0 or n00 + n10 == 0:
        return {"stat": 0.0, "p": 1.0}
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _ll(k1, k0, prob):
        if prob in (0.0, 1.0):
            return 0.0
        return k1 * np.log(prob) + k0 * np.log(1 - prob)

    ll_null = _ll(n01 + n11, n00 + n10, pi)
    ll_alt = _ll(n01, n00, pi01) + _ll(n11, n10, pi11)
    lr = -2 * (ll_null - ll_alt)
    return {"stat": float(lr), "p": float(1 - stats.chi2.cdf(lr, df=1))}


def christoffersen_cc(exceed: np.ndarray, p: float) -> dict:
    """Combined conditional-coverage LR: Kupiec POF + first-order
    independence, chi2 with 2 df (Christoffersen 1998)."""
    k = kupiec_pof(exceed, p)
    ind = christoffersen_independence(exceed)
    if np.isnan(k["stat"]) or np.isnan(ind["stat"]):
        return {"stat": np.nan, "p": np.nan}
    lr = k["stat"] + ind["stat"]
    return {"stat": float(lr), "p": float(1 - stats.chi2.cdf(lr, df=2))}


def dq_test(exceed: np.ndarray, p: float, var_series: np.ndarray | None = None,
            n_lags: int = 4, extra: np.ndarray | None = None) -> dict:
    """Engle-Manganelli DQ: regress hit_t - p on lagged hits (and VaR level);
    H0: all coefficients zero (correct unconditional level AND no dynamics).
    `extra` adds caller-supplied regressors (e.g. a stress-regime
    indicator, making the test regime-aware)."""
    hit = exceed.astype(float) - p
    n = len(hit)
    if n <= n_lags + 5:
        return {"stat": np.nan, "p": np.nan}
    rows = []
    for lag in range(1, n_lags + 1):
        rows.append(hit[n_lags - lag: n - lag])
    X = [np.ones(n - n_lags), *rows]
    if var_series is not None:
        X.append(var_series[n_lags:])
    if extra is not None:
        extra = np.atleast_2d(np.asarray(extra, dtype=float))
        if extra.shape[0] == n:
            extra = extra.T
        for row in extra:
            X.append(row[n_lags:])
    X = np.column_stack(X)
    y = hit[n_lags:]
    XtX = X.T @ X
    try:
        beta = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        return {"stat": np.nan, "p": np.nan}
    dq = float(beta @ XtX @ beta / (p * (1 - p)))
    k = X.shape[1]
    return {"stat": dq, "p": float(1 - stats.chi2.cdf(dq, df=k))}


def backtest_panel(df: pd.DataFrame, exceed_col: str, p: float,
                   group_col: str = "ticker",
                   stress_col: str | None = None) -> pd.DataFrame:
    """Per-group backtests + pooled exceedance rate; returns one row per
    group. stress_col (0/1 indicator) adds a regime-aware DQ variant with
    the indicator as an extra regressor."""
    rows = []
    for g, gdf in df.groupby(group_col):
        e = gdf[exceed_col].values
        row = {
            group_col: g,
            "rate": e.mean(),
            "kupiec_p": kupiec_pof(e, p)["p"],
            "independence_p": christoffersen_independence(e)["p"],
            "cc_p": christoffersen_cc(e, p)["p"],
            "dq_p": dq_test(e, p)["p"],
            "n": len(e),
        }
        if stress_col is not None:
            row["dq_stress_p"] = dq_test(
                e, p, extra=gdf[stress_col].values.astype(float))["p"]
        rows.append(row)
    return pd.DataFrame(rows)
