"""Forecaster interface.

A forecaster consumes a stock-day panel (columns: ticker, date, log_rv, plus
any state columns) sorted by (ticker, date), and produces next-day point
forecasts of log_rv. Implementations must be strictly causal: the forecast
for (ticker, t+1) may use only rows dated <= t.

The walk-forward engine calls fit() on the training slice and predict() on
the evaluation slice; predict() receives the full frame up to each forecast
date so lag features can be built, but must only emit forecasts for the
requested dates.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def har_lags(panel: pd.DataFrame, col: str = "log_rv") -> pd.DataFrame:
    """Standard HAR regressors per ticker: daily, weekly, monthly components.

    rv_d[t] = col[t]; rv_w[t] = mean(col[t-4..t]); rv_m[t] = mean(col[t-21..t]).
    A row dated t holds regressors observable at close of t, used to forecast t+1.
    """
    out = panel[["ticker", "date", col]].copy()
    g = out.groupby("ticker", group_keys=False)[col]
    out["rv_d"] = out[col]
    out["rv_w"] = g.transform(lambda s: s.rolling(5, min_periods=5).mean())
    out["rv_m"] = g.transform(lambda s: s.rolling(22, min_periods=22).mean())
    return out


class Forecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "Forecaster":
        """train: panel rows whose NEXT-day target is known and in-sample."""

    @abstractmethod
    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Return forecast of log_rv at t+1 for each row (ticker, t) in frame,
        indexed like frame."""


def qlike(y_true_logvar: np.ndarray, y_pred_logvar: np.ndarray) -> np.ndarray:
    """QLIKE loss in variance space from log-variance inputs.

    QLIKE(sigma2, h) = sigma2/h - ln(sigma2/h) - 1  >= 0, minimized at h=sigma2.
    """
    ratio = np.exp(y_true_logvar - y_pred_logvar)
    return ratio - (y_true_logvar - y_pred_logvar) - 1.0
