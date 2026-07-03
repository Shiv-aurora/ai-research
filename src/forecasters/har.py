"""HAR-RV forecaster (Corsi 2009) in log space, estimated per stock by OLS.

log_rv[t+1] = b0 + b1*rv_d[t] + b2*rv_w[t] + b3*rv_m[t] + e[t+1]

Log-space HAR with OLS is the standard strong baseline in the ML-for-RV
literature. Coefficients are fit per ticker on the training slice; tickers
with too little history fall back to pooled coefficients.
"""

import numpy as np
import pandas as pd

from src.forecasters.base import Forecaster, har_lags

REGRESSORS = ["rv_d", "rv_w", "rv_m"]
MIN_OBS_PER_TICKER = 100


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return beta


class HARForecaster(Forecaster):
    name = "har"

    def __init__(self) -> None:
        self.coefs_: dict[str, np.ndarray] = {}
        self.pooled_: np.ndarray | None = None

    def _design(self, panel: pd.DataFrame) -> pd.DataFrame:
        lags = har_lags(panel)
        lags["target"] = lags.groupby("ticker", group_keys=False)["log_rv"].shift(-1)
        return lags

    def fit(self, train: pd.DataFrame) -> "HARForecaster":
        d = self._design(train).dropna(subset=REGRESSORS + ["target"])
        self.pooled_ = _ols(d[REGRESSORS].values, d["target"].values)
        self.coefs_ = {}
        for ticker, g in d.groupby("ticker"):
            if len(g) >= MIN_OBS_PER_TICKER:
                self.coefs_[ticker] = _ols(g[REGRESSORS].values, g["target"].values)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        d = har_lags(frame)
        preds = np.full(len(d), np.nan)
        X = d[REGRESSORS].values
        ok = ~np.isnan(X).any(axis=1)
        Xd = np.column_stack([np.ones(len(d)), X])
        for ticker, idx in d.groupby("ticker").indices.items():
            beta = self.coefs_.get(ticker, self.pooled_)
            rows = idx[ok[idx]]
            preds[rows] = Xd[rows] @ beta
        return pd.Series(preds, index=frame.index, name=self.name)
