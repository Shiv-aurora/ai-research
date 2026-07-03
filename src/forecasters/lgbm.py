"""Pooled LightGBM forecaster of next-day log-RV.

One model over the whole cross-section: HAR-style lags of log_rv plus market
state variables and per-stock trailing percentiles, with sector information
deliberately omitted until the CRSP universe lands (the provisional universe
has no point-in-time sector map). Hyperparameters follow the conservative
depth/regularization used in the archived RIVE experiments, which were tuned
for exactly this target on a smaller panel.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.forecasters.base import Forecaster, har_lags

STATE_COLS = [
    "VIXCLS", "vix_pctl", "term_spread", "credit_spread",
    "mkt_log_rv", "mkt_rv_pctl", "xs_dispersion", "stock_rv_pctl",
]


class LGBMForecaster(Forecaster):
    name = "lgbm"

    def __init__(self, **overrides) -> None:
        params = dict(
            n_estimators=400, learning_rate=0.03, max_depth=5, num_leaves=31,
            min_child_samples=50, colsample_bytree=0.8, subsample=0.8,
            subsample_freq=1, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        params.update(overrides)
        self.params = params
        self.model_: LGBMRegressor | None = None
        self.feature_cols_: list[str] = []

    def _design(self, panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        d = har_lags(panel)
        state_cols = [c for c in STATE_COLS if c in panel.columns]
        d = pd.concat([d, panel[state_cols]], axis=1)
        return d, ["rv_d", "rv_w", "rv_m", *state_cols]

    def fit(self, train: pd.DataFrame) -> "LGBMForecaster":
        d, cols = self._design(train)
        d["target"] = d.groupby("ticker", group_keys=False)["log_rv"].shift(-1)
        d = d.dropna(subset=["rv_m", "target"])
        self.feature_cols_ = cols
        self.model_ = LGBMRegressor(**self.params)
        self.model_.fit(d[cols], d["target"])
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        d, _ = self._design(frame)
        X = d[self.feature_cols_]
        preds = pd.Series(self.model_.predict(X), index=frame.index, name=self.name)
        preds[X["rv_m"].isna()] = np.nan  # insufficient history
        return preds
