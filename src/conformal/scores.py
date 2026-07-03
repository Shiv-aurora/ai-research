"""Nonconformity scores and per-stock standardization.

Pooling calibration across the cross-section is only legitimate if scores are
on a common scale; each stock's residuals are standardized by a strictly
trailing robust scale estimate:

    sigma_hat[i, t] = 1.4826 * MAD( residuals[i, t-window .. t-1] )

(lagged one day — the scale used to standardize day t's score never includes
day t's residual). 1.4826 makes MAD consistent for a Gaussian sd; the
constant is irrelevant to conformal validity but keeps thresholds
interpretable.
"""

import numpy as np
import pandas as pd

MAD_CONST = 1.4826


def trailing_scale(residuals: pd.Series, window: int = 250,
                   min_periods: int = 50) -> pd.Series:
    """Strictly trailing robust scale of a residual stream (single stock)."""
    med = residuals.rolling(window, min_periods=min_periods).median()
    mad = (residuals - med).abs().rolling(window, min_periods=min_periods).median()
    scale = (MAD_CONST * mad).shift(1)
    return scale.replace(0.0, np.nan)


def standardized_scores(g: pd.DataFrame, forecast_col: str,
                        window: int = 250) -> pd.DataFrame:
    """Add residual / sigma_hat / standardized score columns for one stock's
    time-sorted predictions frame."""
    out = g.copy()
    out["residual"] = out["target"] - out[forecast_col]
    out["sigma_hat"] = trailing_scale(out["residual"], window=window)
    out["s_std"] = out["residual"] / out["sigma_hat"]
    return out
