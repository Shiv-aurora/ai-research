"""HAR forecaster tests: causality and recovery of a known HAR process."""

import numpy as np
import pandas as pd

from src.forecasters.base import har_lags, qlike
from src.forecasters.har import HARForecaster


def simulate_har_panel(n_days: int = 1500, n_tickers: int = 3, seed: int = 0,
                       beta=(-0.8, 0.35, 0.3, 0.25)) -> pd.DataFrame:
    """Panel where log_rv follows the HAR recursion with known coefficients."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    frames = []
    for k in range(n_tickers):
        lv = np.full(n_days, -9.0)
        for t in range(22, n_days - 1):
            rv_d = lv[t]
            rv_w = lv[t - 4: t + 1].mean()
            rv_m = lv[t - 21: t + 1].mean()
            lv[t + 1] = (beta[0] + beta[1] * rv_d + beta[2] * rv_w
                         + beta[3] * rv_m + rng.normal(0, 0.3))
        frames.append(pd.DataFrame({"ticker": f"T{k}", "date": dates, "log_rv": lv}))
    return pd.concat(frames, ignore_index=True)


def test_recovers_har_coefficients():
    panel = simulate_har_panel()
    model = HARForecaster().fit(panel)
    beta_hat = model.coefs_["T0"]
    assert np.allclose(beta_hat[1:], [0.35, 0.3, 0.25], atol=0.06)


def test_prediction_is_causal():
    panel = simulate_har_panel()
    model = HARForecaster().fit(panel[panel.date < "2019-01-01"])
    eval_frame = panel[panel.date >= "2019-01-01"].reset_index(drop=True)
    p1 = model.predict(eval_frame)
    # Corrupt the last row's log_rv: only forecasts at/after that row may change.
    corrupted = eval_frame.copy()
    corrupted.loc[corrupted.index[-1], "log_rv"] = 5.0
    p2 = model.predict(corrupted)
    changed = (p1 != p2) & ~(p1.isna() & p2.isna())
    assert changed.sum() <= 1  # only the corrupted row itself


def test_har_beats_naive_on_har_process():
    panel = simulate_har_panel(seed=42)
    train = panel[panel.date < "2019-06-01"]
    test = panel[panel.date >= "2019-06-01"].reset_index(drop=True)
    model = HARForecaster().fit(train)
    pred = model.predict(test)
    target = test.groupby("ticker", group_keys=False)["log_rv"].shift(-1)
    ok = pred.notna() & target.notna()
    naive = test["log_rv"]  # random-walk forecast
    assert qlike(target[ok].values, pred[ok].values).mean() < \
           qlike(target[ok].values, naive[ok].values).mean()


def test_har_lags_windows():
    panel = simulate_har_panel(n_days=60, n_tickers=1)
    lags = har_lags(panel)
    row = lags.iloc[30]
    assert row["rv_w"] == panel["log_rv"].iloc[26:31].mean()
    assert row["rv_m"] == panel["log_rv"].iloc[9:31].mean()
