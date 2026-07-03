"""GRU forecaster smoke tests (small settings; full runs happen in E-scripts)."""

import numpy as np

from src.forecasters.neural import GRUForecaster
from tests.test_har import simulate_har_panel


def test_gru_trains_and_predicts_sanely():
    panel = simulate_har_panel(n_days=700, n_tickers=2, seed=13)
    train = panel[panel.date < "2017-01-01"]
    test = panel[panel.date >= "2017-01-01"].reset_index(drop=True)
    model = GRUForecaster(epochs=2, max_train_windows=5000).fit(train)
    pred = model.predict(test)
    target = test.groupby("ticker", group_keys=False)["log_rv"].shift(-1)
    ok = pred.notna() & target.notna()
    assert ok.sum() > 100
    rmse = np.sqrt(((pred[ok] - target[ok]) ** 2).mean())
    assert rmse < 0.8  # loose: smoke test, tiny training budget


def test_gru_prediction_is_causal():
    panel = simulate_har_panel(n_days=500, n_tickers=1, seed=14)
    model = GRUForecaster(epochs=1, max_train_windows=2000).fit(
        panel[panel.date < "2016-06-01"])
    ev = panel[panel.date >= "2016-06-01"].reset_index(drop=True)
    p1 = model.predict(ev)
    corrupted = ev.copy()
    corrupted.loc[corrupted.index[-1], "log_rv"] = 9.0
    p2 = model.predict(corrupted)
    changed = (p1 != p2) & ~(p1.isna() & p2.isna())
    assert changed.sum() <= 1
