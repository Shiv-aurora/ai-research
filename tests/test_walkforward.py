"""Walk-forward engine tests: alignment, coverage of eval window, causality."""

import numpy as np
import pandas as pd

from src.experiments.walkforward import run_walkforward
from src.forecasters.har import HARForecaster
from tests.test_har import simulate_har_panel


def test_target_alignment_and_window():
    panel = simulate_har_panel(n_days=900, n_tickers=2, seed=1)
    res = run_walkforward(panel, [HARForecaster()], eval_start="2017-06-01",
                          verbose=False)
    preds = res.predictions
    assert preds["date"].min() >= pd.Timestamp("2017-06-01")

    # target at (ticker, t) must equal panel log_rv at the next business day
    merged = preds.merge(
        panel.rename(columns={"log_rv": "lv", "date": "d"}),
        left_on=["ticker", "date"], right_on=["ticker", "d"],
    )
    sample = merged.sample(50, random_state=0)
    for _, row in sample.iterrows():
        nxt = panel[(panel.ticker == row.ticker) & (panel.date > row.date)]
        assert row.target == nxt.iloc[0].log_rv


def test_forecasts_reasonable_on_har_process():
    panel = simulate_har_panel(n_days=1200, n_tickers=2, seed=3)
    res = run_walkforward(panel, [HARForecaster()], eval_start="2018-06-01",
                          verbose=False)
    p = res.predictions.dropna(subset=["har"])
    # On a true HAR process with noise sd 0.3, forecast RMSE should be near 0.3
    rmse = np.sqrt(((p["har"] - p["target"]) ** 2).mean())
    assert 0.25 < rmse < 0.4


def test_fold_log_is_expanding():
    panel = simulate_har_panel(n_days=1200, n_tickers=1, seed=5)
    res = run_walkforward(panel, [HARForecaster()], eval_start="2018-06-01",
                          verbose=False)
    n_train = res.fold_log["n_train"].values
    assert (np.diff(n_train) > 0).all()
