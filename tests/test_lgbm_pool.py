"""LightGBM forecaster and Hedge pool tests."""

import numpy as np
import pandas as pd

from src.forecasters.base import qlike
from src.forecasters.lgbm import LGBMForecaster
from src.forecasters.pool import hedge_combine
from tests.test_har import simulate_har_panel


def test_lgbm_learns_har_process():
    panel = simulate_har_panel(n_days=1500, n_tickers=3, seed=8)
    train = panel[panel.date < "2019-06-01"]
    test = panel[panel.date >= "2019-06-01"].reset_index(drop=True)
    model = LGBMForecaster(n_estimators=150).fit(train)
    pred = model.predict(test)
    target = test.groupby("ticker", group_keys=False)["log_rv"].shift(-1)
    ok = pred.notna() & target.notna()
    rmse = np.sqrt(((pred[ok] - target[ok]) ** 2).mean())
    assert rmse < 0.45  # noise sd is 0.3; tree model should get close


def test_lgbm_prediction_is_causal():
    panel = simulate_har_panel(n_days=800, n_tickers=2, seed=9)
    model = LGBMForecaster(n_estimators=50).fit(panel[panel.date < "2017-06-01"])
    ev = panel[panel.date >= "2017-06-01"].reset_index(drop=True)
    p1 = model.predict(ev)
    corrupted = ev.copy()
    corrupted.loc[corrupted.index[-1], "log_rv"] = 9.0
    p2 = model.predict(corrupted)
    changed = (p1 != p2) & ~(p1.isna() & p2.isna())
    assert changed.sum() <= 1


def test_hedge_converges_to_better_expert():
    rng = np.random.default_rng(3)
    n = 3000
    dates = pd.bdate_range("2015-01-01", periods=n)
    target = rng.normal(-9, 1, n)
    good = target + rng.normal(0, 0.1, n)     # accurate expert
    bad = target + rng.normal(1.5, 0.8, n)    # biased noisy expert
    preds = pd.DataFrame({"ticker": "T0", "date": dates, "target": target,
                          "good": good, "bad": bad})
    out = hedge_combine(preds, ["good", "bad"], eta=2.0)
    w = out.attrs["hedge_weights"]
    assert w["good"][-1] > 0.95
    # pool loss ~ good expert's loss, far below bad expert's
    lp = qlike(out["target"].values, out["pool"].values).mean()
    lg = qlike(out["target"].values, out["good"].values).mean()
    lb = qlike(out["target"].values, out["bad"].values).mean()
    assert lp < lg * 1.5 < lb


def test_hedge_is_causal_in_weights():
    """First-date pool must be the uniform average (no losses seen yet)."""
    dates = pd.bdate_range("2020-01-01", periods=5)
    preds = pd.DataFrame({"ticker": "T0", "date": dates,
                          "target": [0.0] * 5,
                          "a": [1.0] * 5, "b": [3.0] * 5})
    out = hedge_combine(preds, ["a", "b"])
    assert out["pool"].iloc[0] == 2.0
