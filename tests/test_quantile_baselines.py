"""Baselines must hit nominal exceedance/coverage on streams they can model."""

import numpy as np
import pandas as pd
import pytest

from src.forecasters.quantile_baselines import caviar_sav, garch_t_var, har_qreg


def simulate_garch_t(n=3000, omega=0.02, a1=0.08, b1=0.9, nu=6, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    r = np.empty(n)
    s2 = omega / (1 - a1 - b1)
    for t in range(n):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + a1 * r[t] ** 2 + b1 * s2
    return r / 100.0


def test_garch_t_var_exceedance():
    r = simulate_garch_t()
    for alpha in (0.05, 0.01):
        var = garch_t_var(r, alpha, min_train=750, refit_every=500)
        ok = ~np.isnan(var)
        exc = (-r[ok] > var[ok]).mean()
        assert abs(exc - alpha) < (0.012 if alpha == 0.05 else 0.006)


def test_garch_t_var_is_causal():
    r = simulate_garch_t(n=1600)
    v1 = garch_t_var(r, 0.05, min_train=750, refit_every=500)
    r2 = r.copy()
    r2[1200:] = 0.15                      # corrupt the future
    v2 = garch_t_var(r2, 0.05, min_train=750, refit_every=500)
    np.testing.assert_allclose(v1[:1201], v2[:1201])


def test_caviar_sav_exceedance():
    r = simulate_garch_t(seed=3)
    var = caviar_sav(r, 0.05, min_train=750, refit_every=1000)
    ok = ~np.isnan(var)
    exc = (-r[ok] > var[ok]).mean()
    assert abs(exc - 0.05) < 0.015


def test_har_qreg_coverage():
    rng = np.random.default_rng(5)
    n = 2500
    y = np.zeros(n)
    for t in range(1, n):                 # AR(1) in log-RV with noise
        y[t] = 0.9 * y[t - 1] + rng.normal(0, 0.5)
    q = har_qreg(y, [0.05, 0.95], min_train=750, refit_every=500)
    ok = ~np.isnan(q[0.05])
    cov = ((y >= q[0.05]) & (y <= q[0.95]))[ok].mean()
    assert abs(cov - 0.90) < 0.02
    assert (q[0.95][ok] > q[0.05][ok]).all()
