"""Classical quantile/VaR baselines for E2/E3: GARCH-t, CAViaR, HAR-QREG.

All are strictly walk-forward: parameters are fit on an expanding window and
frozen for `refit_every` observations; the forecast for t+1 uses only data
through t. These are per-stock models — the standard implementations the
literature compares against, deliberately without our pooling or regime
machinery.

  garch_t_var      GARCH(1,1) with standardized-t innovations (arch package);
                   VaR_q = sigma_{t+1} * F_t^{-1}(q). The workhorse
                   parametric benchmark.
  caviar_sav       Engle & Manganelli (2004) CAViaR, symmetric-absolute-value
                   spec: q_t = b0 + b1*q_{t-1} + b2*|r_{t-1}|, fit by
                   minimizing pinball loss (multi-start Nelder-Mead).
  har_qreg         Quantile regression of next-day log-RV on HAR lags
                   (statsmodels QuantReg) — direct interval baseline for E2.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REFIT_EVERY = 252


def _quantile_loss(e: np.ndarray, tau: float) -> float:
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1.0) * e)))


# ---------------------------------------------------------------- GARCH-t --

def garch_t_var(returns: np.ndarray, alpha: float, min_train: int = 750,
                refit_every: int = REFIT_EVERY) -> np.ndarray:
    """One-step-ahead VaR quantiles (loss side, positive numbers) for a
    single return stream. Returns array aligned with `returns`: entry t is
    the VaR issued at t-1 close for day t; NaN before min_train."""
    from arch import arch_model

    from scipy import stats
    from arch import arch_model

    r = np.asarray(returns, dtype=float) * 100.0     # arch prefers % units
    n = len(r)
    out = np.full(n, np.nan)
    res = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(min_train, n):
            if res is None or (t - min_train) % refit_every == 0:
                am = arch_model(r[:t], vol="GARCH", p=1, q=1, dist="t")
                res = am.fit(disp="off", show_warning=False)
                p = res.params
                mu, omega = p["mu"], p["omega"]
                a1, b1, nu = p["alpha[1]"], p["beta[1]"], p["nu"]
                # standardized-t quantile (unit variance)
                zq = stats.t.ppf(alpha, nu) * np.sqrt((nu - 2.0) / nu)
                sig2_prev = float(res.conditional_volatility[-1] ** 2)
            else:
                # sigma^2_{t-1} is last step's one-step-ahead variance
                # (deterministic given the past under GARCH)
                sig2_prev = s2
            eps_prev = r[t - 1] - mu
            s2 = omega + a1 * eps_prev**2 + b1 * sig2_prev
            out[t] = -(mu + np.sqrt(s2) * zq) / 100.0
    return out


# ----------------------------------------------------------------- CAViaR --

def _caviar_path(params: np.ndarray, absr: np.ndarray, q0: float) -> np.ndarray:
    """q_{t+1} = b0 + b1*q_t + b2*|r_t| is a linear recursion — computed
    exactly by an IIR filter (C speed; the optimizer calls this hundreds of
    times per fit)."""
    from scipy.signal import lfilter

    b0, b1, b2 = params
    u = b0 + b2 * absr
    path, _ = lfilter([1.0], [1.0, -b1], u, zi=np.array([b1 * q0]))
    return np.concatenate(([q0], path))


def caviar_sav(returns: np.ndarray, alpha: float, min_train: int = 750,
               refit_every: int = REFIT_EVERY, n_starts: int = 8,
               seed: int = 0) -> np.ndarray:
    """Engle-Manganelli SAV CAViaR for the loss quantile: VaR_t such that
    P(-r_t > VaR_t) = alpha. Returns array aligned with `returns` (NaN
    before min_train)."""
    r = np.asarray(returns, dtype=float)
    loss = -r
    absr = np.abs(r)
    n = len(r)
    rng = np.random.default_rng(seed)
    out = np.full(n, np.nan)
    params = None
    for t in range(min_train, n):
        if params is None or (t - min_train) % refit_every == 0:
            q0 = float(np.quantile(loss[:min(300, min_train)], 1 - alpha))

            def obj(p):
                if not (0 <= p[1] < 1):
                    return 1e6
                q = _caviar_path(p, absr[:t - 1], q0)
                return _quantile_loss(loss[1:t] - q[1:], 1 - alpha)

            best, best_val = None, np.inf
            starts = [np.array([0.05 * q0, 0.9, 0.1])]
            starts += [np.array([rng.uniform(0, 0.3) * q0,
                                 rng.uniform(0.7, 0.98),
                                 rng.uniform(0.01, 0.4)])
                       for _ in range(n_starts - 1)]
            for p0 in starts:
                r_opt = minimize(obj, p0, method="Nelder-Mead",
                                 options={"maxiter": 400, "xatol": 1e-5,
                                          "fatol": 1e-7})
                if r_opt.fun < best_val:
                    best, best_val = r_opt.x, r_opt.fun
            params = best
            q_state = _caviar_path(params, absr[:t - 1], q0)[-1]
        # issue forecast for day t using info through t-1
        q_state = params[0] + params[1] * q_state + params[2] * absr[t - 1]
        out[t] = q_state
    return out


# --------------------------------------------------------------- HAR-QREG --

def har_qreg(y: np.ndarray, taus: list[float], min_train: int = 750,
             refit_every: int = REFIT_EVERY) -> dict[float, np.ndarray]:
    """Per-stock quantile regression of y_{t+1} on HAR lags of y (daily,
    weekly, monthly means). Returns {tau: forecasts aligned with y} where
    entry t is the forecast FOR day t issued at t-1 (NaN before min_train)."""
    import statsmodels.api as sm

    y = np.asarray(y, dtype=float)
    n = len(y)
    s = pd.Series(y)
    X = pd.DataFrame({
        "d": s.shift(1),
        "w": s.shift(1).rolling(5).mean(),
        "m": s.shift(1).rolling(22).mean(),
    })
    X = sm.add_constant(X, has_constant="add")
    out = {tau: np.full(n, np.nan) for tau in taus}
    fits = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(min_train, n):
            if not fits or (t - min_train) % refit_every == 0:
                mask = X.iloc[:t].notna().all(axis=1).values
                Xtr, ytr = X.iloc[:t][mask], y[:t][mask]
                for tau in taus:
                    fits[tau] = sm.QuantReg(ytr, Xtr).fit(q=tau)
            row = X.iloc[[t]]
            if row.notna().all(axis=1).item():
                for tau in taus:
                    out[tau][t] = float(np.asarray(fits[tau].predict(row))[0])
    return out
