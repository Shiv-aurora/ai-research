"""Online Gaussian HMM regime estimator with filtered probabilities.

Protocol (strictly causal):
  - Parameters are refit on the expanding window every `refit_every` trading
    days (hmmlearn GaussianHMM, diagonal covariance).
  - Between refits, filtered probabilities P(state_t | obs_{1..t}) are
    computed by the forward recursion under the FROZEN parameters — so the
    membership used on day t depends only on data up to t and parameters fit
    on data before the last refit date.
  - States are canonically ordered by the mean of the first feature (the
    stress coordinate), so regime 0 = calm, K-1 = stress across refits.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def _order_states(model: GaussianHMM) -> np.ndarray:
    return np.argsort(model.means_[:, 0])


def _forward_filtered(obs: np.ndarray, model: GaussianHMM,
                      init: np.ndarray) -> np.ndarray:
    """Filtered state probabilities for each row of obs, starting from
    distribution `init` (posterior at the last seen observation)."""
    from scipy.stats import multivariate_normal

    K = model.n_components
    T = len(obs)
    A = model.transmat_
    out = np.zeros((T, K))
    alpha = init
    for t in range(T):
        pred = alpha @ A
        like = np.array([
            multivariate_normal.pdf(
                obs[t], mean=model.means_[k],
                cov=np.diag(np.atleast_2d(model.covars_[k]).diagonal()),
                allow_singular=True)
            for k in range(K)
        ])
        alpha = pred * np.maximum(like, 1e-300)
        s = alpha.sum()
        alpha = alpha / s if s > 0 else np.full(K, 1.0 / K)
        out[t] = alpha
    return out


def online_hmm_memberships(
    market: pd.DataFrame,
    feature_cols: list[str],
    n_regimes: int = 3,
    refit_every: int = 63,
    min_train: int = 750,
    seed: int = 42,
) -> pd.DataFrame:
    """market: one row per DATE (sorted). Returns (date x K) filtered
    membership probabilities; warmup rows (before min_train) are uniform."""
    X_full = market[feature_cols].values
    ok = ~np.isnan(X_full).any(axis=1)
    n = len(market)
    member = np.full((n, n_regimes), 1.0 / n_regimes)

    model, order, alpha_last = None, None, None
    t = min_train
    while t < n:
        t_end = min(t + refit_every, n)
        train_mask = ok[:t]
        X_train = X_full[:t][train_mask]
        if len(X_train) >= min_train // 2:
            m = GaussianHMM(n_components=n_regimes, covariance_type="diag",
                            n_iter=100, random_state=seed)
            try:
                m.fit(X_train)
                model, order = m, _order_states(m)
                # rebuild filtering posterior at t-1 under the new params
                hist_probs = _forward_filtered(
                    X_train[-min(len(X_train), 500):], model,
                    np.full(n_regimes, 1.0 / n_regimes))
                alpha_last = hist_probs[-1]
            except Exception:  # noqa: BLE001 - keep previous params on failure
                pass
        if model is not None:
            block = X_full[t:t_end]
            block_ok = ~np.isnan(block).any(axis=1)
            clean = np.where(block_ok[:, None], block,
                             np.nanmean(X_full[:t], axis=0))
            probs = _forward_filtered(clean, model, alpha_last)
            alpha_last = probs[-1]
            member[t:t_end] = probs[:, order]
        t = t_end

    return pd.DataFrame(member, index=market.index,
                        columns=[f"regime_{k}" for k in range(n_regimes)])
