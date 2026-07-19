"""E19: crisis inference on the full calendar, with episode-aware
resampling and multiplicity control.

Answers the review's demand that the headline stress inference not
depend on compressing selected stress dates into event time. For each
baseline b, the daily paired coverage difference d_t = cov_RC(t) -
cov_b(t) over the COMPLETE calendar is regressed on a stress indicator:

    d_t = b0 + b1 * 1{stress_t} + u_t

so b0 is the calm-day gap (should be ~0) and b1 the stress increment.
Inference on b1 four ways:
  (a) calendar-time HAC (Newey-West, Bartlett, rule lag);
  (b) cluster-robust SEs, clusters = stress episodes (gap > 20 trading
      days) for stress days and calendar quarters for non-stress days;
  (c) wild cluster bootstrap (Rademacher, 9,999 draws) over the same
      clusters, p-value for H0: b1 = 0;
  (d) leave-one-episode-out range of b1.

PRIMARY ENDPOINT (prespecified here, before secondary tables): the
stress-day UPPER-tail coverage difference RC - ACI (the risk-relevant
direction), same regression on covered_hi. Everything else is
secondary and enters the Romano-Wolf stepdown:
  max-t stepdown over all baselines' b1 (moving-block bootstrap over
  calendar days, block = 20), reported as adjusted p-values.

Also: non-inferiority of RC's average interval score vs ACI — paired
daily interval-score difference with HAC CI against a prespecified
margin of 2% of ACI's mean daily interval score.

Requires reports/e2_daily_coverage.parquet with covered / covered_hi /
is_ columns (written by e2_full.py).

Usage: .venv/bin/python scripts/e19_crisis_inference.py
Output: reports/e19_primary.csv, reports/e19_stress_regression.csv,
        reports/e19_romano_wolf.csv, reports/e19_noninferiority.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeding import seed_everything

OURS = "rc_adaptive"
PRIMARY_BASELINE = "aci"
EPISODE_GAP = 20
N_WILD = 9_999
N_RW = 4_999
BLOCK = 20
NONINF_MARGIN_FRAC = 0.02


def nw_var(x: np.ndarray, lag: int | None = None) -> float:
    n = len(x)
    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    xc = x - x.mean()
    v = xc @ xc / n
    for L in range(1, lag + 1):
        v += 2 * (1 - L / (lag + 1)) * (xc[L:] @ xc[:-L]) / n
    return v / n


def fit_stress_reg(d: np.ndarray, stress: np.ndarray) -> tuple:
    """OLS of d on [1, stress]; returns (b0, b1, residuals)."""
    X = np.column_stack([np.ones(len(d)), stress.astype(float)])
    beta, *_ = np.linalg.lstsq(X, d, rcond=None)
    return float(beta[0]), float(beta[1]), d - X @ beta, X


def hac_beta_se(X: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Newey-West sandwich for OLS betas in calendar time."""
    n, k = X.shape
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    g = X * u[:, None]
    S = g.T @ g / n
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        G = g[L:].T @ g[:-L] / n
        S += w * (G + G.T)
    Sxx_inv = np.linalg.inv(X.T @ X / n)
    V = Sxx_inv @ S @ Sxx_inv / n
    return np.sqrt(np.diag(V))


def cluster_se(X: np.ndarray, u: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """CR1 cluster-robust SEs."""
    n, k = X.shape
    g = X * u[:, None]
    S = np.zeros((k, k))
    ids = pd.unique(cl)
    for c in ids:
        m = g[cl == c].sum(axis=0)
        S += np.outer(m, m)
    G = len(ids)
    adj = (G / (G - 1)) * ((n - 1) / (n - k))
    Sxx_inv = np.linalg.inv(X.T @ X)
    V = adj * Sxx_inv @ S @ Sxx_inv
    return np.sqrt(np.diag(V))


def wild_cluster_p(d: np.ndarray, stress: np.ndarray, cl: np.ndarray,
                   rng: np.random.Generator) -> float:
    """Wild cluster bootstrap p for H0: b1 = 0 (Rademacher, null
    imposed)."""
    # restricted fit under H0 (b1 = 0): intercept only
    b0_r = d.mean()
    u_r = d - b0_r
    X = np.column_stack([np.ones(len(d)), stress.astype(float)])
    _, b1_obs, u, _ = fit_stress_reg(d, stress)
    se_obs = cluster_se(X, u, cl)[1]
    t_obs = b1_obs / se_obs
    ids = pd.unique(cl)
    idx_by = {c: np.where(cl == c)[0] for c in ids}
    t_boot = np.empty(N_WILD)
    for i in range(N_WILD):
        w = rng.choice([-1.0, 1.0], size=len(ids))
        u_b = u_r.copy()
        for c, wc in zip(ids, w):
            u_b[idx_by[c]] = u_r[idx_by[c]] * wc
        d_b = b0_r + u_b
        b0b, b1b, ub, Xb = fit_stress_reg(d_b, stress)
        seb = cluster_se(Xb, ub, cl)[1]
        t_boot[i] = b1b / seb
    return float((np.abs(t_boot) >= abs(t_obs)).mean())


def episodes_and_clusters(dates: pd.DatetimeIndex, stress: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Episode ids on stress days; quarter ids elsewhere."""
    ep = np.full(len(dates), -1)
    eid, last_pos = -1, None
    pos = np.arange(len(dates))
    for i in pos[stress]:
        if last_pos is None or i - last_pos > EPISODE_GAP:
            eid += 1
        ep[i] = eid
        last_pos = i
    q = dates.year.astype(str) + "Q" + dates.quarter.astype(str)
    cl = np.where(stress, [f"ep{e}" for e in ep], "q" + q)
    return ep, np.asarray(cl)


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    daily = pd.read_parquet(PROJECT_ROOT / "reports"
                            / "e2_daily_coverage.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    wide_cov = daily.pivot(index="date", columns="method",
                           values="covered").sort_index()
    wide_hi = daily.pivot(index="date", columns="method",
                          values="covered_hi").sort_index()
    wide_is = daily.pivot(index="date", columns="method",
                          values="is_").sort_index()
    vix = daily.groupby("date")["vix_pctl"].first().sort_index()
    stress = (vix > 0.95).fillna(False).values
    dates = wide_cov.index
    ep, cl = episodes_and_clusters(dates, stress)
    n_ep = ep.max() + 1
    print(f"calendar days {len(dates)}, stress days {int(stress.sum())}, "
          f"episodes {n_ep}")

    # ---- PRIMARY ENDPOINT: upper-tail stress gap RC - ACI
    d_hi = (wide_hi[OURS] - wide_hi[PRIMARY_BASELINE]).values
    b0, b1, u, X = fit_stress_reg(d_hi, stress)
    se_h = hac_beta_se(X, u)
    se_c = cluster_se(X, u, cl)
    p_wild = wild_cluster_p(d_hi, stress, cl, rng)
    loo = []
    for e in range(n_ep):
        keep = ep != e
        _, b1e, _, _ = fit_stress_reg(d_hi[keep], stress[keep])
        loo.append(b1e)
    primary = pd.DataFrame([{
        "endpoint": "stress upper-tail coverage, RC - ACI",
        "b0_calm_gap": b0, "b1_stress_increment": b1,
        "hac_se": se_h[1], "hac_p": 2 * (1 - stats.norm.cdf(abs(b1 / se_h[1]))),
        "cluster_se": se_c[1],
        "cluster_p": 2 * (1 - stats.t.cdf(abs(b1 / se_c[1]), df=n_ep - 1)),
        "wild_cluster_p": p_wild,
        "loo_min": min(loo), "loo_max": max(loo)}])
    print("\n=== PRIMARY endpoint ===")
    print(primary.round(4).to_string(index=False))

    # ---- secondary: two-sided coverage stress regression, all baselines
    rows = []
    b1s, gaps = {}, {}
    for m in wide_cov.columns:
        if m == OURS:
            continue
        d = (wide_cov[OURS] - wide_cov[m]).values
        gaps[m] = d
        b0m, b1m, um, Xm = fit_stress_reg(d, stress)
        seh = hac_beta_se(Xm, um)
        sec = cluster_se(Xm, um, cl)
        pw = wild_cluster_p(d, stress, cl, rng)
        b1s[m] = (b1m, seh[1])
        rows.append({"baseline": m, "b0_calm": b0m, "b1_stress": b1m,
                     "hac_se": seh[1],
                     "hac_t": b1m / seh[1],
                     "cluster_se": sec[1],
                     "cluster_t": b1m / sec[1],
                     "wild_cluster_p": pw})
    reg = pd.DataFrame(rows)
    print("\n=== stress-increment regression (two-sided coverage) ===")
    print(reg.round(4).to_string(index=False))

    # ---- Romano-Wolf stepdown over baselines (moving-block bootstrap)
    names = list(b1s.keys())
    t_obs = np.array([b1s[m][0] / b1s[m][1] for m in names])
    n = len(dates)
    n_blocks = int(np.ceil(n / BLOCK))
    t_boot = np.empty((N_RW, len(names)))
    G = np.column_stack([gaps[m] for m in names])
    for i in range(N_RW):
        starts = rng.integers(0, n, n_blocks)
        take = np.concatenate([np.arange(s, min(s + BLOCK, n))
                               for s in starts])[:n]
        Gb, sb = G[take], stress[take]
        for j, m in enumerate(names):
            b0b, b1b, ub, Xb = fit_stress_reg(Gb[:, j], sb)
            seb = hac_beta_se(Xb, ub)[1]
            t_boot[i, j] = (b1b - b1s[m][0]) / seb   # centered
    order = np.argsort(-np.abs(t_obs))
    adj = np.zeros(len(names))
    remaining = list(order)
    prev = 0.0
    for k_i in order:
        maxt = np.abs(t_boot[:, remaining]).max(axis=1)
        p = float((maxt >= abs(t_obs[k_i])).mean())
        adj[k_i] = max(p, prev)
        prev = adj[k_i]
        remaining.remove(k_i)
    rw = pd.DataFrame({"baseline": names, "t": t_obs,
                       "p_rw_adjusted": adj}).sort_values(
        "p_rw_adjusted")
    print("\n=== Romano-Wolf adjusted p-values (stress increment) ===")
    print(rw.round(4).to_string(index=False))

    # ---- non-inferiority of average interval score vs ACI
    d_is = (wide_is[OURS] - wide_is[PRIMARY_BASELINE]).values
    margin = NONINF_MARGIN_FRAC * float(wide_is[PRIMARY_BASELINE].mean())
    m_hat = d_is.mean()
    se_is = np.sqrt(nw_var(d_is))
    ci = (m_hat - 1.96 * se_is, m_hat + 1.96 * se_is)
    noninf = pd.DataFrame([{
        "mean_is_diff": m_hat, "hac_se": se_is,
        "ci_lo": ci[0], "ci_hi": ci[1], "margin": margin,
        "noninferior": ci[1] < margin}])
    print("\n=== non-inferiority: RC - ACI daily interval score "
          f"(margin = {NONINF_MARGIN_FRAC:.0%} of ACI mean) ===")
    print(noninf.round(4).to_string(index=False))

    rep = PROJECT_ROOT / "reports"
    primary.to_csv(rep / "e19_primary.csv", index=False)
    reg.to_csv(rep / "e19_stress_regression.csv", index=False)
    rw.to_csv(rep / "e19_romano_wolf.csv", index=False)
    noninf.to_csv(rep / "e19_noninferiority.csv", index=False)
    print("\nsaved -> reports/e19_*.csv")


if __name__ == "__main__":
    main()
