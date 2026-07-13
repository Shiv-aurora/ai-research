"""E16: numeric theorem bounds + robustness of the headline inference.

(a) Propositions 1-2 evaluated at the paper's constants and the observed
    per-regime stock-day counts. The point is honesty: report where the
    a-priori finite-sample bound is informative and where it is vacuous
    (deviation bound >= 1 means no constraint on a miss frequency).

(b) Episode-block bootstrap of the stress coverage gap (rc_adaptive minus
    each baseline): stress days are grouped into contiguous episodes
    (gap > 20 trading days starts a new episode); episodes are resampled
    with replacement 10,000 times. Complements the HAC test with
    resampling at the level where dependence actually lives.

(c) HAC lag sensitivity: the headline stress gap's t-statistic under
    lag = rule-of-thumb (Newey-West 4(n/100)^(2/9)), 0, 5, 10, 22, 66.

Requires reports/e2_daily_coverage.parquet (written by e2_full.py).

Usage: .venv/bin/python scripts/e16_bounds_inference.py
Output: reports/e16_bound_values.csv, reports/e16_episode_bootstrap.csv,
        reports/e16_hac_sensitivity.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeding import seed_everything

B = 12.0                    # score bound (Assumption 1)
ETA_MAX = 0.064             # adaptive grid maximum
ETA_CORR = 0.002            # corrector rate
ETAS_HAND = [0.001, 0.0015, 0.003, 0.01]   # fixed-rate path (Prop 1)
CUTS = [0.5, 0.8, 0.95]
REGIMES = ["calm", "normal", "elevated", "stress"]
N_BOOT = 10_000
EPISODE_GAP = 20            # trading days separating stress episodes


def regime_of(v: np.ndarray) -> np.ndarray:
    return np.searchsorted(CUTS, v, side="right")


def bound_table(panel: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Per-regime N_k and n_bar on the method's usable sample; a-priori
    deviation bounds for Prop 1 (per fixed rate) and Prop 2."""
    d = preds.dropna(subset=["target", "pool"]).merge(
        panel[["ticker", "date", "vix_pctl"]], on=["ticker", "date"])
    d = d.dropna(subset=["vix_pctl"])
    d["k"] = regime_of(d["vix_pctl"].values)
    n_bar = int(d.groupby("date").size().max())
    rows = []
    for k, name in enumerate(REGIMES):
        N_k = int((d["k"] == k).sum())
        row = {"regime": name, "N_k": N_k, "n_bar": n_bar,
               "prop2_bound":
               (2 * B + (ETA_MAX + ETA_CORR) * n_bar) / (ETA_CORR * N_k)}
        eta_k = ETAS_HAND[k]
        row["prop1_eta"] = eta_k
        row["prop1_bound"] = (2 * B + eta_k * n_bar) / (eta_k * N_k)
        rows.append(row)
    return pd.DataFrame(rows)


def episodes_of(dates: pd.DatetimeIndex, all_days: pd.DatetimeIndex
                ) -> np.ndarray:
    """Label contiguous stress-day runs; a gap of > EPISODE_GAP trading
    days starts a new episode."""
    pos = np.searchsorted(all_days, dates)
    breaks = np.diff(pos) > EPISODE_GAP
    return np.concatenate([[0], np.cumsum(breaks)])


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    proc = PROJECT_ROOT / cfg["data"]["processed_path"]
    panel = pd.read_parquet(proc / "rv_panel.parquet")
    preds = pd.read_parquet(proc / "e0_predictions.parquet")

    bounds = bound_table(panel, preds)
    print("=== a-priori per-regime deviation bounds ===")
    print(bounds.round(4).to_string(index=False))
    print("(bound >= 1.0 constrains nothing: a miss frequency cannot "
          "deviate from target by more than 1)")

    daily = pd.read_parquet(PROJECT_ROOT / "reports"
                            / "e2_daily_coverage.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    all_days = pd.DatetimeIndex(np.sort(daily["date"].unique()))
    stress = daily[daily["vix_pctl"] > 0.95].copy()

    ours = stress[stress.method == "rc_adaptive"].set_index("date")
    boot_rows, rng = [], np.random.default_rng(cfg["seed"])
    for m in sorted(stress.method.unique()):
        if m == "rc_adaptive":
            continue
        b = stress[stress.method == m].set_index("date")
        common = ours.index.intersection(b.index).sort_values()
        # weighted daily gap (weights: stock count per day)
        gap = (ours.loc[common, "covered"] - b.loc[common, "covered"]).values
        w = ours.loc[common, "n"].values.astype(float)
        ep = episodes_of(common, all_days)
        E = ep.max() + 1
        obs = float(np.average(gap, weights=w))
        # resample episodes with replacement
        stats_b = np.empty(N_BOOT)
        idx_by_ep = [np.where(ep == e)[0] for e in range(E)]
        for i in range(N_BOOT):
            take = rng.integers(0, E, E)
            sel = np.concatenate([idx_by_ep[e] for e in take])
            stats_b[i] = np.average(gap[sel], weights=w[sel])
        lo_ci, hi_ci = np.quantile(stats_b, [0.025, 0.975])
        p = 2 * min((stats_b <= 0).mean(), (stats_b >= 0).mean())
        boot_rows.append({"baseline": m, "gap": obs, "n_episodes": E,
                          "ci_lo": lo_ci, "ci_hi": hi_ci, "p_boot": p})
    boot = pd.DataFrame(boot_rows)
    print("\n=== episode-block bootstrap: rc_adaptive - baseline, "
          "stress coverage ===")
    print(boot.round(4).to_string(index=False))

    # HAC lag sensitivity for the headline gap (vs aci)
    b = stress[stress.method == "aci"].set_index("date")
    common = ours.index.intersection(b.index).sort_values()
    gap = (ours.loc[common, "covered"] - b.loc[common, "covered"]).values
    n = len(gap)
    rule = int(np.floor(4 * (n / 100) ** (2 / 9)))
    hac_rows = []
    for lag in [rule, 0, 5, 10, 22, 66]:
        x = gap - gap.mean()
        var = (x @ x) / n
        for L in range(1, lag + 1):
            var += 2 * (1 - L / (lag + 1)) * (x[L:] @ x[:-L]) / n
        t = gap.mean() / np.sqrt(var / n)
        hac_rows.append({"lag": lag, "is_rule": lag == rule,
                         "t": t, "p": 2 * (1 - stats.norm.cdf(abs(t)))})
    hac = pd.DataFrame(hac_rows)
    print("\n=== HAC lag sensitivity (rc_adaptive - aci, stress) ===")
    print(hac.round(4).to_string(index=False))

    rep = PROJECT_ROOT / "reports"
    bounds.to_csv(rep / "e16_bound_values.csv", index=False)
    boot.to_csv(rep / "e16_episode_bootstrap.csv", index=False)
    hac.to_csv(rep / "e16_hac_sensitivity.csv", index=False)
    print("\nsaved -> reports/e16_bound_values.csv, "
          "reports/e16_episode_bootstrap.csv, "
          "reports/e16_hac_sensitivity.csv")


if __name__ == "__main__":
    main()
