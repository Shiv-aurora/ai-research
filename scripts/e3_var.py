"""E3: one-day Value-at-Risk from calibrated volatility — the utility section.

Setup: at close of day t the pool forecasts next-day log variance; predicted
vol sigma_pred = exp(pool/2). The standardized loss is z_{t+1} =
-ret_{t+1}/sigma_pred. A VaR(p) forecast is q_t * sigma_pred where q_t is an
estimate of the (1-p) quantile of z. Methods differ only in how q_t is set:

  normal       q = z_{1-p} (Gaussian)                      [parametric]
  fhs          q = trailing 500d empirical quantile of z    [filtered hist. sim.]
  garch_t      per-stock GARCH(1,1)-t VaR on raw returns    [classical benchmark]
  caviar       Engle-Manganelli SAV CAViaR on raw returns   [classical benchmark]
  aci          per-stock one-sided conformal tracking       [marginal adaptive]
  pooled_k1    pooled one-sided adaptive, no regimes        [pooling-only arm]
  rc_panel     pooled regime-conditional one-sided (ours)   [this paper]
  rc_adaptive  same, DtACI-per-regime adaptive rates        [this paper, no tuning]

garch_t/caviar forecast return quantiles directly; their VaR is converted to
z units (divide by sigma_pred) so all methods share the exceedance test.

Evaluation: pooled exceedance rates overall and by VIX regime; a
date-clustered calm-vs-stress balance z-test per method; share of stocks
passing Kupiec, Christoffersen INDEPENDENCE, combined conditional
coverage (CC, 2 df), DQ, and a regime-aware DQ with a stress indicator
regressor, all at 5% significance.

Usage: .venv/bin/python scripts/e3_var.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal.panel_hierarchical import run_panel_mondrian
from src.data.ohlc import load_returns
from src.forecasters.quantile_baselines import caviar_sav, garch_t_var
from src.utils.parallel import pmap
from src.data.universe import get_universe
from src.eval.dm_hac import hac_mean_se
from src.eval.var_backtests import backtest_panel
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeding import seed_everything

CUTS = [0.5, 0.8, 0.95]
# Extreme quantiles need LARGER per-observation steps, not smaller: the
# tracker must climb into the fat tail against a weak (1%) miss signal.
# (Verified by sweep; the DtACI-per-regime layer will select these online.)
ETAS = {0.05: [0.001, 0.0015, 0.003, 0.01],
        0.01: [0.002, 0.003, 0.006, 0.02]}
WARMUP_DAYS = 250


def aligned_bins(market: pd.DataFrame) -> pd.DataFrame:
    v = market["vix_pctl"].values
    k = np.searchsorted(CUTS, v, side="right")
    pi = np.zeros((len(v), 4))
    ok = ~np.isnan(v)
    pi[ok, k[ok]] = 1.0
    pi[~ok] = 0.25
    return pd.DataFrame(pi, index=market.index,
                        columns=[f"regime_{j}" for j in range(4)])


def _stock_streams(args):
    """Worker: FHS/ACI/GARCH-t/CAViaR columns for one name (module-level
    for spawn; the heavy GARCH and CAViaR fits parallelize across names)."""
    g, alpha = args
    g = g.copy()
    if len(g) <= WARMUP_DAYS + 50:
        return None
    g["q_fhs"] = (g["z"].rolling(500, min_periods=250)
                  .quantile(1 - alpha).shift(1))
    g["q_aci"] = one_sided_aci_stream(g["z"].values, alpha, WARMUP_DAYS)
    g["q_garch"] = garch_t_var(g["ret_next"].values, alpha,
                               min_train=WARMUP_DAYS) / g["sigma_pred"].values
    g["q_caviar"] = caviar_sav(g["ret_next"].values, alpha,
                               min_train=WARMUP_DAYS) / g["sigma_pred"].values
    g["warm"] = np.arange(len(g)) < WARMUP_DAYS
    return g


def one_sided_aci_stream(z: np.ndarray, alpha: float, warmup: int,
                         eta: float = 0.05) -> np.ndarray:
    """Per-stock one-sided quantile tracker; returns issued thresholds."""
    q = float(np.quantile(z[:warmup], 1 - alpha))
    out = np.empty(len(z))
    for t in range(len(z)):
        out[t] = q
        if t >= warmup:
            q += eta * ((1.0 if z[t] > q else 0.0) - alpha)
    return out


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    panel = pd.read_parquet(PROJECT_ROOT / cfg["data"]["processed_path"] / "rv_panel.parquet")
    preds = pd.read_parquet(PROJECT_ROOT / cfg["data"]["processed_path"] / "e0_predictions.parquet")

    rets = load_returns(get_universe(cfg), cfg["data"]["start_date"],
                        "2026-01-01", verbose=False)
    rets = rets.sort_values(["ticker", "date"])
    # align: prediction row (i, t) is scored against the NEXT day's return
    rets["pred_date"] = rets.groupby("ticker")["date"].shift(1)
    nxt = rets.dropna(subset=["pred_date"])[["ticker", "pred_date", "ret"]]
    nxt = nxt.rename(columns={"pred_date": "date", "ret": "ret_next"})

    df = preds.merge(nxt, on=["ticker", "date"], how="inner")
    df["sigma_pred"] = np.exp(df["pool"] / 2.0)
    df["z"] = -df["ret_next"] / df["sigma_pred"]
    df = df.dropna(subset=["z"])
    state = panel[["ticker", "date", "vix_pctl"]]
    market = panel.groupby("date")[["vix_pctl"]].first().sort_index()
    member = aligned_bins(market)
    print(f"VaR panel: {len(df):,} stock-days")

    all_rows = []
    for alpha in (0.05, 0.01):
        # --- parametric normal ---
        base = df.merge(state, on=["ticker", "date"], how="left").copy()
        base["q_normal"] = stats.norm.ppf(1 - alpha)

        # --- FHS/ACI/GARCH-t/CAViaR per stock, across cores ---
        groups = [(g, alpha)
                  for _, g in base.sort_values("date").groupby("ticker")]
        parts = [p for p in pmap(_stock_streams, groups) if p is not None]
        base = pd.concat(parts)

        # --- ours: pooled regime-conditional one-sided on z ---
        rc = run_panel_mondrian(base, member, "pool", alpha=alpha,
                                eta_by_regime=ETAS[alpha],
                                warmup_days=WARMUP_DAYS, one_sided=True,
                                score_col="z")
        rc = rc.rename(columns={"q_hi": "q_rc"})
        base = base.merge(rc[["ticker", "date", "q_rc"]],
                          on=["ticker", "date"], how="left")

        rca = run_panel_mondrian(base, member, "pool", alpha=alpha,
                                 adaptive=True, warmup_days=WARMUP_DAYS,
                                 one_sided=True, score_col="z")
        rca = rca.rename(columns={"q_hi": "q_rca"})
        base = base.merge(rca[["ticker", "date", "q_rca"]],
                          on=["ticker", "date"], how="left")

        # pooled K=1 adaptive (no regimes): isolates what the regime
        # layer adds on the risk head
        member1 = pd.DataFrame(1.0, index=member.index,
                               columns=["regime_0"])
        rk1 = run_panel_mondrian(base, member1, "pool", alpha=alpha,
                                 adaptive=True, warmup_days=WARMUP_DAYS,
                                 one_sided=True, score_col="z")
        rk1 = rk1.rename(columns={"q_hi": "q_rck1"})
        base = base.merge(rk1[["ticker", "date", "q_rck1"]],
                          on=["ticker", "date"], how="left")

        d = base[(~base["warm"]) & base["q_fhs"].notna() & base["q_rc"].notna()
                 & base["q_garch"].notna() & base["q_caviar"].notna()].copy()
        d["stress_ind"] = (d["vix_pctl"] > 0.95).astype(float)
        print(f"\n=== VaR {int((1-alpha)*100)}% (n={len(d):,}) ===")
        header = f"{'method':<12} {'rate':>7} {'calm':>7} {'stress':>7} " \
                 f"{'bal_z':>7} {'kup%':>6} {'ind%':>6} {'cc%':>6} " \
                 f"{'dq%':>6} {'dqS%':>6}"
        print(header)
        for m, qcol in [("normal", "q_normal"), ("fhs", "q_fhs"),
                        ("garch_t", "q_garch"), ("caviar", "q_caviar"),
                        ("aci", "q_aci"), ("pooled_k1", "q_rck1"),
                        ("rc_panel", "q_rc"), ("rc_adaptive", "q_rca")]:
            e = d["z"] > d[qcol]
            calm = e[d.vix_pctl <= 0.5].mean()
            stress = e[d.vix_pctl > 0.95].mean()
            # calm-vs-stress balance: date-clustered z-test of equal
            # exceedance rates in the two slices
            sc = hac_mean_se(e[d.vix_pctl <= 0.5].astype(float),
                             d.loc[d.vix_pctl <= 0.5, "date"])
            ss = hac_mean_se(e[d.vix_pctl > 0.95].astype(float),
                             d.loc[d.vix_pctl > 0.95, "date"])
            bal_z = ((ss["mean"] - sc["mean"])
                     / np.sqrt(ss["se"] ** 2 + sc["se"] ** 2))
            bt = backtest_panel(d.assign(exc=e), "exc", alpha,
                                stress_col="stress_ind")
            pass_k = (bt["kupiec_p"] > 0.05).mean()
            pass_i = (bt["independence_p"] > 0.05).mean()
            pass_cc = (bt["cc_p"] > 0.05).mean()
            pass_d = (bt["dq_p"] > 0.05).mean()
            pass_ds = (bt["dq_stress_p"] > 0.05).mean()
            print(f"{m:<12} {e.mean():>7.4f} {calm:>7.4f} {stress:>7.4f} "
                  f"{bal_z:>7.2f} {pass_k:>6.2f} {pass_i:>6.2f} "
                  f"{pass_cc:>6.2f} {pass_d:>6.2f} {pass_ds:>6.2f}")
            all_rows.append({"alpha": alpha, "method": m, "rate": e.mean(),
                             "rate_calm": calm, "rate_stress": stress,
                             "balance_z": bal_z,
                             "balance_p": 2 * (1 - stats.norm.cdf(abs(bal_z))),
                             "kupiec_pass": pass_k,
                             "independence_pass": pass_i,
                             "cc_pass": pass_cc, "dq_pass": pass_d,
                             "dq_stress_pass": pass_ds})

    out = PROJECT_ROOT / "reports"
    pd.DataFrame(all_rows).to_csv(out / "e3_var_summary.csv", index=False)
    print(f"\nsaved -> {out / 'e3_var_summary.csv'}")


if __name__ == "__main__":
    main()
