"""E3: one-day Value-at-Risk from calibrated volatility — the utility section.

Setup: at close of day t the pool forecasts next-day log variance; predicted
vol sigma_pred = exp(pool/2). The standardized loss is z_{t+1} =
-ret_{t+1}/sigma_pred. A VaR(p) forecast is q_t * sigma_pred where q_t is an
estimate of the (1-p) quantile of z. Methods differ only in how q_t is set:

  normal       q = z_{1-p} (Gaussian)                      [parametric]
  fhs          q = trailing 500d empirical quantile of z    [filtered hist. sim.]
  aci          per-stock one-sided conformal tracking       [marginal adaptive]
  rc_panel     pooled regime-conditional one-sided (ours)   [this paper]

Evaluation: pooled exceedance rates overall and by VIX regime; share of
stocks passing Kupiec, Christoffersen, and DQ at 5% significance.

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
from src.data.universe import get_universe
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

        # --- FHS + ACI per stock ---
        parts = []
        for _, g in base.sort_values("date").groupby("ticker"):
            g = g.copy()
            if len(g) <= WARMUP_DAYS + 50:
                continue
            g["q_fhs"] = (g["z"].rolling(500, min_periods=250)
                          .quantile(1 - alpha).shift(1))
            g["q_aci"] = one_sided_aci_stream(g["z"].values, alpha, WARMUP_DAYS)
            g["warm"] = np.arange(len(g)) < WARMUP_DAYS
            parts.append(g)
        base = pd.concat(parts)

        # --- ours: pooled regime-conditional one-sided on z ---
        rc = run_panel_mondrian(base, member, "pool", alpha=alpha,
                                eta_by_regime=ETAS[alpha],
                                warmup_days=WARMUP_DAYS, one_sided=True,
                                score_col="z")
        rc = rc.rename(columns={"q_hi": "q_rc"})
        base = base.merge(rc[["ticker", "date", "q_rc"]],
                          on=["ticker", "date"], how="left")

        d = base[(~base["warm"]) & base["q_fhs"].notna() & base["q_rc"].notna()]
        print(f"\n=== VaR {int((1-alpha)*100)}% (n={len(d):,}) ===")
        header = f"{'method':<10} {'rate':>7} {'calm':>7} {'stress':>7} " \
                 f"{'kupiec%':>8} {'christ%':>8} {'dq%':>6}"
        print(header)
        for m, qcol in [("normal", "q_normal"), ("fhs", "q_fhs"),
                        ("aci", "q_aci"), ("rc_panel", "q_rc")]:
            e = d["z"] > d[qcol]
            calm = e[d.vix_pctl <= 0.5].mean()
            stress = e[d.vix_pctl > 0.95].mean()
            bt = backtest_panel(d.assign(exc=e), "exc", alpha)
            pass_k = (bt["kupiec_p"] > 0.05).mean()
            pass_c = (bt["christoffersen_p"] > 0.05).mean()
            pass_d = (bt["dq_p"] > 0.05).mean()
            print(f"{m:<10} {e.mean():>7.4f} {calm:>7.4f} {stress:>7.4f} "
                  f"{pass_k:>8.2f} {pass_c:>8.2f} {pass_d:>6.2f}")
            all_rows.append({"alpha": alpha, "method": m, "rate": e.mean(),
                             "rate_calm": calm, "rate_stress": stress,
                             "kupiec_pass": pass_k, "christoffersen_pass": pass_c,
                             "dq_pass": pass_d})

    out = PROJECT_ROOT / "reports"
    pd.DataFrame(all_rows).to_csv(out / "e3_var_summary.csv", index=False)
    print(f"\nsaved -> {out / 'e3_var_summary.csv'}")


if __name__ == "__main__":
    main()
