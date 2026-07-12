"""Panel soft-Mondrian: pooled per-regime thresholds + per-stock offsets.

The per-stock calibrator (mondrian_soft.py) starves in rare regimes: a stock
visits 'stress' only ~150 times in 15 years, so its stress threshold never
converges. Here per-regime thresholds are POOLED across the cross-section of
standardized scores (a stock-day contributes one update; stress regimes get
~100x more updates), with a small per-stock additive offset (L2-shrunk to 0)
absorbing residual idiosyncrasy.

Daily protocol (strictly causal):
  1. At day t every stock i is issued the interval
         m_it +/- (q_side(pi_t) + delta_i) * sigma_hat_it
     using thresholds/offsets as of the end of day t-1.
  2. After all of day t's outcomes are observed, pooled thresholds take one
     OGD step PER STOCK-DAY (sum-based update: eta is a per-observation
     step). This is the point of pooling — convergence is governed by the
     count of stock-days in a regime (~15k for stress on a 100-name panel),
     not the count of days (~150 per stock). A mean-based daily step would
     reduce gradient variance but keep single-stream convergence speed.
     Each stock's offset then takes its own small step.

Adaptive learning rates (adaptive=True): instead of one hand-tuned eta per
regime, each regime keeps a bank of expert thresholds — one per candidate
rate in eta_grid — updated DtACI-style (Gibbs & Candes 2022) on the expert's
OWN miss indicators, and combined by exponential weights on the pinball loss
of each expert's threshold against the day's realized scores. All expert and
weight updates are gated by pi_t(k), so a regime only learns on its own days.
This removes every hand-tuned rate: the aggregator discovers online that
stress regimes need fast rates and extreme quantiles need large steps.

Because the issued threshold is a weight-AVERAGE of expert thresholds, slow
experts that sit permanently low in a shifting regime drag it down even when
the weights are right. A per-regime additive corrector c_k therefore updates
on the ISSUED interval's misses (the same issued-interval rule as the fixed
path) at one universal per-observation rate eta_corr — the expert bank
supplies regime-speed adaptation, the corrector anchors realized coverage of
what is actually issued (quantile-tracking + integrator structure, cf.
conformal PID control, Angelopoulos et al. 2023).

Reduction: hard membership + a single-rate grid + eta_corr=0 reproduces the
fixed path EXACTLY (regression-tested), because with one expert the weights
are trivial and the own-threshold miss equals the issued-interval miss
whenever pi is one-hot.

Output schema matches run_aci (q_lo/q_hi are in STANDARDIZED units; width in
raw log-RV units is (q_lo + q_hi + 2*delta) * sigma_hat — reported directly).
"""

import numpy as np
import pandas as pd

from src.conformal.scores import standardized_scores

DEFAULT_ETA_GRID = (0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064)


def _expert_weights(logw: np.ndarray) -> np.ndarray:
    w = np.exp(logw - logw.max(axis=1, keepdims=True))
    return w / w.sum(axis=1, keepdims=True)


def run_panel_mondrian(
    preds: pd.DataFrame,
    membership: pd.DataFrame,      # date-indexed (K columns), soft or hard
    forecast_col: str,
    alpha: float = 0.10,
    eta_by_regime: list[float] | float = 0.002,   # PER-OBSERVATION step
    eta_offset: float = 0.0,   # headline method: NO per-stock offsets
    offset_l2: float = 0.0,    # (guarantees cover the pooled component;
                               # offsets survive only as an e6 ablation arm)
    warmup_days: int = 100,
    scale_window: int = 250,
    one_sided: bool = False,
    score_col: str | None = None,
    adaptive: bool = False,
    eta_grid: tuple = DEFAULT_ETA_GRID,
    meta_eta: float = 5.0,
    mix_sigma: float = 0.01,
    eta_corr: float = 0.002,
    average_errors: bool = False,   # fixed-rate path only (e13 ablation)
) -> pd.DataFrame:
    """Pooled panel calibrator over a walk-forward predictions frame.

    one_sided=True tracks only the upper threshold at level alpha (the risk
    head: P(score > q) <= alpha). score_col overrides the default
    MAD-standardized forecast residual with a caller-supplied score column
    (e.g. standardized returns for the VaR application); it must already be
    on a cross-sectionally comparable scale. adaptive=True replaces
    eta_by_regime with DtACI-style per-regime rate aggregation over eta_grid
    (eta_by_regime is then ignored); the issued effective rate per regime is
    returned in result.attrs["adaptive"].
    """
    K = membership.shape[1]
    if np.isscalar(eta_by_regime):
        eta_by_regime = [float(eta_by_regime)] * K
    eta = np.asarray(eta_by_regime)
    a_side = alpha if one_sided else alpha / 2.0
    tau = 1.0 - a_side

    # per-stock standardized scores (or caller-supplied score column)
    if score_col is None:
        parts = [standardized_scores(g.sort_values("date"), forecast_col,
                                     window=scale_window)
                 for _, g in preds.groupby("ticker")]
        df = pd.concat(parts)
    else:
        df = preds.copy()
        df["s_std"] = df[score_col]
        df["sigma_hat"] = 1.0
    df = df.dropna(subset=["s_std"]).sort_values(["date", "ticker"])
    df = df.reset_index(drop=True)

    dates = df["date"].unique()
    warmup_end = dates[min(warmup_days, len(dates) - 1)]
    warm_scores = df.loc[df["date"] <= warmup_end, "s_std"].values
    q0_lo = (np.nan if one_sided
             else float(np.quantile(-warm_scores, 1 - a_side)))
    q0_hi = float(np.quantile(warm_scores, 1 - a_side))
    q_lo = np.full(K, q0_lo)
    q_hi = np.full(K, q0_hi)

    if adaptive:
        E = len(eta_grid)
        etas_g = np.asarray(eta_grid, dtype=float)
        Q_hi = np.full((K, E), q0_hi)
        LW_hi = np.zeros((K, E))
        Q_lo = np.full((K, E), q0_lo)
        LW_lo = np.zeros((K, E))
        c_hi = np.zeros(K)                 # issued-interval correctors
        c_lo = np.zeros(K)
        eta_track_dates: list[str] = []
        eta_track_hi: list[list[float]] = []

    tickers = df["ticker"].unique()
    delta = pd.Series(0.0, index=tickers)

    date_groups = df.groupby("date", sort=True)
    out_qlo = np.zeros(len(df))
    out_qhi = np.zeros(len(df))
    out_cov = np.zeros((len(df), 3), dtype=bool)
    warm_flag = np.zeros(len(df), dtype=bool)

    for d, block in date_groups:
        idx = block.index.values
        pi = membership.reindex([d]).values
        if np.isnan(pi).any():
            pi = np.full((1, K), 1.0 / K)
        pi = pi[0] / pi[0].sum()

        if adaptive:
            w_hi = _expert_weights(LW_hi)
            qh = float(pi @ ((w_hi * Q_hi).sum(axis=1) + c_hi))
            ql = np.nan if one_sided else float(
                pi @ ((_expert_weights(LW_lo) * Q_lo).sum(axis=1) + c_lo))
            eta_track_dates.append(str(pd.Timestamp(d).date()))
            eta_track_hi.append((w_hi @ etas_g).tolist())
        else:
            ql = np.nan if one_sided else float(pi @ q_lo)
            qh = float(pi @ q_hi)
        dl = delta.reindex(block["ticker"]).values

        s = block["s_std"].values
        cov_lo = np.ones(len(s), dtype=bool) if one_sided else (-s) <= ql + dl
        cov_hi = s <= qh + dl
        covered = cov_lo & cov_hi

        out_qlo[idx] = ql + dl
        out_qhi[idx] = qh + dl
        out_cov[idx, 0] = covered
        out_cov[idx, 1] = cov_lo
        out_cov[idx, 2] = cov_hi

        if d <= warmup_end:
            warm_flag[idx] = True
            continue

        # pooled per-regime update: one per-observation step per stock-day
        # (sum over the cross-section; eta is a per-observation step size)
        n = len(block)
        if adaptive:
            # per-expert own-threshold misses and pinball losses, (n, K, E)
            def _bank_step(scores, Q, LW):
                eff = Q[None, :, :] + dl[:, None, None]
                diff = scores[:, None, None] - eff
                Q += (etas_g[None, :] * pi[:, None]
                      * ((diff > 0).sum(axis=0) - a_side * n))
                pb = np.where(diff > 0, tau * diff,
                              (tau - 1.0) * diff).mean(axis=0)
                LW -= meta_eta * pi[:, None] * pb
                # uniform mixing gated by pi as well: a regime's meta-weights
                # evolve in ITS OWN regime-time — ungated mixing would wash
                # rare-regime weights back to uniform on the 90%+ of days
                # that carry no signal for them
                w = _expert_weights(LW)
                mix = mix_sigma * pi[:, None]
                w = (1.0 - mix) * w + mix / len(etas_g)
                return Q, np.log(w)

            Q_hi, LW_hi = _bank_step(s, Q_hi, LW_hi)
            c_hi += eta_corr * pi * ((~cov_hi).sum() - a_side * n)
            if not one_sided:
                Q_lo, LW_lo = _bank_step(-s, Q_lo, LW_lo)
                c_lo += eta_corr * pi * ((~cov_lo).sum() - a_side * n)
        else:
            # average_errors=True divides the summed excess by n (a per-DAY
            # step on the mean error) — the e13 mechanism-identification arm
            denom = n if average_errors else 1
            if not one_sided:
                q_lo += eta * pi * ((~cov_lo).sum() - a_side * n) / denom
            q_hi += eta * pi * ((~cov_hi).sum() - a_side * n) / denom

        # per-stock offsets: own miss on either side, shrunk to zero
        own_err = ((~cov_lo).astype(float) + (~cov_hi).astype(float)) / 2.0
        step = eta_offset * (own_err - a_side) - offset_l2 * delta.reindex(
            block["ticker"]).values
        delta.loc[block["ticker"]] = delta.reindex(block["ticker"]).values + step

    df["q_lo"] = out_qlo
    df["q_hi"] = out_qhi
    df["covered"] = out_cov[:, 0]
    df["covered_lo"] = out_cov[:, 1]
    df["covered_hi"] = out_cov[:, 2]
    df["warmup"] = warm_flag
    if one_sided:
        df["width"] = df["q_hi"] * df["sigma_hat"]
    else:
        df["width"] = (df["q_lo"] + df["q_hi"]) * df["sigma_hat"]
    if adaptive:
        # issued effective per-observation rate per regime, per day (upper
        # side) — the paper diagnostic that the aggregator finds fast rates
        # in stress without hand-tuning. Stored as plain lists: parquet-safe.
        df.attrs["adaptive"] = {
            "eta_grid": list(eta_grid),
            "dates": eta_track_dates,
            "eff_eta_hi": eta_track_hi,
            "final_weights_hi": _expert_weights(LW_hi).tolist(),
        }
    return df
