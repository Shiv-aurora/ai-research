"""
February Audit — Test 02: Diebold-Mariano Significance Test
=============================================================
Tests whether RIVE's forecast improvement over baselines is
statistically significant using the Diebold-Mariano (1995) test
with Newey-West HAC standard errors.

Comparisons:
  1. RIVE  vs  HAR-RV-X       (primary claim)
  2. RIVE  vs  GARCH(1,1)-MLE (secondary)
  3. RIVE  vs  EWMA           (secondary)

Also runs:
  4. Mincer-Zarnowitz unbiasedness regression for RIVE

Author: February 2026 Audit
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Diebold-Mariano test ────────────────────────────────────────────

def diebold_mariano(e1, e2, h=1, loss="MSE"):
    """
    Diebold-Mariano test for equal predictive accuracy.

    H0: E[d_t] = 0   (equal accuracy)
    H1: E[d_t] ≠ 0   (different accuracy)

    Args:
        e1: forecast errors from model 1 (np.array)
        e2: forecast errors from model 2 (np.array)
        h:  forecast horizon (for Newey-West bandwidth = h-1)
        loss: "MSE" or "MAE"

    Returns:
        dm_stat, p_value, mean_loss_diff
    """
    if loss == "MSE":
        d = e1 ** 2 - e2 ** 2
    elif loss == "MAE":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    T = len(d)
    d_bar = d.mean()

    # Newey-West HAC variance (bandwidth = h-1 for h-step ahead)
    bandwidth = max(h - 1, 0)
    gamma_0 = np.mean((d - d_bar) ** 2)
    gamma_sum = 0.0
    for k in range(1, bandwidth + 1):
        weight = 1 - k / (bandwidth + 1)  # Bartlett kernel
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += 2 * weight * gamma_k

    var_d = (gamma_0 + gamma_sum) / T

    if var_d <= 0:
        # Fallback: use simple variance
        var_d = np.var(d, ddof=1) / T

    dm_stat = d_bar / np.sqrt(var_d)

    # Two-sided p-value (asymptotic normal)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return dm_stat, p_value, d_bar


def mincer_zarnowitz(actual, forecast):
    """
    Mincer-Zarnowitz regression:  actual_t = α + β * forecast_t + ε_t

    Tests:
      H0: α=0, β=1  (unbiased, efficient forecast)

    Returns:
        alpha, beta, f_stat, f_pvalue, r2
    """
    X = forecast.reshape(-1, 1)
    y = actual

    reg = LinearRegression().fit(X, y)
    alpha = reg.intercept_
    beta = reg.coef_[0]
    r2 = reg.score(X, y)

    # Joint F-test for (α=0, β=1)
    y_pred = reg.predict(X)
    resid = y - y_pred
    n = len(y)
    SSR_unrestricted = np.sum(resid ** 2)

    # Restricted model: actual = 0 + 1*forecast + ε  →  resid = actual - forecast
    SSR_restricted = np.sum((actual - forecast) ** 2)

    # F = ((SSR_r - SSR_u)/q) / (SSR_u/(n-k))
    q = 2  # two restrictions (α=0, β=1)
    k = 2  # intercept + slope
    f_stat = ((SSR_restricted - SSR_unrestricted) / q) / (SSR_unrestricted / (n - k))
    f_pvalue = 1 - stats.f.cdf(f_stat, q, n - k)

    return alpha, beta, f_stat, f_pvalue, r2


# ── Data loading ────────────────────────────────────────────────────

def load_data():
    dp = PROJECT_ROOT / "data" / "processed"
    targets = pd.read_parquet(dp / "targets.parquet")
    residuals = pd.read_parquet(dp / "residuals.parquet")

    for d in [targets, residuals]:
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)

    try:
        news = pd.read_parquet(dp / "news_predictions.parquet")
        news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(None)
        if "news_pred" in news.columns:
            news = news.rename(columns={"news_pred": "news_risk_score"})
    except Exception:
        news = None

    try:
        retail = pd.read_parquet(dp / "retail_predictions.parquet")
        retail["date"] = pd.to_datetime(retail["date"]).dt.tz_localize(None)
    except Exception:
        retail = None

    return targets, residuals, news, retail


def build_rive_df(targets, residuals, news, retail):
    df = targets.copy()
    df = df.merge(residuals[["date", "ticker", "pred_tech"]],
                  on=["date", "ticker"], how="left")
    df["tech_pred"] = df["pred_tech"]

    if news is not None and "news_risk_score" in news.columns:
        df = df.merge(news[["date", "ticker", "news_risk_score"]],
                      on=["date", "ticker"], how="left")
    df["news_risk_score"] = df.get("news_risk_score",
                                    pd.Series(0.2, index=df.index)).fillna(0.2)

    if retail is not None and "retail_risk_score" in retail.columns:
        df = df.merge(retail[["date", "ticker", "retail_risk_score"]],
                      on=["date", "ticker"], how="left")
    df["retail_risk_score"] = df.get("retail_risk_score",
                                      pd.Series(0.2, index=df.index)).fillna(0.2)

    dow = df["date"].dt.dayofweek
    df["is_monday"] = (dow == 0).astype(int)
    df["is_friday"] = (dow == 4).astype(int)
    df["is_q4"] = (df["date"].dt.quarter == 4).astype(int)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    for tk in df["ticker"].unique():
        m = df["ticker"] == tk
        tv = df.loc[m, "target_log_var"]
        df.loc[m, "vol_ma5"]  = tv.rolling(5,  min_periods=1).mean().shift(1)
        df.loc[m, "vol_ma10"] = tv.rolling(10, min_periods=1).mean().shift(1)
        df.loc[m, "vol_std5"] = tv.rolling(5,  min_periods=2).std().shift(1)

    mu = df["target_log_var"].mean()
    df["vol_ma5"]  = df["vol_ma5"].fillna(mu)
    df["vol_ma10"] = df["vol_ma10"].fillna(mu)
    df["vol_std5"] = df["vol_std5"].fillna(0)
    df["news_x_retail"] = df["news_risk_score"] * df["retail_risk_score"]
    return df


def garch_mle_forecast_single(returns, train_end_idx, refit_every=20):
    """GARCH(1,1)-Normal MLE for a single ticker."""
    train = returns[:train_end_idx]
    train = train[np.isfinite(train)]
    if len(train) < 200:
        return None

    try:
        mdl = arch_model(train, mean="Constant", vol="Garch",
                         p=1, q=1, dist="normal", rescale=False)
        res = mdl.fit(disp="off", show_warning=False)
    except Exception:
        return None

    n_test = len(returns) - train_end_idx
    fc = np.empty(n_test)
    buf = list(train)

    for i in range(n_test):
        if i > 0 and i % refit_every == 0:
            try:
                b = np.array(buf); b = b[np.isfinite(b)]
                mdl2 = arch_model(b, mean="Constant", vol="Garch",
                                  p=1, q=1, dist="normal", rescale=False)
                res = mdl2.fit(disp="off", show_warning=False)
            except Exception:
                pass

        fv = res.forecast(horizon=1).variance.values[-1, 0]
        fc[i] = np.log(fv / 1e4 + 1e-10)

        r = returns[train_end_idx + i] if (train_end_idx + i) < len(returns) else 0.0
        if np.isfinite(r):
            buf.append(r)

    return fc


def ewma_forecast(returns, train_end_idx, lam=0.94):
    n = len(returns)
    s2 = np.zeros(n)
    s2[0] = np.nanvar(returns[:min(20, train_end_idx)])
    if not np.isfinite(s2[0]):
        s2[0] = 1e-4
    for t in range(1, n):
        r = returns[t - 1] if np.isfinite(returns[t - 1]) else 0.0
        s2[t] = lam * s2[t - 1] + (1 - lam) * r ** 2
    log_s2 = np.log(s2 / 1e4 + 1e-10)
    fc = np.roll(log_s2, 1); fc[0] = log_s2[0]
    return fc[train_end_idx:]


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 72)
    print("  FEBRUARY AUDIT — TEST 02: DIEBOLD-MARIANO SIGNIFICANCE TEST")
    print("=" * 72)

    targets, residuals, news, retail = load_data()
    targets = targets.sort_values(["ticker", "date"])
    targets["returns"] = targets.groupby("ticker")["close"].pct_change() * 100

    cutoff = pd.to_datetime("2023-01-01")

    # ── 1. Build aligned forecasts for all models ────────────────────
    #    We need per-observation forecasts aligned on (date, ticker)
    #    so the DM test compares the same observations.

    rive_df = build_rive_df(targets, residuals, news, retail)
    rive_feats = ["tech_pred", "news_risk_score", "retail_risk_score",
                  "is_friday", "is_monday", "is_q4",
                  "vol_ma5", "vol_ma10", "vol_std5", "news_x_retail"]

    har_feats_all = ["prev_day_rv", "rv_5d_mean", "rv_20d_mean",
                     "returns_sq_lag_1", "VIX_close", "rsi_14"]
    if "returns_sq_lag_1" not in targets.columns:
        targets["returns_sq_lag_1"] = targets["returns"] ** 2
    har_feats = [f for f in har_feats_all if f in targets.columns]

    # ── Train HAR-RV-X ───────────────────────────────────────────────
    print("\nTraining HAR-RV-X...")
    tr_har = targets[targets["date"] < cutoff].dropna(
        subset=har_feats + ["target_log_var"])
    tr_har = tr_har[np.isfinite(tr_har["target_log_var"])]
    har_model = Ridge(alpha=1.0)
    har_model.fit(tr_har[har_feats].fillna(0), tr_har["target_log_var"])

    # ── Train RIVE ───────────────────────────────────────────────────
    print("Training RIVE...")
    r_tr = rive_df[rive_df["date"] < cutoff].dropna(
        subset=rive_feats + ["target_log_var"])
    r_tr = r_tr[np.isfinite(r_tr["target_log_var"])]
    y_tr = r_tr["target_log_var"].values
    lo, hi = np.percentile(y_tr, 2), np.percentile(y_tr, 98)
    rive_model = Ridge(alpha=100.0)
    rive_model.fit(r_tr[rive_feats].fillna(0), np.clip(y_tr, lo, hi))

    # ── Generate per-ticker aligned forecasts ────────────────────────
    print("Generating aligned per-ticker forecasts (GARCH MLE + EWMA)...\n")
    tickers = sorted(targets["ticker"].unique())

    records = []  # list of dicts with date, ticker, actual, pred_*

    for i, tk in enumerate(tickers):
        tdf = targets[targets["ticker"] == tk].sort_values("date")
        train_n = (tdf["date"] < cutoff).sum()
        if train_n < 200 or len(tdf) - train_n < 50:
            continue

        test_idx = tdf.index[train_n:]
        test_dates = tdf.loc[test_idx, "date"].values
        actuals = tdf.loc[test_idx, "target_log_var"].values
        rets = tdf["returns"].values

        # GARCH MLE
        garch_fc = garch_mle_forecast_single(rets, train_n)
        # EWMA
        ewma_fc = ewma_forecast(rets, train_n)

        if garch_fc is None or len(garch_fc) != len(actuals):
            continue
        if len(ewma_fc) != len(actuals):
            continue

        # HAR-RV-X for this ticker's test rows
        te_har = targets.loc[test_idx].dropna(subset=har_feats + ["target_log_var"])
        if len(te_har) == 0:
            continue
        har_pred = har_model.predict(te_har[har_feats].fillna(0))

        # RIVE for this ticker's test rows
        rdf_tk = rive_df[(rive_df["ticker"] == tk) & (rive_df["date"] >= cutoff)]
        rdf_tk = rdf_tk.dropna(subset=rive_feats + ["target_log_var"])
        if len(rdf_tk) == 0:
            continue
        rive_pred = rive_model.predict(rdf_tk[rive_feats].fillna(0))

        # Align everything on the HAR test dates (most restrictive)
        har_dates = set(te_har["date"].values)
        rive_dates = set(rdf_tk["date"].values)
        common_dates = har_dates & rive_dates

        for j, (d, act, gf, ef) in enumerate(
                zip(test_dates, actuals, garch_fc, ewma_fc)):
            if d not in common_dates or not np.isfinite(act):
                continue
            # find HAR and RIVE predictions for this date
            har_mask = te_har["date"].values == d
            rive_mask = rdf_tk["date"].values == d
            if har_mask.sum() != 1 or rive_mask.sum() != 1:
                continue
            records.append({
                "date": d, "ticker": tk,
                "actual": act,
                "pred_garch": gf,
                "pred_ewma": ef,
                "pred_har": har_pred[np.where(har_mask)[0][0]],
                "pred_rive": rive_pred[np.where(rive_mask)[0][0]],
            })

        if (i + 1) % 10 == 0 or i == len(tickers) - 1:
            print(f"  [{i+1}/{len(tickers)}] tickers done  ({len(records):,} aligned obs)")

    aligned = pd.DataFrame(records)
    print(f"\n  Total aligned observations: {len(aligned):,}")

    # ── 2. Compute forecast errors ───────────────────────────────────
    actual = aligned["actual"].values
    e_rive  = actual - aligned["pred_rive"].values
    e_har   = actual - aligned["pred_har"].values
    e_garch = actual - aligned["pred_garch"].values
    e_ewma  = actual - aligned["pred_ewma"].values

    # Quick R² check
    print(f"\n  Aligned R² check:")
    for name, pred_col in [("GARCH-MLE", "pred_garch"), ("EWMA", "pred_ewma"),
                            ("HAR-RV-X", "pred_har"), ("RIVE", "pred_rive")]:
        r2 = r2_score(actual, aligned[pred_col].values)
        print(f"    {name:12s}: {r2*100:.2f}%")

    # ── 3. Diebold-Mariano tests ────────────────────────────────────
    print("\n" + "=" * 72)
    print("  DIEBOLD-MARIANO TEST RESULTS")
    print("  H0: equal predictive accuracy (two-sided)")
    print("=" * 72)

    comparisons = [
        ("RIVE vs HAR-RV-X",   e_har,   e_rive,  "PRIMARY"),
        ("RIVE vs GARCH-MLE",  e_garch, e_rive,  "SECONDARY"),
        ("RIVE vs EWMA",       e_ewma,  e_rive,  "SECONDARY"),
    ]

    dm_results = []
    for label, e_base, e_test, importance in comparisons:
        dm_mse, p_mse, d_mse = diebold_mariano(e_base, e_test, h=1, loss="MSE")
        dm_mae, p_mae, d_mae = diebold_mariano(e_base, e_test, h=1, loss="MAE")

        sig_mse = "***" if p_mse < 0.001 else "**" if p_mse < 0.01 else "*" if p_mse < 0.05 else "n.s."
        sig_mae = "***" if p_mae < 0.001 else "**" if p_mae < 0.01 else "*" if p_mae < 0.05 else "n.s."

        print(f"\n  [{importance}] {label}")
        print(f"  {'─'*50}")
        print(f"    MSE loss:  DM = {dm_mse:+.4f},  p = {p_mse:.6f}  {sig_mse}")
        print(f"               mean(d_t) = {d_mse:.6f}  "
              f"({'baseline worse' if d_mse > 0 else 'RIVE worse'})")
        print(f"    MAE loss:  DM = {dm_mae:+.4f},  p = {p_mae:.6f}  {sig_mae}")
        print(f"               mean(d_t) = {d_mae:.6f}  "
              f"({'baseline worse' if d_mae > 0 else 'RIVE worse'})")

        dm_results.append({
            "Comparison": label, "Importance": importance,
            "DM_MSE": dm_mse, "p_MSE": p_mse, "sig_MSE": sig_mse,
            "DM_MAE": dm_mae, "p_MAE": p_mae, "sig_MAE": sig_mae,
        })

    # ── 4. Mincer-Zarnowitz regression ──────────────────────────────
    print("\n" + "=" * 72)
    print("  MINCER-ZARNOWITZ UNBIASEDNESS TEST")
    print("  H0: α=0, β=1  (forecast is unbiased and efficient)")
    print("=" * 72)

    for name, pred_col in [("RIVE", "pred_rive"), ("HAR-RV-X", "pred_har"),
                            ("GARCH-MLE", "pred_garch")]:
        alpha, beta, f_stat, f_pval, r2 = mincer_zarnowitz(
            actual, aligned[pred_col].values)
        sig = "REJECT H0" if f_pval < 0.05 else "FAIL TO REJECT"
        print(f"\n  {name}:")
        print(f"    α = {alpha:.4f},  β = {beta:.4f},  R² = {r2:.4f}")
        print(f"    F-test (α=0,β=1): F = {f_stat:.2f},  p = {f_pval:.6f}  → {sig}")

    # ── 5. Summary verdict ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)

    primary = dm_results[0]
    if primary["p_MSE"] < 0.05 and primary["DM_MSE"] > 0:
        print("  RIVE significantly outperforms HAR-RV-X (p < 0.05, MSE loss)")
        print("  The improvement is STATISTICALLY SIGNIFICANT.")
    elif primary["p_MSE"] < 0.10 and primary["DM_MSE"] > 0:
        print("  RIVE outperforms HAR-RV-X at 10% level (marginally significant)")
    else:
        print("  RIVE vs HAR-RV-X: NOT statistically significant at 5% level")
        print("  The R² improvement may be due to sampling variation.")

    print("=" * 72)

    # Save
    out = pd.DataFrame(dm_results)
    out_path = Path(__file__).parent / "dm_test_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
