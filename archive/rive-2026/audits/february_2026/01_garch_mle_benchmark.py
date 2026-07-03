"""
February Audit — Test 01: Proper GARCH(1,1) via MLE
=====================================================
Fits GARCH(1,1) per-ticker using Maximum Likelihood Estimation
(arch library) with expanding-window out-of-sample forecasts.

This replaces the January test that used hardcoded parameters.

Models tested:
  1. GARCH(1,1)  — Normal innovations, MLE-fitted per ticker
  2. GARCH(1,1)  — Student-t innovations
  3. GJR-GARCH   — Asymmetric leverage effect
  4. EWMA         — RiskMetrics λ=0.94 (unchanged, already correct)
  5. HAR-RV-X     — Enhanced HAR baseline (Ridge α=1.0, 6 features)
  6. RIVE         — Full coordinator (Ridge α=100, 10 features)

Author: February 2026 Audit
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Data loading (mirrors january tests pattern) ────────────────────

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


# ── GARCH MLE per-ticker ────────────────────────────────────────────

def garch_mle_forecast(returns, train_end_idx, vol_model="Garch",
                       p=1, o=0, q=1, dist="normal", refit_every=20):
    """
    Fit GARCH-family model via MLE and produce expanding-window forecasts.

    Args:
        returns:       np.array of percentage returns
        train_end_idx: int, first index of test period
        vol_model:     "Garch" or "EGARCH"
        p, o, q:       GARCH order (o>0 for GJR)
        dist:          "normal", "t", or "skewt"
        refit_every:   re-estimate every N observations

    Returns:
        forecasts (np.array), fitted_params (dict) or (None, None)
    """
    train = returns[:train_end_idx]
    train = train[np.isfinite(train)]
    if len(train) < 200:
        return None, None

    try:
        mdl = arch_model(train, mean="Constant", vol=vol_model,
                         p=p, o=o, q=q, dist=dist, rescale=False)
        res = mdl.fit(disp="off", show_warning=False)
    except Exception:
        return None, None

    params = dict(res.params)
    n_test = len(returns) - train_end_idx
    forecasts = np.empty(n_test)
    data_buf = list(train)

    for i in range(n_test):
        # periodic refit on expanding window
        if i > 0 and i % refit_every == 0:
            try:
                buf = np.array(data_buf)
                buf = buf[np.isfinite(buf)]
                mdl2 = arch_model(buf, mean="Constant", vol=vol_model,
                                  p=p, o=o, q=q, dist=dist, rescale=False)
                res = mdl2.fit(disp="off", show_warning=False)
                params = dict(res.params)
            except Exception:
                pass

        fc = res.forecast(horizon=1)
        var_pct2 = fc.variance.values[-1, 0]          # in (%²)
        forecasts[i] = np.log(var_pct2 / 1e4 + 1e-10) # → log(daily var)

        actual_ret = returns[train_end_idx + i] if (train_end_idx + i) < len(returns) else 0.0
        if np.isfinite(actual_ret):
            data_buf.append(actual_ret)

    return forecasts, params


def ewma_forecast(returns, train_end_idx, lam=0.94):
    """EWMA (RiskMetrics).  No fitting needed — λ=0.94 is fixed."""
    n = len(returns)
    s2 = np.zeros(n)
    s2[0] = np.nanvar(returns[:min(20, train_end_idx)])
    if not np.isfinite(s2[0]):
        s2[0] = 1e-4
    for t in range(1, n):
        r = returns[t - 1] if np.isfinite(returns[t - 1]) else 0.0
        s2[t] = lam * s2[t - 1] + (1 - lam) * r ** 2
    log_s2 = np.log(s2 / 1e4 + 1e-10)
    # shift by 1 so forecast[t] uses info up to t-1
    fc = np.roll(log_s2, 1)
    fc[0] = log_s2[0]
    return fc[train_end_idx:]


# ── HAR-RV-X and RIVE helpers ──────────────────────────────────────

def build_rive_df(targets, residuals, news, retail):
    """Merge all features into a single frame."""
    df = targets.copy()
    df = df.merge(residuals[["date", "ticker", "pred_tech"]],
                  on=["date", "ticker"], how="left")
    df["tech_pred"] = df["pred_tech"]

    if news is not None and "news_risk_score" in news.columns:
        df = df.merge(news[["date", "ticker", "news_risk_score"]],
                      on=["date", "ticker"], how="left")
    df["news_risk_score"] = df.get("news_risk_score", pd.Series(0.2, index=df.index)).fillna(0.2)

    if retail is not None and "retail_risk_score" in retail.columns:
        df = df.merge(retail[["date", "ticker", "retail_risk_score"]],
                      on=["date", "ticker"], how="left")
    df["retail_risk_score"] = df.get("retail_risk_score", pd.Series(0.2, index=df.index)).fillna(0.2)

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


# ── Main test ───────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 72)
    print("  FEBRUARY AUDIT — TEST 01: PROPER GARCH(1,1) MLE BENCHMARK")
    print("=" * 72)

    targets, residuals, news, retail = load_data()
    targets = targets.sort_values(["ticker", "date"])
    targets["returns"] = targets.groupby("ticker")["close"].pct_change() * 100

    cutoff = pd.to_datetime("2023-01-01")
    tickers = sorted(targets["ticker"].unique())

    # ── A. GARCH-family per-ticker forecasts ─────────────────────────
    garch_configs = {
        "GARCH(1,1)-N":   dict(vol_model="Garch", p=1, o=0, q=1, dist="normal"),
        "GARCH(1,1)-t":   dict(vol_model="Garch", p=1, o=0, q=1, dist="t"),
        "GJR-GARCH":      dict(vol_model="Garch", p=1, o=1, q=1, dist="normal"),
    }

    # storage: model → list of (forecast, actual) arrays
    model_forecasts = {name: ([], []) for name in garch_configs}
    model_forecasts["EWMA"] = ([], [])
    param_log = {name: [] for name in garch_configs}

    print(f"\nFitting {len(garch_configs)} GARCH variants + EWMA across {len(tickers)} tickers...")
    print("(MLE per ticker, expanding window, refit every 20 obs)\n")

    for i, tk in enumerate(tickers):
        tdf = targets[targets["ticker"] == tk].sort_values("date")
        train_n = (tdf["date"] < cutoff).sum()
        if train_n < 200 or len(tdf) - train_n < 50:
            continue

        rets = tdf["returns"].values
        actuals = tdf["target_log_var"].values[train_n:]

        # GARCH variants
        for name, cfg in garch_configs.items():
            fc, params = garch_mle_forecast(rets, train_n, **cfg)
            if fc is not None and len(fc) == len(actuals):
                ok = np.isfinite(actuals) & np.isfinite(fc)
                model_forecasts[name][0].extend(fc[ok])
                model_forecasts[name][1].extend(actuals[ok])
                if params:
                    param_log[name].append(params)

        # EWMA
        fc_ewma = ewma_forecast(rets, train_n)
        if len(fc_ewma) == len(actuals):
            ok = np.isfinite(actuals) & np.isfinite(fc_ewma)
            model_forecasts["EWMA"][0].extend(fc_ewma[ok])
            model_forecasts["EWMA"][1].extend(actuals[ok])

        if (i + 1) % 10 == 0 or i == len(tickers) - 1:
            print(f"  [{i+1}/{len(tickers)}] tickers processed")

    # ── B. HAR-RV-X baseline ─────────────────────────────────────────
    print("\nFitting HAR-RV-X baseline (Ridge α=1.0, 6 features)...")
    har_feats = ["prev_day_rv", "rv_5d_mean", "rv_20d_mean",
                 "returns_sq_lag_1", "VIX_close", "rsi_14"]
    har_feats = [f for f in har_feats if f in targets.columns]

    if "returns_sq_lag_1" not in targets.columns and "close" in targets.columns:
        targets["returns_sq_lag_1"] = (targets["returns"] ** 2)

    har_feats_avail = [f for f in har_feats if f in targets.columns]

    tr = targets[targets["date"] < cutoff].dropna(subset=har_feats_avail + ["target_log_var"])
    te = targets[targets["date"] >= cutoff].dropna(subset=har_feats_avail + ["target_log_var"])
    tr = tr[np.isfinite(tr["target_log_var"])]
    te = te[np.isfinite(te["target_log_var"])]

    har_model = Ridge(alpha=1.0)
    har_model.fit(tr[har_feats_avail].fillna(0), tr["target_log_var"])
    har_pred = har_model.predict(te[har_feats_avail].fillna(0))
    har_actual = te["target_log_var"].values

    # ── C. RIVE coordinator ──────────────────────────────────────────
    print("Fitting RIVE coordinator (Ridge α=100, 10 features)...")
    rive_df = build_rive_df(targets, residuals, news, retail)
    rive_feats = ["tech_pred", "news_risk_score", "retail_risk_score",
                  "is_friday", "is_monday", "is_q4",
                  "vol_ma5", "vol_ma10", "vol_std5", "news_x_retail"]

    r_tr = rive_df[rive_df["date"] < cutoff].dropna(subset=rive_feats + ["target_log_var"])
    r_te = rive_df[rive_df["date"] >= cutoff].dropna(subset=rive_feats + ["target_log_var"])
    r_tr = r_tr[np.isfinite(r_tr["target_log_var"])]
    r_te = r_te[np.isfinite(r_te["target_log_var"])]

    y_tr = r_tr["target_log_var"].values
    lo, hi = np.percentile(y_tr, 2), np.percentile(y_tr, 98)
    y_tr_w = np.clip(y_tr, lo, hi)

    rive_model = Ridge(alpha=100.0)
    rive_model.fit(r_tr[rive_feats].fillna(0), y_tr_w)
    rive_pred = rive_model.predict(r_te[rive_feats].fillna(0))
    rive_actual = r_te["target_log_var"].values

    # ── D. Results table ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RESULTS: ALL MODELS")
    print("=" * 72)

    rows = []
    for name in list(garch_configs.keys()) + ["EWMA"]:
        fc_arr = np.array(model_forecasts[name][0])
        ac_arr = np.array(model_forecasts[name][1])
        if len(fc_arr) > 0:
            r2  = r2_score(ac_arr, fc_arr)
            mae = mean_absolute_error(ac_arr, fc_arr)
            rows.append({"Model": name, "R2": r2, "MAE": mae, "N": len(fc_arr)})

    rows.append({"Model": f"HAR-RV-X ({len(har_feats_avail)}feat)",
                 "R2": r2_score(har_actual, har_pred),
                 "MAE": mean_absolute_error(har_actual, har_pred),
                 "N": len(har_actual)})
    rows.append({"Model": "RIVE (10feat)",
                 "R2": r2_score(rive_actual, rive_pred),
                 "MAE": mean_absolute_error(rive_actual, rive_pred),
                 "N": len(rive_actual)})

    res_df = pd.DataFrame(rows)

    print(f"\n{'Model':<22s} {'R²':>10s} {'MAE':>10s} {'N':>10s}")
    print("-" * 56)
    for _, r in res_df.iterrows():
        tag = " <-- RIVE" if "RIVE" in r["Model"] else ""
        print(f"  {r['Model']:<20s} {r['R2']*100:>8.2f}%  {r['MAE']:>8.4f}  {int(r['N']):>8,}{tag}")

    # ── E. Estimated GARCH parameters ────────────────────────────────
    print("\n" + "-" * 72)
    print("  ESTIMATED GARCH(1,1)-N PARAMETERS (median across tickers)")
    print("-" * 72)

    if param_log["GARCH(1,1)-N"]:
        all_params = pd.DataFrame(param_log["GARCH(1,1)-N"])
        for col in all_params.columns:
            vals = all_params[col].dropna()
            if len(vals) > 0:
                print(f"    {col:12s}:  median={vals.median():.6f}  "
                      f"mean={vals.mean():.6f}  std={vals.std():.6f}")

    # ── F. Improvement summary ───────────────────────────────────────
    rive_r2 = r2_score(rive_actual, rive_pred)
    best_garch_name = res_df[res_df["Model"].str.contains("GARCH|GJR")]["R2"].idxmax()
    best_garch_row = res_df.loc[best_garch_name]

    print("\n" + "=" * 72)
    print("  IMPROVEMENT SUMMARY")
    print("=" * 72)
    print(f"  Best GARCH variant:  {best_garch_row['Model']}  →  R² = {best_garch_row['R2']*100:.2f}%")
    print(f"  HAR-RV-X baseline:   R² = {r2_score(har_actual, har_pred)*100:.2f}%")
    print(f"  RIVE coordinator:    R² = {rive_r2*100:.2f}%")
    print()
    print(f"  RIVE vs best GARCH:  {(rive_r2 - best_garch_row['R2'])*100:+.2f} pp")
    print(f"  RIVE vs HAR-RV-X:    {(rive_r2 - r2_score(har_actual, har_pred))*100:+.2f} pp")
    print()

    if rive_r2 > best_garch_row["R2"]:
        print("  VERDICT: RIVE outperforms all GARCH variants (MLE-fitted)")
    else:
        print("  WARNING: A GARCH variant matches or beats RIVE!")

    print("=" * 72)

    # Save results for DM test consumption
    out_path = Path(__file__).parent / "garch_mle_results.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
