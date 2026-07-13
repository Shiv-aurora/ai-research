"""Generate all paper figures (vector PDF) from reports/*.csv artifacts.

Figures are regenerated, never hand-edited; every number in a figure traces
to a CSV produced by a scripts/e*.py experiment. Palette is a CVD-safe
Okabe-Ito subset (validated): blue #0072B2 (our method), vermillion #D55E00
(baseline), green #009E73 (oracle/third series). Nominal targets are dashed
gray reference lines.

Usage: .venv/bin/python scripts/make_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "paper2" / "figures"
REP = PROJECT_ROOT / "reports"

BLUE, VERM, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#7f7f7f"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "pdf.fonttype": 42,
})

REGIMES = ["calm", "normal", "elevated", "stress"]
CUTS = [0.5, 0.8, 0.95]


def regime_hac_se(method: str) -> dict:
    """Per-regime coverage mean and Newey-West SE over the daily
    cross-sectional means (reports/e2_daily_coverage.parquet, common
    sample) — the same construction as the paper's clustered inference."""
    daily = pd.read_parquet(REP / "e2_daily_coverage.parquet")
    daily = daily[daily["method"] == method].dropna(subset=["vix_pctl"])
    k = np.searchsorted(CUTS, daily["vix_pctl"].values, side="right")
    out = {}
    for j, name in enumerate(REGIMES):
        x = daily.loc[k == j, "covered"].values.astype(float)
        n = len(x)
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
        xc = x - x.mean()
        var = xc @ xc / n
        for L in range(1, lag + 1):
            var += 2 * (1 - L / (lag + 1)) * (xc[L:] @ xc[:-L]) / n
        out[name] = (float(x.mean()), float(np.sqrt(var / n)))
    return out


def fig_coverage_by_regime():
    """Figure 1: the phenomenon and the repair, by regime (E2).\n\n    Point-and-whisker (95% date-clustered HAC CIs), not bars: the\n    review correctly noted truncated bar axes amplify differences."""
    e2 = pd.read_csv(REP / "e2_full_summary.csv").set_index("method")
    cols = [f"cov_{r}" for r in REGIMES]
    aci = e2.loc["aci", cols].values.astype(float)
    ours = e2.loc["rc_adaptive", cols].values.astype(float)

    se_aci = regime_hac_se("aci")
    se_ours = regime_hac_se("rc_adaptive")
    x = np.arange(len(REGIMES))
    off = 0.16
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.axhline(0.90, ls="--", lw=0.9, color=GRAY, zorder=2)
    ax.text(3.42, 0.902, "nominal 90%", color=GRAY, fontsize=7,
            ha="right", va="bottom")
    for xo, vals, ses, color, label, dx in [
            (x - off, aci, se_aci, VERM, "ACI (per stock)", -0.055),
            (x + off, ours, se_ours, BLUE, "RC (ours)", 0.055)]:
        err = 1.96 * np.array([ses[r][1] for r in REGIMES])
        ax.errorbar(xo, vals, yerr=err, fmt="o", ms=4.2, color=color,
                    elinewidth=1.1, capsize=2.4, label=label, zorder=3)
        for xi, v in zip(xo, vals):
            ax.text(xi + dx, v, f"{100*v:.1f}", ha="left" if dx > 0
                    else "right", va="center", fontsize=6.7, color="#333")
    ax.set_xticks(x, [r.capitalize() for r in REGIMES])
    ax.set_xlim(-0.85, 3.5)
    ax.set_ylabel("Empirical coverage")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_coverage_by_regime.pdf")
    plt.close(fig)


def fig_dse_profile():
    """Days-since-stress-entry: the day-2 pit survives hindsight regimes\n    (E6b), with 95% date-clustered CIs and date counts (E18) on the\n    causal-bins series."""
    dse = pd.read_csv(REP / "e6b_oracle_dse.csv", index_col=0)
    ci = pd.read_csv(REP / "e18_dse_ci.csv", dtype={"dse": str})
    ci = ci.set_index("dse").reindex([str(i) for i in dse.index]
                                     if dse.index.dtype == object
                                     else dse.index.astype(str))
    x = np.arange(len(dse.index))
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.axhline(0.90, ls="--", lw=0.9, color=GRAY)
    if ci["se"].notna().all():
        ax.errorbar(x, dse["bins_k4"], yerr=1.96 * ci["se"].values,
                    fmt="none", ecolor=BLUE, elinewidth=0.9, capsize=2.2,
                    alpha=0.65, zorder=2)
    ax.plot(x, dse["bins_k4"], "-o", color=BLUE, ms=3.5, lw=1.6,
            label="VIX bins (causal)")
    ax.plot(x, dse["hmm_filtered"], "-s", color=VERM, ms=3.2, lw=1.4,
            label="HMM filtered (causal)")
    ax.plot(x, dse["hmm_oracle"], "-^", color=GREEN, ms=3.5, lw=1.4,
            label="HMM smoothed (hindsight)")
    ax.annotate("day-2 pit", xy=(1, float(dse['bins_k4'].iloc[1]) - 0.006),
                xytext=(0.05, 0.732), fontsize=7, color="#333",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#666"))
    labels = [f"{i}\n({int(nd)}d)" for i, nd in
              zip(dse.index, ci["n_dates"].values)]
    ax.set_xticks(x, labels)
    ax.set_xlabel("Days since stress entry (distinct dates below)")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.62, 0.97)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_dse_oracle.pdf")
    plt.close(fig)


def fig_decomposition():
    """Mechanism decomposition (E6, E6b, E2): dot plot with 95% CIs where\n    the common-sample daily series exists (per-stock RC and hindsight\n    arms come from full-panel ablation runs without one)."""
    e6 = pd.read_csv(REP / "e6_ablations.csv").set_index("config")
    e6b = pd.read_csv(REP / "e6b_oracle_regimes.csv").set_index("membership")
    e2 = pd.read_csv(REP / "e2_full_summary.csv").set_index("method")
    rows = [
        ("Per-stock ACI", float(e2.loc["aci", "cov_stress"]), VERM),
        ("Per-stock RC (no pooling)", float(e6.loc["per_stock", "cov_stress"]), VERM),
        ("Pooled, K=1 (no regimes)", float(e6.loc["K=1", "cov_stress"]), BLUE),
        ("Pooled, K=4 (ours)", float(e6.loc["K=4", "cov_stress"]), BLUE),
        ("Pooled, K=4, hindsight regimes", float(e6b.loc["hmm_oracle", "cov_stress"]), GREEN),
    ]
    ses = {"Per-stock ACI": regime_hac_se("aci")["stress"][1],
           "Pooled, K=1 (no regimes)":
               regime_hac_se("pooled_k1")["stress"][1],
           "Pooled, K=4 (ours)":
               regime_hac_se("rc_adaptive")["stress"][1]}
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    colors = [r[2] for r in rows][::-1]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.axvline(0.90, ls="--", lw=0.9, color=GRAY, zorder=2)
    for yi, v, c, lab in zip(y, vals, colors, labels):
        if lab in ses:
            ax.errorbar([v], [yi], xerr=[1.96 * ses[lab]], fmt="none",
                        ecolor=c, elinewidth=1.0, capsize=2.4, zorder=3)
        ax.plot([v], [yi], "o", ms=4.6, color=c, zorder=4)
        ax.text(v, yi + 0.32, f"{100*v:.1f}", ha="center", va="bottom",
                fontsize=6.8, color="#333", zorder=5)
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlabel("Stress-regime coverage (nominal 90%)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_episodes():
    """Crisis episodes: ACI vs ours, two-sided coverage (E9)."""
    e9 = pd.read_csv(REP / "e9_stress_windows.csv")
    e9 = e9.iloc[::-1].reset_index(drop=True)   # earliest at top after flip
    y = np.arange(len(e9))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for yi, a, o in zip(y, e9["aci_cov"], e9["rc_adaptive_cov"]):
        ax.plot([a, o], [yi, yi], "-", color="#bbb", lw=1.2, zorder=2)
    ax.plot(e9["aci_cov"], y, "o", color=VERM, ms=4.5, label="ACI", zorder=3)
    ax.plot(e9["rc_adaptive_cov"], y, "o", color=BLUE, ms=4.5,
            label="RC (ours)", zorder=3)
    ax.axvline(0.90, ls="--", lw=0.9, color=GRAY, zorder=1)
    ax.set_yticks(y, e9["episode"])
    ax.set_xlim(0.72, None)
    ax.set_xlabel("Coverage in episode (nominal 90%)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_episodes.pdf")
    plt.close(fig)


def fig_var_breach():
    """Stress-day VaR breach rates vs nominal budgets, 7 methods (E3)."""
    e3 = pd.read_csv(REP / "e3_var_summary.csv")
    order = ["normal", "fhs", "garch_t", "caviar", "aci", "rc_panel", "rc_adaptive"]
    names = {"normal": "Normal", "fhs": "FHS", "garch_t": "GARCH-t",
             "caviar": "CAViaR", "aci": "ACI", "rc_panel": "RC (hand-tuned)",
             "rc_adaptive": "RC (adaptive)"}
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.1), sharey=True)
    for ax, alpha, nom in zip(axes, [0.05, 0.01], [5.0, 1.0]):
        sub = e3[e3["alpha"] == alpha].set_index("method")
        vals = 100 * sub.loc[order, "rate_stress"].values.astype(float)
        colors = [BLUE if m.startswith("rc_") else VERM for m in order]
        ax.barh(y, vals, 0.62, color=colors, zorder=3)
        ax.axvline(nom, ls="--", lw=0.9, color=GRAY, zorder=4)
        ax.set_xlabel(f"{100 * (1 - alpha):.0f}\\% VaR\n(budget {nom:.0f}%)"
                      .replace("\\%", "%"))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [names[m] for m in order])
    axes[1].tick_params(left=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_var_breach.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_tsfm_repair():
    """Chronos native band vs conformalized band, coverage by regime (E4)."""
    raw = pd.read_csv(REP / "e4_tsfm_raw_coverage.csv").set_index("vix_pctl")
    rep = pd.read_csv(REP / "e4_tsfm_repaired_coverage.csv").set_index("vix_pctl")
    x = np.arange(len(REGIMES))
    off = 0.14
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.axhline(0.80, ls="--", lw=0.9, color=GRAY, zorder=2)
    ax.text(-0.52, 0.8025, "nominal 80%", color=GRAY, fontsize=7,
            ha="left", va="bottom")
    ax.plot(x - off, raw.loc[REGIMES, "mean"], "o", ms=4.6, color=VERM,
            label="Chronos native 80% band", zorder=3)
    ax.plot(x + off, rep.loc[REGIMES, "mean"], "o", ms=4.6, color=BLUE,
            label="Conformalized (ours)", zorder=3)
    for xi, v in zip(x - off, raw.loc[REGIMES, "mean"]):
        ax.text(xi - 0.05, v, f"{100*v:.1f}", ha="right", va="center",
                fontsize=6.7, color="#333")
    for xi, v in zip(x + off, rep.loc[REGIMES, "mean"]):
        ax.text(xi + 0.05, v, f"{100*v:.1f}", ha="left", va="center",
                fontsize=6.7, color="#333")
    ax.set_xticks(x, [r.capitalize() for r in REGIMES])
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0.745, 0.83)
    ax.set_ylabel("Empirical coverage")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncols=2, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tsfm_repair.pdf")
    plt.close(fig)


def fig_alpha_sweep():
    """Stress coverage deficit (points below nominal) across alpha (E12)."""
    e12 = pd.read_csv(REP / "e12_alpha_sweep.csv")
    alphas = [0.05, 0.1, 0.2]
    x = np.arange(len(alphas))
    w = 0.34
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    for off, method, color, label in [(-w / 2, "aci", VERM, "ACI"),
                                      (w / 2, "rc_adaptive", BLUE, "RC (ours)")]:
        d = [100 * ((1 - a) - float(
                e12[(e12["alpha"] == a) & (e12["method"] == method)]
                ["cov_stress"].iloc[0]))
             for a in alphas]
        ax.bar(x + off, d, w, color=color, label=label, zorder=3)
        for xi, v in zip(x + off, d):
            ax.text(xi, v + 0.1, f"{v:.1f}", ha="center", fontsize=6.7,
                    color="#333")
    ax.set_xticks(x, [f"{100 * (1 - a):.0f}% nominal" for a in alphas])
    ax.set_ylim(0, 8.2)
    ax.set_ylabel("Stress coverage deficit (pp)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncols=2, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_alpha_sweep.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_coverage_by_regime()
    fig_dse_profile()
    fig_decomposition()
    fig_episodes()
    fig_var_breach()
    fig_tsfm_repair()
    fig_alpha_sweep()
    print(f"figures -> {OUT}")
    for f in sorted(OUT.glob("*.pdf")):
        print(" ", f.name)


if __name__ == "__main__":
    main()
