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


def fig_coverage_by_regime():
    """Figure 1: the phenomenon and the repair, by regime (E2)."""
    e2 = pd.read_csv(REP / "e2_full_summary.csv").set_index("method")
    cols = [f"cov_{r}" for r in REGIMES]
    aci = e2.loc["aci", cols].values.astype(float)
    ours = e2.loc["rc_adaptive", cols].values.astype(float)

    x = np.arange(len(REGIMES))
    w = 0.34
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.bar(x - w / 2, aci, w, color=VERM, label="ACI (per stock)", zorder=3)
    ax.bar(x + w / 2, ours, w, color=BLUE, label="RC (ours)", zorder=3)
    ax.axhline(0.90, ls="--", lw=0.9, color=GRAY, zorder=2)
    ax.text(3.42, 0.902, "nominal 90%", color=GRAY, fontsize=7,
            ha="right", va="bottom")
    for xi, v in zip(x - w / 2, aci):
        ax.text(xi, v + 0.004, f"{100*v:.1f}", ha="center", fontsize=6.7,
                color="#333")
    for xi, v in zip(x + w / 2, ours):
        ax.text(xi, v + 0.004, f"{100*v:.1f}", ha="center", fontsize=6.7,
                color="#333")
    ax.set_xticks(x, [r.capitalize() for r in REGIMES])
    ax.set_ylim(0.80, 0.925)
    ax.set_ylabel("Empirical coverage")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_coverage_by_regime.pdf")
    plt.close(fig)


def fig_dse_profile():
    """Days-since-stress-entry: the day-2 pit survives oracle regimes (E6b)."""
    dse = pd.read_csv(REP / "e6b_oracle_dse.csv", index_col=0)
    x = np.arange(len(dse.index))
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.axhline(0.90, ls="--", lw=0.9, color=GRAY)
    ax.plot(x, dse["bins_k4"], "-o", color=BLUE, ms=3.5, lw=1.6,
            label="VIX bins (causal)")
    ax.plot(x, dse["hmm_filtered"], "-s", color=VERM, ms=3.2, lw=1.4,
            label="HMM filtered (causal)")
    ax.plot(x, dse["hmm_oracle"], "-^", color=GREEN, ms=3.5, lw=1.4,
            label="HMM smoothed (oracle)")
    ax.annotate("day-2 pit", xy=(1, float(dse['bins_k4'].iloc[1]) - 0.006),
                xytext=(0.05, 0.732), fontsize=7, color="#333",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#666"))
    ax.set_xticks(x, dse.index)
    ax.set_xlabel("Days since entering the stress regime")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.70, 0.95)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_dse_oracle.pdf")
    plt.close(fig)


def fig_decomposition():
    """Mechanism decomposition: what carries stress coverage (E6, E6b, E2)."""
    e6 = pd.read_csv(REP / "e6_ablations.csv").set_index("config")
    e6b = pd.read_csv(REP / "e6b_oracle_regimes.csv").set_index("membership")
    e2 = pd.read_csv(REP / "e2_full_summary.csv").set_index("method")
    rows = [
        ("Per-stock ACI", float(e2.loc["aci", "cov_stress"]), VERM),
        ("Per-stock RC (no pooling)", float(e6.loc["per_stock", "cov_stress"]), VERM),
        ("Pooled, K=1 (no regimes)", float(e6.loc["K=1", "cov_stress"]), BLUE),
        ("Pooled, K=4 (ours)", float(e6.loc["K=4", "cov_stress"]), BLUE),
        ("Pooled, K=4, oracle regimes", float(e6b.loc["hmm_oracle", "cov_stress"]), GREEN),
    ]
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    colors = [r[2] for r in rows][::-1]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.barh(y, vals, 0.62, color=colors, zorder=3)
    ax.axvline(0.90, ls="--", lw=0.9, color=GRAY, zorder=4)
    for yi, v in zip(y, vals):
        ax.text(v - 0.002, yi, f"{100*v:.1f}", ha="right", va="center",
                fontsize=6.8, color="white", zorder=5)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.78, 0.915)
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_coverage_by_regime()
    fig_dse_profile()
    fig_decomposition()
    fig_episodes()
    print(f"figures -> {OUT}")
    for f in sorted(OUT.glob("*.pdf")):
        print(" ", f.name)


if __name__ == "__main__":
    main()
