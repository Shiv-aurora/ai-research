"""E9: named crisis episodes — where the method has to earn its keep.

Coverage of per-stock ACI vs pooled regime-conditional adaptive (ours)
inside well-known stress windows. Complements the regime-sliced tables: a
regime bin aggregates many episodes; referees (and practitioners) want to
see 2020-03 by itself. Eval sample starts 2010 (walk-forward burn-in), so
the GFC is not available.

Usage: .venv/bin/python scripts/e9_stress_windows.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal.aci import run_aci_panel
from src.conformal.panel_hierarchical import run_panel_mondrian
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeding import seed_everything

CUTS = [0.5, 0.8, 0.95]
EPISODES = {
    "2011 US downgrade":   ("2011-08-01", "2011-10-31"),
    "2015 CNY deval":      ("2015-08-15", "2015-09-30"),
    "2018 Volmageddon":    ("2018-02-01", "2018-02-28"),
    "2018 Q4 selloff":     ("2018-12-01", "2018-12-31"),
    "2020 COVID":          ("2020-02-15", "2020-04-30"),
    "2022 bear market":    ("2022-01-01", "2022-10-31"),
    "2024 yen unwind":     ("2024-08-01", "2024-08-31"),
    "2025 tariff shock":   ("2025-04-01", "2025-04-30"),
}


def aligned_bins(market: pd.DataFrame) -> pd.DataFrame:
    v = market["vix_pctl"].values
    k = np.searchsorted(CUTS, v, side="right")
    pi = np.zeros((len(v), 4))
    ok = ~np.isnan(v)
    pi[ok, k[ok]] = 1.0
    pi[~ok] = 0.25
    return pd.DataFrame(pi, index=market.index,
                        columns=[f"regime_{j}" for j in range(4)])


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    proc = PROJECT_ROOT / cfg["data"]["processed_path"]
    panel = pd.read_parquet(proc / "rv_panel.parquet")
    preds = pd.read_parquet(proc / "e0_predictions.parquet")
    market = panel.groupby("date")[["vix_pctl"]].first().sort_index()
    member = aligned_bins(market)

    aci = run_aci_panel(preds.dropna(subset=["pool"]), "pool", alpha=0.10,
                        eta=0.05, warmup=100)
    rc = run_panel_mondrian(preds, member, "pool", alpha=0.10,
                            adaptive=True, warmup_days=100)

    rows = []
    for name, (a, b) in EPISODES.items():
        row = {"episode": name, "window": f"{a}..{b}"}
        for m, res in [("aci", aci), ("rc_adaptive", rc)]:
            d = res[(~res.warmup) & res.date.between(a, b)]
            row[f"{m}_cov"] = d["covered"].mean()
            row[f"{m}_cov_up"] = d["covered_hi"].mean()
            row["n"] = len(d)
        rows.append(row)
        print(f"{name:<20} n={row['n']:>6,}  "
              f"aci {row['aci_cov']:.3f}/{row['aci_cov_up']:.3f}  "
              f"ours {row['rc_adaptive_cov']:.3f}/{row['rc_adaptive_cov_up']:.3f}"
              f"   (cov/upper, target .90/.95)")

    out = pd.DataFrame(rows).set_index("episode")
    out.to_csv(PROJECT_ROOT / "reports" / "e9_stress_windows.csv")
    print(f"\nsaved -> reports/e9_stress_windows.csv")


if __name__ == "__main__":
    main()
