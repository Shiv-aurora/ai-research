"""Zero-shot Chronos-Bolt quantile forecasts for the whole panel.

For every stock-day (i, t) with enough history, feed the trailing 512 days
of log_rv ENDING AT t and predict day t+1, extracting quantiles
[.05, .25, .5, .75, .95]. Zero-shot: no fitting, so the only leakage surface
is context alignment (context ends at t; verified by construction below).

TORCH PROCESS: runs alone (never with lightgbm — see scripts/run_tests.sh).
Output: data/processed_v2/tsfm_predictions.parquet

Usage: .venv/bin/python scripts/tsfm_predict.py [--model amazon/chronos-bolt-small]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import PROJECT_ROOT, load_config

CONTEXT = 512
# Chronos-Bolt only supports quantile levels in [0.1, 0.9] (it clamps
# anything beyond — a paper-worthy limitation in itself). Native 80% band.
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="amazon/chronos-bolt-small")
    parser.add_argument("--start", default=None,
                        help="only predict rows with date >= start")
    args = parser.parse_args()

    from chronos import BaseChronosPipeline

    cfg = load_config()
    panel = pd.read_parquet(
        PROJECT_ROOT / cfg["data"]["processed_path"] / "rv_panel.parquet")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    eval_start = pd.Timestamp(args.start or cfg["evaluation"]["online_eval_start"])

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = BaseChronosPipeline.from_pretrained(args.model, device_map=device,
                                               torch_dtype=torch.float32)
    print(f"model={args.model} device={device}")

    # Build per-ticker arrays once
    series = {t: g["log_rv"].values for t, g in panel.groupby("ticker")}
    dates_by_ticker = {t: g["date"].values for t, g in panel.groupby("ticker")}
    pos_by_ticker = {t: pd.Series(np.arange(len(d)), index=d)
                     for t, d in dates_by_ticker.items()}

    all_dates = np.sort(panel.loc[panel["date"] >= eval_start, "date"].unique())
    rows = []
    for di, d in enumerate(all_dates):
        contexts, meta = [], []
        for t, pos in pos_by_ticker.items():
            if d not in pos.index:
                continue
            p = int(pos.loc[d])
            if p + 1 < CONTEXT:  # not enough history
                continue
            ctx = series[t][p + 1 - CONTEXT: p + 1]   # ends AT t inclusive
            contexts.append(torch.tensor(ctx, dtype=torch.float32))
            meta.append(t)
        if not contexts:
            continue
        with torch.no_grad():
            q, _ = pipe.predict_quantiles(
                contexts, prediction_length=1,
                quantile_levels=QUANTILES)
        q = q[:, 0, :].cpu().numpy()   # (n_series, n_quantiles)
        for j, t in enumerate(meta):
            rows.append((t, d, *q[j]))
        if (di + 1) % 200 == 0:
            print(f"  {di + 1}/{len(all_dates)} dates", flush=True)

    out = pd.DataFrame(rows, columns=["ticker", "date",
                                      *[f"q{int(x*100):02d}" for x in QUANTILES]])
    path = PROJECT_ROOT / cfg["data"]["processed_path"] / "tsfm_predictions.parquet"
    out.to_parquet(path, index=False)
    print(f"saved {len(out):,} rows -> {path}")


if __name__ == "__main__":
    main()
