"""Build the aligned RV panel from the configured source.

Usage:
    .venv/bin/python scripts/build_rv_panel.py [--refresh]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.panel import build_panel
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="re-download all upstream data, ignoring caches")
    args = parser.parse_args()

    cfg = load_config()
    panel = build_panel(cfg, refresh=args.refresh)

    # Completeness report
    per_ticker = panel.groupby("ticker").agg(
        first=("date", "min"), last=("date", "max"), days=("date", "size")
    )
    short = per_ticker[per_ticker["days"] < 2500]
    print(f"\ntickers: {len(per_ticker)}, median days: {per_ticker['days'].median():.0f}")
    if not short.empty:
        print(f"short histories (<2500 days):\n{short}")
    na_share = panel.isna().mean()
    print("\nNaN share (columns > 0):")
    print(na_share[na_share > 0].round(4).to_string() or "  none")


if __name__ == "__main__":
    main()
