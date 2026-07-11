"""Build the point-in-time top-100 universe from CRSP via the WRDS REST API.

For each formation year Y in 2005..2025, take the last December month-end of
year Y-1, rank US common stocks (shrcd 10/11) on NYSE/AMEX/NASDAQ (exchcd
1/2/3) by market cap aggregated at the company (permco) level, and keep the
top `size` companies. Each company is represented by its largest-cap share
class (permno). Companies that later delist stay in the list for the years
they qualified — this is the survivorship-bias-free universe.

Outputs (data/raw/wrds/):
  universe_pit_membership.parquet  one row per (year, company): permno, permco,
                                   ticker/comnam at formation, cap, rank
  universe_pit_delist.parquet      msedelist rows for every permno ever selected
  universe_pit_summary.csv         per-year turnover / sanity report

Requires WRDS_API_TOKEN in the environment. Queries are logged verbatim to
docs/wrds_queries.md.

Usage:
    .venv/bin/python scripts/build_pit_universe.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.wrds_api import get_table
from src.utils.config import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "data" / "raw" / "wrds"
FIRST_YEAR, LAST_YEAR = 2005, 2025
SIZE = 100


def load_msenames(session: requests.Session) -> pd.DataFrame:
    """Full crsp.msenames, downloaded once and cached (the API silently
    ignores filters on shrcd/nameendt — all filtering happens locally)."""
    cache = OUT_DIR / "msenames_full.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    names = get_table(
        "crsp.msenames", None,
        log_purpose="PIT universe: full msenames history (filtered locally; "
                    "API ignores shrcd/nameendt filters)",
        output_note="data/raw/wrds/msenames_full.parquet",
        session=session, verbose=True,
    )
    names.to_parquet(cache, index=False)
    return names


def formation_snapshot(year: int, names_all: pd.DataFrame,
                       session: requests.Session) -> pd.DataFrame:
    """Top-SIZE companies by market cap at the December month-end before `year`."""
    dec = f"{year - 1}-12-01"
    dec_end = f"{year - 1}-12-31"
    cache = OUT_DIR / f"msf_{year - 1}12.parquet"
    if cache.exists():
        msf = pd.read_parquet(cache)
    else:
        msf = get_table(
            "crsp.msf",
            {"date__gte": dec, "date__lte": dec_end},
            log_purpose=f"PIT universe formation {year}: December {year-1} "
                        f"month-end prices/shares",
            output_note="data/raw/wrds/universe_pit_membership.parquet",
            session=session,
        )
        msf.to_parquet(cache, index=False)
    # msf has one row per permno for the month; price can be negative
    # (bid/ask midpoint convention) or null (use altprc). shrout is thousands.
    msf = msf[["permno", "permco", "date", "prc", "altprc", "shrout"]].copy()
    for c in ("prc", "altprc", "shrout"):
        msf[c] = pd.to_numeric(msf[c], errors="coerce")
    price = msf["prc"].abs().fillna(msf["altprc"].abs())
    msf["cap_musd"] = price * msf["shrout"] / 1000.0
    msf = msf.dropna(subset=["cap_musd"])
    msf = msf[msf["cap_musd"] > 0]

    # Point-in-time eligible name record: common stock (shrcd 10/11) on
    # NYSE/AMEX/NASDAQ (exchcd 1/2/3), name row in force on the formation date.
    n = names_all.copy()
    names = n[
        (n["namedt"].astype(str) <= dec_end)
        & (n["nameendt"].astype(str) >= dec_end)
        & (pd.to_numeric(n["shrcd"], errors="coerce").isin([10, 11]))
        & (pd.to_numeric(n["exchcd"], errors="coerce").isin([1, 2, 3]))
    ]
    names = names[["permno", "ticker", "comnam", "shrcd", "exchcd"]].drop_duplicates("permno")

    snap = msf.merge(names, on="permno", how="inner")

    # Aggregate share classes to the company level (permco); the company's
    # ticker/name come from its largest-cap class.
    snap = snap.sort_values("cap_musd", ascending=False)
    comp = snap.groupby("permco", as_index=False).agg(
        permno=("permno", "first"), ticker=("ticker", "first"),
        comnam=("comnam", "first"), cap_musd=("cap_musd", "sum"),
    )
    comp = comp.sort_values("cap_musd", ascending=False).head(SIZE).reset_index(drop=True)
    comp.insert(0, "year", year)
    comp["rank"] = np.arange(1, len(comp) + 1)
    return comp


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    names_all = load_msenames(session)

    years = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        snap = formation_snapshot(year, names_all, session)
        assert len(snap) == SIZE, f"{year}: only {len(snap)} companies ranked"
        years.append(snap)
        print(f"{year}: top {SIZE} spans ${snap.cap_musd.min()/1e3:.1f}B–"
              f"${snap.cap_musd.max()/1e3:.1f}B  (#1 {snap.iloc[0].ticker} "
              f"{snap.iloc[0].comnam.title()})", flush=True)

    membership = pd.concat(years, ignore_index=True)
    membership.to_parquet(OUT_DIR / "universe_pit_membership.parquet", index=False)

    # Delisting records for every company ever selected (documents that dead
    # names are retained, and gives the panel an authoritative end date).
    permnos = sorted(membership["permno"].unique())
    delist = get_table(
        "crsp.msedelist", {"permno__in": permnos},
        log_purpose="PIT universe: delisting records for all selected permnos",
        output_note="data/raw/wrds/universe_pit_delist.parquet",
        session=session,
    )
    if not delist.empty:
        delist = delist[["permno", "dlstdt", "dlstcd"]]
    delist.to_parquet(OUT_DIR / "universe_pit_delist.parquet", index=False)

    # Sanity/turnover report
    by_year = {y: set(g["permno"]) for y, g in membership.groupby("year")}
    rows = []
    for y in range(FIRST_YEAR + 1, LAST_YEAR + 1):
        rows.append({"year": y, "n_new": len(by_year[y] - by_year[y - 1])})
    turn = pd.DataFrame(rows)
    dead = delist[pd.to_numeric(delist["dlstcd"], errors="coerce") >= 200]
    summary = pd.DataFrame({
        "unique_companies": [membership["permno"].nunique()],
        "mean_annual_turnover": [turn["n_new"].mean()],
        "delisted_members": [dead["permno"].nunique()],
    })
    summary.to_csv(OUT_DIR / "universe_pit_summary.csv", index=False)
    print(f"\nunique companies across {FIRST_YEAR}-{LAST_YEAR}: "
          f"{membership['permno'].nunique()}")
    print(f"mean annual turnover: {turn['n_new'].mean():.1f} names")
    print(f"members with delisting events (dlstcd>=200): {dead['permno'].nunique()}")
    print(f"artifacts -> {OUT_DIR}")


if __name__ == "__main__":
    main()
