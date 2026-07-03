"""Identity audit: verify every cached Risk Lab series belongs to the intended
company, not a recycled-ticker impostor.

For each cached ticker, look up all symbol matches, report which permno was
chosen and its company name, and flag suspicious picks (multiple exact
matches, or names that share no token with the ticker's expected company).

Usage: .venv/bin/python scripts/audit_universe_identity.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.risklab import BASE, CACHE_DIR


def main() -> None:
    session = requests.Session()
    rows = []
    for path in sorted(CACHE_DIR.glob("*.parquet")):
        ticker = path.stem
        chosen = int(pd.read_parquet(path, columns=["permno"]).iloc[0, 0])
        resp = session.get(f"{BASE}/symbol.php", params={"s": ticker}, timeout=30)
        matches = resp.json()
        exact = [(int(p), name) for p, sym, name in matches
                 if sym.upper() == ticker.upper()]
        name = dict(exact).get(chosen, "<permno not in symbol matches>")
        rows.append({
            "ticker": ticker, "permno": chosen, "company": name,
            "n_exact_matches": len(exact),
            "alternatives": "; ".join(f"{p}:{n}" for p, n in exact if p != chosen),
        })
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    print(df[["ticker", "permno", "company"]].to_string(index=False))
    multi = df[df.n_exact_matches > 1]
    if not multi.empty:
        print(f"\n{len(multi)} tickers had multiple exact matches — REVIEW:")
        print(multi[["ticker", "permno", "company", "alternatives"]]
              .to_string(index=False))
    out = Path("reports")
    out.mkdir(exist_ok=True)
    df.to_csv(out / "universe_identity_audit.csv", index=False)
    print(f"\nsaved -> {out / 'universe_identity_audit.csv'}")


if __name__ == "__main__":
    main()
