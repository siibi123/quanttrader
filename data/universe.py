"""Trading universe (P9) — S&P 500 + Nasdaq-100 constituents, fetched once
from their public Wikipedia tables and cached to disk. This is what
PollingFeed scans for quotes; it is deliberately NOT what the decision
cycle trades (that stays a small, explicit watchlist — see app.py's
TRADING_WATCHLIST) — screening 550 names for tradeable setups every 5
minutes is a real, separate feature the owner hasn't asked for yet.
"""
from __future__ import annotations

import io
import json
import os
import time

import pandas as pd
import requests

CACHE_PATH = os.path.join("runtime", "universe.json")
CACHE_TTL_S = 7 * 24 * 3600          # index constituents change rarely
UA = "quanttrader (+https://github.com/siibi123/quanttrader)"

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NDX100_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"

# Honest degraded fallback ONLY: used if there's no cache AND the live
# Wikipedia fetch fails (e.g. first-ever run with no network) — never
# presented as the real ~550-symbol universe, just enough that the feed
# isn't scanning nothing.
FALLBACK_UNIVERSE = sorted([
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    "JPM", "V", "MA", "UNH", "HD", "PG", "COST", "NFLX", "AMD", "ADBE",
    "PEP", "CSCO", "SPY", "QQQ", "XOM", "JNJ", "BAC", "KO", "WMT",
    "DIS", "CRM", "INTC",
])


def _fetch_table(url: str, ticker_col_hints: tuple[str, ...]) -> list[str]:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    for t in tables:
        cols = [str(c) for c in t.columns]
        hit = next((c for c in cols
                   if any(h.lower() in c.lower() for h in ticker_col_hints)),
                  None)
        if hit and len(t) > 50:          # skip small unrelated tables
            return [str(s).strip().upper().replace(".", "-")
                   for s in t[hit].dropna()]
    return []


def _fetch_live() -> list[str]:
    sp500 = _fetch_table(SP500_URL, ("Symbol", "Ticker"))
    ndx100 = _fetch_table(NDX100_URL, ("Ticker", "Symbol"))
    symbols = sorted(set(sp500) | set(ndx100))
    if len(symbols) < 400:               # sanity floor -- wrong table shape
        raise ValueError(f"fetched universe suspiciously small "
                         f"({len(symbols)} symbols)")
    return symbols


def load_universe(force_refresh: bool = False) -> list[str]:
    """Cached union of S&P 500 + Nasdaq-100 tickers (~550, deduped).
    Refreshes at most once every CACHE_TTL_S; on any fetch failure falls
    back to the last good cache (even if stale) rather than leaving the
    feed with nothing, and only reaches for FALLBACK_UNIVERSE if there's
    no cache at all yet."""
    cached = None
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
        except Exception:
            cached = None
        if (not force_refresh and cached
                and time.time() - cached.get("fetched_at", 0) < CACHE_TTL_S):
            return cached["symbols"]

    try:
        symbols = _fetch_live()
    except Exception:
        if cached and cached.get("symbols"):
            return cached["symbols"]
        return FALLBACK_UNIVERSE

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump({"symbols": symbols, "fetched_at": time.time()}, fh)
    return symbols
