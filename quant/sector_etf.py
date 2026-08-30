"""Top-down sector gate: rank GICS ETFs vs SPY, then keep names only
from the leaders. No hardcoded stock lists. No fabricated Fed/CPI.
Unclassified names skip the gate (we cannot pretend we know the sector).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Consumer Staples": "XLP",
}

YAHOO_TO_GICS = {
    "Technology": "Technology",
    "Healthcare": "Healthcare",
    "Financial Services": "Financials",
    "Financials": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Discretionary": "Consumer Discretionary",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Basic Materials": "Materials",
    "Materials": "Materials",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    "Communication Services": "Communication Services",
    "Consumer Defensive": "Consumer Staples",
    "Consumer Staples": "Consumer Staples",
}

RATE_SENSITIVITY = {
    "Financials": 1.0, "Energy": 0.5, "Consumer Staples": 0.2,
    "Healthcare": 0.0, "Industrials": 0.0, "Materials": 0.0,
    "Communication Services": -0.3, "Technology": -0.5,
    "Utilities": -0.8, "Consumer Discretionary": -1.0, "Real Estate": -1.0,
}


def canon_sector(raw: str | None) -> str:
    s = (raw or "").strip()
    if not s or s == "Unclassified":
        return "Unclassified"
    return YAHOO_TO_GICS.get(s, s)


def _sma(c: pd.Series, n: int) -> float:
    if len(c) < n:
        return float(c.mean()) if len(c) else 0.0
    return float(c.iloc[-n:].mean())


def etf_momentum(df: pd.DataFrame) -> float:
    if df is None or len(df) < 20:
        return 50.0
    c = df["Close"]
    price = float(c.iloc[-1])
    s20, s50 = _sma(c, 20), _sma(c, 50)
    s200 = _sma(c, 200) if len(c) >= 200 else s50
    score = 50.0
    if price > s200: score += 15
    if price > s50: score += 10
    if price > s20: score += 10
    if s20 > s50: score += 10
    if s50 > s200: score += 5
    ret20 = (price / float(c.iloc[-20]) - 1) * 100
    score += float(np.clip(ret20 * 2, -15, 15))
    return float(np.clip(score, 0, 100))


def vs_spy(sector_df: pd.DataFrame, spy_df: pd.DataFrame,
           window: int = 20) -> float:
    if (sector_df is None or spy_df is None
            or len(sector_df) < window or len(spy_df) < window):
        return 0.0
    sec = float(sector_df["Close"].iloc[-1] / sector_df["Close"].iloc[-window] - 1)
    spy = float(spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[-window] - 1)
    return round((sec - spy) * 100, 2)


def macro_adj(fed_rate: float | None, cpi: float | None) -> dict[str, float]:
    """Only when real numbers exist. Missing → empty (no 3.75 default)."""
    if fed_rate is None and cpi is None:
        return {}
    rate = float(fed_rate) if fed_rate is not None else None
    inf = float(cpi) if cpi is not None else None
    rate_f = float(np.clip((rate - 3.0) / 4.0, -1, 1)) if rate is not None else 0.0
    inf_f = float(np.clip((inf - 3.0) / 4.0, -1, 1)) if inf is not None else 0.0
    out = {}
    for sec, sens in RATE_SENSITIVITY.items():
        adj = sens * rate_f * 8
        if sens < 0:
            adj -= inf_f * 3
        out[sec] = round(float(np.clip(adj, -10, 10)), 1)
    return out


def breadth_from_universe(data: dict[str, pd.DataFrame],
                          sectors: dict[str, str],
                          window: int = 50) -> dict[str, float]:
    """% of scanned names in each GICS above their SMA — live universe, not a list."""
    tallies: dict[str, list[int]] = {}
    for tkr, df in data.items():
        g = canon_sector(sectors.get(tkr))
        if g == "Unclassified" or df is None or len(df) < window:
            continue
        above = float(df["Close"].iloc[-1]) > _sma(df["Close"], window)
        tallies.setdefault(g, []).append(1 if above else 0)
    return {g: round(sum(v) / len(v) * 100, 1) for g, v in tallies.items() if v}


def rank_etfs(etf_data: dict[str, pd.DataFrame], spy_df: pd.DataFrame,
              breadth: dict[str, float] | None = None,
              fed_rate: float | None = None,
              cpi: float | None = None) -> list[dict]:
    breadth = breadth or {}
    mac = macro_adj(fed_rate, cpi)
    rows = []
    for sector, etf in SECTOR_ETFS.items():
        df = etf_data.get(sector)
        if df is None:
            df = etf_data.get(etf)
        if df is None or len(df) < 20:
            continue
        mom = etf_momentum(df)
        rs = vs_spy(df, spy_df) if spy_df is not None and len(spy_df) >= 20 else 0.0
        brd = breadth.get(sector, 50.0)
        mac_pts = mac.get(sector, 0.0)
        composite = (mom * 0.45
                     + float(np.clip(rs * 3 + 50, 0, 100)) * 0.25
                     + brd * 0.20
                     + float(np.clip(mac_pts * 5 + 50, 0, 100)) * 0.10)
        rows.append({"sector": sector, "etf": etf, "momentum": round(mom, 1),
                     "rel_strength": rs, "breadth_%": brd, "macro_adj": mac_pts,
                     "composite": round(composite, 1)})
    rows.sort(key=lambda x: -x["composite"])
    return rows


def apply_etf_gate(names: list[dict], etf_ranked: list[dict],
                   top_n: int = 3) -> tuple[list[dict], list[dict], list[str]]:
    """Keep classified names in the top N ETF sectors. Unclassified stay."""
    if not etf_ranked or not names:
        return names, [], []
    classified = [n for n in names
                  if canon_sector(n.get("sector")) != "Unclassified"]
    if len(classified) < 5:
        return names, [], [r["sector"] for r in etf_ranked[:top_n]]
    allowed = [r["sector"] for r in etf_ranked[:top_n]]
    allow = set(allowed)
    kept, side = [], []
    for n in names:
        g = canon_sector(n.get("sector"))
        if g == "Unclassified" or g in allow:
            kept.append(n)
        else:
            n2 = dict(n)
            n2["sideline_reason"] = f"{g} not in top {top_n} ETF sectors today"
            side.append(n2)
    return kept, side, allowed
