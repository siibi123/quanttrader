"""Free cross-asset tape from Yahoo — VIX, 10y yield, SPY.

No paid feed. Cached 5 minutes. Used as a book-level risk-off switch:
new entries shrink or stand down when vol is rising and yields jumped.
Exits are never blocked.
"""
from __future__ import annotations

import os
import time

import pandas as pd

_CACHE: dict = {"ts": 0.0, "data": None}
_TTL = 300


def _series_close(raw: pd.DataFrame, ticker: str) -> pd.Series:
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        if "Close" in lvl0:
            s = raw["Close"][ticker] if ticker in raw["Close"].columns else None
        else:
            s = raw[ticker]["Close"] if ticker in raw.columns.get_level_values(0) else None
        return s.dropna() if s is not None else pd.Series(dtype=float)
    return raw["Close"].dropna() if "Close" in raw.columns else pd.Series(dtype=float)


def read_tape(force: bool = False) -> dict:
    now = time.time()
    if os.environ.get("QT_OFFLINE"):
        return {
            "vix": None, "vix_5d": None, "tnx": None, "tnx_5d_bp": None,
            "spy_5d_pct": None, "risk_off": False, "size_mult": 1.0,
            "why": "offline",
        }
    if not force and _CACHE["data"] is not None and now - _CACHE["ts"] < _TTL:
        return _CACHE["data"]
    out = {
        "vix": None, "vix_5d": None, "tnx": None, "tnx_5d_bp": None,
        "spy_5d_pct": None, "risk_off": False, "size_mult": 1.0,
        "why": "tape unavailable",
    }
    try:
        import yfinance as yf
        raw = yf.download(
            ["^VIX", "^TNX", "SPY"], period="4mo", interval="1d",
            progress=False, auto_adjust=True, threads=False,
        )
    except Exception as exc:
        out["why"] = f"yahoo failed: {str(exc)[:80]}"
        _CACHE.update(ts=now, data=out)
        return out

    vix = _series_close(raw, "^VIX")
    tnx = _series_close(raw, "^TNX")
    spy = _series_close(raw, "SPY")
    if len(vix):
        out["vix"] = round(float(vix.iloc[-1]), 2)
        if len(vix) > 5:
            out["vix_5d"] = round(float(vix.iloc[-1] - vix.iloc[-6]), 2)
    if len(tnx):
        out["tnx"] = round(float(tnx.iloc[-1]), 3)
        if len(tnx) > 5:
            out["tnx_5d_bp"] = round(float((tnx.iloc[-1] - tnx.iloc[-6]) * 100), 1)
    if len(spy) > 5:
        out["spy_5d_pct"] = round(
            float(spy.iloc[-1] / spy.iloc[-6] - 1) * 100, 2)

    vol_up = (out["vix"] or 0) >= 22 and (out["vix_5d"] or 0) > 0
    rates_up = (out["tnx_5d_bp"] or 0) >= 20
    spy_down = (out["spy_5d_pct"] or 0) <= -2.5
    if vol_up and (rates_up or spy_down):
        out["risk_off"] = True
        out["size_mult"] = 0.0
        out["why"] = (f"RISK-OFF · VIX {out['vix']} ({out['vix_5d']:+} 5d) "
                      f"· TNX {out['tnx_5d_bp']}bp · SPY {out['spy_5d_pct']}%")
    elif vol_up or rates_up:
        out["risk_off"] = False
        out["size_mult"] = 0.5
        out["why"] = (f"CAUTION · VIX {out['vix']} · TNX {out['tnx_5d_bp']}bp "
                      f"· half-size new entries")
    else:
        out["why"] = (f"CLEAR · VIX {out['vix']} · TNX 5d {out['tnx_5d_bp']}bp "
                      f"· SPY 5d {out['spy_5d_pct']}%")
    _CACHE.update(ts=now, data=out)
    return out
