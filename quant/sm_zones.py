"""Discount / premium zones + market mode.

Mechanical translation of the public swing framework:
buy bullish names in the discount of the last dealing range; stand aside
in premium; trail with ATR rather than chasing. This is our own
implementation of those public rules — not a port of any paid course or
TradingView script.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.levels import fib_levels
from quant.signals import atr, rsi


def classify_zone(df: pd.DataFrame, lookback: int = 126) -> dict:
    """Where price sits in the last swing: BUY_ZONE / DISCOUNT / EQUILIBRIUM / PREMIUM.

    Up-swing buy zone = 0.618–0.786 retracement from the swing high
    (the deep-discount pocket). Deep discount below that still counts.
    """
    px = float(df["Close"].iloc[-1])
    empty = {
        "label": "UNKNOWN", "price": round(px, 2),
        "swing_high": None, "swing_low": None, "equilibrium": None,
        "ote_lo": None, "ote_hi": None, "pos": None,
        "in_discount": False, "in_premium": False, "in_buy_zone": False,
        "up_swing": None,
    }
    if len(df) < 30:
        return empty
    fib = fib_levels(df, lookback=min(lookback, len(df)))
    hi, lo = float(fib["swing_high"]), float(fib["swing_low"])
    rng = hi - lo
    if rng <= 1e-9:
        return empty
    pos = float(np.clip((px - lo) / rng, 0.0, 1.0))
    eq = (hi + lo) / 2.0
    ote_lo = hi - 0.786 * rng
    ote_hi = hi - 0.618 * rng
    if ote_lo > ote_hi:
        ote_lo, ote_hi = ote_hi, ote_lo
    in_discount = pos <= 0.50
    in_premium = pos >= 0.55
    in_buy_zone = bool(fib["up_swing"] and (ote_lo <= px <= ote_hi or pos <= 0.25))
    if in_buy_zone:
        label = "BUY_ZONE"
    elif in_discount:
        label = "DISCOUNT"
    elif in_premium:
        label = "PREMIUM"
    else:
        label = "EQUILIBRIUM"
    return {
        "label": label, "price": round(px, 2),
        "swing_high": round(hi, 2), "swing_low": round(lo, 2),
        "equilibrium": round(eq, 2),
        "ote_lo": round(ote_lo, 2), "ote_hi": round(ote_hi, 2),
        "pos": round(pos, 3),
        "in_discount": in_discount, "in_premium": in_premium,
        "in_buy_zone": in_buy_zone, "up_swing": bool(fib["up_swing"]),
    }


def market_mode(df: pd.DataFrame) -> dict:
    """EXPANSION / COMPRESSION / CAPITULATION from ATR regime + RSI2 panic."""
    if len(df) < 25:
        return {"mode": "UNKNOWN", "atr_pctile": None, "rsi2": None}
    a = atr(df).dropna()
    a_now = float(a.iloc[-1])
    a_ago = float(a.iloc[-20]) if len(a) > 20 else a_now
    window = a.iloc[-60:] if len(a) >= 60 else a
    pctile = float((window <= a_now).mean()) if len(window) else 0.5
    r2 = float(rsi(df["Close"], 2).iloc[-1])
    c = df["Close"]
    drop = (float(c.iloc[-1]) / float(c.iloc[-4]) - 1.0) if len(c) > 4 else 0.0
    if r2 < 10 or drop < -0.08:
        mode = "CAPITULATION"
    elif a_now > a_ago * 1.25 and pctile > 0.65:
        mode = "EXPANSION"
    elif pctile < 0.30:
        mode = "COMPRESSION"
    else:
        mode = "EXPANSION" if a_now >= a_ago else "COMPRESSION"
    return {"mode": mode, "atr_pctile": round(pctile, 2),
            "rsi2": round(r2, 1)}
