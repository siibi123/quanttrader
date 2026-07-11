"""B-Xtrender — institutional-grade port & upgrade.

Original: Bharat Jhunjhunwala (IFTA Journal), Pine port by @Puppytherapy.
  short osc = RSI( EMA(close,5) - EMA(close,20), 15 ) - 50
  long  osc = RSI( EMA(close,20), 15 ) - 50
  signal    = Tillson T3(short osc, 5, b=0.7)

Upgrades here:
  * exact Wilder RSI (TradingView-faithful)
  * turn signals on the T3 line (local trough/peak flips)
  * regular divergence detection (price vs oscillator pivots)
  * multi-timeframe confluence (weekly long-term oscillator)
  * event study — real forward returns after each historical signal
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Building blocks (TradingView-faithful)
# ---------------------------------------------------------------------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi_wilder(s: pd.Series, n: int) -> pd.Series:
    """TradingView rsi(): Wilder RMA smoothing of gains/losses."""
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def t3(s: pd.Series, n: int, b: float = 0.7) -> pd.Series:
    """Tillson T3 — six cascaded EMAs with volume-factor coefficients."""
    e1 = ema(s, n); e2 = ema(e1, n); e3 = ema(e2, n)
    e4 = ema(e3, n); e5 = ema(e4, n); e6 = ema(e5, n)
    c1 = -b ** 3
    c2 = 3 * b ** 2 + 3 * b ** 3
    c3 = -6 * b ** 2 - 3 * b - 3 * b ** 3
    c4 = 1 + 3 * b + b ** 3 + 3 * b ** 2
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


# ---------------------------------------------------------------------------
# Core oscillators
# ---------------------------------------------------------------------------

def bxtrender(df: pd.DataFrame, s1: int = 5, s2: int = 20, s3: int = 15,
              l1: int = 20, l2: int = 15) -> pd.DataFrame:
    """Return short/long oscillators, T3 signal line and turn markers."""
    close = df["Close"]
    short_osc = rsi_wilder(ema(close, s1) - ema(close, s2), s3) - 50
    long_osc = rsi_wilder(ema(close, l1), l2) - 50
    sig = t3(short_osc, 5)

    up_turn = (sig > sig.shift(1)) & (sig.shift(1) < sig.shift(2))
    dn_turn = (sig < sig.shift(1)) & (sig.shift(1) > sig.shift(2))

    return pd.DataFrame({
        "short_osc": short_osc,
        "long_osc": long_osc,
        "t3": sig,
        "t3_rising": sig > sig.shift(1),
        "buy_turn": up_turn,
        "sell_turn": dn_turn,
    }, index=df.index)


def score_bx(df: pd.DataFrame) -> pd.Series:
    """Sub-score in [-1, +1] for the composite model.

    Direction from the long oscillator, timing from the short osc + T3 slope.
    """
    bx = bxtrender(df)
    long_dir = np.tanh(bx["long_osc"] / 25.0)
    short_dir = np.tanh(bx["short_osc"] / 25.0)
    slope = np.where(bx["t3_rising"], 0.3, -0.3)
    raw = 0.45 * long_dir + 0.35 * short_dir + slope
    return pd.Series(raw, index=df.index).clip(-1, 1).fillna(0)


# ---------------------------------------------------------------------------
# Institutional upgrades
# ---------------------------------------------------------------------------

def _pivots(s: pd.Series, k: int = 3) -> tuple[list[int], list[int]]:
    """Indices of confirmed local highs and lows (k bars each side)."""
    v = s.values
    highs, lows = [], []
    for i in range(k, len(v) - k):
        win = v[i - k:i + k + 1]
        if v[i] == win.max() and (win.argmax() == k):
            highs.append(i)
        if v[i] == win.min() and (win.argmin() == k):
            lows.append(i)
    return highs, lows


def detect_divergence(df: pd.DataFrame, lookback: int = 120) -> dict:
    """Regular divergences between price and the short oscillator.

    Bearish: price higher high, oscillator lower high.
    Bullish: price lower low, oscillator higher low.
    """
    bx = bxtrender(df)
    price = df["Close"].iloc[-lookback:]
    osc = bx["short_osc"].iloc[-lookback:]

    p_hi, p_lo = _pivots(price, k=3)
    out = {"bearish": False, "bullish": False, "detail": ""}

    if len(p_hi) >= 2:
        i1, i2 = p_hi[-2], p_hi[-1]
        if (price.iloc[i2] > price.iloc[i1]
                and osc.iloc[i2] < osc.iloc[i1] and osc.iloc[i1] > 0):
            out["bearish"] = True
            out["detail"] = (f"Price HH {price.iloc[i1]:.2f}→{price.iloc[i2]:.2f} "
                             f"but oscillator LH {osc.iloc[i1]:.1f}→{osc.iloc[i2]:.1f}")
    if len(p_lo) >= 2:
        i1, i2 = p_lo[-2], p_lo[-1]
        if (price.iloc[i2] < price.iloc[i1]
                and osc.iloc[i2] > osc.iloc[i1] and osc.iloc[i1] < 0):
            out["bullish"] = True
            out["detail"] = (f"Price LL {price.iloc[i1]:.2f}→{price.iloc[i2]:.2f} "
                             f"but oscillator HL {osc.iloc[i1]:.1f}→{osc.iloc[i2]:.1f}")
    return out


def weekly_alignment(df: pd.DataFrame) -> dict:
    """Compute the long-term oscillator on WEEKLY bars — the boss timeframe."""
    w = df.resample("W-FRI").agg({"Open": "first", "High": "max",
                                  "Low": "min", "Close": "last",
                                  "Volume": "sum"}).dropna()
    if len(w) < 40:
        return {"weekly_osc": None, "weekly_rising": None}
    bx_w = bxtrender(w)
    osc = float(bx_w["long_osc"].iloc[-1])
    rising = bool(bx_w["long_osc"].iloc[-1] > bx_w["long_osc"].iloc[-2])
    return {"weekly_osc": round(osc, 1), "weekly_rising": rising}


def event_study(df: pd.DataFrame, horizons=(5, 10, 20)) -> pd.DataFrame:
    """What ACTUALLY happened after every historical turn signal on this ticker."""
    bx = bxtrender(df)
    close = df["Close"]
    rows = []
    for name, mask in (("Buy turns", bx["buy_turn"]),
                       ("Sell turns", bx["sell_turn"])):
        idx = np.where(mask.values)[0]
        idx = idx[idx > 50]                       # skip warm-up
        row = {"signal": name, "count": len(idx)}
        for hzn in horizons:
            valid = idx[idx + hzn < len(close)]
            if len(valid) == 0:
                row[f"avg {hzn}d %"] = None
                row[f"win {hzn}d %"] = None
                continue
            fwd = close.values[valid + hzn] / close.values[valid] - 1.0
            row[f"avg {hzn}d %"] = round(float(fwd.mean()) * 100, 2)
            row[f"win {hzn}d %"] = round(float((fwd > 0).mean()) * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows)
