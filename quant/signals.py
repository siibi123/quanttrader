"""Signal engine — technical indicators combined into a composite quant score.

Every indicator produces a sub-score in [-1, +1].
The composite is a weighted average, mapped to BUY / HOLD / SELL.
All indicators use only past data (no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, 12) - ema(close, 26)
    signal = ema(line, 9)
    return line, signal, line - signal


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def bollinger_z(close: pd.Series, n: int = 20) -> pd.Series:
    m = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return (close - m) / sd.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Sub-scores (each in [-1, +1])
# ---------------------------------------------------------------------------

def score_trend(df: pd.DataFrame) -> pd.Series:
    """Trend: price vs SMA50/SMA200 + golden/death cross direction."""
    c = df["Close"]
    s50, s200 = sma(c, 50), sma(c, 200)
    above50 = np.sign(c - s50)
    above200 = np.sign(c - s200)
    cross = np.sign(s50 - s200)
    return pd.Series((above50 + above200 + cross) / 3.0, index=df.index)


def score_momentum(df: pd.DataFrame) -> pd.Series:
    """12-1 momentum (classic academic factor), squashed with tanh."""
    c = df["Close"]
    mom = c.shift(21) / c.shift(252) - 1.0  # skip last month, look at prior 11
    return np.tanh(mom * 3.0).fillna(0)


def score_rsi(df: pd.DataFrame) -> pd.Series:
    """RSI regime: >50 bullish, <50 bearish; extremes fade slightly."""
    r = rsi(df["Close"])
    base = (r - 50) / 50.0                      # -1..+1
    fade = np.where(r > 75, -0.3, np.where(r < 25, 0.3, 0.0))
    return (base + fade).clip(-1, 1)


def score_macd(df: pd.DataFrame) -> pd.Series:
    """MACD histogram sign, normalised by price."""
    _, _, hist = macd(df["Close"])
    norm = hist / df["Close"] * 100
    return np.tanh(norm * 2.0).fillna(0)


def score_meanrev(df: pd.DataFrame) -> pd.Series:
    """Mean reversion: fade extreme Bollinger z-scores."""
    z = bollinger_z(df["Close"])
    return (-z / 2.5).clip(-1, 1).fillna(0)


def score_volume(df: pd.DataFrame) -> pd.Series:
    """Volume confirmation: surge in direction of the daily move."""
    v_ratio = df["Volume"] / df["Volume"].rolling(20).mean()
    direction = np.sign(df["Close"].pct_change())
    conf = np.where(v_ratio > 1.5, direction * 0.8,
                    np.where(v_ratio > 1.0, direction * 0.3, 0.0))
    return pd.Series(conf, index=df.index).fillna(0)


def vol_regime(df: pd.DataFrame) -> pd.Series:
    """Volatility regime filter: 1 = calm (trade), 0.5 = elevated, 0.25 = storm."""
    a = atr(df) / df["Close"]
    pct = a.rolling(252, min_periods=60).rank(pct=True)
    return pd.Series(np.where(pct > 0.9, 0.25, np.where(pct > 0.7, 0.5, 1.0)),
                     index=df.index)


WEIGHTS = {
    "trend": 0.25,
    "momentum": 0.20,
    "bxtrender": 0.15,
    "macd": 0.125,
    "rsi": 0.10,
    "meanrev": 0.10,
    "volume": 0.075,
}

BUY_TH, SELL_TH = 0.25, -0.25


def composite(df: pd.DataFrame) -> pd.DataFrame:
    """Return df of sub-scores + composite score + signal label per bar."""
    from .bxtrender import score_bx
    parts = {
        "trend": score_trend(df),
        "momentum": score_momentum(df),
        "bxtrender": score_bx(df),
        "macd": score_macd(df),
        "rsi": score_rsi(df),
        "meanrev": score_meanrev(df),
        "volume": score_volume(df),
    }
    out = pd.DataFrame(parts, index=df.index)
    raw = sum(out[k] * w for k, w in WEIGHTS.items())
    out["score"] = raw * vol_regime(df)          # dampen in vol storms
    out["signal"] = np.where(out["score"] >= BUY_TH, "BUY",
                     np.where(out["score"] <= SELL_TH, "SELL", "HOLD"))
    return out


def latest_snapshot(df: pd.DataFrame) -> dict:
    """Latest composite reading for the screener table."""
    comp = composite(df)
    last = comp.iloc[-1]
    a = atr(df).iloc[-1]
    price = df["Close"].iloc[-1]
    return {
        "price": round(float(price), 2),
        "score": round(float(last["score"]), 3),
        "signal": str(last["signal"]),
        "trend": round(float(last["trend"]), 2),
        "momentum": round(float(last["momentum"]), 2),
        "bxtrender": round(float(last["bxtrender"]), 2),
        "macd": round(float(last["macd"]), 2),
        "rsi_score": round(float(last["rsi"]), 2),
        "meanrev": round(float(last["meanrev"]), 2),
        "atr": round(float(a), 2),
        "ret_1m": round(float(df["Close"].pct_change(21).iloc[-1] * 100), 1),
        "ret_3m": round(float(df["Close"].pct_change(63).iloc[-1] * 100), 1)
        if len(df) > 63 else None,
        "off_52w_high": round(float(price / df["High"].rolling(min(252, len(df))).max().iloc[-1] - 1) * 100, 1),
    }
