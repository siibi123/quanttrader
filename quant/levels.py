"""Price-structure analytics — Fibonacci retracements & Hurst exponent."""
from __future__ import annotations

import numpy as np
import pandas as pd

FIB_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)


def fib_levels(df: pd.DataFrame, lookback: int = 126) -> dict:
    """Auto-detect the dominant swing in `lookback` bars and return fib levels.

    If the swing low came before the swing high -> uptrend swing, retracements
    measured down from the high. Otherwise downtrend swing, measured up.
    """
    win = df.iloc[-lookback:]
    hi_pos = int(np.argmax(win["High"].values))
    lo_pos = int(np.argmin(win["Low"].values))
    hi = float(win["High"].iloc[hi_pos])
    lo = float(win["Low"].iloc[lo_pos])
    up_swing = lo_pos < hi_pos                      # low first, then high

    levels = {}
    rng = hi - lo
    for r in FIB_RATIOS:
        price = hi - rng * r if up_swing else lo + rng * r
        levels[f"{r:.3f}".rstrip("0").rstrip(".") or "0"] = round(price, 2)

    return {
        "up_swing": up_swing,
        "swing_high": round(hi, 2),
        "swing_low": round(lo, 2),
        "high_date": win.index[hi_pos],
        "low_date": win.index[lo_pos],
        "levels": levels,
    }


def hurst(df: pd.DataFrame, max_lag: int = 100) -> float:
    """Hurst exponent via rescaled variance of lagged differences.

    H > 0.5 -> trending (momentum works), H < 0.5 -> mean-reverting
    (fade extremes), H ~ 0.5 -> random walk (no memory).
    """
    prices = np.log(df["Close"].dropna().values)
    if len(prices) < max_lag * 2:
        max_lag = max(20, len(prices) // 4)
    lags = range(2, max_lag)
    tau = [np.std(prices[lag:] - prices[:-lag]) for lag in lags]
    tau = np.maximum(tau, 1e-12)
    h = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
    return round(float(np.clip(h, 0.0, 1.0)), 3)
