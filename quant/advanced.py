"""Advanced desk analytics — EWMA vol forecast, Kelly sizing, S/R, regimes."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_vol(df: pd.DataFrame, lam: float = 0.94) -> dict:
    """RiskMetrics EWMA volatility forecast (lambda = 0.94, daily).

    Returns tomorrow's expected daily move and annualised vol.
    """
    rets = np.log(df["Close"] / df["Close"].shift(1)).dropna().values
    var = rets[0] ** 2
    for r in rets[1:]:
        var = lam * var + (1 - lam) * r ** 2
    sigma_d = float(np.sqrt(var))
    price = float(df["Close"].iloc[-1])
    return {
        "sigma_daily_pct": round(sigma_d * 100, 2),
        "sigma_annual_pct": round(sigma_d * np.sqrt(252) * 100, 1),
        "expected_move_1d": round(price * sigma_d, 2),
    }


def kelly(p_win: float, rr: float) -> dict:
    """Kelly fraction for a binary bet: f* = p − (1−p)/b."""
    p = min(max(p_win, 0.0), 1.0)
    b = max(rr, 1e-9)
    f = p - (1 - p) / b
    return {
        "kelly_pct": round(f * 100, 1),
        "half_kelly_pct": round(f * 50, 1),
        "edge_positive": f > 0,
    }


def support_resistance(df: pd.DataFrame, lookback: int = 252,
                       k: int = 5, n_levels: int = 4,
                       tol: float = 0.015) -> list[dict]:
    """Cluster swing pivots into the most-touched support/resistance zones."""
    win = df.iloc[-lookback:]
    highs, lows = [], []
    hv, lv = win["High"].values, win["Low"].values
    for i in range(k, len(win) - k):
        if hv[i] == hv[i - k:i + k + 1].max():
            highs.append(hv[i])
        if lv[i] == lv[i - k:i + k + 1].min():
            lows.append(lv[i])
    pivots = sorted(highs + lows)
    if not pivots:
        return []

    clusters: list[list[float]] = []
    for p in pivots:
        if clusters and p <= clusters[-1][-1] * (1 + tol):
            clusters[-1].append(p)
        else:
            clusters.append([p])

    price = float(df["Close"].iloc[-1])
    levels = [{"price": round(float(np.mean(c)), 2), "touches": len(c),
               "kind": "support" if np.mean(c) < price else "resistance"}
              for c in clusters if len(c) >= 2]
    levels.sort(key=lambda x: -x["touches"])
    return levels[:n_levels]


def regime_quadrant(df: pd.DataFrame) -> dict:
    """Classify the current market regime: trend x volatility quadrant."""
    c = df["Close"]
    sma200 = c.rolling(200).mean()
    bull = bool(c.iloc[-1] > sma200.iloc[-1])
    rets = c.pct_change().dropna()
    vol_now = float(rets.iloc[-21:].std() * np.sqrt(252))
    vol_hist = float(rets.rolling(21).std().dropna().quantile(0.7) * np.sqrt(252))
    calm = vol_now < vol_hist
    name = ("🟢 Bull · Calm" if bull and calm else
            "🟡 Bull · Storm" if bull else
            "🔵 Bear · Calm" if calm else
            "🔴 Bear · Storm")
    playbook = {
        "🟢 Bull · Calm": "Best regime for longs — trend signals shine, full size allowed.",
        "🟡 Bull · Storm": "Uptrend but violent — halve size, widen stops, expect shakeouts.",
        "🔵 Bear · Calm": "Quiet downtrend — shorts/cash; long signals need extra proof.",
        "🔴 Bear · Storm": "Crash conditions — capital preservation mode. Most edges die here.",
    }[name]
    return {"regime": name, "playbook": playbook,
            "vol_now_pct": round(vol_now * 100, 1)}
