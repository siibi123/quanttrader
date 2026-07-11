"""Monte Carlo engine — simulate thousands of possible futures.

Geometric Brownian Motion calibrated to the ticker's own history
(EWMA-weighted drift & volatility so recent behaviour matters more).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calibrate(df: pd.DataFrame, lookback: int = 252) -> tuple[float, float]:
    """Annualised (mu, sigma) from log returns, recent-weighted."""
    rets = np.log(df["Close"] / df["Close"].shift(1)).dropna().iloc[-lookback:]
    w = np.exp(np.linspace(-1.0, 0.0, len(rets)))          # recent = heavier
    w /= w.sum()
    mu_d = float((rets * w).sum())
    var_d = float((w * (rets - mu_d) ** 2).sum())
    return mu_d * 252, np.sqrt(var_d * 252)


def simulate(df: pd.DataFrame, days: int = 30, n_paths: int = 2000,
             seed: int = 42) -> np.ndarray:
    """Return (n_paths, days+1) matrix of simulated prices, col 0 = spot."""
    mu, sigma = calibrate(df)
    s0 = float(df["Close"].iloc[-1])
    dt = 1.0 / 252.0
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, days))
    steps = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    paths = s0 * np.exp(np.cumsum(steps, axis=1))
    return np.hstack([np.full((n_paths, 1), s0), paths])


def cone(paths: np.ndarray,
         pcts=(5, 25, 50, 75, 95)) -> dict[int, np.ndarray]:
    """Percentile bands across time for the probability cone chart."""
    return {p: np.percentile(paths, p, axis=0) for p in pcts}


def trade_odds(paths: np.ndarray, entry: float, stop: float, target: float,
               direction: int) -> dict:
    """First-touch simulation: does each path hit target or stop first?

    direction: +1 long, -1 short.
    """
    if direction >= 0:
        hit_t = paths >= target
        hit_s = paths <= stop
    else:
        hit_t = paths <= target
        hit_s = paths >= stop

    def first_true(m):
        idx = np.argmax(m, axis=1)
        idx[~m.any(axis=1)] = m.shape[1] + 1     # never touched
        return idx

    t_idx, s_idx = first_true(hit_t), first_true(hit_s)
    p_target = float((t_idx < s_idx).mean())
    p_stop = float((s_idx < t_idx).mean())
    p_neither = max(0.0, 1.0 - p_target - p_stop)

    # Terminal P&L distribution (per share, direction-adjusted)
    pnl = (paths[:, -1] - entry) * direction
    var95 = float(np.percentile(pnl, 5))
    cvar95 = float(pnl[pnl <= var95].mean()) if (pnl <= var95).any() else var95

    return {
        "p_target_first": round(p_target * 100, 1),
        "p_stop_first": round(p_stop * 100, 1),
        "p_neither": round(p_neither * 100, 1),
        "p_profit_end": round(float((pnl > 0).mean()) * 100, 1),
        "exp_pnl_share": round(float(pnl.mean()), 2),
        "var95_share": round(var95, 2),
        "cvar95_share": round(cvar95, 2),
    }
