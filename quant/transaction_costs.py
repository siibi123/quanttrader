"""Transaction cost model — spread + square-root market impact.

This platform has no real Level-1 bid/ask feed (candles only), so the
spread is ESTIMATED via Corwin & Schultz (2012), "A Simple Way to
Estimate Bid-Ask Spreads from Daily High and Low Prices", Journal of
Finance — a well-documented estimator built from real OHLC data, not
fabricated. Market impact uses the standard square-root law (Almgren-
Chriss style): impact = spread + volatility * sqrt(order_size /
daily_volume).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def corwin_schultz_spread(df: pd.DataFrame, window: int = 20) -> float:
    """Estimated bid-ask spread as a fraction of price, from adjacent-day
    high/low ranges (paired with the PRIOR day, not the next one, so this
    never looks ahead)."""
    h, l = df["High"], df["Low"]
    h_prev, l_prev = h.shift(1), l.shift(1)
    k = 3 - 2 * np.sqrt(2)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.log(h / l) ** 2 + np.log(h_prev / l_prev) ** 2
        hi2 = np.maximum(h, h_prev)
        lo2 = np.minimum(l, l_prev)
        gamma = np.log(hi2 / lo2) ** 2
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    s = s.clip(lower=0).rolling(window, min_periods=5).mean()
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else 0.001    # 10bps floor if unestimable


def expected_trade_cost(df: pd.DataFrame, order_shares: int, price: float,
                        vol_window: int = 20, adv_window: int = 20) -> dict:
    """Spread + square-root market impact, in % and $ of the order.

    impact = spread + volatility * sqrt(order_size / daily_volume)
    """
    spread_pct = corwin_schultz_spread(df)
    rets = df["Close"].pct_change().dropna()
    vol_pct = (float(rets.iloc[-vol_window:].std()) if len(rets) >= vol_window
              else float(rets.std()) if len(rets) else 0.0)
    adv = float(df["Volume"].iloc[-adv_window:].mean())
    participation = (max(order_shares, 0) / adv) if adv > 0 else 1.0
    impact_pct = spread_pct + vol_pct * np.sqrt(max(participation, 0))
    notional = order_shares * price
    return {
        "spread_pct": round(spread_pct * 100, 4),
        "volatility_pct": round(vol_pct * 100, 4),
        "avg_daily_volume": round(adv, 0),
        "participation_rate_pct": round(participation * 100, 3),
        "expected_cost_pct": round(impact_pct * 100, 4),
        "expected_cost_$": round(impact_pct * notional, 2),
        "notional_$": round(notional, 2),
    }
