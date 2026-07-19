"""Correlation regime monitor — rolling pairwise correlation across
holdings, a hard alert when it spikes, and an early-warning trend
detector so a book de-risks BEFORE correlations fully converge toward 1
(and the drawdown that tends to follow), not after the fact.

Distinct from the v0.3-era quant.risk.correlation_heat (a static,
full-history snapshot with a 0.6 warning threshold, unchanged here):
this uses a genuine rolling 20-day window and tracks the TREND of that
rolling average over time, not just its current level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALERT_THRESHOLD = 0.7
TREND_LOOKBACK = 20        # how many rolling-window snapshots to trend over

# "de-risk BEFORE the drawdown" as an actual sizing policy, same shape as
# quant.regime_gate.REGIME_POLICY and core.circuit_breaker's multiplier —
# stacks with those, doesn't replace them.
CORRELATION_POLICY = {
    "normal": {"size_multiplier": 1.0, "new_trades_allowed": True},
    "early_warning": {"size_multiplier": 0.5, "new_trades_allowed": True},
    "alert": {"size_multiplier": 0.0, "new_trades_allowed": False},
}


def rolling_avg_correlation(returns: dict[str, pd.Series],
                            window: int = 20) -> pd.Series:
    """Time series of the average pairwise correlation across `returns`'
    holdings, each point computed from the trailing `window` days as of
    that date (a rolling-of-rolling — not a single static snapshot)."""
    R = pd.DataFrame(returns).dropna()
    if R.shape[1] < 2:
        return pd.Series(dtype=float)
    n = len(R)
    idx, out = [], []
    for i in range(window, n + 1):
        chunk = R.iloc[i - window:i]
        corr = chunk.corr().values
        iu = np.triu_indices_from(corr, 1)
        out.append(float(np.nanmean(corr[iu])))
        idx.append(R.index[i - 1])
    return pd.Series(out, index=idx)


def correlation_regime(returns: dict[str, pd.Series], window: int = 20,
                       alert_threshold: float = ALERT_THRESHOLD,
                       trend_lookback: int = TREND_LOOKBACK) -> dict:
    """Current rolling avg correlation + whether it's trending toward 1
    (early warning) + a hard alert at/above `alert_threshold`. Honest
    error, not a fabricated read, if there isn't enough overlapping
    history across at least 2 holdings."""
    series = rolling_avg_correlation(returns, window=window)
    if len(series) < trend_lookback:
        return {"error": f"need >= {window + trend_lookback} overlapping "
                         f"return observations across >= 2 holdings"}
    current = float(series.iloc[-1])
    recent = series.iloc[-trend_lookback:]
    slope = float(np.polyfit(np.arange(len(recent)), recent.values, 1)[0])

    alert = current >= alert_threshold
    converging = slope > 0.005 and current > 0.4 and not alert
    state = "alert" if alert else ("early_warning" if converging else "normal")
    verdict = ("🔴 ALERT — correlations elevated, book is effectively one "
              "trade" if alert else
              "🟡 EARLY WARNING — correlations rising toward 1, de-risk "
              "before it fully converges" if converging else "🟢 normal")
    return {
        "current_avg_correlation": round(current, 3),
        "trend_slope_per_day": round(slope, 5),
        "alert": alert,
        "converging_early_warning": converging,
        "state": state,
        "policy": CORRELATION_POLICY[state],
        "verdict": verdict,
    }
