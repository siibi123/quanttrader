"""Kalman filter dynamic hedge ratio for pairs trading.

Standard 2-state (intercept, slope) formulation: y_t = alpha_t + beta_t*x_t
+ eps_t, where [alpha_t, beta_t] follows a random walk. This is the
textbook Kalman-filter pairs hedge ratio (Chan, "Algorithmic Trading",
ch.5) — the hedge ratio adapts to regime changes instead of a static OLS
beta re-fit on a rolling window. One forward pass, no lookahead: at each
t the filter only uses data up to t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kalman_hedge_ratio(y: pd.Series, x: pd.Series, delta: float = 1e-4,
                       ve: float = 1e-3) -> pd.DataFrame:
    """Recursive (alpha, beta) estimate + spread z-score.

    delta: state (random-walk) variance scale — higher = hedge ratio
    adapts faster but noisier. ve: observation noise variance.
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    n = len(df)
    if n < 20:
        return pd.DataFrame()

    theta = np.zeros(2)                       # [alpha, beta]
    P = np.eye(2) * 1.0
    Vw = delta / (1 - delta) * np.eye(2)       # state (process) noise cov

    alphas, betas, spreads, spread_std = [], [], [], []
    yv, xv = df["y"].values, df["x"].values
    for i in range(n):
        H = np.array([1.0, xv[i]])
        P = P + Vw                             # predict
        yhat = H @ theta
        e = yv[i] - yhat
        S = float(H @ P @ H + ve)
        K = P @ H / S                          # update
        theta = theta + K * e
        P = P - np.outer(K, H) @ P
        alphas.append(theta[0]); betas.append(theta[1])
        spreads.append(e); spread_std.append(np.sqrt(S))

    out = df.copy()
    out["alpha"], out["beta"] = alphas, betas
    out["spread"] = spreads
    out["spread_z"] = np.array(spreads) / np.array(spread_std)
    return out


def pair_signal(y: pd.Series, x: pd.Series, entry_z: float = 2.0,
                exit_z: float = 0.5, **kwargs) -> dict:
    """Latest hedge ratio + spread z-score + a plain mean-reversion signal."""
    kf = kalman_hedge_ratio(y, x, **kwargs)
    if kf.empty:
        return {"error": "need >= 20 overlapping observations"}
    last = kf.iloc[-1]
    z = float(last["spread_z"])
    if abs(z) >= entry_z:
        signal = "SHORT SPREAD" if z > 0 else "LONG SPREAD"
    elif abs(z) <= exit_z:
        signal = "FLAT / EXIT"
    else:
        signal = "HOLD / NO ACTION"
    return {
        "hedge_ratio": round(float(last["beta"]), 4),
        "intercept": round(float(last["alpha"]), 4),
        "spread_z": round(z, 2),
        "signal": signal,
    }
