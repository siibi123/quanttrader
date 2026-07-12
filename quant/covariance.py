"""Ledoit-Wolf shrinkage covariance + the min-variance portfolio it enables.

Sample covariance on N stocks with limited history is noisy and often
ill-conditioned once N approaches the observation count — exactly a
retail book's situation (a handful of tickers, ~1-2y of daily bars).
Shrinking toward a structured target before any optimizer sees the matrix
fixes that (scikit-learn's LedoitWolf — the reference implementation of
Ledoit & Wolf 2004, not reimplemented by hand: the closed-form shrinkage
intensity has several easy-to-flip normalization constants and this is
risk math, not a place to guess).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def shrunk_covariance(returns: pd.DataFrame) -> dict:
    """Ledoit-Wolf shrunk covariance/correlation on a returns matrix (cols=tickers)."""
    r = returns.dropna()
    if len(r) < 30 or r.shape[1] < 2:
        return {"error": "need >= 30 observations and >= 2 tickers"}
    lw = LedoitWolf().fit(r.values)
    cov = pd.DataFrame(lw.covariance_, index=r.columns, columns=r.columns)
    std = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(std, std)
    return {
        "tickers": list(r.columns),
        "shrinkage": round(float(lw.shrinkage_), 4),
        "covariance": cov,
        "correlation": pd.DataFrame(corr, index=r.columns, columns=r.columns),
    }


def min_variance_weights(returns: pd.DataFrame, long_only: bool = True) -> dict:
    """Global minimum-variance portfolio on the Ledoit-Wolf shrunk covariance.

    long_only uses a simple clip-and-renormalize projection, not a full QP
    solver — good enough for a small retail watchlist, honestly approximate
    for larger ones.
    """
    sc = shrunk_covariance(returns)
    if "error" in sc:
        return sc
    cov = sc["covariance"].values
    n = cov.shape[0]
    ones = np.ones(n)
    raw = np.linalg.pinv(cov) @ ones
    if long_only and (raw < 0).any():
        w = np.clip(raw, 0, None)
        w = w / w.sum() if w.sum() > 0 else ones / n
    else:
        w = raw / raw.sum()
    port_var = float(w @ cov @ w)
    return {
        "tickers": sc["tickers"],
        "weights": {t: round(float(wi), 4) for t, wi in zip(sc["tickers"], w)},
        "shrinkage": sc["shrinkage"],
        "expected_daily_vol_pct": round(float(np.sqrt(max(port_var, 0))) * 100, 3),
        "expected_annual_vol_pct": round(
            float(np.sqrt(max(port_var, 0) * 252)) * 100, 1),
    }
