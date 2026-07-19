"""Monte Carlo portfolio stress test — correlated paths of the CURRENT
book, not independent per-symbol simulations (which understate tail
risk, since correlated assets crash together). Reuses quant.covariance's
Ledoit-Wolf shrinkage (P2) for a robust covariance/correlation estimate.

Buy-and-hold assumption over the horizon (no rebalancing) — a standard
simplification for this kind of report, not a claim about how the
platform actually trades within the window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .covariance import shrunk_covariance

HIGH_RISK_P10DD_THRESHOLD = 25.0    # P(10% DD) >= this -> cut next week's size


def simulate_portfolio(returns: dict[str, pd.Series], dollars: dict[str, float],
                       horizon_days: int = 21, n_paths: int = 10_000,
                       seed: int = 7) -> dict:
    """Correlated GBM simulation of the whole book's dollar value over
    `horizon_days` (default 21 ~ one trading month)."""
    tickers = [t for t in dollars if t in returns and dollars[t] > 0]
    if not tickers:
        return {"error": "no positive-dollar holdings with return history to simulate"}

    r_df = pd.DataFrame({t: returns[t] for t in tickers}).dropna()
    if len(r_df) < 30:
        return {"error": "need >= 30 overlapping daily returns per holding"}

    n = len(tickers)
    mu_d = r_df.mean().values
    if n > 1:
        sc = shrunk_covariance(r_df)
        if "error" in sc:
            return sc
        cov = sc["covariance"].values
        sigma_d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sigma_d, sigma_d)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        chol = np.linalg.cholesky(corr + 1e-8 * np.eye(n))
    else:
        sigma_d = r_df.std().values
        chol = np.eye(1)

    dollars0 = np.array([dollars[t] for t in tickers], dtype=float)
    total0 = float(dollars0.sum())

    rng = np.random.default_rng(seed)
    port_paths = np.zeros((n_paths, horizon_days + 1))
    port_paths[:, 0] = total0
    asset_value = np.tile(dollars0, (n_paths, 1))
    for day in range(1, horizon_days + 1):
        z = rng.standard_normal((n_paths, n)) @ chol.T
        day_rets = mu_d + sigma_d * z
        asset_value = asset_value * (1 + day_rets)
        port_paths[:, day] = asset_value.sum(axis=1)

    terminal = port_paths[:, -1]
    running_peak = np.maximum.accumulate(port_paths, axis=1)
    max_dd = ((running_peak - port_paths) / running_peak).max(axis=1)

    worst_week_pct = None
    if horizon_days >= 5:
        weekly = port_paths[:, 5::5] / port_paths[:, :-5:5] - 1
        if weekly.size:
            worst_week_pct = round(float(np.percentile(weekly, 5)) * 100, 2)

    var95 = float(np.percentile(terminal, 5))
    cvar_mask = terminal <= var95
    cvar95 = float(terminal[cvar_mask].mean()) if cvar_mask.any() else var95

    return {
        "n_paths": n_paths, "horizon_days": horizon_days,
        "tickers": tickers, "starting_value_$": round(total0, 0),
        "p_10pct_drawdown_%": round(float((max_dd >= 0.10).mean()) * 100, 1),
        "p_20pct_drawdown_%": round(float((max_dd >= 0.20).mean()) * 100, 1),
        "expected_worst_week_%": worst_week_pct,
        "var95_$": round(total0 - var95, 0),
        "cvar95_$": round(total0 - cvar95, 0),
        "median_terminal_$": round(float(np.median(terminal)), 0),
    }


def risk_budget_from_stress(stress: dict,
                            high_risk_threshold: float = HIGH_RISK_P10DD_THRESHOLD
                            ) -> dict:
    """Feeds the stress test into next week's sizing: elevated P(10% DD)
    cuts new-entry size in half until the next run says otherwise."""
    if "error" in stress:
        return {"size_multiplier": 1.0, "elevated_risk": False}
    elevated = stress["p_10pct_drawdown_%"] >= high_risk_threshold
    return {"size_multiplier": 0.5 if elevated else 1.0, "elevated_risk": elevated}
