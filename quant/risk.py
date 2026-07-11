"""Risk management — the quant models that keep a $5K account alive.

Everything here answers ONE question a desk asks before every trade:
"how much can this realistically cost me, and am I over the line?"

  * Position risk      — $ and % at risk to the stop (the only sizing that matters)
  * Portfolio heat     — total simultaneous risk if every stop hits at once
  * Parametric VaR/CVaR — 1-day 95/99% loss estimate on the whole book
  * Correlation-adjusted heat — naive heat lies when positions move together
  * Risk of ruin       — probability of losing X% given your edge & bet size
  * Kelly ladder       — full / half / quarter Kelly with the growth-vs-pain tradeoff
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def position_risk(entry: float, stop: float, shares: int) -> dict:
    per_share = abs(entry - stop)
    return {
        "risk_$": round(per_share * shares, 2),
        "risk_per_share": round(per_share, 2),
        "notional_$": round(entry * shares, 2),
    }


def portfolio_var(positions: list[dict], returns: dict[str, pd.Series],
                  account: float, horizon_days: int = 1,
                  conf: float = 0.95) -> dict:
    """Parametric (variance-covariance) VaR/CVaR on the open book.

    positions: [{ticker, shares, entry}], returns: {ticker: daily ret series}.
    """
    tks = [p["ticker"] for p in positions if p["ticker"] in returns]
    if not tks:
        return {}
    w_dollar = np.array([next(p["shares"] * p["entry"] for p in positions
                              if p["ticker"] == t) for t in tks])
    R = pd.DataFrame({t: returns[t] for t in tks}).dropna()
    if len(R) < 30:
        return {}
    cov = R.cov().values * horizon_days
    port_var_dollar = float(np.sqrt(w_dollar @ cov @ w_dollar))
    z = stats.norm.ppf(conf)
    var = z * port_var_dollar
    # CVaR (expected shortfall) for a normal dist
    cvar = port_var_dollar * stats.norm.pdf(z) / (1 - conf)
    return {
        "VaR_$": round(var, 0),
        "VaR_%": round(var / account * 100, 2),
        "CVaR_$": round(cvar, 0),
        "CVaR_%": round(cvar / account * 100, 2),
        "conf": int(conf * 100),
        "horizon": horizon_days,
        "gross_exposure_%": round(w_dollar.sum() / account * 100, 1),
    }


def correlation_heat(positions: list[dict], returns: dict[str, pd.Series],
                     account: float) -> dict:
    """Naive heat assumes independence; real heat accounts for correlation."""
    tks = [p["ticker"] for p in positions if p["ticker"] in returns]
    if len(tks) < 2:
        return {}
    R = pd.DataFrame({t: returns[t] for t in tks}).dropna()
    if len(R) < 30:
        return {}
    corr = R.corr()
    risks = np.array([next((p["entry"] - p["stop"]) * p["shares"]
                          for p in positions if p["ticker"] == t) for t in tks])
    naive = float(risks.sum())
    combined = float(np.sqrt(risks @ corr.values @ risks))
    avg_corr = float(corr.values[np.triu_indices_from(corr.values, 1)].mean())
    return {
        "naive_heat_$": round(naive, 0),
        "corr_adj_heat_$": round(combined, 0),
        "avg_correlation": round(avg_corr, 2),
        "diversification_benefit_%": round((1 - combined / naive) * 100, 0)
        if naive > 0 else 0,
        "warning": avg_corr > 0.6,
    }


def risk_of_ruin(win_rate: float, avg_win: float, avg_loss: float,
                 risk_per_trade_pct: float, ruin_pct: float = 0.30,
                 n_sims: int = 5000, n_trades: int = 200,
                 seed: int = 7) -> dict:
    """Monte Carlo probability of drawing down `ruin_pct` given the edge."""
    if avg_loss <= 0:
        return {}
    rng = np.random.default_rng(seed)
    payoff = avg_win / avg_loss
    ruined = 0
    for _ in range(n_sims):
        eq = 1.0
        peak = 1.0
        for _ in range(n_trades):
            bet = risk_per_trade_pct / 100
            if rng.random() < win_rate:
                eq *= 1 + bet * payoff
            else:
                eq *= 1 - bet
            peak = max(peak, eq)
            if eq <= peak * (1 - ruin_pct):
                ruined += 1
                break
    p = ruined / n_sims
    return {
        "ruin_threshold_%": int(ruin_pct * 100),
        "prob_of_ruin_%": round(p * 100, 1),
        "payoff_ratio": round(payoff, 2),
        "expectancy_R": round(win_rate * payoff - (1 - win_rate), 3),
        "verdict": ("🟢 Robust" if p < 0.05 else "🟡 Survivable" if p < 0.20
                    else "🔴 Dangerous — cut size"),
    }


def kelly_ladder(win_rate: float, rr: float) -> dict:
    b = max(rr, 1e-9)
    f = win_rate - (1 - win_rate) / b
    return {
        "full_kelly_%": round(f * 100, 1),
        "half_kelly_%": round(f * 50, 1),
        "quarter_kelly_%": round(f * 25, 1),
        "edge": f > 0,
    }
