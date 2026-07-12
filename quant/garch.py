"""GARCH(1,1) volatility forecasting via the `arch` package (Bollerslev 1986).

Thin, honest wrapper: fit on daily returns, forecast next-period
conditional vol, and surface the persistence/half-life a desk actually
asks about — "is this vol spike about to mean-revert, or is the regime
sticky?"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model


def garch_forecast(close: pd.Series, horizon: int = 1, p: int = 1, q: int = 1,
                   dist: str = "normal") -> dict:
    """Fit GARCH(p,q) on daily returns, forecast conditional vol `horizon` out."""
    r = close.pct_change().dropna() * 100     # arch wants returns in % for numerical stability
    if len(r) < 100:
        return {"error": "need >= 100 return observations"}
    try:
        res = arch_model(r, vol="Garch", p=p, q=q, dist=dist, mean="Zero").fit(disp="off")
        fc = res.forecast(horizon=horizon, reindex=False)
    except Exception as e:
        return {"error": f"GARCH fit failed: {e}"}

    var_path = fc.variance.values[-1]
    vol_1d_pct = float(np.sqrt(var_path[0]))
    alpha = float(res.params.get("alpha[1]", np.nan))
    beta = float(res.params.get("beta[1]", np.nan))
    persistence = (alpha + beta if not (np.isnan(alpha) or np.isnan(beta))
                  else None)
    half_life = (float(np.log(0.5) / np.log(persistence))
                if persistence is not None and 0 < persistence < 1 else None)
    regime = ("🔴 explosive / near unit-root" if persistence and persistence > 0.995
             else "🟡 sticky vol regime" if persistence and persistence > 0.9
             else "🟢 mean-reverting vol")
    return {
        "vol_1d_pct": round(vol_1d_pct, 3),
        "vol_annual_pct": round(vol_1d_pct * np.sqrt(252), 1),
        "alpha": round(alpha, 4) if alpha == alpha else None,
        "beta": round(beta, 4) if beta == beta else None,
        "persistence": round(persistence, 4) if persistence is not None else None,
        "half_life_days": round(half_life, 1) if half_life else None,
        "regime": regime,
    }
