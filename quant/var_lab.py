"""VaR Lab — 4 methods (historical / parametric / Cornish-Fisher / EWMA),
horizon scaling by sqrt(t), optional CVaR. Matches the industry-standard
parameterization (conf 90-99.9, lookback 30-500, horizon 1-30)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def var_suite(close: pd.Series, method: str = "historical",
              conf: float = 0.95, horizon: int = 1, lookback: int = 252,
              value: float = 100_000.0, cvar: bool = True) -> dict:
    r = close.pct_change().dropna().iloc[-int(lookback):]
    if len(r) < 30:
        return {"error": "need >=30 observations"}
    mu, sig = float(r.mean()), float(r.std())
    z = stats.norm.ppf(1 - conf)                      # negative
    if method == "historical":
        var_pct = -float(np.percentile(r, (1 - conf) * 100))
        tail = r[r <= np.percentile(r, (1 - conf) * 100)]
        cvar_pct = -float(tail.mean()) if len(tail) else var_pct
    elif method == "parametric":
        var_pct = -(mu + z * sig)
        cvar_pct = sig * stats.norm.pdf(z) / (1 - conf) - mu
    elif method == "cornish_fisher":
        S, K = float(stats.skew(r)), float(stats.kurtosis(r))  # excess K
        zcf = (z + (z**2 - 1) * S / 6 + (z**3 - 3 * z) * K / 24
               - (2 * z**3 - 5 * z) * S**2 / 36)
        var_pct = -(mu + zcf * sig)
        cvar_pct = var_pct * 1.15                     # CF tail approx
    elif method == "ewma":
        lam, v = 0.94, float(r.iloc[0]) ** 2
        for x in r.iloc[1:].values:
            v = lam * v + (1 - lam) * x * x
        sig_e = float(np.sqrt(v))
        var_pct = -(z * sig_e)
        cvar_pct = sig_e * stats.norm.pdf(z) / (1 - conf)
    else:
        return {"error": f"unknown method {method}"}
    s = np.sqrt(max(int(horizon), 1))
    var_pct, cvar_pct = max(var_pct, 0) * s, max(cvar_pct, 0) * s
    out = {"method": method, "conf": conf, "horizon_d": horizon,
           "lookback": len(r), "VaR_pct": round(var_pct * 100, 3),
           "VaR_$": round(var_pct * value, 0)}
    if cvar:
        out["CVaR_pct"] = round(cvar_pct * 100, 3)
        out["CVaR_$"] = round(cvar_pct * value, 0)
    return out
