"""Volatility surface — normalizes an options chain (a few tolerated
column-naming conventions, since the exact LSE /options/chain shape isn't
hardcoded-guessed) into a strike x DTE x IV grid ready for a Plotly Surface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_chain(chain: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, coerce strike/iv/dte to numeric. Accepts a `dte`
    or `days_to_expiry` column, or derives DTE from an expiry/expiration/
    exp_date date column. Returns an empty DataFrame if the chain doesn't
    carry what's needed — callers must handle that honestly, not fake it.
    """
    if chain is None or not len(chain):
        return pd.DataFrame()
    c = chain.copy()
    c.columns = [str(x).lower() for x in c.columns]
    if not {"strike", "iv"}.issubset(c.columns):
        return pd.DataFrame()
    c["strike"] = pd.to_numeric(c["strike"], errors="coerce")
    c["iv"] = pd.to_numeric(c["iv"], errors="coerce")
    if "dte" in c.columns:
        c["dte"] = pd.to_numeric(c["dte"], errors="coerce")
    elif "days_to_expiry" in c.columns:
        c["dte"] = pd.to_numeric(c["days_to_expiry"], errors="coerce")
    else:
        exp_col = next((k for k in ("expiry", "expiration", "exp_date")
                        if k in c.columns), None)
        if exp_col is None:
            return pd.DataFrame()
        exp = pd.to_datetime(c[exp_col], errors="coerce")
        c["dte"] = (exp - pd.Timestamp.now()).dt.days
    if "delta" in c.columns:
        c["delta"] = pd.to_numeric(c["delta"], errors="coerce")
    if "type" in c.columns:
        c["type"] = c["type"].astype(str).str.lower().str[0]     # 'c' / 'p'
    return c.dropna(subset=["strike", "iv", "dte"])


def build_surface_grid(chain: pd.DataFrame) -> dict:
    """strike x DTE x IV grid, mean-aggregated where multiple quotes share
    a (strike, dte) cell (e.g. call and put both quoting the same strike).
    """
    c = normalize_chain(chain)
    if c.empty:
        return {"error": "chain missing strike/iv and dte-or-expiry columns"}
    piv = c.pivot_table(index="dte", columns="strike", values="iv", aggfunc="mean")
    piv = piv.sort_index().reindex(sorted(piv.columns), axis=1)
    return {
        "dtes": [int(d) for d in piv.index],
        "strikes": [float(s) for s in piv.columns],
        "iv_grid": [[None if pd.isna(v) else round(float(v), 4) for v in row]
                    for row in piv.values],
        "contracts_used": int(len(c)),
    }
