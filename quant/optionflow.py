"""Options flow analytics — call/put premium share, volume/premium spike
z-scores vs trailing norms, largest prints.

Built on LSEProvider.options_flow(), which wraps a REAL endpoint
(GET /options/flow — actual trade prints with premium/IV/greeks at print
time) verified 2026-07-12 by installing the real lse-data==0.14.0
package and reading its client.py source directly. This is not a
chain-delta proxy — the roadmap's documented fallback for if the
endpoint didn't exist, which it does.

Honest gap: neither options_flow() nor options() is documented to carry
an explicit open-interest field (only "today's volume and premium
totals" per the SDK's own chain docstring), so OI-based z-scores aren't
implemented here — that would be guessing at a field that isn't
verified to exist, which this project's rules forbid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _prep(flow: pd.DataFrame) -> pd.DataFrame:
    if flow is None or not len(flow):
        return pd.DataFrame()
    f = flow.copy()
    f.columns = [str(c).lower() for c in f.columns]
    if "type" in f.columns:
        f["type"] = f["type"].astype(str).str.lower().str[0]
    for c in ("premium", "volume", "strike"):
        if c in f.columns:
            f[c] = pd.to_numeric(f[c], errors="coerce")
    return f


def premium_share(flow: pd.DataFrame) -> dict:
    """Call vs put share of total premium traded in this flow snapshot."""
    f = _prep(flow)
    if f.empty or "type" not in f.columns or "premium" not in f.columns:
        return {"error": "flow missing type or premium column"}
    calls = float(f.loc[f["type"] == "c", "premium"].sum())
    puts = float(f.loc[f["type"] == "p", "premium"].sum())
    total = calls + puts
    if total <= 0:
        return {"error": "no premium in this flow snapshot"}
    return {"call_premium": round(calls, 0), "put_premium": round(puts, 0),
           "call_share_pct": round(calls / total * 100, 1),
           "put_share_pct": round(puts / total * 100, 1)}


def flow_spike(today_flow: pd.DataFrame,
              history_flow: list[pd.DataFrame]) -> dict:
    """Volume/premium z-score of TODAY's flow vs the trailing norm (one
    flow snapshot per prior day in history_flow, ~20 trading days).
    Honest: errors out rather than inventing a baseline from <10 days."""
    f = _prep(today_flow)
    if f.empty:
        return {"error": "no flow today"}
    today_vol = float(f["volume"].sum()) if "volume" in f.columns else float(len(f))
    today_prem = float(f["premium"].sum()) if "premium" in f.columns else None

    hist_vols, hist_prems = [], []
    for h in history_flow:
        hp = _prep(h)
        if hp.empty:
            continue
        hist_vols.append(float(hp["volume"].sum())
                         if "volume" in hp.columns else float(len(hp)))
        if "premium" in hp.columns:
            hist_prems.append(float(hp["premium"].sum()))
    if len(hist_vols) < 10:
        return {"error": "need >= 10 days of flow history for a z-score"}

    vol_mu, vol_sd = float(np.mean(hist_vols)), float(np.std(hist_vols))
    out = {"today_volume": today_vol, "hist_mean_volume": round(vol_mu, 0),
          "volume_z": round((today_vol - vol_mu) / vol_sd, 2) if vol_sd > 0 else 0.0,
          "n_history_days": len(hist_vols)}
    if today_prem is not None and len(hist_prems) >= 10:
        prem_mu, prem_sd = float(np.mean(hist_prems)), float(np.std(hist_prems))
        out["premium_z"] = (round((today_prem - prem_mu) / prem_sd, 2)
                            if prem_sd > 0 else 0.0)
    return out


def largest_prints(flow: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """The single biggest premium prints in this flow snapshot."""
    f = _prep(flow)
    if f.empty or "premium" not in f.columns:
        return []
    top = f.sort_values("premium", ascending=False).head(top_n)
    return [{"strike": row.get("strike"), "type": row.get("type"),
            "premium": round(float(row["premium"]), 0)
                      if pd.notna(row["premium"]) else None,
            "expiry": str(row.get("expiry", ""))}
           for _, row in top.iterrows()]
