"""Volatility surface interpreter — rule-based, deterministic plain-text
reads of an options chain's shape. No LLM calls: every finding traces to
a specific number pulled straight from the chain (skew in vol points,
term-structure spread in vol points, smile deviation vs a local median).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .vol_surface import normalize_chain


def interpret_surface(chain: pd.DataFrame, spot: float | None = None) -> dict:
    c = normalize_chain(chain)
    if c.empty:
        return {"error": "chain missing strike/iv and dte-or-expiry columns",
               "findings": []}

    spot = float(spot) if spot else float(c["strike"].median())
    near_dte = int(c["dte"].min())
    near = c[c["dte"] == near_dte]
    findings = []

    # ---- 1. Skew: 25-delta if greeks are present, else a ~10% OTM strike proxy
    skew_pts = None
    if "delta" in near.columns and "type" in near.columns:
        puts = near[(near["type"] == "p") & (near["delta"].between(-0.35, -0.15))]
        calls = near[(near["type"] == "c") & (near["delta"].between(0.15, 0.35))]
        if len(puts) and len(calls):
            skew_pts = float(puts["iv"].mean() - calls["iv"].mean()) * 100
    if skew_pts is None:
        otm_puts = near[near["strike"] < spot * 0.93]
        otm_calls = near[near["strike"] > spot * 1.07]
        if len(otm_puts) and len(otm_calls):
            skew_pts = float(otm_puts["iv"].mean() - otm_calls["iv"].mean()) * 100
    if skew_pts is not None:
        if skew_pts > 5:
            findings.append(f"Steep put skew (+{skew_pts:.1f} vol pts at {near_dte}d) "
                            f"— market paying up for downside protection.")
        elif skew_pts < -2:
            findings.append(f"Inverted skew ({skew_pts:+.1f} vol pts at {near_dte}d) "
                            f"— calls richer than puts; unusual, check for a "
                            f"squeeze/melt-up setup or thin liquidity.")
        else:
            findings.append(f"Skew unremarkable ({skew_pts:+.1f} vol pts at {near_dte}d).")

    # ---- 2. Term structure: nearest vs furthest ATM IV
    term_pts = None
    dtes_sorted = sorted(c["dte"].unique())
    if len(dtes_sorted) >= 2:
        def _atm_iv(df):
            if df.empty:
                return None
            idx = (df["strike"] - spot).abs().idxmin()
            return float(df.loc[idx, "iv"])
        near_atm = _atm_iv(c[c["dte"] == dtes_sorted[0]])
        far_atm = _atm_iv(c[c["dte"] == dtes_sorted[-1]])
        if near_atm is not None and far_atm is not None:
            term_pts = (near_atm - far_atm) * 100
            if term_pts > 3:
                findings.append(f"Term structure INVERTED: {dtes_sorted[0]}d ATM IV is "
                                f"{term_pts:.1f} pts above {dtes_sorted[-1]}d — a "
                                f"near-term event (earnings/macro print) is priced in.")
            elif term_pts < -8:
                findings.append(f"Steep contango ({dtes_sorted[-1]}d IV "
                                f"{-term_pts:.1f} pts above {dtes_sorted[0]}d) — calm, "
                                f"normal term structure, no near-term event priced.")
            else:
                findings.append(f"Term structure roughly flat "
                                f"({term_pts:+.1f} pts, {dtes_sorted[0]}d vs "
                                f"{dtes_sorted[-1]}d).")

    # ---- 3. Smile anomalies at the near expiry
    anomalies = []
    if len(near) >= 5:
        s = near.sort_values("strike")
        smoothed = s["iv"].rolling(3, center=True, min_periods=1).median()
        dev = (s["iv"] - smoothed).abs()
        thresh = max(float(dev.median()) * 4, 0.03)
        for idx, row in s[dev > thresh].iterrows():
            anomalies.append({"strike": float(row["strike"]),
                              "iv_pct": round(float(row["iv"]) * 100, 1),
                              "deviation_pts": round(float(dev.loc[idx]) * 100, 1)})
        if anomalies:
            findings.append(
                f"{len(anomalies)} smile anomaly strike(s) at {near_dte}d (IV out of "
                f"line with neighbors — thin liquidity or unusual single-strike flow): "
                + ", ".join(f"${a['strike']:.0f} ({a['iv_pct']}% IV, "
                            f"+{a['deviation_pts']} pts)" for a in anomalies[:3]))

    return {
        "spot": round(spot, 2),
        "near_dte": near_dte,
        "skew_pts": round(skew_pts, 2) if skew_pts is not None else None,
        "term_structure_pts": round(term_pts, 2) if term_pts is not None else None,
        "smile_anomalies": anomalies,
        "findings": findings,
    }
