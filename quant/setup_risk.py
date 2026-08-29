"""Per-trade risk budget.

The owner's slider is a CEILING. Each ticket earns a fraction of it from
setup quality, this ticker's own forward history, overlap with the open
book, spread/cost, and near-term macro events.

Stops: thesis invalidation (below the buy zone / swing low) with ATR as
a floor (don't get wicked) and a cap (don't risk the farm). ATR is a
ruler, not the thesis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.transaction_costs import expected_trade_cost
from quant.verdict import MODELS


def model_agreement(comp_row) -> tuple[int, float]:
    n = 0
    for m in MODELS:
        try:
            if float(comp_row[m]) > 0:
                n += 1
        except Exception:
            pass
    return n, n / max(len(MODELS), 1)


def structure_stop(price: float, atr: float, zone: dict | None,
                   stop_atr_floor: float = 2.5) -> dict:
    """Invalidation just below the zone/swing; ATR clamps 1×–4×."""
    zone = zone or {}
    price, atr = float(price), float(atr or 0.0)
    atr_stop = price - stop_atr_floor * atr if atr > 0 else price * 0.94
    candidates = []
    for key in ("ote_lo", "swing_low"):
        v = zone.get(key)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if 0 < fv < price:
            candidates.append(fv * 0.997)
    if not candidates or atr <= 0:
        return {"stop": round(atr_stop, 2), "stop_atr": stop_atr_floor,
                "kind": "ATR"}
    struct = max(candidates)
    dist = price - struct
    if dist < 1.0 * atr:
        return {"stop": round(price - 1.0 * atr, 2), "stop_atr": 1.0,
                "kind": "structure-tight→1ATR"}
    if dist > 4.0 * atr:
        return {"stop": round(price - 4.0 * atr, 2), "stop_atr": 4.0,
                "kind": "structure-far→4ATR"}
    return {"stop": round(struct, 2),
            "stop_atr": round(dist / atr, 2),
            "kind": "structure"}


def ticker_edge_from_comp(comp: pd.DataFrame, close: pd.Series,
                          horizon: int = 10) -> dict:
    """Did BUY on THIS name actually pay over the next `horizon` days?"""
    if comp is None or close is None or len(close) < 80:
        return {"mult": 1.0, "note": None}
    s = comp["signal"].values
    c = close.values
    n = len(c)
    rets = []
    for i in range(40, n - horizon):
        if s[i] == "BUY":
            rets.append(c[i + horizon] / c[i] - 1.0)
    if len(rets) < 8:
        return {"mult": 1.0, "note": None}
    arr = np.asarray(rets, dtype=float)
    mean = float(arr.mean())
    win = float((arr > 0).mean())
    if mean <= 0:
        return {"mult": 0.40,
                "note": f"this ticker's BUY lost fwd ({mean:.1%}, n={len(rets)})"}
    if mean >= 0.015 and win >= 0.55:
        return {"mult": 1.15,
                "note": f"ticker paid +{mean:.1%} / 10d, win {win:.0%}"}
    return {"mult": 1.0, "note": f"ticker 10d {mean:.1%}"}


def book_overlap_mult(candidate: pd.DataFrame,
                      held: dict[str, pd.DataFrame]) -> tuple[float, str]:
    """Haircut when the new name is highly correlated with what's already held."""
    if not held or candidate is None or len(candidate) < 40:
        return 1.0, ""
    c = candidate["Close"].pct_change().dropna()
    corrs = []
    for df in held.values():
        if df is None or len(df) < 40:
            continue
        r = df["Close"].pct_change().dropna()
        aligned = pd.concat([c, r], axis=1, join="inner").dropna()
        if len(aligned) < 30:
            continue
        val = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        if np.isfinite(val):
            corrs.append(val)
    if not corrs:
        return 1.0, ""
    avg, mx = float(np.mean(corrs)), float(np.max(corrs))
    if mx >= 0.80:
        return 0.40, f"crowded vs book (corr {mx:.2f}) — cut"
    if avg >= 0.55:
        return 0.65, f"avg corr to book {avg:.2f} — reduced"
    return 1.0, f"book corr {avg:.2f}"


def high_impact_soon(macro: dict | None, hours: int = 48) -> bool:
    events = (macro or {}).get("upcoming_events") or []
    keys = ("cpi", "fomc", "payroll", "nfp", "rate decision", "fed ", "nonfarm")
    now = pd.Timestamp.utcnow()
    for e in events:
        name = str(e.get("event") or "").lower()
        if not any(k in name for k in keys):
            continue
        dt = pd.to_datetime(e.get("date"), utc=True, errors="coerce")
        if pd.isna(dt):
            continue
        delta = (dt.to_pydatetime() - now.to_pydatetime()).total_seconds()
        if 0 <= delta <= hours * 3600:
            return True
    return False


def size_setup(*, urgency: str, zone: dict | None, mode: dict | None,
               weekly: dict | None, atr: float, price: float,
               account: float, risk_cap_pct: float,
               comp_row=None, fundamental: dict | None = None,
               df: pd.DataFrame | None = None,
               comp: pd.DataFrame | None = None) -> dict:
    cap = max(0.1, float(risk_cap_pct))
    zone = zone or {}
    mode = mode or {}
    weekly = weekly or {}
    fund = fundamental or {}
    reasons: list[str] = []

    if urgency == "🟢 ACTIONABLE" and zone.get("in_buy_zone"):
        mult, stop_atr_floor = 1.00, 3.0
        reasons.append("full size: 5/5 + buy zone")
    elif urgency == "🟢 ACTIONABLE":
        mult, stop_atr_floor = 0.75, 2.5
        reasons.append("0.75×: 5/5 in discount")
    elif urgency == "🟡 FAST SETUP":
        mult, stop_atr_floor = 0.40, 1.5
        reasons.append("0.40× scalp")
    else:
        return {"risk_pct": 0.0, "risk_mult": 0.0, "stop_atr": 2.5,
                "stop": round(float(price) * 0.94, 2), "stop_kind": "none",
                "shares": 0, "dollar_risk": 0.0, "reasons": ["no trade"],
                "why_short": "no trade"}

    osc = weekly.get("weekly_osc")
    if osc is not None:
        if float(osc) > 10:
            mult *= 1.10
            reasons.append("weekly B-X strong")
        elif float(osc) <= 0:
            mult *= 0.50
            reasons.append("weekly B-X weak — halved")

    mlabel = mode.get("mode")
    if mlabel == "COMPRESSION":
        mult *= 0.70
        reasons.append("compression: cut")
    elif mlabel == "EXPANSION":
        reasons.append("expansion")

    if comp_row is not None:
        n_ag, _ = model_agreement(comp_row)
        if n_ag >= 5:
            mult *= 1.05
            reasons.append(f"models {n_ag}/7 long")
        elif n_ag <= 3:
            mult *= 0.70
            reasons.append(f"models only {n_ag}/7 — cut")

    if comp is not None and df is not None:
        edge = ticker_edge_from_comp(comp, df["Close"])
        if edge["mult"] != 1.0:
            mult *= edge["mult"]
        if edge["note"]:
            reasons.append(edge["note"])

    bear, bull = fund.get("bearish_pct"), fund.get("bullish_pct")
    if bear is not None and float(bear) >= 60:
        mult *= 0.50
        reasons.append(f"news bearish {float(bear):.0f}% — halved")
    elif bull is not None and float(bull) >= 65:
        mult *= 1.10
        reasons.append(f"news bullish {float(bull):.0f}%")

    bcm = fund.get("book_corr_mult")
    if bcm is not None and float(bcm) < 0.99:
        mult *= float(bcm)
        if fund.get("book_corr_note"):
            reasons.append(str(fund["book_corr_note"]))

    if fund.get("high_impact_soon"):
        mult *= 0.50
        reasons.append("high-impact macro ≤48h — halved")

    mult = float(np.clip(mult, 0.25, 1.0))
    risk_pct = round(cap * mult, 3)

    st = structure_stop(price, atr, zone, stop_atr_floor=stop_atr_floor)
    stop = st["stop"]
    stop_dist = max(float(price) - stop, 1e-9)
    shares = int((account * risk_pct / 100) / stop_dist) if stop_dist > 0 else 0

    if df is not None and shares >= 1:
        try:
            cost = expected_trade_cost(df, shares, float(price))
            cost_vs_risk = (cost["expected_cost_$"] / (shares * stop_dist)
                            if shares and stop_dist else 0)
            if cost["spread_pct"] >= 0.80 or cost_vs_risk >= 0.25:
                shares = max(int(shares * 0.5), 0)
                reasons.append(
                    f"cost haircut (spread {cost['spread_pct']:.2f}%, "
                    f"cost/risk {cost_vs_risk:.0%})")
        except Exception:
            pass

    dollar_risk = round(shares * stop_dist, 2) if shares else 0.0
    reasons.append(f"stop {st['kind']}")
    return {
        "risk_pct": risk_pct,
        "risk_mult": round(mult, 3),
        "stop_atr": st["stop_atr"],
        "stop": stop,
        "stop_kind": st["kind"],
        "shares": max(shares, 0),
        "dollar_risk": dollar_risk,
        "reasons": reasons,
        "why_short": "; ".join(reasons[:4]),
    }
