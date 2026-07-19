"""Execution quality — slippage per fill vs the decision price, rolled
up into a report (average slippage, worst fills, total cost drag).

Honest caveat: PaperBroker currently applies a FIXED slippage constant
(0.05%) rather than a market-condition-dependent model, so today every
fill's slippage_pct will cluster near that same fixed value — this
report is real infrastructure over real fill data, but it will only get
more informative if/when the paper broker's slippage model itself
becomes more realistic (a separate, not-yet-requested change).
"""
from __future__ import annotations

import time

import pandas as pd


def slippage_report(fills: list[dict], lookback_days: int = 7,
                    worst_n: int = 5) -> dict:
    """Report over fills in the trailing `lookback_days`. Honest empty
    result if there's nothing to report yet — never fabricates a number."""
    if not fills:
        return {"error": "no fills yet"}
    cutoff = time.time() - lookback_days * 86400
    recent = [f for f in fills if f.get("ts", 0) >= cutoff
             and f.get("decision_price") and "slippage_pct" in f]
    if not recent:
        return {"error": f"no fills with recorded slippage in the last "
                         f"{lookback_days} days (older fills predate P7d "
                         f"and were never given a decision_price)"}
    df = pd.DataFrame(recent)
    total_cost_dollars = float((df["slippage_pct"] / 100 * df["decision_price"]
                               * df["qty"]).sum())
    worst = df.sort_values("slippage_pct", ascending=False).head(worst_n)
    return {
        "n_fills": len(df),
        "lookback_days": lookback_days,
        "avg_slippage_pct": round(float(df["slippage_pct"].mean()), 4),
        "median_slippage_pct": round(float(df["slippage_pct"].median()), 4),
        "worst_slippage_pct": round(float(df["slippage_pct"].max()), 4),
        "total_cost_drag_$": round(total_cost_dollars, 2),
        "worst_fills": worst[["ts", "ticker", "side", "qty", "decision_price",
                              "price", "slippage_pct"]].to_dict("records"),
    }
