"""Daily setup scanner — the whole universe through the Playbook gates.

Answers the desk's morning question: "what is tradeable TODAY?"
Every ticker gets the 5-gate check + dip-setup check; output is ranked by
actionability. This is the 'few trades each day, from all the data' engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .advanced import regime_quadrant
from .bxtrender import bxtrender
from .signals import atr, composite, rsi, sma

RISK_PROFILES = {
    "🛡️ Conservative": {"risk_pct": 1.0, "max_pos": 3, "heat_cap": 3.0,
                        "conviction_min": 55},
    "⚖️ Balanced": {"risk_pct": 1.5, "max_pos": 4, "heat_cap": 5.0,
                    "conviction_min": 50},
    "🔥 Aggressive": {"risk_pct": 2.0, "max_pos": 6, "heat_cap": 8.0,
                      "conviction_min": 45},
}


def scan_setups(data: dict[str, pd.DataFrame], account: float = 5000.0,
                risk_pct: float = 1.0) -> pd.DataFrame:
    """Light playbook pass on every ticker. Fast: no MC, no backtests."""
    rows = []
    for tkr, df in data.items():
        try:
            if len(df) < 220:
                continue
            c = df["Close"]
            price = float(c.iloc[-1])
            a = float(atr(df).iloc[-1])
            if a <= 0:
                continue
            comp = composite(df)
            sig = str(comp["signal"].iloc[-1])
            score = float(comp["score"].iloc[-1])
            bx = bxtrender(df).iloc[-1]
            s200 = float(sma(c, 200).iloc[-1])
            r2 = float(rsi(c, 2).iloc[-1])
            reg = regime_quadrant(df)

            g = [price > s200,
                 sig == "BUY",
                 float(bx["long_osc"]) > 0,
                 bool(bx["t3_rising"]),
                 "Storm" not in reg["regime"]]
            greens = sum(g)
            dip = price > s200 and r2 < 10

            if greens == 5:
                setup, urgency, rank = "TREND ENTRY", "🟢 ENTER", 0
            elif dip:
                setup, urgency, rank = "DIP SCALP", "🟡 FAST", 1
            elif greens == 4:
                setup, urgency, rank = "1 gate away", "👀 STALK", 2
            else:
                continue                      # not actionable today

            stop = price - 2.5 * a
            shares = int((account * risk_pct / 100) / (2.5 * a))
            rows.append({
                "ticker": tkr, "setup": setup, "urgency": urgency,
                "price": round(price, 2), "stop": round(stop, 2),
                "shares": shares,
                "cost $": round(shares * price, 0),
                "risk $": round(shares * 2.5 * a, 0),
                "score": round(score, 2),
                "BX": f"{bx['long_osc']:+.0f}{'↑' if bx['t3_rising'] else '↓'}",
                "RSI2": round(r2),
                "gates": f"{greens}/5",
                "_rank": rank,
            })
        except Exception:
            continue
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["_rank", "score"],
                              ascending=[True, False]).drop(columns="_rank")
    return out.reset_index(drop=True)
