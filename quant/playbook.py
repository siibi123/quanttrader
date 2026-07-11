"""Playbook — the WHEN engine. One panel that answers, at all times:
enter now? wait for what? holding — do what? exit now — why?

It runs a checklist of gates (the same ones the backtest engine trades),
shows which are green and which are blocking, and produces ONE instruction
with an urgency level. This is the terminal's 'what do I do' function.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .advanced import regime_quadrant, support_resistance
from .bxtrender import bxtrender
from .levels import fib_levels
from .signals import atr, composite, rsi, sma


def build_playbook(df: pd.DataFrame, account: float = 5000.0,
                   risk_pct: float = 1.0,
                   in_position: bool = False,
                   entry: float | None = None,
                   stop: float | None = None) -> dict:
    c = df["Close"]
    price = float(c.iloc[-1])
    a = float(atr(df).iloc[-1])
    comp = composite(df)
    score = float(comp["score"].iloc[-1])
    sig = str(comp["signal"].iloc[-1])
    bx = bxtrender(df).iloc[-1]
    s200 = float(sma(c, 200).iloc[-1]) if len(c) >= 200 else np.nan
    r2 = float(rsi(c, 2).iloc[-1])
    reg = regime_quadrant(df)
    fib = fib_levels(df)
    sr = support_resistance(df)

    # ---------------- ENTRY GATES ----------------
    gates = [
        ("Price above 200-bar average (regime)",
         not np.isnan(s200) and price > s200,
         f"price ${price:,.2f} vs ${s200:,.2f}" if not np.isnan(s200) else "n/a"),
        ("Composite signal is BUY",
         sig == "BUY", f"score {score:+.2f} → {sig}"),
        ("B-Xtrender long oscillator positive",
         float(bx["long_osc"]) > 0, f"{bx['long_osc']:+.0f}"),
        ("B-Xtrender T3 rising",
         bool(bx["t3_rising"]), "rising" if bx["t3_rising"] else "falling"),
        ("Volatility regime not a storm",
         "Storm" not in reg["regime"], reg["regime"]),
    ]
    dip_setup = (not np.isnan(s200) and price > s200 and r2 < 10)
    greens = sum(1 for _, ok, _ in gates if ok)

    # levels for a fresh entry
    stop_new = price - 2.5 * a
    shares = int((account * risk_pct / 100) / (2.5 * a)) if a > 0 else 0
    scale1 = price + 2.5 * a
    scale2 = price + 5.0 * a

    # ---------------- DECISION ----------------
    if in_position and entry and stop:
        r_unit = abs(entry - stop) if abs(entry - stop) > 1e-9 else 2.5 * a
        r_now = (price - entry) / r_unit
        trail = price - 2.5 * a
        actions = []
        urgency = "🟢 CALM"
        if price <= stop:
            instruction = f"EXIT NOW — stop ${stop:,.2f} violated"
            urgency = "🔴 IMMEDIATE"
        elif sig == "SELL":
            instruction = "EXIT — composite flipped to SELL"
            urgency = "🟠 TODAY"
        elif float(bx["long_osc"]) < 0 and not bx["t3_rising"]:
            instruction = ("TIGHTEN — B-X turned fully negative; "
                           f"raise stop to ${max(stop, trail):,.2f}")
            urgency = "🟠 TODAY"
        elif r_now >= 2 :
            instruction = (f"SCALE — trade is +{r_now:.1f}R: bank ⅓, "
                           f"stop to ${entry + r_unit:,.2f} (entry+1R)")
            urgency = "🟡 SOON"
        elif r_now >= 1:
            instruction = (f"PROTECT — +{r_now:.1f}R: stop to breakeven "
                           f"${entry:,.2f}; consider first scale at "
                           f"${entry + 2 * r_unit:,.2f}")
            urgency = "🟡 SOON"
        else:
            instruction = (f"HOLD — {r_now:+.1f}R · stop ${stop:,.2f} · "
                           f"let the setup work")
        return {"mode": "MANAGE", "instruction": instruction,
                "urgency": urgency, "r_now": round(r_now, 2),
                "gates": gates, "greens": greens,
                "trail_suggestion": round(trail, 2),
                "regime": reg["regime"]}

    # flat: enter, stalk or stand down
    if greens == 5:
        instruction = (f"ENTER — all 5 gates green: buy {shares} shares "
                       f"≈ ${price:,.2f}, stop ${stop_new:,.2f}, "
                       f"scale ⅓ at ${scale1:,.2f} and ${scale2:,.2f}")
        urgency = "🟢 ACTIONABLE"
    elif dip_setup:
        pocket = (fib["levels"]["0.618"] if fib else None)
        instruction = (f"DIP SETUP — RSI2={r2:.0f} panic in an uptrend: "
                       f"scalp entry ≈ ${price:,.2f}, stop ${stop_new:,.2f},"
                       f" exit on RSI2 > 65"
                       + (f" · golden pocket ${pocket:,.2f}" if pocket else ""))
        urgency = "🟡 FAST SETUP"
    elif greens >= 3:
        missing = [name for name, ok, _ in gates if not ok]
        instruction = ("STALK — close but blocked by: " +
                       "; ".join(missing[:2]) +
                       ". Set an alert, don't force it.")
        urgency = "🟡 WATCH"
    else:
        instruction = (f"STAND DOWN — only {greens}/5 gates green. "
                       "No setup exists; capital preservation is the trade.")
        urgency = "⚪ NO TRADE"

    nearest_sup = max((lv["price"] for lv in sr if lv["price"] < price),
                      default=None)
    nearest_res = min((lv["price"] for lv in sr if lv["price"] > price),
                      default=None)
    return {"mode": "ENTRY", "instruction": instruction, "urgency": urgency,
            "gates": gates, "greens": greens,
            "plan": {"shares": shares, "entry": round(price, 2),
                     "stop": round(stop_new, 2),
                     "scale1": round(scale1, 2), "scale2": round(scale2, 2)},
            "nearest_support": nearest_sup, "nearest_resistance": nearest_res,
            "regime": reg["regime"]}
