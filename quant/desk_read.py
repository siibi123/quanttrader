"""Chart-symbol desk read — attribution, walk-forward verdict, vs buy-hold.

Not a screener clone. Answers three PM questions on the name you are
looking at: who is driving the score, does the edge survive folds, and
did trading beat sitting in the name.
"""
from __future__ import annotations

import pandas as pd

from quant.signals import WEIGHTS


def attribution(comp: pd.DataFrame) -> dict:
    last = comp.iloc[-1]
    rows = []
    for name, w in WEIGHTS.items():
        raw = float(last[name]) if name in last.index else 0.0
        c = raw * w
        if raw > 0.05:
            side = "LONG"
        elif raw < -0.05:
            side = "SHORT"
        else:
            side = "FLAT"
        rows.append({"model": name, "raw": round(raw, 3), "weight": w,
                     "contrib": round(float(c), 3), "side": side})
    rows.sort(key=lambda x: abs(x["contrib"]), reverse=True)
    n_long = sum(1 for r in rows if r["side"] == "LONG")
    n_short = sum(1 for r in rows if r["side"] == "SHORT")
    leader = rows[0]
    trend = next((r for r in rows if r["model"] == "trend"), None)
    if n_long >= 5:
        crowd = "BROAD"
    elif (leader["model"] in ("volume", "rsi", "meanrev")
          and trend and trend["side"] != "LONG"
          and float(last["score"]) > 0):
        crowd = "COUNTERTREND"
    elif n_long <= 2 and float(last["score"]) > 0:
        crowd = "NARROW"
    else:
        crowd = "MIXED"
    return {
        "score": round(float(last["score"]), 3),
        "signal": str(last["signal"]),
        "leader": leader["model"],
        "crowd": crowd,
        "n_long": n_long,
        "n_short": n_short,
        "rows": rows,
        "line": (f"{last['signal']} led by {leader['model']} "
                 f"({leader['contrib']:+.2f}) · {n_long}/7 long · {crowd}"),
    }


def robustness(wf: pd.DataFrame, metrics: dict | None) -> dict:
    metrics = metrics or {}
    if wf is None or len(wf) == 0 or "Sharpe" not in wf.columns:
        return {"label": "N/A",
                "line": "Not enough history for walk-forward on this timeframe.",
                "activity": None, "n_pos": 0, "n": 0}
    sharpes = pd.to_numeric(wf["Sharpe"], errors="coerce").dropna()
    n = len(sharpes)
    last_s = float(sharpes.iloc[-1])
    n_pos = int((sharpes > 0).sum())
    if last_s <= 0 and n_pos <= 1:
        label = "FRAGILE"
        line = (f"Last fold Sharpe {last_s:.2f}; {n_pos}/{n} folds positive. "
                "The backtest does not survive the latest window — do not "
                "size this name as if the full-sample number is real.")
    elif n_pos >= max(3, n - 1) and last_s > 0:
        label = "ROBUST"
        line = (f"{n_pos}/{n} folds Sharpe>0, last fold {last_s:.2f}. "
                "Edge is not a single-regime fluke.")
    else:
        label = "UNSTABLE"
        line = (f"{n_pos}/{n} folds positive, last Sharpe {last_s:.2f}. "
                "Treat as a maybe — half size at most.")
    strat = metrics.get("CAGR %")
    bh = metrics.get("Buy&Hold CAGR %")
    activity = None
    if strat is not None and bh is not None:
        if float(strat) < float(bh):
            activity = (f"Strategy {strat}% CAGR vs sitting in the name {bh}%. "
                        "Activity lost. Stand down unless playbook is 5/5 "
                        "in the discount zone.")
        else:
            activity = (f"Strategy {strat}% CAGR vs buy-and-hold {bh}%. "
                        "Trading this name earned its keep vs doing nothing.")
    return {"label": label, "line": line, "activity": activity,
            "n_pos": n_pos, "n": n, "last_sharpe": last_s}
