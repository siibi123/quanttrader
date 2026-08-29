"""Fund-style scorecard — what a PM looks at, not vanity P&L.

Alpha vs SPY (session), expectancy, heat (dollars to stops), hit rate.
Labeled honestly when the window is this process, not inception.
"""
from __future__ import annotations


def book_heat(positions: dict, marks: dict) -> dict:
    """$ and % of equity at risk if every stop is hit (longs)."""
    risk = 0.0
    gross = 0.0
    for t, p in (positions or {}).items():
        px = float(marks.get(t, p.get("avg_price") or 0) or p.get("avg_price") or 0)
        stop = p.get("stop")
        stop = float(stop) if stop is not None else px * 0.94
        qty = float(p.get("qty") or 0)
        risk += max(px - stop, 0.0) * qty
        gross += px * qty
    return {"heat_$": round(risk, 2), "gross_$": round(gross, 2)}


def trade_stats(fills: list[dict]) -> dict:
    sells = [f for f in fills if f.get("side") == "SELL"]
    if not sells:
        return {"n_exits": 0, "win_rate": None, "expectancy_$": None,
                "profit_factor": None, "avg_win_$": None, "avg_loss_$": None}
    wins = [f["realized"] for f in sells if f.get("realized", 0) > 0]
    losses = [f["realized"] for f in sells if f.get("realized", 0) < 0]
    gw, gl = sum(wins), -sum(losses)
    n = len(sells)
    return {
        "n_exits": n,
        "win_rate": round(len(wins) / n * 100, 1),
        "expectancy_$": round(sum(f.get("realized", 0) for f in sells) / n, 2),
        "profit_factor": (round(gw / gl, 2) if gl > 0 else (None if gw <= 0 else "inf")),
        "avg_win_$": round(sum(wins) / len(wins), 2) if wins else None,
        "avg_loss_$": round(sum(losses) / len(losses), 2) if losses else None,
    }


def vs_benchmark(book_ret_pct: float, spy_ret_pct: float | None) -> dict:
    if spy_ret_pct is None:
        return {"book_pct": round(book_ret_pct, 2), "spy_pct": None,
                "excess_pct": None}
    return {"book_pct": round(book_ret_pct, 2),
            "spy_pct": round(spy_ret_pct, 2),
            "excess_pct": round(book_ret_pct - spy_ret_pct, 2)}
