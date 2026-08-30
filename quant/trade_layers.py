"""Three-layer closed-trade writeup.

Layer A — facts we actually have (no 'check the news' homework).
Layer B — measurement vs the name and vs SPY when SPY exists.
Layer C — graded attribution. A stop that fires is hygiene, not a win.
"""
from __future__ import annotations


def closed_trade_layers(fill: dict, *, entry_fill: dict | None = None,
                        news_headline: str | None = None,
                        spy_ret_pct: float | None = None) -> dict:
    ticker = fill.get("ticker", "?")
    side = fill.get("side", "")
    qty = fill.get("qty", 0)
    exit_px = float(fill.get("price") or 0)
    realized = float(fill.get("realized") or 0)
    reason = str(fill.get("reason") or "").strip()
    entry_px = float((entry_fill or {}).get("price") or 0)
    entry_why = str((entry_fill or {}).get("reason") or "").strip()

    facts = [f"{ticker} {side} {qty} @ ${exit_px:,.2f}",
             f"realized ${realized:+,.2f}"]
    if reason:
        facts.append(f"exit log: {reason}")
    if news_headline:
        facts.append(f"headline: {news_headline}")

    measurement: list[str] = []
    price_chg = None
    if entry_px > 0 and exit_px > 0:
        price_chg = (exit_px / entry_px - 1) * 100
        measurement.append(f"price change {price_chg:+.2f}% "
                           f"(entry ${entry_px:,.2f} → ${exit_px:,.2f})")
    if spy_ret_pct is not None and price_chg is not None:
        measurement.append(
            f"SPY session {spy_ret_pct:+.2f}% · excess "
            f"{price_chg - spy_ret_pct:+.2f}pp")
    if entry_why:
        measurement.append(f"entry thesis: {entry_why[:200]}")

    rlow, elow = reason.lower(), entry_why.lower()
    chased = any(w in elow for w in ("premium", "overpay", "chase", "don't overpay"))
    if realized < 0 and chased:
        grade, note = "LOW", (
            "Lost after a premium/chase entry. The thesis was the error, "
            "not the stop.")
    elif realized < 0 and any(w in rlow for w in
                              ("stop", "exit now", "violated", "exit —")):
        grade, note = "HYGIENE", (
            "Stop contained the loss. That is risk working — not evidence "
            "the model was right.")
    elif realized > 0 and any(w in elow for w in ("5/5", "buy zone", "enter")):
        grade, note = "MEDIUM", (
            "Won with a stated playbook entry. One ticket is not a proof.")
    elif realized > 0:
        grade, note = "MEDIUM", "Profitable. Not enough to call the model correct."
    elif realized == 0:
        grade, note = "N/A", "Flat round-trip."
    else:
        grade, note = "LOW", "Loss without a clean stop/thesis match."

    return {"ticker": ticker, "realized": realized, "facts": facts,
            "measurement": measurement, "grade": grade, "note": note}


def pair_exits(fills: list[dict]) -> list[tuple[dict, dict | None]]:
    """Match each SELL to the most recent prior BUY of the same ticker."""
    last_buy: dict[str, dict] = {}
    out = []
    for f in sorted(fills, key=lambda x: x.get("ts") or 0):
        t, side = f.get("ticker"), f.get("side")
        if side == "BUY":
            last_buy[t] = f
        elif side == "SELL":
            out.append((f, last_buy.get(t)))
    return out
