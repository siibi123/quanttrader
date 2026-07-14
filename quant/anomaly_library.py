"""Curated market-anomaly reference library with real academic citations.

Two uses:
  1. ANOMALIES is static reference material for the future LLMOrchestrator's
     system prompt (alongside state.to_ai_context()) — grounding
     suggestions in documented research instead of vibes.
  2. match_anomalies() filters to whichever anomalies' trigger condition is
     actually met by TODAY's computed numbers, for right-now audit/state
     wiring via RuleOrchestrator.research() — every citation attached to a
     live symbol traces to a real number on that symbol, not a vague vibe.

Every trigger is a pure function of a context dict; a missing key just
means that anomaly can never fire (honest — never fabricates a field we
don't actually have, e.g. market cap or earnings-date data this platform
doesn't fetch yet).
"""
from __future__ import annotations

ANOMALIES: list[dict] = [
    {"name": "Momentum",
     "citation": "Jegadeesh & Titman (1993), 'Returns to Buying Winners and "
                "Selling Losers: Implications for Stock Market Efficiency', "
                "Journal of Finance",
     "finding": "3-12 month winners keep outperforming losers over the "
               "following 3-12 months.",
     "trigger": lambda ctx: ctx.get("score") is not None
                and abs(ctx["score"]) >= 0.35 and ctx.get("agree_frac", 0) >= 0.7},
    {"name": "Short-term reversal",
     "citation": "Jegadeesh (1990), 'Evidence of Predictable Behavior of "
                "Security Returns', Journal of Finance",
     "finding": "Extreme 1-2 week moves tend to partially reverse over "
               "the following weeks.",
     "trigger": lambda ctx: ctx.get("rsi2") is not None
                and (ctx["rsi2"] <= 10 or ctx["rsi2"] >= 90)},
    {"name": "Low-volatility anomaly",
     "citation": "Ang, Hodrick, Xing & Zhang (2006), 'The Cross-Section of "
                "Volatility and Expected Returns', Journal of Finance",
     "finding": "Low-volatility stocks have historically delivered better "
               "risk-adjusted returns than high-volatility ones.",
     "trigger": lambda ctx: ctx.get("ewma_ann_vol_pct") is not None
                and ctx["ewma_ann_vol_pct"] < 15},
    {"name": "Post-earnings-announcement drift (PEAD)",
     "citation": "Bernard & Thomas (1989), 'Post-Earnings-Announcement "
                "Drift: Delayed Price Response or Risk Premium?', Journal "
                "of Accounting Research",
     "finding": "Prices keep drifting in the direction of an earnings "
               "surprise for weeks after the print.",
     "trigger": lambda ctx: ctx.get("near_earnings") is True},
    {"name": "Turn-of-month effect",
     "citation": "Ariel (1987), 'A Monthly Effect in Stock Returns', "
                "Journal of Financial Economics",
     "finding": "Returns cluster around the last and first few trading "
               "days of the calendar month.",
     "trigger": lambda ctx: ctx.get("trading_day_of_month") is not None
                and (ctx["trading_day_of_month"] <= 3
                     or ctx["trading_day_of_month"] >= 19)},
    {"name": "January effect",
     "citation": "Rozeff & Kinney (1976), 'Capital Market Seasonality: The "
                "Case of Stock Returns', Journal of Financial Economics",
     "finding": "Small-cap stocks have historically outperformed in "
               "January, concentrated in the first few trading days.",
     "trigger": lambda ctx: ctx.get("month") == 1
                and ctx.get("trading_day_of_month", 99) <= 5},
    {"name": "Size effect",
     "citation": "Banz (1981), 'The Relationship Between Return and Market "
                "Value of Common Stocks', Journal of Financial Economics",
     "finding": "Small-cap stocks have historically earned higher average "
               "returns than large-caps, even risk-adjusted.",
     "trigger": lambda ctx: ctx.get("market_cap_usd") is not None
                and ctx["market_cap_usd"] < 2_000_000_000},
    {"name": "Value effect",
     "citation": "Fama & French (1992), 'The Cross-Section of Expected "
                "Stock Returns', Journal of Finance",
     "finding": "Stocks with low price-to-book ratios have historically "
               "outperformed high price-to-book 'growth' stocks.",
     "trigger": lambda ctx: ctx.get("price_to_book") is not None
                and ctx["price_to_book"] < 1.0},
    {"name": "Disposition effect (behavioral)",
     "citation": "Shefrin & Statman (1985), 'The Disposition to Sell "
                "Winners Too Early and Ride Losers Too Long: Theory and "
                "Evidence', Journal of Finance",
     "finding": "Retail investors systematically sell winners too early "
               "and hold losers too long — a bias to correct for, not "
               "trade on.",
     "trigger": lambda ctx: ctx.get("r_now") is not None
                and -1.5 <= ctx["r_now"] <= -0.5},
]


def match_anomalies(context: dict) -> list[dict]:
    """Anomalies whose trigger condition is met by today's numbers."""
    out = []
    for a in ANOMALIES:
        try:
            if a["trigger"](context):
                out.append({"name": a["name"], "citation": a["citation"],
                           "finding": a["finding"]})
        except Exception:
            continue
    return out
