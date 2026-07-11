"""Verdict engine — turns all signals into ONE trading decision.

Philosophy (how an actual quant desk thinks):
1. Signal strength alone is not enough — models must AGREE.
2. A signal that never worked on this ticker historically deserves no trust.
3. High-volatility regimes kill edges — stand aside.
4. No trade without risk/reward: defined stop, defined target, RR >= 1.3.
5. "NO TRADE" is the default. A trade must EARN its conviction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BTConfig, run_backtest
from .signals import BUY_TH, SELL_TH, atr, composite, vol_regime

MODELS = ["trend", "momentum", "bxtrender", "macd", "rsi", "meanrev", "volume"]


def analyze(df: pd.DataFrame, account: float = 5000.0,
            risk_pct: float = 1.0, skew: float | None = None,
            flow_call_share: float | None = None) -> dict:
    """Full desk-style analysis of one ticker. Returns a verdict dict."""
    comp = composite(df)
    last = comp.iloc[-1]
    score = float(last["score"])
    price = float(df["Close"].iloc[-1])
    a = float(atr(df).iloc[-1])
    regime = float(vol_regime(df).iloc[-1])

    direction = 1 if score >= BUY_TH else (-1 if score <= SELL_TH else 0)

    # --- 1. Model agreement ------------------------------------------------
    signs = np.sign([float(last[m]) for m in MODELS])
    agree = int((signs == direction).sum()) if direction != 0 else 0
    agree_frac = agree / len(MODELS)

    # --- 2. Historical edge on THIS ticker ---------------------------------
    sharpe = 0.0
    n_trades = 0
    try:
        bt = run_backtest(df, BTConfig(starting_cash=account,
                                       risk_per_trade=risk_pct / 100))
        sharpe = float(bt.metrics["Sharpe"])
        n_trades = int(bt.metrics["Trades"])
    except Exception:
        pass
    edge_ok = sharpe > 0.3 and n_trades >= 5

    # --- 3. Levels & risk/reward -------------------------------------------
    look = min(63, len(df) - 1)
    if direction >= 0:
        stop = price - 2.5 * a
        risk_dist = price - stop
        swing = float(df["High"].rolling(look).max().iloc[-1])
        # At/near new highs there is no overhead resistance — use a
        # measured-move target (2R) instead of punishing the breakout.
        target = max(swing, price + 2.0 * risk_dist)
    else:
        stop = price + 2.5 * a
        risk_dist = stop - price
        swing = float(df["Low"].rolling(look).min().iloc[-1])
        target = min(swing, price - 2.0 * risk_dist)
    risk = abs(price - stop)
    reward = abs(target - price)
    rr = round(reward / risk, 2) if risk > 0 else 0.0

    # --- 4. Options sentiment (optional) ------------------------------------
    skew_adj = 0.0
    if skew is not None:
        # heavy put skew argues against fresh longs / supports shorts
        if direction == 1 and skew > 8:
            skew_adj = -8.0
        elif direction == -1 and skew > 8:
            skew_adj = +5.0
        elif direction == 1 and skew < 2:
            skew_adj = +3.0

    # --- 4b. Unusual-flow tilt (live options positioning) ---------------------
    flow_adj = 0.0
    if flow_call_share is not None:
        if direction == 1 and flow_call_share >= 0.65:
            flow_adj = +5.0
        elif direction == 1 and flow_call_share <= 0.35:
            flow_adj = -5.0
        elif direction == -1 and flow_call_share <= 0.35:
            flow_adj = +5.0

    # --- 5. Conviction (0-100) ----------------------------------------------
    conviction = 100 * (
        0.40 * min(abs(score) / 0.50, 1.0)      # signal strength
        + 0.25 * agree_frac                     # model agreement
        + 0.15 * regime                         # calm regime
        + 0.20 * min(max(sharpe, 0) / 1.2, 1.0) # proven edge here
    ) + skew_adj + flow_adj
    conviction = float(np.clip(conviction, 0, 100))

    # --- 6. Verdict ----------------------------------------------------------
    reasons_pro, reasons_con = [], []

    if direction == 1:
        reasons_pro.append(f"Composite score {score:+.2f} above BUY threshold")
    elif direction == -1:
        reasons_pro.append(f"Composite score {score:+.2f} below SELL threshold")
    else:
        reasons_con.append(f"Composite score {score:+.2f} is in the dead zone "
                           f"({SELL_TH} to {BUY_TH}) — no directional edge")

    if direction != 0:
        if agree >= 5:
            reasons_pro.append(f"{agree}/{len(MODELS)} models agree on direction")
        else:
            reasons_con.append(f"Only {agree}/{len(MODELS)} models agree — mixed signals")

    if regime >= 1.0:
        reasons_pro.append("Calm volatility regime — edges work best here")
    elif regime >= 0.5:
        reasons_con.append("Elevated volatility — position sizes should shrink")
    else:
        reasons_con.append("Volatility storm — historically the worst time to trade signals")

    if edge_ok:
        reasons_pro.append(f"Signal has real history on this ticker "
                           f"(Sharpe {sharpe:.2f}, {n_trades} trades)")
    else:
        reasons_con.append(f"Weak historical edge on this ticker "
                           f"(Sharpe {sharpe:.2f}, {n_trades} trades)")

    if rr >= 1.8:
        reasons_pro.append(f"Attractive risk/reward {rr}:1 to the {look}-day level")
    elif rr >= 1.3:
        reasons_pro.append(f"Acceptable risk/reward {rr}:1")
    else:
        reasons_con.append(f"Poor risk/reward {rr}:1 — target too close to stop")

    if flow_call_share is not None and abs(flow_adj) > 0:
        (reasons_pro if flow_adj > 0 else reasons_con).append(
            f"Unusual options flow: {flow_call_share*100:.0f}% of fresh premium "
            f"in calls ({flow_adj:+.0f} conviction)")
    if skew is not None and abs(skew_adj) > 0:
        (reasons_pro if skew_adj > 0 else reasons_con).append(
            f"Options skew {skew:+.1f} pts adjusts conviction {skew_adj:+.0f}")

    tradeable = (direction != 0 and conviction >= 55 and rr >= 1.3
                 and regime > 0.25)
    if tradeable:
        verdict = "LONG" if direction == 1 else "SHORT"
    else:
        verdict = "NO TRADE"

    # --- 7. Sizing ------------------------------------------------------------
    risk_dollars = account * risk_pct / 100
    shares = int(min(risk_dollars / risk, account / price)) if risk > 0 else 0

    return {
        "verdict": verdict,
        "conviction": round(conviction),
        "score": round(score, 3),
        "price": round(price, 2),
        "entry": round(price, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "rr": rr,
        "atr": round(a, 2),
        "regime": regime,
        "agree": agree,
        "sharpe": round(sharpe, 2),
        "n_trades": n_trades,
        "shares": shares,
        "risk_dollars": round(shares * risk, 0),
        "reasons_pro": reasons_pro,
        "reasons_con": reasons_con,
    }
