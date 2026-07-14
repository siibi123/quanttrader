"""Sector & target engine — the WHAT-TO-BUY-TODAY layer.

Ranks sectors first (is this group worth exposure today), then names
within each sector. The base score is quant.verdict's already-computed
technical conviction (momentum, model agreement, proven edge, regime,
risk/reward) — this module only ADDS tilts on top: news sentiment,
recent large options prints, and a simple macro-rate-trend tilt. Every
ranked name keeps verdict's own entry/stop/target/rr and reasons, plus
the tilt reasons that moved its score — never a bare number with no why.
Names that don't clear a tradeable verdict, or whose tilts actively
fight the technical call, land in the avoid list instead of the ranking.
"""
from __future__ import annotations

import pandas as pd

from .advanced import support_resistance
from .verdict import analyze as verdict_analyze

MACRO_TILT_PTS = 3.0
SENTIMENT_TILT_PTS = 8.0
FLOW_TILT_PTS = 5.0
FLOW_CONFLUENCE_TILT_PTS = 6.0


def score_name(df: pd.DataFrame, account: float = 5000.0, risk_pct: float = 1.0,
               sentiment: dict | None = None, flow: dict | None = None,
               macro_trend: str | None = None,
               flow_confluence: dict | None = None) -> dict:
    """One ticker's verdict + tilts -> one target-engine score."""
    v = verdict_analyze(df, account=account, risk_pct=risk_pct)
    direction = 1 if v["verdict"] == "LONG" else (-1 if v["verdict"] == "SHORT" else 0)
    tilt = 0.0
    tilt_reasons: list[str] = []

    if sentiment and direction != 0:
        b, r = sentiment.get("bullish_pct"), sentiment.get("bearish_pct")
        if b is not None and r is not None:
            s = (b - r) / 100                                  # -1..+1
            adj = SENTIMENT_TILT_PTS * s * direction
            if abs(adj) >= 1.5:
                tilt += adj
                tilt_reasons.append(f"news sentiment {b:.0f}%/{r:.0f}% "
                                    f"bull/bear ({adj:+.1f})")

    if flow and direction != 0:
        n_prints = len(flow.get("prints", []))
        if n_prints:
            adj = FLOW_TILT_PTS * direction
            tilt += adj
            tilt_reasons.append(f"{n_prints} large option print(s) recently "
                               f"({adj:+.1f})")

    if macro_trend and direction != 0:
        # simple, well-known heuristic: rate hikes are a headwind for risk
        # assets broadly, cuts a tailwind — not a precise model, a tilt.
        macro_dir = -1 if macro_trend == "up" else (1 if macro_trend == "down" else 0)
        if macro_dir:
            adj = MACRO_TILT_PTS * macro_dir * direction
            tilt += adj
            tilt_reasons.append(f"macro rate trend {macro_trend} ({adj:+.1f})")

    if flow_confluence and direction != 0:
        fc_verdict = flow_confluence.get("verdict")
        agree = ((fc_verdict == "CONFLUENCE LONG" and direction == 1)
                or (fc_verdict == "CONFLUENCE SHORT" and direction == -1))
        disagree = ((fc_verdict == "CONFLUENCE LONG" and direction == -1)
                   or (fc_verdict == "CONFLUENCE SHORT" and direction == 1)
                   or fc_verdict == "CONFLICT")
        if agree:
            tilt += FLOW_CONFLUENCE_TILT_PTS
            tilt_reasons.append(f"flow confluence {fc_verdict} agrees "
                               f"({FLOW_CONFLUENCE_TILT_PTS:+.1f})")
        elif disagree:
            tilt -= FLOW_CONFLUENCE_TILT_PTS
            tilt_reasons.append(f"flow confluence {fc_verdict} disagrees "
                               f"({-FLOW_CONFLUENCE_TILT_PTS:+.1f})")

    target_score = round(min(max(v["conviction"] + tilt, 0), 100), 1)
    conflict = bool(tilt_reasons) and (tilt < 0) and direction != 0

    sr = support_resistance(df)
    price = float(df["Close"].iloc[-1])
    avoid_above = min((lv["price"] for lv in sr if lv["price"] > price), default=None)
    avoid_below = max((lv["price"] for lv in sr if lv["price"] < price), default=None)

    return {**v, "target_score": target_score, "tilt_reasons": tilt_reasons,
           "conflict": conflict,
           "avoid_above": round(avoid_above, 2) if avoid_above else None,
           "avoid_below": round(avoid_below, 2) if avoid_below else None}


def rank_sectors_and_names(data: dict[str, pd.DataFrame], sectors: dict[str, str],
                          account: float = 5000.0, risk_pct: float = 1.0,
                          sentiment_by_ticker: dict | None = None,
                          flow_by_ticker: dict | None = None,
                          macro_trend: str | None = None,
                          flow_confluence_by_ticker: dict | None = None) -> dict:
    """Rank sectors (by average target_score of their tradeable names),
    then names within each sector. NO TRADE verdicts and tilt-conflicted
    names go to `avoid` instead of the ranking."""
    sentiment_by_ticker = sentiment_by_ticker or {}
    flow_by_ticker = flow_by_ticker or {}
    flow_confluence_by_ticker = flow_confluence_by_ticker or {}
    names, avoid = [], []

    for tkr, df in data.items():
        if len(df) < 220:
            continue
        try:
            s = score_name(df, account=account, risk_pct=risk_pct,
                          sentiment=sentiment_by_ticker.get(tkr),
                          flow=flow_by_ticker.get(tkr), macro_trend=macro_trend,
                          flow_confluence=flow_confluence_by_ticker.get(tkr))
        except Exception:
            continue
        s["ticker"] = tkr
        s["sector"] = sectors.get(tkr, "Unclassified")
        if s["verdict"] == "NO TRADE":
            avoid.append({"ticker": tkr, "sector": s["sector"],
                         "reason": "; ".join(s["reasons_con"][:2]) or "no edge today"})
        elif s["conflict"]:
            avoid.append({"ticker": tkr, "sector": s["sector"],
                         "reason": f"{s['verdict']} technically, but "
                                  + "; ".join(s["tilt_reasons"])})
            names.append(s)                    # still shown ranked, flagged
        else:
            names.append(s)

    sec_rows: dict[str, dict] = {}
    for n in names:
        d = sec_rows.setdefault(n["sector"], {"sector": n["sector"],
                                              "n_names": 0, "scores": []})
        d["n_names"] += 1
        d["scores"].append(n["target_score"])
    sector_ranked = sorted(
        [{"sector": d["sector"], "n_names": d["n_names"],
          "avg_target_score": round(sum(d["scores"]) / len(d["scores"]), 1)}
         for d in sec_rows.values()],
        key=lambda x: -x["avg_target_score"])
    names_ranked = sorted(names, key=lambda x: -x["target_score"])
    return {"sectors": sector_ranked, "names": names_ranked, "avoid": avoid,
           "n_scanned": len(names) + len(avoid)}
