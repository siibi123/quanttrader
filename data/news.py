"""News ingestion — Finnhub headlines + sentiment (NEWS_API_KEY).

Finnhub's free tier: `/company-news` (recent headlines per symbol) and
`/news-sentiment` (an aggregate bullish/bearish score + buzz). Both are
well-documented public REST endpoints (https://finnhub.io/docs/api), not
a probe-then-lock situation like the LSE vault. Stubs cleanly — empty
list / empty dict, `working = False` — when NEWS_API_KEY is unset. Never
fabricates a headline or a sentiment number.
"""
from __future__ import annotations

import time

import requests


class NewsProvider:
    BASE = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str = ""):
        self.key = api_key
        self.working = bool(api_key)

    def _get(self, path: str, params: dict):
        if not self.key:
            return None
        try:
            p = {k: v for k, v in params.items() if v is not None}
            p["token"] = self.key
            r = requests.get(f"{self.BASE}{path}", params=p, timeout=15)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def company_news(self, symbol: str, days: int = 3, limit: int = 15) -> list[dict]:
        if not self.key:
            return []
        today = time.strftime("%Y-%m-%d")
        frm = time.strftime("%Y-%m-%d", time.localtime(time.time() - days * 86400))
        rows = self._get("/company-news", {"symbol": symbol, "from": frm, "to": today})
        if not isinstance(rows, list):
            return []
        return [{"symbol": symbol, "headline": r.get("headline", ""),
                "source": r.get("source", ""), "url": r.get("url", ""),
                "ts": r.get("datetime", 0)} for r in rows[:limit]]

    def sentiment(self, symbol: str) -> dict:
        if not self.key:
            return {}
        d = self._get("/news-sentiment", {"symbol": symbol})
        if not isinstance(d, dict) or "sentiment" not in d:
            return {}
        s = d.get("sentiment") or {}
        buzz = d.get("buzz") or {}
        return {"symbol": symbol,
               "bullish_pct": round(float(s.get("bullishPercent", 0)) * 100, 1),
               "bearish_pct": round(float(s.get("bearishPercent", 0)) * 100, 1),
               "buzz_articles_week": buzz.get("articlesInLastWeek"),
               "buzz_z": buzz.get("buzz")}
