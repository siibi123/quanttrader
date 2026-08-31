"""RTH interior gate — no new entries in the open auction or the close.

Yahoo daily bars do not resolve 09:31 vs 09:44. This only fires on a live
cycle during regular hours. Exits are never gated.
"""
from __future__ import annotations

from datetime import datetime

from core.state import market_status


def allow_new_entries(now: datetime | None = None) -> dict:
    ms = market_status(now)
    if ms["session"] != "open":
        return {"ok": False, "why": f"session is {ms['session']} — no new entries"}
    t = ms["et_time"]
    minutes = t.hour * 60 + t.minute
    open_m, close_m = 9 * 60 + 30, 16 * 60
    if minutes < open_m + 15:
        return {"ok": False, "why": "first 15 minutes of RTH — no new entries"}
    if minutes >= close_m - 10:
        return {"ok": False, "why": "last 10 minutes of RTH — no new entries"}
    return {"ok": True, "why": "RTH interior"}
