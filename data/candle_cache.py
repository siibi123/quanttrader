"""In-process OHLCV cache so a 518-name cycle does not re-hit Yahoo.

TTL: 5 minutes while the US cash session is open, 30 minutes when closed
(weekend / overnight bars do not move). Callers always get a copy.
"""
from __future__ import annotations

import threading
import time

import pandas as pd

from core.state import market_status

_OPEN_TTL = 300.0
_CLOSED_TTL = 1800.0


def _ttl() -> float:
    try:
        ms = market_status()
        if isinstance(ms, dict) and ms.get("open"):
            return _OPEN_TTL
        return _CLOSED_TTL
    except Exception:
        return _OPEN_TTL


class CandleCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._rows: dict[tuple, tuple[float, pd.DataFrame]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, symbol: str, interval: str, lookback: str) -> pd.DataFrame | None:
        key = (str(symbol).upper(), str(interval), str(lookback))
        now = time.time()
        with self._lock:
            row = self._rows.get(key)
            if not row:
                self.misses += 1
                return None
            ts, df = row
            if now - ts > _ttl() or df is None or df.empty:
                self._rows.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            return df.copy()

    def put(self, symbol: str, interval: str, lookback: str, df: pd.DataFrame) -> None:
        if df is None or not len(df):
            return
        key = (str(symbol).upper(), str(interval), str(lookback))
        with self._lock:
            self._rows[key] = (time.time(), df.copy())

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._rows),
                "hits": self.hits,
                "misses": self.misses,
                "ttl_s": int(_ttl()),
            }


CACHE = CandleCache()
