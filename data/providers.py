"""Data layer — provider abstraction with honest fallbacks.

DataProvider is the contract; the rest of the platform never knows or cares
where bars come from. Chain: LSE (free key, if verified) → Yahoo (fallback)
→ Fake (tests). PollingFeed publishes ticks to the bus on a background
thread. NOTE on streaming: LSE WebSockets are NOT verified to exist; when/if
their docs confirm a WS endpoint, a StreamingFeed drops in beside
PollingFeed without touching anything else — that seam is the point of this
design.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import requests

from core.state import Event, EventBus, GlobalState


class DataProvider(ABC):
    name = "base"

    @abstractmethod
    def get_candles(self, symbol: str, interval: str = "1d",
                    lookback: str = "2y") -> pd.DataFrame: ...

    @abstractmethod
    def get_quote(self, symbol: str) -> dict: ...


# ---------------------------------------------------------------------------

class YahooProvider(DataProvider):
    name = "yahoo"

    def get_candles(self, symbol, interval="1d", lookback="2y"):
        import yfinance as yf
        for attempt in range(3):
            try:
                df = yf.Ticker(symbol).history(period=lookback,
                                               interval=interval,
                                               auto_adjust=False)
                if not df.empty:
                    df = df.rename(columns=str.title)
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    return df[["Open", "High", "Low", "Close",
                               "Volume"]].dropna()
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        return pd.DataFrame()

    def get_quote(self, symbol):
        import yfinance as yf
        try:
            fi = yf.Ticker(symbol).fast_info
            px = float(fi["last_price"])
            prev = float(fi.get("previous_close") or px)
            return {"symbol": symbol, "price": px,
                    "chg_pct": round((px / prev - 1) * 100, 2)}
        except Exception:
            return {}


class LSEProvider(DataProvider):
    """London Strategic Edge — free-key API. Endpoint pattern is DISCOVERED
    via probe (their docs are JS-rendered), then locked. Never guesses in
    production: unconfigured/unprobed -> empty results -> chain falls back.
    """
    name = "lse"
    CANDLE_PATTERNS = [
        "{base}/candles?symbol={sym}&resolution={res}&api_key={key}",
        "{base}/candles/{sym}?resolution={res}&api_key={key}",
        "{base}/v1/candles?symbol={sym}&resolution={res}&api_key={key}",
        "{base}/history?symbol={sym}&resolution={res}&api_key={key}",
    ]
    HEADER_STYLES = [{}, {"Authorization": "Bearer {key}"},
                     {"X-API-Key": "{key}"}]

    def __init__(self, api_key: str, base_url: str = ""):
        self.key = api_key
        self.bases = ([base_url] if base_url else []) + [
            "https://api.londonstrategicedge.com",
            "https://londonstrategicedge.com/api",
        ]
        self.working: dict | None = None

    def probe(self, symbol="AAPL") -> list[dict]:
        results = []
        for base in self.bases:
            for pat in self.CANDLE_PATTERNS:
                for hs in self.HEADER_STYLES:
                    url = pat.format(base=base.rstrip("/"), sym=symbol,
                                     res="1d",
                                     key=self.key if not hs else "")
                    h = {k: v.format(key=self.key) for k, v in hs.items()}
                    try:
                        r = requests.get(url, headers=h, timeout=8)
                        ok = r.status_code == 200
                    except Exception:
                        ok, r = False, None
                    results.append({"url": url.replace(self.key, "***"),
                                    "ok": ok,
                                    "status": r.status_code if r else None})
                    if ok:
                        self.working = {"pattern": pat, "base": base,
                                        "headers": hs}
                        return results
        return results

    @staticmethod
    def _normalize(j) -> pd.DataFrame:
        rows = None
        if isinstance(j, dict):
            if all(k in j for k in ("t", "o", "h", "l", "c")):
                df = pd.DataFrame({"Open": j["o"], "High": j["h"],
                                   "Low": j["l"], "Close": j["c"],
                                   "Volume": j.get("v", [0]*len(j["t"]))})
                ts = pd.Series(j["t"])
                idx = pd.to_datetime(ts, unit="s", errors="coerce") \
                    if pd.api.types.is_numeric_dtype(ts) and ts.max() > 1e9 \
                    else pd.to_datetime(ts, errors="coerce")
                df.index = pd.DatetimeIndex(idx).tz_localize(None)
                return df.astype(float).dropna().sort_index()
            for k in ("candles", "data", "results", "bars"):
                if isinstance(j.get(k), list):
                    rows = j[k]
                    break
        elif isinstance(j, list):
            rows = j
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        cmap = {}
        for want, alts in {"Open": ("open", "o"), "High": ("high", "h"),
                           "Low": ("low", "l"), "Close": ("close", "c"),
                           "Volume": ("volume", "v")}.items():
            for a in alts:
                if a in df.columns:
                    cmap[a] = want
                    break
        tcol = next((c for c in ("time", "t", "ts", "timestamp", "date")
                     if c in df.columns), None)
        if not tcol or len(cmap) < 4:
            return pd.DataFrame()
        df = df.rename(columns=cmap)
        ts = df[tcol]
        idx = pd.to_datetime(ts, unit="s", errors="coerce") \
            if pd.api.types.is_numeric_dtype(ts) and ts.max() > 1e9 \
            else pd.to_datetime(ts, errors="coerce")
        df.index = pd.DatetimeIndex(idx).tz_localize(None)
        if "Volume" not in df:
            df["Volume"] = 0.0
        return df[["Open", "High", "Low", "Close",
                   "Volume"]].astype(float).dropna().sort_index()

    def get_candles(self, symbol, interval="1d", lookback="2y"):
        if not self.working or not self.key:
            return pd.DataFrame()
        w = self.working
        url = w["pattern"].format(base=w["base"].rstrip("/"), sym=symbol,
                                  res=interval,
                                  key=self.key if not w["headers"] else "")
        h = {k: v.format(key=self.key) for k, v in w["headers"].items()}
        try:
            r = requests.get(url, headers=h, timeout=12)
            return self._normalize(r.json()) if r.status_code == 200 \
                else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def get_quote(self, symbol):
        df = self.get_candles(symbol, "1d")
        if df.empty:
            return {}
        px = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else px
        return {"symbol": symbol, "price": px,
                "chg_pct": round((px / prev - 1) * 100, 2)}


class FakeProvider(DataProvider):
    """Deterministic GBM data for tests — no network."""
    name = "fake"

    def __init__(self, mu=0.0008, vol=0.012, seed=7):
        self.mu, self.vol, self.seed = mu, vol, seed
        self._tick = 0

    def get_candles(self, symbol, interval="1d", lookback="2y"):
        n = 500
        rng = np.random.default_rng(self.seed + hash(symbol) % 1000)
        close = 100 * np.exp(np.cumsum(rng.normal(self.mu, self.vol, n)))
        return pd.DataFrame(
            {"Open": close, "High": close * 1.005, "Low": close * 0.995,
             "Close": close, "Volume": 1e6},
            index=pd.bdate_range("2024-01-01", periods=n))

    def get_quote(self, symbol):
        self._tick += 1
        base = float(self.get_candles(symbol)["Close"].iloc[-1])
        return {"symbol": symbol,
                "price": round(base * (1 + 0.001 * (self._tick % 5 - 2)), 2),
                "chg_pct": 0.1 * (self._tick % 5 - 2)}


class CompositeProvider(DataProvider):
    """Ordered fallback chain; records which provider actually served."""
    name = "composite"

    def __init__(self, providers: list[DataProvider],
                 state: GlobalState | None = None):
        self.providers, self._state = providers, state

    def get_candles(self, symbol, interval="1d", lookback="2y"):
        for p in self.providers:
            df = p.get_candles(symbol, interval, lookback)
            if len(df):
                if self._state:
                    self._state.set(f"feed.served_by.{symbol}", p.name,
                                    source="data")
                return df
        return pd.DataFrame()

    def get_quote(self, symbol):
        for p in self.providers:
            q = p.get_quote(symbol)
            if q:
                return q
        return {}


# ---------------------------------------------------------------------------

class PollingFeed:
    """Background thread: poll quotes -> publish 'tick' events -> state.

    Honest platform note: on Streamlit Community Cloud this runs while the
    app process is awake. On a VPS (Hetzner phase) it runs 24/7 unchanged.
    """

    def __init__(self, bus: EventBus, state: GlobalState,
                 provider: DataProvider, symbols: list[str],
                 interval_s: int = 30):
        self._bus, self._state, self._provider = bus, state, provider
        self.symbols = symbols
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._state.set("feed.status",
                        {"running": True, "symbols": self.symbols,
                         "interval_s": self.interval_s}, source="feed")

    def stop(self):
        self._stop.set()
        self._state.set("feed.status", {"running": False}, source="feed")

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive()
                    and not self._stop.is_set())

    def _run(self):
        while not self._stop.is_set():
            for s in list(self.symbols):
                if self._stop.is_set():
                    break
                q = self._provider.get_quote(s)
                if q:
                    self._state.set(f"quotes.{s}", q, source="feed")
                    self._bus.publish(Event("tick", q, source="feed"))
            self._stop.wait(self.interval_s)
