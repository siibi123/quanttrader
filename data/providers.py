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
    """London Strategic Edge — VERIFIED contract (extracted from their
    official SDK, github.com/londonstrategicedge/lse-data v0.14.0):

      GET https://api.londonstrategicedge.com/vault/candles
          ?symbol=AAPL&timeframe=1d&limit=5000&order=asc[&start=&end=]
      headers: x-api-key: <key>, User-Agent: <custom>  (their CDN blocks
      the default Python UA — requests' UA is fine, we set ours anyway)

      Rows: {ts|timestamp, open, high, low, close, volume?} — bar-open time.
      Timeframes: 1s 5s 15s 30s 1m 3m 5m 15m 30m 1h 4h 1d 1w 1mo.
      Options chain w/ greeks: GET /vault/options/chain?underlying=...
      WebSocket (verified to exist): wss://data-ws.londonstrategicedge.com
    """
    name = "lse"
    VAULT = "https://api.londonstrategicedge.com/vault"
    # Verified to exist in their SDK; NOT used yet — polling stays primary
    # until a StreamingFeed phase (CLAUDE.md roadmap #7) proves lifecycle
    # safety on our hosting. Do not wire without owner sign-off.
    WS_URL_ROADMAP = "wss://data-ws.londonstrategicedge.com"
    UA = "quanttrader (+https://github.com/siibi123/quanttrader)"
    TF_MAP = {"1h": "1h", "1d": "1d", "1wk": "1w", "1w": "1w",
              "1mo": "1mo", "1m": "1m", "5m": "5m", "15m": "15m",
              "4h": "4h"}

    def __init__(self, api_key: str, base_url: str = ""):
        self.key = api_key
        self.base = (base_url or self.VAULT).rstrip("/")
        self.working = bool(api_key)      # verified contract; key = enabled

    def _get(self, path: str, params: dict) -> list | dict | None:
        if not self.key:
            return None
        try:
            r = requests.get(f"{self.base}{path}",
                             params={k: v for k, v in params.items()
                                     if v is not None},
                             headers={"x-api-key": self.key,
                                      "User-Agent": self.UA},
                             timeout=30)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def get_candles(self, symbol, interval="1d", lookback="2y"):
        tf = self.TF_MAP.get(interval, "1d")
        rows = self._get("/candles", {"symbol": symbol, "timeframe": tf,
                                      "limit": 5000, "order": "desc"})
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        tcol = "timestamp" if "timestamp" in df.columns else             ("ts" if "ts" in df.columns else None)
        need = {"open", "high", "low", "close"}
        if not tcol or not need.issubset(df.columns):
            return pd.DataFrame()
        df["volume"] = df.get("volume", 0.0)
        ts = df[tcol]
        idx = pd.to_datetime(ts, unit="s", errors="coerce") \
            if pd.api.types.is_numeric_dtype(ts) else \
            pd.to_datetime(ts, errors="coerce")
        df.index = pd.DatetimeIndex(idx).tz_localize(None)
        df = df.rename(columns={"open": "Open", "high": "High",
                                "low": "Low", "close": "Close",
                                "volume": "Volume"})
        return df[["Open", "High", "Low", "Close",
                   "Volume"]].astype(float).dropna().sort_index()

    def get_quote(self, symbol):
        df = self.get_candles(symbol, "1m")
        if df.empty:
            df = self.get_candles(symbol, "1d")
        if df.empty:
            return {}
        px = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else px
        return {"symbol": symbol, "price": px,
                "chg_pct": round((px / prev - 1) * 100, 2)}

    def options_chain(self, underlying: str, max_dte: int | None = 45
                      ) -> pd.DataFrame:
        """Current chain, one row per contract, WITH iv and greeks."""
        rows = self._get("/options/chain",
                         {"underlying": underlying, "limit": 5000,
                          "max_dte": max_dte})
        return pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()

    def usage(self) -> dict:
        return self._get("/usage", {}) or {}


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
