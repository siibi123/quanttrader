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

from core.state import Event, EventBus, GlobalState, market_status
from data.candle_cache import CACHE as _CANDLE_CACHE


class DataProvider(ABC):
    name = "base"

    @abstractmethod
    def get_candles(self, symbol: str, interval: str = "1d",
                    lookback: str = "2y") -> pd.DataFrame: ...

    @abstractmethod
    def get_quote(self, symbol: str) -> dict: ...


def _lookback_days(lookback: str) -> int:
    """Parse a yfinance-style period string ("2y", "720d", "6mo", "10y",
    "max") into an approximate day count, for client-side trimming."""
    s = (lookback or "2y").strip().lower()
    if s in ("max", ""):
        return 36_500                      # ~100y, effectively unbounded
    digits = "".join(c for c in s if c.isdigit())
    n = int(digits) if digits else 0
    if s.endswith("mo"):
        return n * 31
    if s.endswith("wk"):
        return n * 7
    if s.endswith("d"):
        return n
    if s.endswith("y"):
        return n * 365
    return n or 730


def _trim_to_lookback(df: pd.DataFrame, lookback: str) -> pd.DataFrame:
    """Defense-in-depth: cut a provider's response down to the requested
    lookback window even if that provider's own API ignored the parameter
    (LSEProvider's /candles only supports limit+order, no start/end filter
    verified in the SDK — this is what was silently returning ~5000 bars
    of history, e.g. AAPL back to 2008, when the chart asked for "2y")."""
    if df.empty:
        return df
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=_lookback_days(lookback))
    return df[df.index >= cutoff]


def filter_price_outliers(df: pd.DataFrame, window: int = 20,
                          mult: float = 3.0) -> pd.DataFrame:
    """Drop bars whose High/Low ratio blows out past `mult`x the rolling
    `window`-day average of that ratio — catches bad-tick/vendor data
    artifacts (the vertical-spike chart anomalies) without touching the
    research/signal pipeline's bar count, which the engine's indicators
    (e.g. the 200-day SMA gate) depend on. Chart-display use only."""
    if df.empty or len(df) < window + 1:
        return df
    ratio = df["High"] / df["Low"].replace(0, np.nan)
    roll_avg = ratio.rolling(window, min_periods=window).mean()
    range_spike = (ratio > roll_avg * mult).fillna(False)
    # Hampel on close-to-close: a single print that jumps >5σ of the
    # last `window` returns is almost always a vendor glitch on Yahoo.
    rets = df["Close"].pct_change()
    med = rets.rolling(window, min_periods=window).median()
    mad = (rets - med).abs().rolling(window, min_periods=window).median()
    sigma = (1.4826 * mad).replace(0, np.nan)
    close_spike = ((rets - med).abs() > 5.0 * sigma).fillna(False)
    return df[~(range_spike | close_spike)]


# ---------------------------------------------------------------------------

class YahooProvider(DataProvider):
    name = "yahoo"

    def get_candles(self, symbol, interval="1d", lookback="2y"):
        import yfinance as yf
        for attempt in range(3):
            try:
                # auto_adjust=False everywhere: Close must be the real,
                # un-split-adjusted-away broker price, never Yahoo's
                # dividend/split-adjusted "Adj Close" substituted in.
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

    def get_candles_batch(self, symbols: list[str], interval: str = "1d",
                          lookback: str = "2y") -> dict[str, pd.DataFrame]:
        """One yf.download() call for the whole universe instead of N
        separate yf.Ticker(...).history() round-trips — the rate-limit
        protection for universe/sector scans."""
        import yfinance as yf
        out: dict[str, pd.DataFrame] = {}
        if not symbols:
            return out
        try:
            raw = yf.download(tickers=" ".join(symbols), period=lookback,
                              interval=interval, group_by="ticker",
                              auto_adjust=False, threads=True,
                              progress=False)
        except Exception:
            return out
        if raw is None or raw.empty:
            return out
        for s in symbols:
            try:
                sub = raw[s] if len(symbols) > 1 else raw
                sub = sub.rename(columns=str.title)
                sub = sub[["Open", "High", "Low", "Close",
                           "Volume"]].dropna()
                sub.index = pd.to_datetime(sub.index).tz_localize(None)
                if len(sub):
                    out[s] = _trim_to_lookback(sub, lookback)
            except Exception:
                continue
        return out

    def get_quotes_batch(self, symbols: list[str]) -> dict[str, dict]:
        """Batched quote fetch for a chunk of symbols via one yf.download()
        call — today's still-forming daily bar as `price`, the prior
        close as the % change baseline. This is deliberately less precise
        than get_quote()'s yf.Ticker.fast_info (which stays the path for
        the small priority set — positions + trading watchlist); it's
        what makes staggered ~50-symbol universe scanning affordable
        (P9 rate-limit protection)."""
        import yfinance as yf
        out: dict[str, dict] = {}
        if not symbols:
            return out
        try:
            raw = yf.download(tickers=" ".join(symbols), period="5d",
                              interval="1d", group_by="ticker",
                              auto_adjust=False, threads=True,
                              progress=False)
        except Exception:
            return out
        if raw is None or raw.empty:
            return out
        for s in symbols:
            try:
                sub = raw[s] if len(symbols) > 1 else raw
                closes = sub["Close"].dropna()
                if not len(closes):
                    continue
                px = float(closes.iloc[-1])
                prev = float(closes.iloc[-2]) if len(closes) > 1 else px
                out[s] = {"symbol": s, "price": px,
                         "chg_pct": round((px / prev - 1) * 100, 2)
                                    if prev else 0.0}
            except Exception:
                continue
        return out


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

      Verified 2026-07-12 by installing the real `lse-data` v0.14.0
      package from PyPI and reading lse/client.py + lse/vault.py source
      directly (WebFetch on their GitHub repo gave THREE mutually
      contradictory endpoint lists across three fetches — not trustworthy
      for something this consequential, so ground truth came from the
      actual installed package instead):
        Macro series (rates, CPI, bond yields): GET /vault/series
            ?symbol=cpi_yoy|fdtr|US10Y|...&dataset=&start=&end=&order=&limit=
        Macro events (CPI/NFP/rate decisions/GDP): GET /vault/ref/economic_calendar
            ?region=&event=&start=&end=&released=&order=&limit=
        Options flow (real trade prints, not a proxy): GET /vault/options/flow
            ?underlying=&type=&min_premium=&max_dte=&start=&end=&order=&limit=
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
        # options_chain() rate-limit state (RATE LIMIT PROTECTION, P8):
        # chains are the heaviest call in the SDK (full contract list +
        # greeks), capped to once per 10 minutes per underlying.
        self._chain_last_fetch: dict[str, float] = {}
        self._chain_cache: dict[str, pd.DataFrame] = {}

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
        df = df[["Open", "High", "Low", "Close",
                "Volume"]].astype(float).dropna().sort_index()
        # the vault's /candles endpoint only supports limit+order, not a
        # start/end filter (per the verified SDK contract above) — trim
        # client-side so a "2y" request can't silently come back as the
        # full 5000-bar history (this was the AAPL-back-to-2008 chart bug).
        return _trim_to_lookback(df, lookback)

    def get_quote(self, symbol):
        live = self.get_candles(symbol, "1m")
        if live.empty:
            live = self.get_candles(symbol, "1d")
        if live.empty:
            return {}
        px = float(live["Close"].iloc[-1])
        # % change must be vs the previous DAILY close, never the previous
        # 1-minute bar (that comparison is ~0.00% almost every tick and was
        # the reason the UI always showed a flat 0.00% change).
        daily = self.get_candles(symbol, "1d")
        if len(daily) >= 2:
            prev = float(daily["Close"].iloc[-2])
        elif len(daily) == 1:
            prev = float(daily["Close"].iloc[-1])
        else:
            prev = px
        return {"symbol": symbol, "price": px,
                "chg_pct": round((px / prev - 1) * 100, 2)}

    def options_chain(self, underlying: str, max_dte: int | None = 45
                      ) -> pd.DataFrame:
        """Current chain, one row per contract, WITH iv and greeks.

        Rate-limited to once per 10 minutes per underlying (RATE LIMIT
        PROTECTION, P8) — a repeat call inside the window returns the
        last cached chain instead of hitting the vault again."""
        now = time.time()
        if (now - self._chain_last_fetch.get(underlying, 0) < 600
                and underlying in self._chain_cache):
            return self._chain_cache[underlying]
        rows = self._get("/options/chain",
                         {"underlying": underlying, "limit": 5000,
                          "max_dte": max_dte})
        out = pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()
        self._chain_last_fetch[underlying] = now
        self._chain_cache[underlying] = out
        return out

    def usage(self) -> dict:
        return self._get("/usage", {}) or {}

    def macro_series(self, symbol: str, dataset: str | None = None,
                     start: str | None = None, end: str | None = None,
                     order: str = "asc", limit: int = 5000) -> pd.DataFrame:
        """One (date, value) observation series — any macro economics
        series or bond yield tenor (e.g. "cpi_yoy", "fdtr", "US10Y")."""
        rows = self._get("/series", {"symbol": symbol, "dataset": dataset,
                                     "start": start, "end": end,
                                     "order": order, "limit": limit})
        return pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()

    def economic_calendar(self, region: str | None = None,
                          event: str | None = None, start: str | None = None,
                          end: str | None = None, released_only: bool = False,
                          order: str = "asc", limit: int = 5000) -> pd.DataFrame:
        """Macro economic events — CPI, NFP, rate decisions, GDP."""
        rows = self._get("/ref/economic_calendar",
                         {"region": region, "event": event, "start": start,
                          "end": end, "released": 1 if released_only else None,
                          "order": order, "limit": limit})
        return pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()

    def company_profiles(self, symbol: str | None = None,
                         limit: int = 5000) -> pd.DataFrame:
        """Company reference profiles — sector, industry, description.
        Verified from the official SDK's client.company_profiles()
        (GET /ref/company_profiles)."""
        rows = self._get("/ref/company_profiles", {"symbol": symbol, "limit": limit})
        return pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()

    def options_flow(self, underlying: str | None = None,
                     type: str | None = None, min_premium: float | None = None,
                     max_dte: int | None = None, start: str | None = None,
                     end: str | None = None, order: str = "desc",
                     limit: int = 5000) -> pd.DataFrame:
        """Recent option prints (time & sales): trade, premium, IV and
        greeks at print time — a real feed, not a chain-delta proxy."""
        rows = self._get("/options/flow",
                         {"underlying": underlying, "type": type,
                          "min_premium": min_premium, "max_dte": max_dte,
                          "start": start, "end": end, "order": order,
                          "limit": limit})
        return pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()


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
        hit = _CANDLE_CACHE.get(symbol, interval, lookback)
        if hit is not None:
            return hit
        for p in self.providers:
            df = p.get_candles(symbol, interval, lookback)
            if len(df):
                _CANDLE_CACHE.put(symbol, interval, lookback, df)
                if self._state:
                    self._state.set(f"feed.served_by.{symbol}", p.name,
                                    source="data")
                    self._state.set("feed.cache", _CANDLE_CACHE.stats(),
                                    source="data")
                return df
        return pd.DataFrame()

    def get_quote(self, symbol):
        for p in self.providers:
            q = p.get_quote(symbol)
            if q:
                return q
        return {}

    def get_candles_batch(self, symbols: list[str], interval: str = "1d",
                          lookback: str = "2y") -> dict[str, pd.DataFrame]:
        """LSE has no documented batch/multi-symbol candles endpoint (the
        verified SDK contract is per-symbol /candles only), so universe-
        wide batching goes straight to Yahoo's single yf.download() call;
        anything it didn't return falls back to the normal per-symbol
        provider chain."""
        out: dict[str, pd.DataFrame] = {}
        missing: list[str] = []
        for s in symbols:
            hit = _CANDLE_CACHE.get(s, interval, lookback)
            if hit is not None:
                out[s] = hit
            else:
                missing.append(s)
        yahoo = next((p for p in self.providers
                     if isinstance(p, YahooProvider)), None)
        if yahoo and missing:
            fetched = yahoo.get_candles_batch(missing, interval, lookback)
            for s, df in fetched.items():
                if len(df):
                    _CANDLE_CACHE.put(s, interval, lookback, df)
                    out[s] = df
        for s in missing:
            if not len(out.get(s, pd.DataFrame())):
                df = self.get_candles(s, interval, lookback)
                if len(df):
                    out[s] = df
        if self._state:
            self._state.set("feed.cache", _CANDLE_CACHE.stats(), source="data")
        return out

    def get_quotes_batch(self, symbols: list[str]) -> dict[str, dict]:
        """Same LSE-has-no-batch-endpoint reasoning as get_candles_batch —
        goes straight to Yahoo. Deliberately no per-symbol fallback loop
        here (unlike get_candles_batch): this is the low-priority
        universe-scan tier specifically to AVOID N individual round-
        trips, so a partial/empty batch just gets retried next pass
        rather than defeating the point via a fallback loop."""
        yahoo = next((p for p in self.providers
                     if isinstance(p, YahooProvider)), None)
        return yahoo.get_quotes_batch(symbols) if yahoo else {}


# ---------------------------------------------------------------------------

class PollingFeed:
    """Background thread: poll quotes -> publish 'tick' events -> state.

    Two-tier scanning (P9): `symbols` is the FULL trading universe
    (~550 S&P 500 + Nasdaq-100 names, see data/universe.py) — scanning
    all of it every interval_s the old way (one get_quote() round-trip
    per symbol) would be both slow and a rate-limit problem. Instead:
      * `priority_fn()` (positions + the small trading watchlist) gets
        the precise per-symbol get_quote() every interval_s (30s).
      * the full universe is scanned in `batch_size`-symbol chunks via
        one get_quotes_batch() call, ONE chunk per tick — with
        interval_s=30s and batch_size=50, a ~550-symbol universe (~11
        batches) completes a full pass roughly every 5-6 minutes, i.e.
        spread across the scheduler's 5-minute decision-cycle window
        rather than hitting Yahoo with 550 requests at once.

    Honest platform note: on Streamlit Community Cloud this runs while the
    app process is awake. On a VPS (Hetzner phase) it runs 24/7 unchanged.
    """

    def __init__(self, bus: EventBus, state: GlobalState,
                 provider: DataProvider, symbols: list[str],
                 interval_s: int = 30, market_hours_gate: bool = True,
                 priority_fn=None, batch_size: int = 50):
        self._bus, self._state, self._provider = bus, state, provider
        self.symbols = symbols
        self.interval_s = interval_s
        # gate real network calls to pre/open/after sessions and skip
        # entirely when the market's closed (rate-limit budget); tests
        # that just want to verify tick delivery on FakeProvider pass
        # False here since they don't care about real wall-clock hours.
        self.market_hours_gate = market_hours_gate
        self._priority_fn = priority_fn or (lambda: [])
        self.batch_size = batch_size
        self._batch_cursor = 0
        self._scanned_this_pass = 0
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

    def _batches(self) -> list[list[str]]:
        syms = list(self.symbols)
        return [syms[i:i + self.batch_size]
               for i in range(0, len(syms), self.batch_size)]

    def _publish_quotes(self, quotes: dict[str, dict]) -> bool:
        got_any = False
        for s, q in quotes.items():
            if q:
                got_any = True
                self._state.set(f"quotes.{s}", q, source="feed")
                self._bus.publish(Event("tick", q, source="feed"))
        return got_any

    def _run(self):
        fail_count = 0
        while not self._stop.is_set():
            if self.market_hours_gate:
                if market_status()["session"] == "closed":
                    # save rate-limit budget: no network calls while shut
                    self._stop.wait(min(self.interval_s, 60))
                    continue
                throttle = self._state.get("feed.throttled")
                if throttle and time.time() < throttle.get("retry_at", 0):
                    self._stop.wait(self.interval_s)
                    continue

            got_any = False

            # tier 1: positions + trading watchlist, precise, every tick
            priority = [s for s in self._priority_fn() if s]
            if priority:
                for s in priority:
                    if self._stop.is_set():
                        break
                    got_any |= self._publish_quotes(
                        {s: self._provider.get_quote(s)})

            # tier 2: one ~50-symbol universe batch per tick -- with
            # interval_s=30s that spreads a ~550-symbol universe (~11
            # batches) across roughly one 5-minute decision-cycle window
            batches = self._batches()
            if batches:
                if self._batch_cursor == 0:
                    self._scanned_this_pass = 0
                chunk = batches[self._batch_cursor % len(batches)]
                batch_fn = getattr(self._provider, "get_quotes_batch", None)
                quotes = (batch_fn(chunk) if batch_fn else
                         {s: self._provider.get_quote(s) for s in chunk})
                got_any |= self._publish_quotes(quotes)
                self._scanned_this_pass += len(chunk)
                self._batch_cursor = (self._batch_cursor + 1) % len(batches)
                self._state.set(
                    "feed.universe_scan_progress",
                    {"scanned": self._scanned_this_pass,
                     "total": len(self.symbols)}, source="feed")

            if got_any:
                fail_count = 0
                if self._state.get("feed.throttled"):
                    self._state.set("feed.throttled", None, source="feed")
            elif priority or batches:
                fail_count += 1
                if fail_count >= 2:                # 2 empty passes in a row
                    self._state.set(
                        "feed.throttled",
                        {"since": time.time(), "retry_at": time.time() + 120},
                        source="feed")
            self._stop.wait(self.interval_s)
