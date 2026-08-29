"""Core primitives: config, event bus, global state.

Design contract (the whole platform hangs off these three):
  * Config     — every credential and limit from .env, nothing hardcoded.
  * EventBus   — thread-safe pub/sub. EVERYTHING that happens is an Event.
  * GlobalState— thread-safe key/value tree. Every tick, calculation and UI
                 action commits here; `to_ai_context()` exports a compact,
                 curated snapshot that gets injected into the AI's context —
                 this is how the AI "knows" everything without being told.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from datetime import time as _dtime
from typing import Any, Callable
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:                                    # dotenv optional
    pass


def _env(name: str, default: str = "") -> str:
    """os.getenv first (local .env / VPS), st.secrets second (Streamlit
    Cloud). Never raises when Streamlit or the secret is absent."""
    v = os.getenv(name)
    if v:
        return v
    try:
        import streamlit as _st
        return str(_st.secrets.get(name, default))
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    lse_api_key: str = field(default_factory=lambda: _env("LSE_API_KEY"))
    lse_base_url: str = field(default_factory=lambda: _env("LSE_BASE_URL"))
    news_api_key: str = field(default_factory=lambda: _env("NEWS_API_KEY"))
    anthropic_api_key: str = field(
        default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    starting_cash: float = field(
        default_factory=lambda: float(_env("STARTING_CASH", "10000")))
    max_position_pct: float = field(
        default_factory=lambda: float(_env("RISK_MAX_POSITION_PCT", "25")))
    max_gross_exposure_pct: float = field(default_factory=lambda: float(
        _env("RISK_MAX_GROSS_EXPOSURE_PCT", "120")))
    max_daily_loss_pct: float = field(
        default_factory=lambda: float(_env("RISK_MAX_DAILY_LOSS_PCT", "3")))
    max_var_pct: float = field(
        default_factory=lambda: float(_env("RISK_MAX_VAR_PCT", "2.5")))
    runtime_dir: str = field(default_factory=lambda: _env("RUNTIME_DIR",
                                                          "runtime"))
    # Owner-declared total capital (AUM) — 0 means "unset", fall back to the
    # paper broker's own live equity. Lets the position-size cap below be
    # sized against the owner's real account, not just the paper sandbox.
    aum: float = field(default_factory=lambda: float(_env("PORTFOLIO_AUM", "0")))
    max_position_mode: str = field(
        default_factory=lambda: _env("RISK_MAX_POSITION_MODE", "pct"))  # "pct" | "fixed"
    max_position_fixed_usd: float = field(
        default_factory=lambda: float(_env("RISK_MAX_POSITION_FIXED_USD", "0")))
    max_book_heat_pct: float = field(
        default_factory=lambda: float(_env("RISK_MAX_BOOK_HEAT_PCT", "6")))


# ---------------------------------------------------------------------------
# Market hours
# ---------------------------------------------------------------------------

ET = ZoneInfo("America/New_York")


def market_status(now: datetime | None = None) -> dict:
    """Classify the current US-equity session in Eastern time: pre-market
    (4:00-9:30), open (9:30-16:00), after-hours (16:00-20:00), else closed.

    Honest limitation: weekday-only — there's no exchange holiday calendar
    dependency in this platform, so a market holiday still reads as a
    normal weekday session here. Good enough to gate polling/scheduling
    cadence; not a substitute for a real trading calendar.
    """
    now_et = (now or datetime.now(ET)).astimezone(ET)
    if now_et.weekday() >= 5:                      # Sat/Sun
        return {"session": "closed", "label": "Closed (weekend)",
                "et_time": now_et}
    t = now_et.time()
    if _dtime(9, 30) <= t < _dtime(16, 0):
        session, label = "open", "Market Open"
    elif _dtime(4, 0) <= t < _dtime(9, 30):
        session, label = "pre", "Pre-Market"
    elif _dtime(16, 0) <= t < _dtime(20, 0):
        session, label = "after", "After-Hours"
    else:
        session, label = "closed", "Closed"
    return {"session": session, "label": label, "et_time": now_et}


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class Event:
    type: str                      # e.g. "tick", "broker.fill", "risk.veto"
    payload: dict = field(default_factory=dict)
    source: str = "system"
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"type": self.type, "payload": self.payload,
                "source": self.source, "ts": self.ts}


class EventBus:
    """Thread-safe pub/sub. Subscribe to a type or '*' for everything."""

    def __init__(self, history: int = 2000):
        self._subs: dict[str, list[Callable[[Event], None]]] = {}
        self._lock = threading.RLock()
        self.history: deque[Event] = deque(maxlen=history)

    def subscribe(self, event_type: str, fn: Callable[[Event], None]) -> None:
        with self._lock:
            self._subs.setdefault(event_type, []).append(fn)

    def publish(self, event: Event) -> None:
        with self._lock:
            self.history.append(event)
            targets = list(self._subs.get(event.type, [])) + \
                list(self._subs.get("*", []))
        for fn in targets:                 # call outside the lock
            try:
                fn(event)
            except Exception:
                pass                       # a bad subscriber never kills the bus

    def recent(self, n: int = 50, event_type: str | None = None) -> list[Event]:
        with self._lock:
            evs = list(self.history)
        if event_type:
            evs = [e for e in evs if e.type == event_type]
        return evs[-n:]


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

class GlobalState:
    """Thread-safe nested store. Dot-paths: state.set("quotes.AAPL", {...}).

    Every mutation publishes a 'state.changed' event, so any component
    (including the AI orchestrator) can react to anything.
    """

    def __init__(self, bus: EventBus):
        self._data: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._bus = bus

    def set(self, path: str, value: Any, source: str = "system") -> None:
        keys = path.split(".")
        with self._lock:
            node = self._data
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = value
        self._bus.publish(Event("state.changed",
                                {"path": path}, source=source))

    def get(self, path: str, default: Any = None) -> Any:
        keys = path.split(".")
        with self._lock:
            node = self._data
            for k in keys:
                if not isinstance(node, dict) or k not in node:
                    return default
                node = node[k]
            return node

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._data, default=str))

    # ---- the AI's senses --------------------------------------------------
    AI_KEYS = ["session", "feed", "quotes", "portfolio", "risk",
               "signals", "research", "options", "news", "macro",
               "sector_scan", "flow", "regime", "execution_quality",
               "correlation_regime", "portfolio_stress", "daily_report", "ui"]

    def to_ai_context(self, max_chars: int = 6000) -> str:
        """Curated, compact JSON of everything the AI needs to be
        state-aware. Injected into the orchestrator every step."""
        snap = self.snapshot()
        ctx = {k: snap[k] for k in self.AI_KEYS if k in snap}
        s = json.dumps(ctx, default=str, separators=(",", ":"))
        return s[:max_chars]
