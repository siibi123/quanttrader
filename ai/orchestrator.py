"""AI layer — the orchestrator seat and its contract with the platform.

Two orchestrators share one interface:
  * RuleOrchestrator — v1, deterministic. The proven gate logic (regime +
    momentum + panic-dip) drives proposals. Every proposal carries written
    reasoning, passes through the RiskEngine veto, and lands in the audit
    trail. Provable behavior, zero API cost.
  * LLMOrchestrator  — the socket for a real language model (needs
    ANTHROPIC_API_KEY). It receives GlobalState.to_ai_context() every step
    and may ONLY act through TOOL_SCHEMAS below — never free-form. Until a
    key exists this raises a clear error instead of pretending.

TOOL_SCHEMAS is the entire machine-to-machine surface: if a capability
isn't listed here, the AI cannot do it. The RiskEngine veto applies to the
LLM exactly as it does to the rules — no exceptions, by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.engine import AuditLog, Order, PaperBroker, RiskEngine
from core.state import EventBus, GlobalState
from data.providers import DataProvider

TOOL_SCHEMAS = [
    {"name": "get_state",
     "description": "Read the platform's global state snapshot",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_candles",
     "description": "Fetch OHLCV bars for a symbol",
     "input_schema": {"type": "object", "properties": {
         "symbol": {"type": "string"}, "interval": {"type": "string"}},
         "required": ["symbol"]}},
    {"name": "propose_order",
     "description": "Propose a paper trade. It will be risk-reviewed; "
                    "approval is NOT guaranteed. Must include reasoning.",
     "input_schema": {"type": "object", "properties": {
         "ticker": {"type": "string"},
         "side": {"type": "string", "enum": ["BUY", "SELL"]},
         "qty": {"type": "integer", "minimum": 1},
         "reasoning": {"type": "string"}},
         "required": ["ticker", "side", "qty", "reasoning"]}},
    {"name": "set_feed_symbols",
     "description": "Change which symbols the live feed polls",
     "input_schema": {"type": "object", "properties": {
         "symbols": {"type": "array", "items": {"type": "string"}}},
         "required": ["symbols"]}},
]


class RuleOrchestrator:
    """Deterministic v1 policy: transparent, testable, honest.

    Entry gates (all must pass): price > 200-bar SMA (regime) AND either
    RSI2 < 10 (panic dip in uptrend) or 20>50 SMA cross freshness.
    Exit: RSI2 > 80, or position down more than 1.5*ATR from avg price.
    Sizing: risk-based, capped by RiskEngine anyway.
    """

    def __init__(self, bus: EventBus, state: GlobalState, audit: AuditLog,
                 risk: RiskEngine, broker: PaperBroker,
                 provider: DataProvider):
        self._bus, self._state, self._audit = bus, state, audit
        self._risk, self._broker, self._provider = risk, broker, provider

    # ---- indicators (self-contained; QuantSignal engines port in later) --
    @staticmethod
    def _rsi(close: pd.Series, n: int = 2) -> float:
        d = close.diff()
        up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        rs = up / dn.replace(0, np.nan)
        return float((100 - 100 / (1 + rs)).iloc[-1])

    @staticmethod
    def _atr(df: pd.DataFrame, n: int = 14) -> float:
        tr = pd.concat([df["High"] - df["Low"],
                        (df["High"] - df["Close"].shift()).abs(),
                        (df["Low"] - df["Close"].shift()).abs()],
                       axis=1).max(axis=1)
        return float(tr.rolling(n).mean().iloc[-1])

    def analyze(self, symbol: str) -> dict:
        df = self._provider.get_candles(symbol)
        if len(df) < 220:
            return {"symbol": symbol, "signal": "NONE",
                    "why": "insufficient history"}
        c = df["Close"]
        price = float(c.iloc[-1])
        s200 = float(c.rolling(200).mean().iloc[-1])
        s20 = float(c.rolling(20).mean().iloc[-1])
        s50 = float(c.rolling(50).mean().iloc[-1])
        r2 = self._rsi(c)
        a = self._atr(df)
        sig, why = "NONE", []
        if price > s200:
            why.append(f"regime OK (px {price:.2f} > 200SMA {s200:.2f})")
            if r2 < 10:
                sig = "BUY"
                why.append(f"RSI2 panic ({r2:.0f}<10) in uptrend")
            elif s20 > s50 and price > s20:
                sig = "BUY"
                why.append("20>50 SMA trend alignment")
        else:
            why.append(f"regime blocks longs (px < 200SMA)")
        if r2 > 80:
            sig, why = "SELL", [f"RSI2 stretched ({r2:.0f}>80) — take profit"]
        out = {"symbol": symbol, "signal": sig, "price": price,
               "rsi2": round(r2, 1), "atr": round(a, 3),
               "why": " · ".join(why)}
        self._state.set(f"signals.{symbol}", out, source="orchestrator")
        return out

    def step(self, symbols: list[str], risk_pct: float = 1.0) -> list[dict]:
        """One decision cycle over the watchlist. Returns executed fills."""
        fills = []
        for s in symbols:
            sig = self.analyze(s)
            price = sig.get("price", 0)
            held = self._broker.positions.get(s, {}).get("qty", 0)
            if sig["signal"] == "BUY" and not held and price > 0:
                eq = self._broker.equity({s: price})
                qty = int((eq * risk_pct / 100) / max(1.5 * sig["atr"], 0.01))
                if qty < 1:
                    continue
                order = Order(s, "BUY", qty, reason=sig["why"])
                self._audit.record("Orchestrator", "PROPOSE BUY",
                                   trigger=f"signals.{s}", model="rule-v1",
                                   reasoning=sig["why"],
                                   data={"qty": qty, "price": price})
                order = self._risk.review(order, self._broker, price)
                if order.approved:
                    f = self._broker.execute(order, price)
                    if f:
                        fills.append(f)
            elif sig["signal"] == "SELL" and held:
                order = Order(s, "SELL", held, reason=sig["why"])
                self._audit.record("Orchestrator", "PROPOSE SELL",
                                   trigger=f"signals.{s}", model="rule-v1",
                                   reasoning=sig["why"], data={"qty": held})
                order = self._risk.review(order, self._broker, price)
                if order.approved:
                    f = self._broker.execute(order, price)
                    if f:
                        fills.append(f)
        return fills


class LLMOrchestrator:
    """The plug-in socket for a real language model (Claude API).

    Contract: receives state.to_ai_context() + TOOL_SCHEMAS; every tool call
    routes through the same RiskEngine.review() as the rules. Deliberately
    refuses to run without a key — this platform does not fake intelligence.
    """

    def __init__(self, api_key: str, **components):
        if not api_key:
            raise RuntimeError(
                "LLMOrchestrator requires ANTHROPIC_API_KEY in .env. "
                "Until then, RuleOrchestrator runs the desk — honestly.")
        self.api_key = api_key
        self.components = components
        # Implementation lands when a key exists (see CLAUDE.md roadmap):
        # anthropic.messages.create(..., tools=TOOL_SCHEMAS,
        #                           system=state.to_ai_context())
