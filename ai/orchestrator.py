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
from quant.playbook import build_playbook
from quant.risk import correlation_heat, portfolio_var
from quant.surface_interpreter import interpret_surface
from quant.verdict import analyze as qs_verdict

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

    def analyze(self, symbol: str, equity: float = 10000.0,
                risk_pct: float = 1.0, held: dict | None = None) -> dict:
        """QuantSignal fusion: the 5-gate Playbook + 7-model verdict drive
        the signal; the reasoning IS the playbook instruction."""
        df = self._provider.get_candles(symbol)
        if len(df) < 220:
            return {"symbol": symbol, "signal": "NONE",
                    "why": "insufficient history"}
        price = float(df["Close"].iloc[-1])
        if held and held.get("qty", 0) > 0:
            pb = build_playbook(df, account=equity, risk_pct=risk_pct,
                                in_position=True,
                                entry=float(held["avg_price"]),
                                stop=float(held.get(
                                    "stop", held["avg_price"] * 0.94)))
            sig = "SELL" if any(k in pb["instruction"]
                                for k in ("EXIT", "TIGHTEN")) else "NONE"
            out = {"symbol": symbol, "signal": sig, "price": price,
                   "urgency": pb["urgency"], "mode": "MANAGE",
                   "gates": f"{pb['greens']}/5",
                   "why": f"PLAYBOOK {pb['urgency']}: {pb['instruction']}"}
        else:
            pb = build_playbook(df, account=equity, risk_pct=risk_pct)
            sig = "BUY" if pb["urgency"] in ("🟢 ACTIONABLE",
                                             "🟡 FAST SETUP") else "NONE"
            out = {"symbol": symbol, "signal": sig, "price": price,
                   "urgency": pb["urgency"], "mode": "ENTRY",
                   "gates": f"{pb['greens']}/5",
                   "shares": pb.get("plan", {}).get("shares", 0),
                   "why": f"PLAYBOOK {pb['urgency']}: {pb['instruction']}"}
        self._state.set(f"signals.{symbol}", out, source="orchestrator")
        return out

    def correlation_watch(self, prices: dict) -> dict:
        """The correlation engine on the LIVE book — QuantSignal's
        risk math guarding QuantTrader's positions every cycle."""
        pos = self._broker.positions
        if len(pos) < 2:
            return {}
        eq = self._broker.equity(prices)
        plist, rets = [], {}
        for t, p in pos.items():
            df = self._provider.get_candles(t)
            if len(df) < 60:
                continue
            rets[t] = df["Close"].pct_change().dropna()
            plist.append({"ticker": t, "shares": int(p["qty"]),
                          "entry": float(p["avg_price"]),
                          "stop": float(p["avg_price"]) * 0.94})
        ch = correlation_heat(plist, rets, eq) or {}
        pv = portfolio_var(plist, rets, eq) or {}
        out = {**ch, **{f"var_{k}": v for k, v in pv.items()}}
        if out:
            self._state.set("risk.book", out, source="risk")
            warn = ch.get("warning")
            self._audit.record(
                "Research", "CORRELATION WATCH",
                model="corr-adjusted heat + parametric VaR",
                reasoning=(f"avg pairwise corr {ch.get('avg_correlation')}"
                           f" · heat ${ch.get('naive_heat_$')}→"
                           f"${ch.get('corr_adj_heat_$')} · 1-day VaR "
                           f"{pv.get('VaR_%','—')}%"
                           + (" · ⚠️ CROWDED BOOK — positions are "
                              "effectively one trade" if warn else "")),
                data=out)
        return out

    def research(self, symbol: str) -> dict:
        """Autonomous quant pass: EWMA vol + Monte Carlo odds -> state+audit."""
        df = self._provider.get_candles(symbol)
        if len(df) < 60:
            return {}
        r = df["Close"].pct_change().dropna()
        lam, var = 0.94, float(r.iloc[0]) ** 2
        for x in r.iloc[1:].values:
            var = lam * var + (1 - lam) * x * x
        sig_d = float(np.sqrt(var))
        rng = np.random.default_rng(7)
        paths = np.exp(np.cumsum(
            rng.normal(float(r.mean()), sig_d, (2000, 20)), axis=1))
        out = {"symbol": symbol,
               "ewma_ann_vol_pct": round(sig_d * np.sqrt(252) * 100, 1),
               "p_up_20d_pct": round(float((paths[:, -1] > 1).mean()) * 100, 1),
               "exp_move_20d": round(
                   float(df["Close"].iloc[-1]) * sig_d * np.sqrt(20), 2)}
        self._state.set(f"research.{symbol}", out, source="research")
        self._audit.record(
            "Research", "VOL+MONTECARLO", trigger=symbol,
            model="EWMA(l=.94) + GBM-MC(2000x20d)",
            reasoning=f"{symbol}: ann vol {out['ewma_ann_vol_pct']}% · "
                      f"P(up in 20d) {out['p_up_20d_pct']}% · expected "
                      f"1s move +/-${out['exp_move_20d']}",
            data=out)
        return out

    def ingest_chain(self, symbol: str, chain: pd.DataFrame) -> dict:
        """Distill an options chain WITH greeks into the Global State."""
        if chain is None or not len(chain):
            return {}
        c = chain.copy()
        c.columns = [str(x).lower() for x in c.columns]
        g = {}
        for k in ("delta", "gamma", "theta", "vega", "iv"):
            if k in c.columns:
                g[k] = pd.to_numeric(c[k], errors="coerce")
        out = {"symbol": symbol, "contracts": int(len(c)),
               "greeks_present": sorted(g.keys())}
        if "iv" in g:
            out["median_iv"] = round(float(g["iv"].median()), 4)
        if "type" in c.columns:
            t = c["type"].astype(str).str.lower()
            out["call_share_pct"] = round(
                float((t.str.startswith("c")).mean()) * 100, 1)
        if "gamma" in g and "strike" in c.columns:
            gx = g["gamma"].abs().groupby(
                pd.to_numeric(c["strike"], errors="coerce")).sum()
            if len(gx):
                out["max_gamma_strike"] = float(gx.idxmax())

        spot = (self._state.get(f"quotes.{symbol}") or {}).get("price")
        surf = interpret_surface(chain, spot=spot)
        if "error" not in surf:
            out["surface"] = surf

        self._state.set(f"options.{symbol}", out, source="research")
        self._audit.record(
            "Research", "OPTIONS CHAIN", trigger=symbol,
            model="LSE /options/chain (precomputed greeks)",
            reasoning=f"{symbol}: {out['contracts']} contracts · greeks "
                      f"{','.join(out['greeks_present']) or 'none'} · "
                      f"median IV {out.get('median_iv', 'n/a')} · max-gamma "
                      f"strike {out.get('max_gamma_strike', 'n/a')}",
            data=out)
        if "error" not in surf and surf["findings"]:
            self._audit.record(
                "Research", "VOL SURFACE", trigger=symbol,
                model="surface_interpreter (rule-based, deterministic)",
                reasoning=f"{symbol} @{surf['near_dte']}d: " +
                          " ".join(surf["findings"]),
                data=surf)
        return out

    def step(self, symbols: list[str], risk_pct: float = 1.0) -> list[dict]:
        """One decision cycle over the watchlist. Returns executed fills."""
        fills = []
        prices_seen = {}
        for s in symbols:
            held_pos = self._broker.positions.get(s)
            eq0 = self._broker.equity(prices_seen)
            sig = self.analyze(s, equity=eq0, risk_pct=risk_pct,
                               held=held_pos)
            price = sig.get("price", 0)
            if price:
                prices_seen[s] = price
            held = held_pos.get("qty", 0) if held_pos else 0
            if sig["signal"] == "BUY" and not held and price > 0:
                qty = int(sig.get("shares") or 0)
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
        self.correlation_watch(prices_seen)
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
