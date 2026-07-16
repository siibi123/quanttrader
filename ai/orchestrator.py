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

import time

import numpy as np
import pandas as pd

from core.engine import AuditLog, Order, PaperBroker, RiskEngine
from core.state import Event, EventBus, GlobalState
from core.strategy_registry import StrategyRegistry
from data.news import NewsProvider
from data.providers import DataProvider, LSEProvider
from quant.anomaly_library import match_anomalies
from quant.flow_confluence import confluence
from quant.playbook import build_playbook
from quant.risk import correlation_heat, portfolio_var
from quant.sector_engine import rank_sectors_and_names
from quant.signals import BUY_TH, SELL_TH, composite, rsi
from quant.surface_interpreter import interpret_surface
from quant.verdict import MODELS
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

    P7a mandatory gate: when a StrategyRegistry is wired in, every BUY/
    SELL signal from this strategy is logged; NEW entries only execute
    once the strategy is promoted PAPER (>= 30 settled signals, bootstrap
    CI on forward returns excludes zero). Exits are never gated — closing
    risk on an existing position is always allowed regardless of status.
    """

    STRATEGY_NAME = "rule_v1_playbook_verdict"

    def __init__(self, bus: EventBus, state: GlobalState, audit: AuditLog,
                 risk: RiskEngine, broker: PaperBroker,
                 provider: DataProvider, news: NewsProvider | None = None,
                 lse: LSEProvider | None = None,
                 registry: StrategyRegistry | None = None):
        self._bus, self._state, self._audit = bus, state, audit
        self._risk, self._broker, self._provider = risk, broker, provider
        self._news, self._lse = news, lse
        self._registry = registry

    def _settle_price(self, symbol: str):
        q = self._provider.get_quote(symbol)
        return q.get("price") if q else None

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
        """Autonomous quant pass: EWMA vol + Monte Carlo odds -> state+audit,
        plus any curated academic anomaly whose trigger condition matches
        today's numbers on this symbol (quant.anomaly_library)."""
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

        comp = composite(df)
        score = float(comp["score"].iloc[-1])
        direction = 1 if score >= BUY_TH else (-1 if score <= SELL_TH else 0)
        signs = np.sign([float(comp[m].iloc[-1]) for m in MODELS])
        agree_frac = (float((signs == direction).sum()) / len(MODELS)
                     if direction != 0 else 0.0)
        today = time.localtime()
        ctx = {"score": score, "agree_frac": agree_frac,
              "rsi2": float(rsi(df["Close"], 2).iloc[-1]),
              "ewma_ann_vol_pct": out["ewma_ann_vol_pct"],
              "month": today.tm_mon, "trading_day_of_month": today.tm_mday}
        anomalies = match_anomalies(ctx)
        if anomalies:
            out["anomalies"] = anomalies

        self._state.set(f"research.{symbol}", out, source="research")
        self._audit.record(
            "Research", "VOL+MONTECARLO", trigger=symbol,
            model="EWMA(l=.94) + GBM-MC(2000x20d)",
            reasoning=f"{symbol}: ann vol {out['ewma_ann_vol_pct']}% · "
                      f"P(up in 20d) {out['p_up_20d_pct']}% · expected "
                      f"1s move +/-${out['exp_move_20d']}",
            data=out)
        if anomalies:
            self._audit.record(
                "Research", "ANOMALY MATCH", trigger=symbol,
                model="anomaly_library (curated, rule-matched)",
                reasoning=f"{symbol}: " + " | ".join(
                    f"{a['name']} ({a['citation']})" for a in anomalies),
                data={"anomalies": anomalies})
        return out

    def scan_news(self, symbol: str) -> dict:
        """Headlines + sentiment -> state.news, audit, and an interrupt
        event on a strong sentiment reading. Cleanly empty — no fake
        headlines, no fake score — if NEWS_API_KEY is unset."""
        if not self._news or not self._news.working:
            return {}
        headlines = self._news.company_news(symbol, days=3, limit=10)
        sent = self._news.sentiment(symbol)
        out = {"symbol": symbol, "headlines": headlines, **sent}
        if not headlines and not sent:
            return out
        self._state.set(f"news.{symbol}", out, source="news")
        self._audit.record(
            "News", "HEADLINES+SENTIMENT", trigger=symbol,
            model="Finnhub company-news + news-sentiment",
            reasoning=(f"{symbol}: {len(headlines)} headline(s) in 3d"
                      + (f" · bullish {sent['bullish_pct']}% / bearish "
                         f"{sent['bearish_pct']}%" if sent else "")),
            data=out)
        if sent and (sent.get("bullish_pct", 0) >= 70
                    or sent.get("bearish_pct", 0) >= 70):
            self._bus.publish(Event("news.interrupt",
                                    {"symbol": symbol, **sent}, source="news"))
        return out

    def scan_macro(self, series: list[str] | None = None) -> dict:
        """Rates/CPI/economic-calendar snapshot -> state.macro + audit.
        Symbols are the LSE SDK's own documented examples (cpi_yoy, fdtr,
        US10Y). Empty/honest if the LSE key is unset or the vault has
        nothing for a given series — never fabricates a number."""
        if not self._lse or not self._lse.key:
            return {}
        series = series or ["cpi_yoy", "fdtr", "US10Y"]
        out: dict = {}
        for s in series:
            df = self._lse.macro_series(s, limit=2, order="desc")
            if not len(df):
                continue
            df.columns = [str(c).lower() for c in df.columns]
            val_col = next((c for c in ("value", "close") if c in df.columns), None)
            dt_col = next((c for c in ("date", "timestamp") if c in df.columns), None)
            if not val_col:
                continue
            latest = float(df[val_col].iloc[0])
            prior = float(df[val_col].iloc[1]) if len(df) > 1 else None
            trend = ("up" if prior is not None and latest > prior else
                    "down" if prior is not None and latest < prior else "flat")
            out[s] = {"latest": latest, "prior": prior, "trend": trend,
                      "as_of": str(df[dt_col].iloc[0]) if dt_col else ""}
        cal = self._lse.economic_calendar(region="US", order="asc", limit=10)
        upcoming = []
        if len(cal):
            cal.columns = [str(c).lower() for c in cal.columns]
            ev_col = next((c for c in ("event", "name", "title")
                          if c in cal.columns), None)
            dt_col = next((c for c in ("date", "start", "timestamp")
                          if c in cal.columns), None)
            for _, row in cal.head(5).iterrows():
                upcoming.append({"event": str(row.get(ev_col, "")) if ev_col else "",
                                "date": str(row.get(dt_col, "")) if dt_col else ""})
        if upcoming:
            out["upcoming_events"] = upcoming
        if not out:
            return out
        self._state.set("macro", out, source="macro")
        self._audit.record(
            "Research", "MACRO SCAN",
            model="LSE /series + /ref/economic_calendar",
            reasoning="Macro snapshot: " + ", ".join(
                f"{k}={v['latest']}" for k, v in out.items()
                if k != "upcoming_events")
                + (f" · {len(upcoming)} upcoming event(s)" if upcoming else ""),
            data=out)
        return out

    def scan_flow(self, symbol: str, min_premium: float = 100_000) -> dict:
        """Recent large option prints on `symbol` -> state.flow_alerts +
        audit + an interrupt event. Real prints from LSE /options/flow, not
        a chain-delta proxy. Distinct from the fuller statistical flow
        engine (quant/optionflow.py, P6b) that consumes this same feed."""
        if not self._lse or not self._lse.key:
            return {}
        df = self._lse.options_flow(underlying=symbol, min_premium=min_premium,
                                    order="desc", limit=20)
        if not len(df):
            return {}
        df.columns = [str(c).lower() for c in df.columns]
        prem_col = next((c for c in ("premium", "notional")
                        if c in df.columns), None)
        prints = []
        for _, row in df.head(10).iterrows():
            prints.append({
                "strike": row.get("strike"), "type": row.get("type"),
                "premium": (float(row[prem_col])
                           if prem_col and pd.notna(row.get(prem_col)) else None),
                "expiry": str(row.get("expiry", ""))})
        out = {"symbol": symbol, "min_premium": min_premium, "prints": prints}
        self._state.set(f"flow_alerts.{symbol}", out, source="flow")
        self._audit.record(
            "Research", "FLOW ALERT", trigger=symbol,
            model="LSE /options/flow (real prints, not a proxy)",
            reasoning=f"{symbol}: {len(prints)} print(s) >= "
                      f"${min_premium:,.0f} premium in the recent tape",
            data=out)
        self._bus.publish(Event("flow.interrupt", out, source="flow"))
        return out

    def scan_flow_confluence(self, symbol: str) -> dict:
        """One CONFLUENCE read per symbol -> state.flow.{symbol}, audit,
        and a VPIN-toxicity caution folded into RiskEngine's reasoning
        trail (informational only, never a veto — that stays a hard-veto
        decision the owner makes explicitly, not this method).

        Options positioning uses today's LSE options_flow() snapshot.
        Flow-spike z-scoring needs >= 10 days of daily flow history;
        options_flow() only covers a trailing week, so that baseline
        isn't built here yet — premium_share (call/put split) alone
        drives the options-positioning read for now."""
        df = self._provider.get_candles(symbol)
        if len(df) < 40:
            return {}
        flow_today = None
        if self._lse and self._lse.key:
            flow_today = self._lse.options_flow(underlying=symbol, max_dte=45,
                                                limit=500)
        out = confluence(df, flow_today)
        out["symbol"] = symbol
        self._state.set(f"flow.{symbol}", out, source="flow")
        self._audit.record(
            "Research", "FLOW CONFLUENCE", trigger=symbol,
            model="quant.flow_confluence (BVC/CVD/VPIN + options premium share)",
            reasoning=(f"{symbol}: {out['verdict']} · tape "
                      f"{out['tape_score']:+.2f} · options "
                      f"{out['options_score']:+.2f} · " +
                      " | ".join(out["tape_reasons"] + out["options_reasons"])),
            data=out)
        if out.get("toxic_caution"):
            self._audit.record(
                "RiskEngine", "CAUTION FLAG", trigger=symbol,
                model="VPIN toxicity (informational only, not a veto)",
                reasoning=(f"{symbol}: VPIN toxicity "
                          f"{out.get('vpin_percentile')}pct (>=85th) — "
                          f"elevated informed-trading risk; RiskEngine's "
                          f"actual checks are unchanged, this is advisory"),
                data={"vpin_percentile": out.get("vpin_percentile")})
        return out

    def sector_scan(self, symbols: list[str], account: float = 5000.0,
                    risk_pct: float = 1.0) -> dict:
        """Multi-factor sector/name ranking (quant.sector_engine): verdict's
        technical conviction tilted by whatever news sentiment, large
        option prints, and macro rate-trend readings are already cached in
        state (from scan_news/scan_flow/scan_macro — this does not fetch
        those itself). Sector comes from LSE company_profiles when the key
        is set, else 'Unclassified' — never guessed."""
        data = {s: self._provider.get_candles(s) for s in symbols}
        data = {s: df for s, df in data.items() if len(df) >= 220}
        if not data:
            return {}

        sectors = {}
        if self._lse and self._lse.key:
            for s in data:
                prof = self._lse.company_profiles(symbol=s)
                if len(prof):
                    prof.columns = [str(c).lower() for c in prof.columns]
                    if "sector" in prof.columns:
                        sectors[s] = str(prof["sector"].iloc[0])

        sentiment_by, flow_by, confluence_by = {}, {}, {}
        for s in data:
            n = self._state.get(f"news.{s}")
            if n and n.get("bullish_pct") is not None:
                sentiment_by[s] = n
            f = self._state.get(f"flow_alerts.{s}")
            if f:
                flow_by[s] = f
            fc = self._state.get(f"flow.{s}")
            if fc:
                confluence_by[s] = fc

        rate = (self._state.get("macro") or {}).get("fdtr") or {}
        macro_trend = rate.get("trend")

        out = rank_sectors_and_names(data, sectors, account=account,
                                     risk_pct=risk_pct,
                                     sentiment_by_ticker=sentiment_by,
                                     flow_by_ticker=flow_by,
                                     macro_trend=macro_trend,
                                     flow_confluence_by_ticker=confluence_by)
        self._state.set("sector_scan", out, source="research")
        top_sec = out["sectors"][0]["sector"] if out["sectors"] else "none"
        top_names = ", ".join(f"{n['ticker']} ({n['target_score']})"
                              for n in out["names"][:3])
        self._audit.record(
            "Research", "SECTOR SCAN",
            model="quant.sector_engine (verdict + sentiment/flow/macro tilts)",
            reasoning=(f"Scanned {out['n_scanned']} names · top sector "
                      f"{top_sec} · top names: {top_names or 'none tradeable'}"
                      f" · {len(out['avoid'])} flagged to avoid"),
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
        may_enter = True
        if self._registry:
            self._registry.settle_signals(self.STRATEGY_NAME, self._settle_price)
            promo = self._registry.evaluate_promotion(self.STRATEGY_NAME)
            may_enter = self._registry.status(self.STRATEGY_NAME) \
                == StrategyRegistry.STATUS_PAPER

        for s in symbols:
            held_pos = self._broker.positions.get(s)
            eq0 = self._broker.equity(prices_seen)
            sig = self.analyze(s, equity=eq0, risk_pct=risk_pct,
                               held=held_pos)
            price = sig.get("price", 0)
            if price:
                prices_seen[s] = price
            held = held_pos.get("qty", 0) if held_pos else 0

            if sig["signal"] in ("BUY", "SELL") and self._registry and price > 0:
                self._registry.log_signal(self.STRATEGY_NAME, s,
                                          sig["signal"], price)

            if sig["signal"] == "BUY" and not held and price > 0:
                if not may_enter:
                    self._audit.record(
                        "Orchestrator", "SIGNAL LOGGED (INCUBATION)",
                        trigger=f"signals.{s}", model="rule-v1",
                        reasoning=(f"{s}: BUY signal logged but NOT traded — "
                                  f"strategy '{self.STRATEGY_NAME}' is still "
                                  f"in INCUBATION (P7a promotion gate)"),
                        data={"symbol": s, "signal": "BUY", "price": price})
                    continue
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
                # exits are never gated by the promotion status
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
