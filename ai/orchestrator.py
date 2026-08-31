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

import os
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
from quant.setup_risk import book_overlap_mult, high_impact_soon
from quant.playbook import build_playbook
from quant.scorecard import book_heat
from quant.sector_etf import (SECTOR_ETFS, apply_etf_gate, breadth_from_universe,
                              rank_etfs)
from quant.trade_layers import closed_trade_layers, pair_exits
from quant.correlation_monitor import CORRELATION_POLICY, correlation_regime
from quant.daily_report import render_morning_briefing, render_report
from quant.execution_quality import slippage_report
from quant.portfolio_stress import risk_budget_from_stress, simulate_portfolio
from quant.regime_gate import REGIME_POLICY, classify_regime
from quant.risk import correlation_heat, portfolio_var
from quant.sector_engine import rank_sectors_and_names
from quant.signals import BUY_TH, SELL_TH, composite, rsi
from quant.surface_interpreter import interpret_surface
from quant.transaction_costs import expected_trade_cost
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


def _cooldown_ok(state: GlobalState, key: str, min_interval_s: float) -> bool:
    """Self-imposed pacing cap for expensive universe/chain fetches (rate-
    limit protection) — distinct from PollingFeed's reactive Yahoo-throttle
    backoff. True if `min_interval_s` has elapsed since the last call
    tagged `key`; does NOT itself record a call, so a caller that decides
    not to proceed (e.g. because it's returning a cached result) doesn't
    reset the clock."""
    last = state.get(f"_ratelimit.{key}")
    return last is None or time.time() - last >= min_interval_s


def _mark_ran(state: GlobalState, key: str) -> None:
    state.set(f"_ratelimit.{key}", time.time(), source="ratelimit")


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

    # P9 universe-scale decision cycle: _regime_gate's HMM refit measured
    # ~11s/symbol (quant.hmm_regime's hand-rolled Baum-Welch EM, no
    # torch/GPU per Iron Rule #8) -- fitting all ~550 universe symbols
    # fresh every 5-minute cycle is a ~100-minute job, not a 5-minute one.
    # step() caches each symbol's regime read for REGIME_REFIT_TTL_S and
    # caps fresh refits to REGIME_REFIT_BUDGET_PER_CYCLE per cycle; symbols
    # without ANY cached read yet default to unrestricted Bull (the same
    # honest fallback _regime_gate already uses for <90 bars of history)
    # rather than blocking, so a cold-started universe trades immediately
    # and gets progressively more accurate regime protection as the cache
    # warms up over the next couple of hours.
    REGIME_REFIT_TTL_S = 1800
    REGIME_REFIT_BUDGET_PER_CYCLE = 20

    def __init__(self, bus: EventBus, state: GlobalState, audit: AuditLog,
                 risk: RiskEngine, broker: PaperBroker,
                 provider: DataProvider, news: NewsProvider | None = None,
                 lse: LSEProvider | None = None,
                 registry: StrategyRegistry | None = None,
                 circuit_breaker=None):
        self._bus, self._state, self._audit = bus, state, audit
        self._risk, self._broker, self._provider = risk, broker, provider
        self._news, self._lse = news, lse
        self._registry = registry
        self._circuit_breaker = circuit_breaker

    def _settle_price(self, symbol: str):
        q = self._provider.get_quote(symbol)
        return q.get("price") if q else None

    def _cost_and_edge(self, symbol: str, qty: int, price: float,
                       equity: float, risk_pct: float, side: str) -> dict:
        """P7b: expected transaction cost (spread + sqrt market impact),
        and — for BUY only — the expected edge (% move to target from
        quant.verdict.analyze(), a simplification, not a full
        probability-weighted EV) so RiskEngine can gate edge < 2x cost."""
        df = self._provider.get_candles(symbol)
        if len(df) < 30:
            return {}
        cost = expected_trade_cost(df, qty, price)
        out = {"cost": cost}
        if side == "BUY":
            try:
                v = qs_verdict(df, account=equity, risk_pct=risk_pct)
                out["edge_pct"] = round(
                    abs(v["target"] / v["entry"] - 1) * 100, 3)
            except Exception:
                pass
        return out

    def _regime_gate(self, symbol: str) -> dict:
        """P7c: classify Bull/Bear/Storm via the P2 HMM and return its
        policy. Storm fires an alert (audit + regime.interrupt event) —
        the policy gates sizing and which entries step() allows."""
        df = self._provider.get_candles(symbol)
        if len(df) < 90:
            return {"regime": "Bull", "policy": REGIME_POLICY["Bull"]}
        rc = classify_regime(df["Close"].pct_change())
        self._state.set(f"regime.{symbol}",
                        {"regime": rc["regime"], "policy": rc["policy"]},
                        source="risk")
        if rc.get("regime") == "Storm":
            self._audit.record(
                "RiskEngine", "STORM REGIME ALERT", trigger=symbol,
                model="quant.regime_gate (P2 HMM Bull/Bear/Storm mapping)",
                reasoning=(f"{symbol}: HMM's highest-volatility state is "
                          f"active — STORM regime. No new trades; tighten "
                          f"stops on any existing position."),
                data=rc)
            self._bus.publish(Event("regime.alert", {"symbol": symbol, **rc},
                                    source="risk"))
        return rc

    def _cached_regime_gate(self, symbol: str) -> dict | None:
        """step()'s universe-scale refit budget: returns the cached
        state.regime.{symbol} read if it's still within
        REGIME_REFIT_TTL_S of _regime_gate()'s last real fit for this
        symbol, else None (caller decides whether this cycle's refit
        budget allows a fresh classification). _regime_gate() itself is
        untouched -- this only reads what it already persists to state,
        so nothing here changes its own tested behavior."""
        cached = self._state.get(f"regime.{symbol}")
        if cached is not None and not _cooldown_ok(
                self._state, f"regime_fit.{symbol}", self.REGIME_REFIT_TTL_S):
            return cached
        return None

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

    def _fundamental(self, symbol: str, df=None,
                     held_candles: dict | None = None) -> dict:
        news = self._state.get(f"news.{symbol}") or {}
        out = {"bullish_pct": news.get("bullish_pct"),
               "bearish_pct": news.get("bearish_pct"),
               "high_impact_soon": high_impact_soon(self._state.get("macro"))}
        if df is not None and held_candles:
            m, note = book_overlap_mult(df, held_candles)
            out["book_corr_mult"] = m
            out["book_corr_note"] = note
        return out

    def analyze(self, symbol: str, equity: float = 10000.0,
                risk_pct: float = 1.0, held: dict | None = None,
                regime: str | None = None,
                candles: pd.DataFrame | None = None,
                require_discount: bool = False,
                held_candles: dict | None = None) -> dict:
        """QuantSignal fusion: the 5-gate Playbook + 7-model verdict drive
        the signal; the reasoning IS the playbook instruction.

        regime: P7c Bull/Bear/Storm from quant.regime_gate. In Storm, an
        existing position's fallback stop tightens from 6% to 3% below
        entry — "tighten all stops" per the regime policy.

        candles: pre-fetched bars (step()'s universe-wide batched fetch,
        one yf.download() for every symbol instead of N round-trips) --
        falls back to a live per-symbol fetch when not supplied, so every
        other caller (research(), tests, direct use) is unaffected."""
        df = candles if candles is not None else self._provider.get_candles(symbol)
        if len(df) < 220:
            return {"symbol": symbol, "signal": "NONE",
                    "why": "insufficient history"}
        price = float(df["Close"].iloc[-1])
        if held and held.get("qty", 0) > 0:
            stop_frac = 0.97 if regime == "Storm" else 0.94
            pb = build_playbook(df, account=equity, risk_pct=risk_pct,
                                in_position=True,
                                entry=float(held["avg_price"]),
                                stop=float(held.get(
                                    "stop", held["avg_price"] * stop_frac)),
                                require_discount=require_discount)
            instr = pb["instruction"]
            if "EXIT" in instr:
                sig = "SELL"
            elif "TIGHTEN" in instr:
                sig = "TRAIL"
            else:
                sig = "NONE"
            why = f"EXIT — {instr}" if sig == "SELL" else (
                f"HOLD/TRAIL — {instr}" if sig == "TRAIL"
                else f"PLAYBOOK {pb['urgency']}: {instr}")
            out = {"symbol": symbol, "signal": sig, "price": price,
                   "urgency": pb["urgency"], "mode": "MANAGE",
                   "gates": f"{pb['greens']}/5",
                   "new_stop": pb.get("trail_suggestion"),
                   "zone": (pb.get("zone") or {}).get("label"),
                   "market_mode": (pb.get("market_mode") or {}).get("mode"),
                   "why": why}
        else:
            pb = build_playbook(df, account=equity, risk_pct=risk_pct,
                                require_discount=require_discount,
                                fundamental=self._fundamental(
                                    symbol, df=df, held_candles=held_candles))
            sig = "BUY" if pb["urgency"] in ("🟢 ACTIONABLE",
                                             "🟡 FAST SETUP") else "NONE"
            z = pb.get("zone") or {}
            md = pb.get("market_mode") or {}
            why = (f"ENTRY — {pb['instruction']}")
            out = {"symbol": symbol, "signal": sig, "price": price,
                   "urgency": pb["urgency"], "mode": "ENTRY",
                   "gates": f"{pb['greens']}/5",
                   "shares": pb.get("plan", {}).get("shares", 0),
                   "stop": pb.get("plan", {}).get("stop"),
                   "zone": z.get("label"),
                   "market_mode": md.get("mode"),
                   "risk_pct": (pb.get("risk_budget") or {}).get("risk_pct"),
                   "why": why}
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

    def correlation_monitor(self) -> dict:
        """P7f: rolling 20-day correlation regime across the live book —
        a hard alert at >=0.7 avg pairwise correlation, and an early-
        warning flag when correlations are trending toward 1 even below
        that threshold (de-risk BEFORE the drawdown, not after). Distinct
        from correlation_watch()'s static, full-history snapshot; this
        also produces a sizing policy that step() applies to new
        entries, stacking with the P7c/P7e gates rather than replacing
        them."""
        pos = self._broker.positions
        if len(pos) < 2:
            return {"policy": CORRELATION_POLICY["normal"]}
        rets = {}
        for t in pos:
            df = self._provider.get_candles(t)
            if len(df) >= 60:
                rets[t] = df["Close"].pct_change().dropna()
        if len(rets) < 2:
            return {"policy": CORRELATION_POLICY["normal"]}
        out = correlation_regime(rets)
        if "error" in out:
            return {**out, "policy": CORRELATION_POLICY["normal"]}
        self._state.set("correlation_regime", out, source="risk")
        if out["alert"] or out["converging_early_warning"]:
            self._audit.record(
                "RiskEngine", "CORRELATION REGIME ALERT",
                model="rolling 20d pairwise correlation + trend",
                reasoning=(f"{out['verdict']}: avg pairwise correlation "
                          f"{out['current_avg_correlation']} (trend "
                          f"{out['trend_slope_per_day']:+.5f}/day across "
                          f"the live book)"),
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

    def daily_report(self, watchlist: list[str] | None = None,
                     scan_universe: list[str] | None = None,
                     reports_dir: str = "reports") -> dict:
        """P7h: assemble + render the daily institutional report ->
        reports/YYYY-MM-DD.md, state.daily_report, audit.

        Honest limitation: "auto-generated at market close" needs a
        scheduler, which doesn't exist yet (CLAUDE.md roadmap "Later" —
        Hetzner/systemd deploy). This is the real report-generation
        logic, triggered on-demand until that infrastructure lands."""
        today = time.strftime("%Y-%m-%d")
        cutoff = time.time() - 24 * 3600
        marks = {t: (self._state.get(f"quotes.{t}") or {}).get(
                    "price", p["avg_price"])
                for t, p in self._broker.positions.items()}
        equity = self._broker.equity(marks)
        day_start = self._broker.day_start_equity
        pnl_today_pct = ((equity / day_start - 1) * 100) if day_start > 0 else 0.0
        pnl_today_dollars = equity - day_start

        fills_today = []
        for f in self._broker.fills:
            if f.get("ts", 0) >= cutoff:
                f2 = dict(f)
                f2["strategy"] = self.STRATEGY_NAME
                fills_today.append(f2)

        risk_limits = {}
        cfg = self._risk.cfg
        gross = self._broker.gross_exposure(marks)
        gross_cap = equity * cfg.max_gross_exposure_pct / 100
        if equity > 0:
            risk_limits["Gross exposure"] = {
                "used": f"${gross:,.0f}", "cap": f"${gross_cap:,.0f}",
                "pct": (gross / gross_cap * 100) if gross_cap > 0 else 0}
            risk_limits["Daily loss"] = {
                "used": f"{pnl_today_pct:+.2f}%",
                "cap": f"-{cfg.max_daily_loss_pct}%",
                "pct": (abs(min(pnl_today_pct, 0)) / cfg.max_daily_loss_pct * 100
                       if cfg.max_daily_loss_pct > 0 else 0)}
        if self._circuit_breaker:
            cbs = self._circuit_breaker.status()
            risk_limits["Drawdown circuit breaker"] = {
                "used": f"{cbs.get('drawdown_pct', 0)}%",
                "cap": "15% (hard stop)",
                "pct": cbs.get("drawdown_pct", 0) / 15 * 100}

        signals_today = {"n": 0, "buy": 0, "sell": 0}
        settled_today = {"n": 0, "win_rate_pct": 0.0, "mean_return_pct": 0.0}
        settled_cumulative = None
        notes = []
        if self._registry:
            st = self._registry._data.get(self.STRATEGY_NAME, {})
            todays_signals = [s for s in st.get("signals", [])
                             if s.get("ts", 0) >= cutoff]
            signals_today["n"] = len(todays_signals)
            signals_today["buy"] = sum(1 for s in todays_signals
                                       if s["direction"] == "BUY")
            signals_today["sell"] = sum(1 for s in todays_signals
                                        if s["direction"] == "SELL")
            settled_today_list = [s for s in st.get("signals", [])
                                  if s.get("settled")
                                  and s.get("settled_ts", 0) >= cutoff]
            if settled_today_list:
                rets = [s["forward_return"] for s in settled_today_list]
                settled_today = {
                    "n": len(rets),
                    "win_rate_pct": round(sum(1 for r in rets if r > 0)
                                          / len(rets) * 100, 1),
                    "mean_return_pct": round(sum(rets) / len(rets) * 100, 2)}
            all_settled = self._registry.settled_returns(self.STRATEGY_NAME)
            if len(all_settled):
                settled_cumulative = {
                    "n": len(all_settled),
                    "win_rate_pct": round(float((all_settled > 0).mean()) * 100, 1),
                    "mean_return_pct": round(float(all_settled.mean()) * 100, 2)}
        else:
            notes.append("No StrategyRegistry wired in — signal quality "
                         "section is unavailable.")

        suggestions = []
        etf_leaders: list[str] = []
        scan_symbols = scan_universe or watchlist
        if scan_symbols:
            scan = self.sector_scan(scan_symbols, account=equity)
            suggestions = scan.get("names", [])
            etf_leaders = scan.get("etf_leaders") or []
        else:
            notes.append("No watchlist passed — tomorrow's candidate "
                        "orders section skipped this run.")

        spy_ret = (self._state.get("benchmark.spy") or {}).get("ret_pct")
        ht = book_heat(self._broker.positions, marks)
        heat_pct = (ht["heat_$"] / equity * 100) if equity else 0.0
        closed_layers = []
        for sell, buy in pair_exits(self._broker.fills):
            if (sell.get("ts") or 0) < cutoff:
                continue
            headline = None
            news = self._state.get(f"news.{sell.get('ticker')}") or {}
            heads = news.get("headlines") or news.get("headline")
            if isinstance(heads, list) and heads:
                headline = str(heads[0].get("headline", heads[0])
                               if isinstance(heads[0], dict) else heads[0])
            elif isinstance(heads, str):
                headline = heads
            closed_layers.append(closed_trade_layers(
                sell, entry_fill=buy, news_headline=headline,
                spy_ret_pct=spy_ret))

        data = {
            "date": today, "equity": equity, "day_start_equity": day_start,
            "pnl_today_$": pnl_today_dollars, "pnl_today_pct": pnl_today_pct,
            "fills_today": fills_today, "risk_limits": risk_limits,
            "signals_today": signals_today, "settled_today": settled_today,
            "settled_cumulative": settled_cumulative,
            "suggestions": suggestions, "notes": notes,
            "spy_return_pct": spy_ret, "heat_pct": heat_pct,
            "closed_layers": closed_layers, "etf_leaders": etf_leaders,
        }
        report_md = render_report(data)

        os.makedirs(reports_dir, exist_ok=True)
        path = os.path.join(reports_dir, f"{today}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report_md)

        out = {"path": path, "markdown": report_md, **data}
        self._state.set("daily_report", {"path": path, "date": today,
                                         "pnl_today_$": pnl_today_dollars,
                                         "pnl_today_pct": pnl_today_pct},
                        source="research")
        self._audit.record(
            "Research", "DAILY REPORT", model="daily_report (P7h)",
            reasoning=(f"Daily report saved to {path}: equity "
                      f"${equity:,.2f}, today {pnl_today_dollars:+,.2f} "
                      f"({pnl_today_pct:+.2f}%), {len(fills_today)} "
                      f"fill(s), {signals_today['n']} signal(s) logged"),
            data={"path": path})
        return out

    def morning_briefing(self, watchlist: list[str] | None = None,
                         scan_universe: list[str] | None = None,
                         reports_dir: str = "reports") -> dict:
        """P8: pre-open snapshot -> reports/YYYY-MM-DD_morning.md — risk
        status (drawdown circuit breaker, correlation regime), a per-
        symbol regime read, and today's ranked candidates (sector_scan).
        Fired by the scheduler at 9:25 ET; read-only, nothing here
        executes a trade."""
        today = time.strftime("%Y-%m-%d")
        marks = {t: (self._state.get(f"quotes.{t}") or {}).get(
                    "price", p["avg_price"])
                for t, p in self._broker.positions.items()}
        equity = self._broker.equity(marks)
        cb = self._circuit_breaker.status() if self._circuit_breaker else None
        corr = self._state.get("correlation_regime")

        regimes: dict[str, str] = {}
        candidates: list = []
        etf_leaders: list[str] = []
        notes: list[str] = []
        if watchlist:
            for s in watchlist:
                regimes[s] = self._regime_gate(s)["regime"]
        scan_symbols = scan_universe or watchlist
        if scan_symbols:
            scan = self.sector_scan(scan_symbols, account=equity)
            candidates = scan.get("names", [])
            etf_leaders = scan.get("etf_leaders") or []
        if not watchlist and not scan_symbols:
            notes.append("No watchlist passed — regime reads and "
                        "candidate scan skipped this run.")

        data = {"date": today, "equity": equity, "circuit_breaker": cb,
               "correlation_regime": corr, "regimes": regimes,
               "candidates": candidates, "notes": notes,
               "etf_leaders": etf_leaders}
        md = render_morning_briefing(data)

        os.makedirs(reports_dir, exist_ok=True)
        path = os.path.join(reports_dir, f"{today}_morning.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(md)

        out = {"path": path, "markdown": md, **data}
        self._state.set("morning_briefing", {"path": path, "date": today},
                        source="research")
        self._audit.record(
            "Research", "MORNING BRIEFING", model="morning_briefing (P8)",
            reasoning=(f"Morning briefing saved to {path}: equity "
                      f"${equity:,.2f}, {len(candidates)} candidate(s) "
                      f"ranked"),
            data={"path": path})
        return out

    def refresh_desk(self, universe: list[str] | None = None,
                     chart_symbol: str | None = None,
                     account: float = 10000.0,
                     risk_pct: float = 1.0) -> dict:
        """Run the research stack the LAB tab used to hide behind buttons.

        Called from every decision cycle. Each job is cooldown-gated so a
        5-minute cycle does not re-Monte-Carlo the book or re-scan 550
        names. Failures never raise into the trading path.
        """
        ran: list[str] = []
        universe = universe or []
        chart_symbol = (chart_symbol
                        or self._state.get("ui.chart_symbol")
                        or "AAPL")
        try:
            if (len(self._broker.positions) >= 2
                    and _cooldown_ok(self._state, "desk.stress", 1800)):
                self.stress_test(n_paths=2000)
                _mark_ran(self._state, "desk.stress")
                ran.append("stress")
        except Exception:
            pass
        try:
            if universe and len(universe) >= 50:
                scan = self.sector_scan(universe, account=account,
                                        risk_pct=risk_pct)
                if scan:
                    held = set(self._broker.positions)
                    alts = [n for n in (scan.get("names") or [])
                            if n.get("ticker") not in held][:8]
                    self._state.set("desk.alternatives", alts, source="desk")
                    ran.append("sector")
        except Exception:
            pass
        try:
            if (self._broker.fills
                    and _cooldown_ok(self._state, "desk.execution", 600)):
                self.execution_quality_report(7)
                _mark_ran(self._state, "desk.execution")
                ran.append("execution")
        except Exception:
            pass
        try:
            if _cooldown_ok(self._state, "desk.flow", 600):
                names = list(self._broker.positions)[:4]
                if chart_symbol and chart_symbol not in names:
                    names.append(chart_symbol)
                for s in names:
                    self.scan_flow_confluence(s)
                _mark_ran(self._state, "desk.flow")
                ran.append("flow")
        except Exception:
            pass
        try:
            if (self._lse and getattr(self._lse, "key", None) and chart_symbol
                    and _cooldown_ok(self._state, "desk.surface", 1800)):
                chain = self._lse.options_chain(chart_symbol)
                self.ingest_chain(chart_symbol, chain)
                _mark_ran(self._state, "desk.surface")
                ran.append("surface")
        except Exception:
            pass
        try:
            if _cooldown_ok(self._state, "desk.spy", 1800):
                sdf = self._provider.get_candles("SPY", interval="1d",
                                                 lookback="6mo")
                if len(sdf):
                    px = float(sdf["Close"].iloc[-1])
                    base = self._state.get("benchmark.spy_base")
                    if not base:
                        self._state.set("benchmark.spy_base", px, source="desk")
                        base = px
                    self._state.set("benchmark.spy", {
                        "price": round(px, 2),
                        "ret_pct": round((px / float(base) - 1) * 100, 2),
                    }, source="desk")
                    _mark_ran(self._state, "desk.spy")
                    ran.append("spy")
        except Exception:
            pass
        stamp = {"ts": time.time(), "ran": ran}
        self._state.set("desk.last_refresh", stamp, source="desk")
        return stamp

    def stress_test(self, horizon_days: int = 21, n_paths: int = 10_000) -> dict:
        """P7g Monte Carlo of the current book. Cached; step() reads it."""
        pos = self._broker.positions
        if len(pos) < 1:
            return {"error": "no open positions to stress test"}
        rets, dollars = {}, {}
        marks = {t: (self._state.get(f"quotes.{t}") or {}).get(
                    "price", p["avg_price"]) for t, p in pos.items()}
        for t, p in pos.items():
            df = self._provider.get_candles(t)
            if len(df) >= 30:
                rets[t] = df["Close"].pct_change().dropna()
                dollars[t] = p["qty"] * marks.get(t, p["avg_price"])
        out = simulate_portfolio(rets, dollars, horizon_days=horizon_days,
                                 n_paths=n_paths)
        if "error" in out:
            return out
        budget = risk_budget_from_stress(out)
        out["risk_budget"] = budget
        self._state.set("portfolio_stress", out, source="risk")
        self._audit.record(
            "RiskEngine", "PORTFOLIO STRESS TEST",
            model=f"Monte Carlo, {n_paths:,} correlated paths, "
                 f"{horizon_days}d horizon",
            reasoning=(f"P(10% DD next {horizon_days}d) "
                      f"{out['p_10pct_drawdown_%']}% · expected worst week "
                      f"{out.get('expected_worst_week_%', '—')}% · 95% VaR "
                      f"${out['var95_$']:,.0f}"
                      + (" · ELEVATED RISK — next week's new-entry size "
                         "cut to 50%" if budget["elevated_risk"] else
                         " · risk budget normal")),
            data=out)
        return out

    def execution_quality_report(self, lookback_days: int = 7) -> dict:
        """P7d: slippage-vs-decision-price report over recent fills ->
        state.execution_quality + audit. Honestly empty if there's
        nothing settled to report yet."""
        out = slippage_report(self._broker.fills, lookback_days=lookback_days)
        if "error" in out:
            return out
        self._state.set("execution_quality", out, source="broker")
        self._audit.record(
            "PaperBroker", "EXECUTION QUALITY REPORT",
            model="slippage vs decision price",
            reasoning=(f"{out['n_fills']} fill(s) over the last "
                      f"{lookback_days}d: avg slippage "
                      f"{out['avg_slippage_pct']:+.3f}%, worst "
                      f"{out['worst_slippage_pct']:+.3f}%, total cost "
                      f"drag ${out['total_cost_drag_$']:,.2f}"),
            data=out)
        return out

    def sector_scan(self, symbols: list[str], account: float = 5000.0,
                    risk_pct: float = 1.0) -> dict:
        """Multi-factor sector/name ranking (quant.sector_engine): verdict's
        technical conviction tilted by whatever news sentiment, large
        option prints, and macro rate-trend readings are already cached in
        state (from scan_news/scan_flow/scan_macro — this does not fetch
        those itself). Sector comes from LSE company_profiles when the key
        is set, else 'Unclassified' — never guessed.

        Rate-limited to once per 5 minutes (RATE LIMIT PROTECTION, P8):
        this IS the "universe scan" — one batched candles fetch across
        every symbol passed in (P9: normally the full ~550-name S&P500+
        Nasdaq100 universe, not just the small decision-cycle watchlist —
        this function itself never places an order, so scanning wider
        doesn't touch the P9 decision-cycle/universe split). A repeat call
        inside the cooldown window returns the last cached state.sector_scan
        instead of re-fetching."""
        if not _cooldown_ok(self._state, "sector_scan", 300):
            cached = self._state.get("sector_scan")
            return {**cached, "throttled": True} if cached else {}
        _mark_ran(self._state, "sector_scan")

        batch_fn = getattr(self._provider, "get_candles_batch", None)
        data = (batch_fn(symbols) if batch_fn else
               {s: self._provider.get_candles(s) for s in symbols})
        data = {s: df for s, df in data.items() if len(df) >= 220}
        n_requested = len(symbols)
        if not data:
            return {}

        sectors = {}
        if self._lse and self._lse.key:
            # One un-filtered profiles pull instead of one call per symbol
            # -- the LSE endpoint already returns the whole reference
            # table (up to `limit`), so N single-symbol round-trips would
            # just be N-1 wasted calls once the universe is hundreds of
            # names instead of a handful.
            prof = self._lse.company_profiles(limit=5000)
            if len(prof):
                prof.columns = [str(c).lower() for c in prof.columns]
                sym_col = next((c for c in ("symbol", "ticker")
                               if c in prof.columns), None)
                if sym_col and "sector" in prof.columns:
                    prof[sym_col] = prof[sym_col].astype(str).str.upper()
                    by_symbol = prof.drop_duplicates(sym_col).set_index(sym_col)
                    for s in data:
                        if s in by_symbol.index:
                            sectors[s] = str(by_symbol.loc[s, "sector"])

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
        etf_leaders: list[str] = []
        try:
            etf_syms = list(SECTOR_ETFS.values()) + ["SPY"]
            batch = (batch_fn(etf_syms) if batch_fn else
                     {s: self._provider.get_candles(s) for s in etf_syms})
            spy_df = batch.get("SPY")
            etf_data = {name: batch[etf] for name, etf in SECTOR_ETFS.items()
                        if etf in batch and len(batch[etf]) >= 20}
            fed = None
            cpi = None
            macro = self._state.get("macro") or {}
            if isinstance(macro.get("fdtr"), dict) and macro["fdtr"].get("latest") is not None:
                fed = float(macro["fdtr"]["latest"])
            if isinstance(macro.get("cpi_yoy"), dict) and macro["cpi_yoy"].get("latest") is not None:
                cpi = float(macro["cpi_yoy"]["latest"])
            brd = breadth_from_universe(data, sectors)
            ranked_etf = rank_etfs(etf_data, spy_df, breadth=brd,
                                   fed_rate=fed, cpi=cpi)
            kept, side, etf_leaders = apply_etf_gate(out.get("names") or [],
                                                     ranked_etf, top_n=3)
            out["names"] = kept
            out["sidelined"] = side
            out["etf_rank"] = ranked_etf
            out["etf_leaders"] = etf_leaders
        except Exception:
            out["etf_leaders"] = []
            out["sidelined"] = []
            out["etf_rank"] = []
        out["n_requested"] = n_requested
        self._state.set("sector_scan", out, source="research")
        top_sec = out["sectors"][0]["sector"] if out["sectors"] else "none"
        top_names = ", ".join(f"{n['ticker']} ({n['target_score']})"
                              for n in out["names"][:3])
        self._audit.record(
            "Research", "SECTOR SCAN",
            model="quant.sector_engine (verdict + sentiment/flow/macro tilts)",
            reasoning=(f"Scanned {out['n_scanned']}/{n_requested} names · "
                      f"top sector {top_sec} · top names: "
                      f"{top_names or 'none tradeable'} · "
                      f"{len(out['avoid'])} flagged to avoid"),
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

    def step(self, symbols: list[str], risk_pct: float = 1.0,
             bypass_incubation: bool = False,
             require_discount: bool = False) -> list[dict]:
        """One decision cycle over the watchlist. Returns executed fills.

        bypass_incubation: owner-controlled escape hatch (sidebar toggle,
        app.py) for a strategy with zero trading history — the P7a gate is
        correct long-term but blocks every entry forever if never bypassed
        starting from zero signals. Signals are still logged and settled
        normally either way, so the registry keeps accumulating real
        history toward promotion even while bypassed; this only skips the
        may_enter check below, never any of the other gates (regime,
        circuit breaker, correlation, cost)."""
        # a state key (not just the audit record below) so the AUDIT tab
        # can show this durably -- at universe scale a single cycle can
        # produce dozens of other audit records (SIGNAL LOGGED/PROPOSE
        # BUY/etc per symbol) that would otherwise bury this in tail(N)
        # before anyone sees it.
        self._state.set("decision_cycle.last_scan",
                        {"n_symbols": len(symbols), "ts": time.time()},
                        source="orchestrator")
        self._audit.record(
            "Orchestrator", "DECISION CYCLE", model="rule-v1",
            reasoning=f"Decision cycle: scanning {len(symbols)} symbols",
            data={"n_symbols": len(symbols)})
        try:
            marks0 = {t: (self._state.get(f"quotes.{t}") or {}).get(
                        "price", p["avg_price"])
                      for t, p in self._broker.positions.items()}
            self.refresh_desk(
                universe=symbols,
                chart_symbol=self._state.get("ui.chart_symbol"),
                account=self._broker.equity(marks0) if marks0 else 10_000.0,
                risk_pct=risk_pct)
        except Exception:
            pass
        fills = []
        prices_seen = {}
        may_enter = True
        try:
            from quant.macro_tape import read_tape
            from quant.session_gate import allow_new_entries
            tape = read_tape()
            sess = allow_new_entries()
            self._state.set("desk.tape", tape, source="orchestrator")
            self._state.set("desk.session_gate", sess, source="orchestrator")
        except Exception:
            tape = {"risk_off": False, "size_mult": 1.0, "why": "tape n/a"}
            sess = {"ok": True, "why": "session gate n/a"}
        incubation_bypassed = False
        if self._registry:
            self._registry.settle_signals(self.STRATEGY_NAME, self._settle_price)
            self._registry.evaluate_promotion(self.STRATEGY_NAME)
            # Forced ON until 20 signals are logged, then the caller's
            # bypass_incubation flag (sidebar toggle) takes over. This is
            # the P7a AAPL-shows-BUY-but-no-fill fix: a Cloud restart used
            # to lose the toggle and silently hold every entry.
            may_enter, incubation_bypassed = \
                self._registry.should_bypass_incubation(
                    self.STRATEGY_NAME, bypass_incubation)

        cb = None
        if self._circuit_breaker:
            marks = {t: (self._state.get(f"quotes.{t}") or {}).get(
                        "price", p["avg_price"])
                    for t, p in self._broker.positions.items()}
            eq_now = self._broker.equity(marks)
            if eq_now > 0:
                cb = self._circuit_breaker.update(eq_now)

        corr_pol = self.correlation_monitor().get(
            "policy", CORRELATION_POLICY["normal"])
        stress_budget = (self._state.get("portfolio_stress.risk_budget")
                        or {"size_multiplier": 1.0, "elevated_risk": False})

        # universe-scale batched candle fetch (see analyze()'s `candles`
        # param) -- one provider round-trip for every symbol instead of N.
        # FakeProvider (tests) has no get_candles_batch, so this is {} and
        # every symbol falls back to analyze()'s own per-symbol fetch,
        # unchanged from before this method existed.
        batch_fn = getattr(self._provider, "get_candles_batch", None)
        candles_by_symbol = batch_fn(symbols) if batch_fn else {}

        regime_refits_used = 0
        for s in symbols:
            held_pos = self._broker.positions.get(s)
            eq0 = self._broker.equity(prices_seen)
            rc = self._cached_regime_gate(s)
            if rc is None:
                if regime_refits_used < self.REGIME_REFIT_BUDGET_PER_CYCLE:
                    rc = self._regime_gate(s)
                    _mark_ran(self._state, f"regime_fit.{s}")
                    regime_refits_used += 1
                else:
                    # this cycle's refit budget is spent -- an unclassified
                    # symbol trades unrestricted rather than being blocked
                    # on a technicality (same honest fallback _regime_gate
                    # itself uses for <90 bars of history).
                    rc = {"regime": "Bull", "policy": REGIME_POLICY["Bull"]}
            pol = rc["policy"]
            sig = self.analyze(s, equity=eq0, risk_pct=risk_pct,
                               held=held_pos, regime=rc["regime"],
                               candles=candles_by_symbol.get(s),
                               require_discount=require_discount,
                               held_candles={t: candles_by_symbol[t]
                                             for t in self._broker.positions
                                             if t in candles_by_symbol
                                             and t != s})
            price = sig.get("price", 0)
            if price:
                prices_seen[s] = price
            held = held_pos.get("qty", 0) if held_pos else 0

            if sig["signal"] in ("BUY", "SELL") and self._registry and price > 0:
                self._registry.log_signal(self.STRATEGY_NAME, s,
                                          sig["signal"], price,
                                          regime=rc["regime"])

            if sig["signal"] == "BUY" and not held and price > 0:
                if not sess.get("ok", True):
                    self._audit.record(
                        "Orchestrator", "STAND DOWN (SESSION)",
                        trigger=f"signals.{s}", model="rule-v1",
                        reasoning=f"{s}: {sess.get('why')}",
                        data={"symbol": s, "signal": "BUY"})
                    continue
                if tape.get("size_mult", 1.0) <= 0:
                    self._audit.record(
                        "Orchestrator", "STAND DOWN (TAPE)",
                        trigger=f"signals.{s}", model="rule-v1",
                        reasoning=f"{s}: {tape.get('why')}",
                        data={"symbol": s, "signal": "BUY", "tape": tape})
                    continue
                if not may_enter:
                    self._audit.record(
                        "Orchestrator", "SIGNAL LOGGED (INCUBATION)",
                        trigger=f"signals.{s}", model="rule-v1",
                        reasoning=(f"{s}: BUY signal logged but NOT traded — "
                                  f"strategy '{self.STRATEGY_NAME}' is still "
                                  f"in INCUBATION (P7a promotion gate)"),
                        data={"symbol": s, "signal": "BUY", "price": price})
                    continue
                if not pol["new_trades_allowed"]:
                    self._audit.record(
                        "Orchestrator", "SIGNAL LOGGED (STORM REGIME)",
                        trigger=f"signals.{s}", model="rule-v1",
                        reasoning=(f"{s}: BUY signal logged but NOT traded "
                                  f"— STORM regime forbids new trades "
                                  f"(P7c regime gate)"),
                        data={"symbol": s, "signal": "BUY", "price": price,
                             "regime": rc["regime"]})
                    continue
                if pol["dip_only"] and sig.get("urgency") != "🟡 FAST SETUP":
                    self._audit.record(
                        "Orchestrator", "SIGNAL LOGGED (BEAR REGIME)",
                        trigger=f"signals.{s}", model="rule-v1",
                        reasoning=(f"{s}: BUY signal logged but NOT traded "
                                  f"— BEAR regime only allows dip-buys at "
                                  f"extremes, this was a trend entry "
                                  f"(P7c regime gate)"),
                        data={"symbol": s, "signal": "BUY", "price": price,
                             "regime": rc["regime"]})
                    continue
                if cb and (cb["halted"] or cb["only_risk_reducing"]):
                    self._audit.record(
                        "Orchestrator", "SIGNAL LOGGED (CIRCUIT BREAKER)",
                        trigger=f"signals.{s}", model="rule-v1",
                        reasoning=(f"{s}: BUY signal logged but NOT traded "
                                  f"— drawdown circuit breaker at "
                                  f"{cb['drawdown_pct']}% from peak "
                                  f"({'HALTED, needs manual reset' if cb['halted'] else 'risk-reducing trades only'})"),
                        data={"symbol": s, "signal": "BUY", "price": price,
                             "circuit_breaker": cb})
                    continue
                if not corr_pol["new_trades_allowed"]:
                    self._audit.record(
                        "Orchestrator", "SIGNAL LOGGED (CORRELATION ALERT)",
                        trigger=f"signals.{s}", model="rule-v1",
                        reasoning=(f"{s}: BUY signal logged but NOT traded "
                                  f"— rolling correlation regime ALERT, "
                                  f"the book is effectively one trade "
                                  f"(P7f correlation monitor)"),
                        data={"symbol": s, "signal": "BUY", "price": price})
                    continue
                qty = int(sig.get("shares") or 0)
                if pol["size_multiplier"] < 1.0:
                    qty = int(qty * pol["size_multiplier"])
                if corr_pol["size_multiplier"] < 1.0:
                    qty = int(qty * corr_pol["size_multiplier"])
                if stress_budget["size_multiplier"] < 1.0:
                    qty = int(qty * stress_budget["size_multiplier"])
                if cb and cb["size_multiplier"] < 1.0:
                    qty = int(qty * cb["size_multiplier"])
                if tape.get("size_mult", 1.0) < 1.0:
                    qty = int(qty * float(tape["size_mult"]))
                if qty < 1:
                    self._audit.record(
                        "Orchestrator", "SIGNAL LOGGED (SIZE ROUNDED TO 0)",
                        trigger=f"signals.{s}", model="rule-v1",
                        reasoning=(f"{s}: BUY signal logged but NOT traded "
                                  f"— position size rounded down to 0 "
                                  f"shares after regime/correlation/stress/"
                                  f"circuit-breaker size multipliers"),
                        data={"symbol": s, "signal": "BUY", "price": price,
                             "raw_shares": sig.get("shares")})
                    continue
                order = Order(s, "BUY", qty, reason=sig["why"],
                              stop=sig.get("stop"))
                ce = self._cost_and_edge(s, qty, price, eq0, risk_pct, "BUY")
                cost_note = ""
                if ce.get("cost"):
                    c = ce["cost"]
                    cost_note = (f" · expected cost {c['expected_cost_pct']}% "
                                f"(${c['expected_cost_$']:,.2f}) vs expected "
                                f"edge {ce.get('edge_pct', '—')}%")
                    order.slippage = float(min(
                        max(c.get("spread_pct", 0.1) / 200.0, 0.0003), 0.008))
                bypass_note = (" · ⚠️ P7a INCUBATION gate BYPASSED (testing "
                               "mode)" if incubation_bypassed else "")
                self._audit.record("Orchestrator", "PROPOSE BUY",
                                   trigger=f"signals.{s}", model="rule-v1",
                                   reasoning=sig["why"] + cost_note + bypass_note,
                                   data={"qty": qty, "price": price,
                                        "incubation_bypassed": incubation_bypassed,
                                        **ce})
                order = self._risk.review(order, self._broker, price,
                                          cost_info={
                                              "expected_cost_pct":
                                                  ce.get("cost", {}).get("expected_cost_pct"),
                                              "expected_edge_pct": ce.get("edge_pct")}
                                          if ce.get("cost") else None)
                if order.approved:
                    f = self._broker.execute(order, price)
                    if f:
                        fills.append(f)
            elif sig["signal"] == "TRAIL" and held:
                new_stop = sig.get("new_stop")
                if new_stop:
                    moved = self._broker.update_stop(s, float(new_stop))
                    self._audit.record(
                        "Orchestrator", "TRAIL STOP",
                        trigger=f"signals.{s}", model="rule-v1",
                        reasoning=(sig["why"] + ("" if moved
                                   else " · stop not raised (already higher)")),
                        data={"symbol": s, "new_stop": new_stop,
                              "moved": moved})
            elif sig["signal"] == "SELL" and held:
                # exits are never gated by the promotion status
                order = Order(s, "SELL", held, reason=sig["why"])
                ce = self._cost_and_edge(s, held, price, eq0, risk_pct, "SELL")
                cost_note = ""
                if ce.get("cost"):
                    c = ce["cost"]
                    cost_note = (f" · expected cost {c['expected_cost_pct']}% "
                                f"(${c['expected_cost_$']:,.2f})")
                self._audit.record("Orchestrator", "PROPOSE SELL",
                                   trigger=f"signals.{s}", model="rule-v1",
                                   reasoning=sig["why"] + cost_note,
                                   data={"qty": held, **ce})
                order = self._risk.review(order, self._broker, price)
                if order.approved:
                    f = self._broker.execute(order, price)
                    if f:
                        fills.append(f)
            elif sig["signal"] == "BUY" and held:
                # not a new entry -- no pyramiding into an existing
                # position. Audited so a BUY in the Signals table with no
                # matching fill in TRADES isn't a silent mystery.
                self._audit.record(
                    "Orchestrator", "SIGNAL IGNORED (ALREADY HELD)",
                    trigger=f"signals.{s}", model="rule-v1",
                    reasoning=(f"{s}: BUY signal but {held} shares already "
                              f"held — no new entry placed (not pyramiding)"),
                    data={"symbol": s, "signal": "BUY", "price": price,
                         "held_qty": held})
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
