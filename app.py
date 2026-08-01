"""QuantTrader — pro terminal UI (v0.2). Thin shell; engine in core/data/ai."""
from __future__ import annotations

import dataclasses
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai.orchestrator import RuleOrchestrator, TOOL_SCHEMAS
from core.engine import AuditLog, PaperBroker, RiskEngine
from core.state import Config, EventBus, GlobalState, market_status
from core.circuit_breaker import DrawdownCircuitBreaker
from core.scheduler import TradingScheduler
from core.strategy_registry import MIN_SIGNALS_TO_PROMOTE, StrategyRegistry
from data.news import NewsProvider
from data.providers import (CompositeProvider, LSEProvider, PollingFeed,
                            YahooProvider, filter_price_outliers)
from data.universe import load_universe

st.set_page_config(page_title="QuantTrader", page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded")

ACCENT = "#22c55e"

# P9: the decision cycle (real paper trades) stays on this small, explicit
# watchlist -- deliberately NOT the full ~550-symbol universe below, which
# only drives the feed's quote scanning. Screening the whole universe for
# tradeable setups every 5 minutes is a real, separate feature (universe
# pre-filter / scanner.scan_setups wiring) that hasn't been requested.
TRADING_WATCHLIST = ["SPY", "QQQ", "AAPL", "NVDA"]
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=IBM+Plex+Mono:wght@400;600&display=swap');
html, body, .stApp {{ background:#0a0a0a !important; color:#d4d4d8;
  font-family:'Inter',sans-serif; }}
#MainMenu, footer {{ visibility:hidden; }}
.block-container {{ padding-top:0.4rem; max-width:100% !important; }}
.qt-nav {{ display:flex; align-items:center; gap:26px; padding:10px 6px;
  border-bottom:1px solid #1c1c1f; }}
.qt-logo {{ font-weight:800; font-size:1.05rem; color:#fff; }}
.qt-logo span {{ color:{ACCENT}; }}
.qt-sub {{ font-size:.6rem; letter-spacing:2px; color:#71717a; }}
.qt-link {{ font-size:.85rem; color:#a1a1aa; }}
.qt-link.active {{ color:#fff; border-bottom:2px solid {ACCENT};
  padding-bottom:8px; }}
.qt-strip {{ display:flex; border:1px solid #1c1c1f; border-left:0;
  border-right:0; margin:0 -1rem 14px -1rem; background:#0d0d0f; }}
.qt-stat {{ flex:1; text-align:center; padding:10px 4px;
  border-right:1px solid #1c1c1f; }}
.qt-stat .v {{ font-family:'IBM Plex Mono',monospace; font-weight:600;
  font-size:1.02rem; color:#fff; }}
.qt-stat .v.g {{ color:{ACCENT}; }} .qt-stat .v.r {{ color:#ef4444; }}
.qt-stat .k {{ font-size:.58rem; letter-spacing:1.5px; color:#71717a; }}
section[data-testid="stSidebar"] {{ background:#0d0d0f !important;
  border-right:1px solid #1c1c1f; }}
section[data-testid="stSidebar"] .stExpander {{ border:1px solid #1c1c1f;
  border-radius:6px; background:#0f0f12; margin-bottom:6px; }}
section[data-testid="stSidebar"] summary {{ font-size:.72rem !important;
  letter-spacing:1.5px; text-transform:uppercase; color:#e4e4e7 !important; }}
.stButton>button {{ border-radius:6px; border:1px solid #27272a;
  background:#141417; color:#d4d4d8; font-weight:600; }}
.stButton>button[kind="primary"] {{ background:{ACCENT}; color:#052e16;
  border:0; font-weight:800; letter-spacing:1px; }}
.qt-panel {{ border:1px solid #1c1c1f; border-radius:8px; background:#0d0d0f;
  padding:12px 16px; margin-bottom:10px; }}
div[data-testid="stDataFrame"] {{ font-family:'IBM Plex Mono',monospace;
  font-size:.82rem; }}
div[data-testid="stMetric"] {{ background:#0d0d0f; border:1px solid #1c1c1f;
  border-radius:8px; padding:8px 12px; }}
.qt-audit {{ border-left:3px solid {ACCENT}; padding:7px 12px; margin:5px 0;
  background:#0f0f12; border-radius:0 6px 6px 0;
  font-family:'IBM Plex Mono',monospace; font-size:.78rem; }}
.qt-audit.veto {{ border-left-color:#ef4444; }}
.qt-audit .who {{ color:{ACCENT}; font-weight:600; }}
.qt-audit.veto .who {{ color:#ef4444; }}
.qt-audit .t {{ color:#52525b; float:right; }}
h3 {{ color:#fff !important; font-size:1rem !important; }}
.stTabs [data-baseweb="tab"] {{ font-size:.8rem; letter-spacing:1.5px;
  text-transform:uppercase; }}
.stTabs [aria-selected="true"] {{ color:{ACCENT} !important; }}
</style>""", unsafe_allow_html=True)

PLOT = dict(paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            font=dict(color="#a1a1aa", family="IBM Plex Mono", size=11),
            xaxis=dict(gridcolor="#161618", rangeslider_visible=False),
            yaxis=dict(gridcolor="#161618", side="right"))


@st.cache_resource
def get_engine():
    cfg = Config()
    bus = EventBus()
    state = GlobalState(bus)
    audit = AuditLog(bus)
    lse = LSEProvider(cfg.lse_api_key, cfg.lse_base_url)
    news = NewsProvider(cfg.news_api_key)
    provider = CompositeProvider([lse, YahooProvider()], state)
    broker = PaperBroker(cfg, bus, state, audit)
    circuit_breaker = DrawdownCircuitBreaker(audit)
    risk = RiskEngine(cfg, bus, state, audit, circuit_breaker=circuit_breaker)
    registry = StrategyRegistry(audit)
    orch = RuleOrchestrator(bus, state, audit, risk, broker, provider,
                            news=news, lse=lse, registry=registry,
                            circuit_breaker=circuit_breaker)
    # P9: fetch-once-and-cache S&P 500 + Nasdaq-100 universe (~550
    # symbols) -- the feed scans ALL of it; the decision cycle keeps
    # trading only TRADING_WATCHLIST (see module-level comment above).
    universe = load_universe()
    feed = PollingFeed(
        bus, state, provider, universe, interval_s=30,
        priority_fn=lambda: sorted(set(broker.positions) |
                                   set(TRADING_WATCHLIST)),
        batch_size=50)
    # risk_pct_fn is a callable, not a fixed value, so the scheduler's
    # background thread always sees whatever the sidebar currently has
    # set (state.ui.risk_pct), not a snapshot frozen at process start.
    scheduler = TradingScheduler(
        orch,
        symbols_fn=lambda: TRADING_WATCHLIST,
        risk_pct_fn=lambda: state.get("ui.risk_pct") or 1.0,
        # P7a bypass default ON (see sidebar STRATEGY PROMOTION expander):
        # with zero signal history the gate would otherwise block every
        # entry forever, so the system trades from day one until the
        # owner turns it off.
        bypass_incubation_fn=lambda: state.get("ui.bypass_incubation", True),
        # sector_scan's candidate ranking (morning briefing/daily report)
        # scans the full universe, not just TRADING_WATCHLIST -- see
        # module comment above; the decision cycle itself is unaffected.
        universe_fn=lambda: universe)
    scheduler.start()
    state.set("session", {"started": time.strftime(
        "%Y-%m-%d %H:%M UTC", time.gmtime())})
    state.set("ui.watchlist", TRADING_WATCHLIST, source="ui")
    return dict(cfg=cfg, bus=bus, state=state, audit=audit, lse=lse,
                news=news, provider=provider, broker=broker, risk=risk,
                registry=registry, circuit_breaker=circuit_breaker,
                orch=orch, feed=feed, scheduler=scheduler, universe=universe)


E = get_engine()
cfg, state, audit = E["cfg"], E["state"], E["audit"]
broker, risk, orch, feed = E["broker"], E["risk"], E["orch"], E["feed"]
registry = E["registry"]
circuit_breaker = E["circuit_breaker"]
scheduler = E["scheduler"]
universe = E["universe"]
quotes = state.get("quotes") or {}


@st.cache_resource
def _autostart_feed(_feed):
    """Process-wide singleton starter: the decorated body only ever runs
    once per process (st.cache_resource), so the feed starts exactly
    once no matter how many browser sessions/reruns hit this line."""
    _feed.start()
    return True


@st.fragment(run_every=1)
def render_cycle_countdown():
    """P9: live countdown to the scheduler's next automatic decision
    cycle — the button above is now just a manual override, this is
    what tells the owner the auto-run is actually the primary path."""
    ms = market_status()
    if ms["session"] != "open":
        st.caption(f"Auto decision cycle paused — market {ms['label'].lower()}.")
        return
    nrt = scheduler.next_run("decision_cycle")
    if nrt is None:
        st.caption("Scheduler not running.")
        return
    remaining = max(0, int((nrt - datetime.now(nrt.tzinfo)).total_seconds()))
    m, s = divmod(remaining, 60)
    st.caption(f"Auto-running every 5 min during market hours · "
              f"next cycle in {m}m {s}s")

# ---------------------------------------------------------------------------
# LEFT RAIL
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='qt-logo'>◆ Quant<span>Trader</span></div>"
                "<div class='qt-sub'>AUTONOMOUS PAPER DESK</div>",
                unsafe_allow_html=True)
    st.write("")
    with st.expander("INSTRUMENT", expanded=True):
        chart_sym = (st.text_input(
            "Chart symbol (feed always scans full universe)",
            "AAPL").strip().upper() or "AAPL")
        tf = st.select_slider("Timeframe", ["1h", "1d", "1wk"], value="1d")
        st.caption(f"Trading watchlist (decision cycle): "
                  f"{', '.join(TRADING_WATCHLIST)}")
    with st.expander("CONFIGURATION", expanded=True):
        st.caption(f"Paper capital · ${cfg.starting_cash:,.0f}")
        rp = st.slider("Risk per position %", 0.5, 3.0, 1.0, 0.25)
        state.set("ui.risk_pct", rp, source="ui")
        deep = st.toggle("Options greeks pass (LSE)",
                         value=bool(cfg.lse_api_key),
                         disabled=not cfg.lse_api_key,
                         help="Pull the chart symbol's chain with "
                              "precomputed greeks each cycle")
        news_pass = st.toggle("News + sentiment pass (Finnhub)",
                              value=bool(cfg.news_api_key),
                              disabled=not cfg.news_api_key,
                              help="Headlines + sentiment per watchlist "
                                   "symbol each cycle")
        macro_pass = st.toggle("Macro + flow pass (LSE)",
                               value=bool(cfg.lse_api_key),
                               disabled=not cfg.lse_api_key,
                               help="Rates/CPI/economic-calendar snapshot "
                                    "plus large options-print alerts on "
                                    "the chart symbol each cycle")
    with st.expander("PORTFOLIO CAPITAL", expanded=True):
        aum_in = st.number_input(
            "Total Portfolio Capital (AUM) $", min_value=0.0,
            value=float(cfg.aum or cfg.starting_cash), step=500.0,
            help="Your real total capital. 0 falls back to the paper "
                 "broker's own live equity. Position-size caps below are "
                 "measured against this number.")
        mode_label = st.radio("Max position size", ["% of AUM", "Fixed $"],
                              horizontal=True,
                              index=1 if cfg.max_position_mode == "fixed"
                              else 0)
        if mode_label == "Fixed $":
            fixed_cap_in = st.number_input(
                "Max $ per position", min_value=0.0,
                value=float(cfg.max_position_fixed_usd or 1000.0), step=100.0)
            pct_cap_in = cfg.max_position_pct
            mode_val = "fixed"
        else:
            pct_cap_in = st.slider("Max % of AUM per position", 1.0, 100.0,
                                   float(cfg.max_position_pct), 1.0)
            fixed_cap_in = cfg.max_position_fixed_usd
            mode_val = "pct"
    with st.expander("RISK MANAGEMENT", expanded=True):
        st.caption(f"Position cap · {pct_cap_in}% of AUM" if mode_val == "pct"
                   else f"Position cap · ${fixed_cap_in:,.0f} fixed")
        st.caption(f"Gross exposure · ≤{cfg.max_gross_exposure_pct}%")
        st.caption(f"Daily loss halt · −{cfg.max_daily_loss_pct}%")
        st.caption(f"VaR ceiling · {cfg.max_var_pct}%")
        st.caption("⛔ RiskEngine veto: ABSOLUTE")
    with st.expander("STRATEGY PROMOTION (P7a)", expanded=True):
        strat_name = orch.STRATEGY_NAME
        s_status = registry.status(strat_name)
        counts = registry.signal_counts(strat_name)
        if s_status == StrategyRegistry.STATUS_PAPER:
            st.caption(f"🟢 PAPER — {strat_name}")
            st.caption("Promoted: entries execute normally.")
        else:
            st.caption(f"🔬 INCUBATION — {strat_name}")
            st.caption(f"{counts['settled']}/{MIN_SIGNALS_TO_PROMOTE} settled "
                       f"signals needed · new entries held back, signals "
                       f"still logged")
        bypass_gate = st.toggle(
            f"P7a gate: INCUBATION (need {MIN_SIGNALS_TO_PROMOTE} signals) "
            "[BYPASS FOR TESTING]",
            value=state.get("ui.bypass_incubation", True),
            help="ON: new-entry signals go straight to risk review and "
                 "execution even while INCUBATION, so the system trades "
                 "from day one with zero history. Signals are still logged "
                 "toward promotion either way. OFF: restores the normal "
                 "P7a gate — new entries wait for promotion. Never affects "
                 "exits, which are always allowed regardless of status.")
        state.set("ui.bypass_incubation", bypass_gate, source="ui")
        if bypass_gate and s_status != StrategyRegistry.STATUS_PAPER:
            st.caption("⚠️ Gate bypassed — entries execute despite INCUBATION.")
        st.caption(f"Signals: {counts['total']} total · {counts['settled']} "
                   f"settled · {counts['pending']} pending")
        last_val = registry.last_validation(strat_name)
        if last_val.get("decision") not in (None, "NOT ENOUGH SIGNALS"):
            bc = last_val.get("bootstrap", {})
            st.caption(f"Last eval: {last_val['decision']} · bootstrap CI "
                       f"[{bc.get('CI90_low_%', '—')}%, "
                       f"{bc.get('CI90_high_%', '—')}%]")
        perf_by_regime = registry.performance_by_regime(strat_name)
        if perf_by_regime:
            st.caption("Per-regime (P7c): " + " · ".join(
                f"{r} n={d['n']} mean={d['mean_return_%']}% "
                f"win={d['win_rate_%']}%"
                for r, d in perf_by_regime.items()))
    with st.expander("REGIME (P7c)", expanded=False):
        reg_info = state.get(f"regime.{chart_sym}")
        if reg_info:
            r_badge = {"Bull": "🟢", "Bear": "🟡", "Storm": "🔴"}.get(
                reg_info["regime"], "⚪")
            st.caption(f"{r_badge} {chart_sym}: {reg_info['regime']} regime")
            pol = reg_info["policy"]
            st.caption(f"Size {pol['size_multiplier']:.0%}"
                       + (" · dip-buys only" if pol["dip_only"] else "")
                       + ("" if pol["new_trades_allowed"] else " · NO NEW TRADES")
                       + (" · stops tightened" if pol["tighten_stops"] else ""))
        else:
            st.caption("Run a decision cycle to classify the current regime.")
    with st.expander("DRAWDOWN CIRCUIT BREAKER (P7e)", expanded=True):
        cbs = circuit_breaker.status()
        dd = cbs.get("drawdown_pct", 0.0)
        if cbs.get("halted"):
            st.caption(f"🔴 HALTED — {dd}% drawdown from peak "
                       f"${cbs.get('peak_equity', 0):,.0f}")
            st.caption("New entries blocked until a manual reset. "
                       "Existing positions can still be exited.")
            reset_reason = st.text_input(
                "Reason for reset (required)", key="cb_reset_reason")
            if st.button("Reset circuit breaker", use_container_width=True):
                if reset_reason.strip():
                    circuit_breaker.manual_reset(reset_reason)
                    st.rerun()
                else:
                    st.warning("A written reason is required to reset.")
        else:
            badge = ("🟡" if cbs.get("only_risk_reducing") else
                     "🟠" if cbs.get("size_multiplier", 1.0) < 1.0 else "🟢")
            st.caption(f"{badge} {dd}% drawdown from peak "
                       f"${cbs.get('peak_equity', 0):,.0f}")
            st.caption(f"Size multiplier: {cbs.get('size_multiplier', 1.0):.0%}"
                       + (" · risk-reducing only"
                          if cbs.get("only_risk_reducing") else ""))
    with st.expander("DATA CHAIN", expanded=False):
        if cfg.lse_api_key:
            st.caption("🟢 LSE vault (verified contract) → Yahoo failsafe")
            st.caption("WS parked for roadmap #7: "
                       f"`{E['lse'].WS_URL_ROADMAP}`")
            if st.button("Vault usage"):
                st.json(E["lse"].usage() or {"note": "no response"})
        else:
            st.caption("⚪ Yahoo only — add LSE_API_KEY in Secrets/.env")
    with st.expander("SCHEDULER (P8)", expanded=False):
        sst = scheduler.status()
        if sst["running"]:
            for j in sst["jobs"]:
                st.caption(f"🟢 {j['id']} · next run {j['next_run'] or '—'}")
        else:
            st.caption("⚫ not running")
        st.caption("Decision cycle every 5min (market open only) · "
                   "morning briefing 9:25 ET · daily report 16:05 ET")
    st.write("")
    # P9: the scheduler is the primary path now (fires every 5min during
    # market hours, see SCHEDULER expander above) -- this button is a
    # manual override for testing/impatience, no longer required for
    # normal operation, so it's de-emphasized (no more `type="primary"`).
    run = st.button("⚡ Force cycle now", use_container_width=True,
                    help="Manual override — the scheduler already runs "
                         "this automatically every 5 minutes during "
                         "market hours.")
    render_cycle_countdown()

    # BUG FIX 3 + auto-feed: start once on app load, no button press
    # needed. _autostart_feed is st.cache_resource (process-wide, runs
    # exactly once); the session_state flag just tracks it for this
    # browser session's status caption below. feed.symbols is the fixed
    # universe baked in at construction (get_engine) — nothing to
    # refresh here anymore, unlike the old user-typed watchlist.
    if "feed_auto_started" not in st.session_state:
        st.session_state.feed_auto_started = _autostart_feed(feed)

    fc1, fc2 = st.columns(2)
    if fc1.button("▶ Feed", use_container_width=True):
        feed.start()
        st.rerun()
    if fc2.button("⏹ Stop", use_container_width=True):
        feed.stop()
        st.rerun()
    throttle = state.get("feed.throttled")
    if throttle and throttle.get("retry_at", 0) > time.time():
        st.caption(f"🟠 throttled — retrying at "
                   f"{time.strftime('%H:%M:%S', time.localtime(throttle['retry_at']))}"
                   f" · universe {len(universe)} symbols · {feed.interval_s}s")
    else:
        st.caption(("🟢 feed running (auto)" if feed.running else
                   "⚫ feed stopped") +
                  f" · universe {len(universe)} symbols · {feed.interval_s}s")
    scan_prog = state.get("feed.universe_scan_progress")
    if scan_prog:
        st.caption(f"Scanned {scan_prog['scanned']}/{scan_prog['total']} "
                  f"symbols this cycle")

E["risk"].cfg = dataclasses.replace(
    cfg, aum=aum_in, max_position_mode=mode_val,
    max_position_pct=pct_cap_in, max_position_fixed_usd=fixed_cap_in)


@st.fragment(run_every=1)
def render_header_clock():
    """REAL-TIME ITEM 6: live ET clock + market session badge, ticking
    once a second independent of the rest of the page."""
    ms = market_status()
    badge = {"open": "🟢", "pre": "🟡", "after": "🟡",
             "closed": "⚫"}.get(ms["session"], "⚫")
    st.markdown(
        f"<div style='text-align:right; font-family:\"IBM Plex Mono\","
        f"monospace; font-size:.8rem; color:#a1a1aa;'>"
        f"{badge} {ms['label']} · {ms['et_time'].strftime('%H:%M:%S')} ET"
        f"</div>", unsafe_allow_html=True)


@st.fragment(run_every=30)
def render_quote_strip():
    """REAL-TIME ITEM 1: the watchlist quote strip, refreshing every 30s
    off GlobalState directly — PollingFeed's background thread keeps
    state.quotes current independent of any Streamlit rerun, so this
    fragment must re-read state itself rather than close over the
    module-level `quotes` snapshot from the last full script run."""
    q_now = state.get("quotes") or {}
    if q_now:
        qc = st.columns(min(len(q_now), 6))
        for col, (s_, q) in zip(qc, list(q_now.items())[:6]):
            col.metric(s_, f"{q.get('price', 0):,.2f}",
                      f"{q.get('chg_pct', 0):+.2f}%")
        st.caption(f"Last updated {time.strftime('%H:%M:%S')}")
    else:
        st.caption("No quotes yet — feed starting…")


@st.fragment(run_every=60)
def render_open_book():
    """REAL-TIME ITEM 2: open-position mark-to-market, refreshing every
    60s off GlobalState directly (same reasoning as render_quote_strip —
    a fragment timer fires without a full script rerun)."""
    q_now = state.get("quotes") or {}
    marks_now = {t: q.get("price", 0) for t, q in q_now.items()}
    if broker.positions:
        eq_now = broker.equity(marks_now)
        exposure_basis = aum_in if aum_in > 0 else eq_now
        rows = [{"ticker": t, "qty": p["qty"],
                "avg": round(p["avg_price"], 2),
                "mark": marks_now.get(t, "—"),
                "P&L $": round((marks_now.get(t, p["avg_price"]) -
                                p["avg_price"]) * p["qty"], 0),
                "% of AUM": round(p["qty"] * marks_now.get(t, p["avg_price"])
                                  / exposure_basis * 100, 1)
                if exposure_basis > 0 else "—"}
               for t, p in broker.positions.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                    hide_index=True)
        st.caption(f"Exposure basis: ${exposure_basis:,.0f} "
                  f"({'declared AUM' if aum_in > 0 else 'live paper equity'})"
                  f" · updated {time.strftime('%H:%M:%S')}")
    else:
        st.caption("Flat.")


# ---------------------------------------------------------------------------
# NAV + LSE-style STATS STRIP: TRADES · WIN% · PF · P&L · DD · SR
# ---------------------------------------------------------------------------
nav_l, nav_r = st.columns([4, 1])
with nav_l:
    st.markdown("""
    <div class="qt-nav">
     <div><span class="qt-logo">◆ Quant<span style="color:#22c55e">Trader</span></span>
     <span class="qt-sub"> MARKET INTELLIGENCE</span></div>
     <span class="qt-link active">Terminal</span><span class="qt-link">Data</span>
     <span class="qt-link">AI Core</span><span class="qt-link">Docs</span>
    </div>""", unsafe_allow_html=True)
with nav_r:
    render_header_clock()

marks = {t: q.get("price", 0) for t, q in quotes.items()}
eq = broker.equity(marks)
ret = (eq / broker.start_equity - 1) * 100

# session equity curve -> DD + SR (session-based, labeled as such)
curve = state.get("portfolio.equity_curve") or []
if quotes and (not curve or abs(curve[-1][1] - eq) > 0.01):
    curve = (curve + [[time.time(), eq]])[-600:]
    state.set("portfolio.equity_curve", curve, source="ui")
vals = np.array([v for _, v in curve], dtype=float) if curve else np.array([])
dd = float(((np.maximum.accumulate(vals) - vals) /
            np.maximum.accumulate(vals)).max() * 100) if len(vals) > 1 else 0.0
steps = np.diff(vals) / vals[:-1] if len(vals) > 2 else np.array([])
sr = float(steps.mean() / steps.std() * np.sqrt(252)) \
    if len(steps) > 10 and steps.std() > 0 else None

sells = [f for f in broker.fills if f["side"] == "SELL"]
wins = [f for f in sells if f["realized"] > 0]
gw = sum(f["realized"] for f in wins)
gl = -sum(f["realized"] for f in sells if f["realized"] < 0)
pf = round(gw / gl, 2) if gl > 0 else ("∞" if gw > 0 else "—")
realized = sum(f["realized"] for f in broker.fills)
vetoes = len([r for r in audit.tail(200) if "VETO" in r.get("action", "")])
winrate = f"{len(wins) / len(sells) * 100:.0f}%" if sells else "—"

st.markdown(f"""
<div class="qt-strip">
 <div class="qt-stat"><div class="v">{len(sells)}</div><div class="k">TRADES</div></div>
 <div class="qt-stat"><div class="v {'g' if sells and len(wins)/max(len(sells),1)>=.5 else ''}">{winrate}</div><div class="k">WIN</div></div>
 <div class="qt-stat"><div class="v">{pf}</div><div class="k">PF</div></div>
 <div class="qt-stat"><div class="v {'g' if realized>=0 else 'r'}">${realized:+,.0f}</div><div class="k">P&L</div></div>
 <div class="qt-stat"><div class="v {'r' if dd>2 else ''}">{dd:.1f}%</div><div class="k">DD</div></div>
 <div class="qt-stat"><div class="v">{f"{sr:.2f}" if sr is not None else "—"}</div><div class="k">SR·SESSION</div></div>
 <div class="qt-stat"><div class="v">${eq:,.0f}</div><div class="k">EQUITY</div></div>
 <div class="qt-stat"><div class="v {'r' if vetoes else 'g'}">{vetoes or 'ARMED'}</div><div class="k">{'VETOES' if vetoes else 'RISK ENGINE'}</div></div>
</div>""", unsafe_allow_html=True)

if run:
    with st.spinner("Research → propose → risk review → execute…"):
        for s_ in TRADING_WATCHLIST:
            orch.research(s_)
            if news_pass and cfg.news_api_key:
                orch.scan_news(s_)
        if deep and cfg.lse_api_key:
            orch.ingest_chain(chart_sym,
                              E["lse"].options_chain(chart_sym))
        if macro_pass and cfg.lse_api_key:
            orch.scan_macro()
            orch.scan_flow(chart_sym)
        new_fills = orch.step(TRADING_WATCHLIST, risk_pct=rp,
                              bypass_incubation=bypass_gate)
    st.toast(f"Forced cycle complete — {len(new_fills)} fill(s) · research + "
             f"news/macro/flow in AUDIT")

# ---------------------------------------------------------------------------
# TABS — CHART | METRICS | TRADES | AUDIT
# ---------------------------------------------------------------------------
t_chart, t_metrics, t_trades, t_lab, t_audit = st.tabs(
    ["CHART", "METRICS", "TRADES", "LAB", "AUDIT"])

with t_chart:
    iv = {"1h": ("1h", "720d"), "1d": ("1d", "2y"),
          "1wk": ("1wk", "10y")}[tf]
    df = E["provider"].get_candles(chart_sym, interval=iv[0], lookback=iv[1])
    df = filter_price_outliers(df)         # BUG FIX 2: drop vendor-data spikes
    if len(df):
        fig = go.Figure(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
            close=df["Close"], name=chart_sym,
            increasing_line_color=ACCENT, increasing_fillcolor=ACCENT,
            decreasing_line_color="#ef4444", decreasing_fillcolor="#7f1d1d"))
        sf = [f for f in broker.fills if f["ticker"] == chart_sym]
        if sf:
            fd = pd.DataFrame(sf)
            fd["dt"] = pd.to_datetime(fd["ts"], unit="s")
            b, s2 = fd[fd["side"] == "BUY"], fd[fd["side"] == "SELL"]
            if len(b):
                fig.add_trace(go.Scatter(x=b["dt"], y=b["price"],
                              mode="markers", name="Entry",
                              marker=dict(symbol="triangle-up", size=13,
                                          color=ACCENT)))
            if len(s2):
                fig.add_trace(go.Scatter(x=s2["dt"], y=s2["price"],
                              mode="markers", name="Exit",
                              marker=dict(symbol="triangle-down", size=13,
                                          color="#f59e0b")))
        last = float(df["Close"].iloc[-1])
        fig.add_hline(y=last, line_color=ACCENT, line_width=1,
                      line_dash="dot", annotation_text=f"{last:,.2f}",
                      annotation_font_color=ACCENT)
        fig.update_layout(height=540, margin=dict(l=6, r=6, t=24, b=6),
                          legend=dict(orientation="h", y=1.06), **PLOT)
        st.plotly_chart(fig, use_container_width=True)
        sig = state.get(f"signals.{chart_sym}")
        res = state.get(f"research.{chart_sym}")
        opt = state.get(f"options.{chart_sym}")
        line = []
        if sig:
            line.append(f"signal <b style='color:{ACCENT}'>{sig['signal']}"
                        f"</b> — {sig.get('why','')}")
        if res:
            line.append(f"vol {res['ewma_ann_vol_pct']}% · P(up 20d) "
                        f"{res['p_up_20d_pct']}% · ±${res['exp_move_20d']}")
        if opt:
            line.append(f"chain {opt['contracts']} contracts · IV "
                        f"{opt.get('median_iv','—')} · max-γ strike "
                        f"{opt.get('max_gamma_strike','—')}")
        if line:
            st.markdown(f"<div class='qt-panel'>🎯 <b>{chart_sym}</b> · "
                        + " &nbsp;|&nbsp; ".join(line) + "</div>",
                        unsafe_allow_html=True)
        st.caption(f"Last updated {time.strftime('%H:%M:%S')}")
    else:
        st.info(f"No data for {chart_sym} — throttled or bad symbol.")

def _readable_cell(v):
    """dict/list cells (e.g. research.{symbol}.anomalies, a list of
    {name, citation, finding} dicts from quant.anomaly_library) render as
    literal "[object Object]" if handed straight to st.dataframe — Arrow
    has no display form for an arbitrary nested object. Flatten to a
    short human-readable string instead."""
    if isinstance(v, list):
        if v and isinstance(v[0], dict):
            key = "name" if "name" in v[0] else next(iter(v[0]), None)
            return ", ".join(str(item.get(key, item)) for item in v)
        return ", ".join(str(x) for x in v) if v else "—"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {v2}" for k, v2 in v.items()) or "—"
    return v


with t_metrics:
    render_quote_strip()
    for title, key in (("Signals", "signals"), ("Research", "research"),
                       ("Options (greeks distilled)", "options")):
        d = state.get(key) or {}
        if d:
            st.markdown(f"### {title}")
            rows = [{k: _readable_cell(v) for k, v in row.items()}
                   for row in d.values()]
            st.dataframe(pd.DataFrame(rows),
                         use_container_width=True, hide_index=True)
    macro_d = state.get("macro") or {}
    if macro_d:
        st.markdown("### Macro")
        lines = [f"{k} = {v['latest']} (as of {v.get('as_of', '—')})"
                for k, v in macro_d.items()
                if k != "upcoming_events" and isinstance(v, dict)]
        if lines:
            st.caption(" · ".join(lines))
        if macro_d.get("upcoming_events"):
            st.caption("Upcoming: " + "; ".join(
                f"{e['event']} ({e['date']})"
                for e in macro_d["upcoming_events"][:5]))

    news_d = state.get("news") or {}
    if news_d:
        st.markdown("### News & Sentiment")
        for sym, n in news_d.items():
            line = f"**{sym}**"
            if n.get("bullish_pct") is not None:
                line += (f" · bullish {n['bullish_pct']}% / "
                        f"bearish {n['bearish_pct']}%")
            st.caption(line)
            for h in (n.get("headlines") or [])[:3]:
                st.caption(f"— {h['headline']} ({h['source']})")

    flow_d = state.get("flow_alerts") or {}
    if flow_d:
        st.markdown("### Flow Alerts (large option prints)")
        for sym, f_ in flow_d.items():
            st.caption(f"**{sym}**: {len(f_['prints'])} print(s) ≥ "
                       f"${f_['min_premium']:,.0f} premium")

    if not (state.get("signals") or state.get("research")):
        st.caption("Run a decision cycle to populate.")
    st.caption(f"AI contract: {len(TOOL_SCHEMAS)} tools · LLM socket awaits "
               "ANTHROPIC_API_KEY · every call risk-reviewed. Signals/"
               f"research/options updated {time.strftime('%H:%M:%S')}.")

with t_trades:
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown("### Open book")
        render_open_book()
    bk = state.get("risk.book") or {}
    if bk:
        warn = bk.get("warning")
        st.markdown(f"<div class='qt-panel'>🕸️ <b>Correlation watch</b> · "
                    f"avg pairwise {bk.get('avg_correlation','—')} · heat "
                    f"${bk.get('naive_heat_$','—')}→"
                    f"${bk.get('corr_adj_heat_$','—')} · VaR "
                    f"{bk.get('var_VaR_%','—')}%"
                    + (" · <span style='color:#ef4444'>⚠️ CROWDED — "
                       "effectively one trade</span>" if warn else "")
                    + "</div>", unsafe_allow_html=True)
    corr_reg = state.get("correlation_regime")
    if corr_reg:
        st.markdown(f"<div class='qt-panel'>📈 <b>Correlation regime "
                    f"(P7f, rolling 20d)</b> · {corr_reg['verdict']} · "
                    f"avg corr {corr_reg['current_avg_correlation']} · "
                    f"trend {corr_reg['trend_slope_per_day']:+.5f}/day"
                    f"</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("### Fills")
        if broker.fills:
            fd = pd.DataFrame(broker.fills[::-1])
            fd["time"] = pd.to_datetime(fd["ts"], unit="s").dt.strftime(
                "%m-%d %H:%M")
            st.dataframe(fd[["time", "ticker", "side", "qty", "price",
                             "realized", "reason"]],
                         use_container_width=True, hide_index=True,
                         height=320)
        else:
            st.caption("No fills yet — only what survives the gates AND "
                       "the veto trades.")

with t_lab:
    from quant.sltp_opt import optimize_sltp
    from quant.var_lab import var_suite
    from quant.vol_surface import build_surface_grid
    lc1, lc2 = st.columns(2)
    with lc1:
        st.markdown("### 🧮 VaR Lab")
        vm = st.selectbox("Method", ["historical", "parametric",
                                     "cornish_fisher", "ewma"])
        vc, vh = st.columns(2)
        conf = vc.slider("Confidence %", 90.0, 99.9, 95.0, 0.5)
        hor = vh.slider("Horizon (days)", 1, 30, 1)
        look = st.slider("Lookback", 30, 500, 252)
        if st.button("Calculate VaR", use_container_width=True):
            d_ = E["provider"].get_candles(chart_sym)
            r_ = var_suite(d_["Close"], method=vm, conf=conf / 100,
                           horizon=hor, lookback=look,
                           value=broker.equity(marks))
            if "error" in r_:
                st.warning(r_["error"])
            else:
                m1, m2 = st.columns(2)
                m1.metric(f"VaR {conf:.0f}% / {hor}d",
                          f"${r_['VaR_$']:,.0f}", f"{r_['VaR_pct']}%")
                m2.metric("CVaR (expected shortfall)",
                          f"${r_['CVaR_$']:,.0f}", f"{r_['CVaR_pct']}%")
                st.caption(f"{vm} · {r_['lookback']} obs · on current "
                           f"paper equity")
    with lc2:
        st.markdown("### 🎯 SL/TP Optimizer")
        rank = st.selectbox("Rank by", ["sharpe", "pf", "win", "return",
                                        "min_dd", "expect", "rr"])
        hold_ = st.slider("Max hold (bars)", 5, 60, 20)
        if st.button("Optimize stops & targets", use_container_width=True):
            with st.spinner("Replaying every model signal × 25 combos…"):
                d_ = E["provider"].get_candles(chart_sym)
                g_ = optimize_sltp(d_, hold=hold_, rank_by=rank,
                                   capital=broker.equity(marks))
            if len(g_):
                b_ = g_.iloc[0]
                st.success(f"Best: SL {b_['SL']}×ATR / TP {b_['TP']}×ATR — "
                           f"{b_['win_pct']}% win · PF {b_['PF']} · "
                           f"Sharpe {b_['sharpe']} · DD {b_['max_dd_pct']}%")
                st.dataframe(g_, use_container_width=True, hide_index=True,
                             height=300)
            else:
                st.info("Not enough BUY signals in history to optimize.")

    st.markdown("### 🌋 Volatility Surface")
    if not cfg.lse_api_key:
        st.caption("Add LSE_API_KEY in Secrets/.env to build the vol surface "
                   "— no chain data without it.")
    elif st.button("Build vol surface", use_container_width=True):
        with st.spinner("Fetching chain + building surface…"):
            chain_ = E["lse"].options_chain(chart_sym)
            ing = orch.ingest_chain(chart_sym, chain_)
            grid = build_surface_grid(chain_)
        if "error" in grid:
            st.warning(grid["error"])
        else:
            fig2 = go.Figure(go.Surface(
                x=grid["strikes"], y=grid["dtes"], z=grid["iv_grid"],
                colorscale="Viridis"))
            fig2.update_layout(
                height=480, margin=dict(l=6, r=6, t=24, b=6),
                scene=dict(xaxis_title="Strike", yaxis_title="DTE",
                          zaxis_title="IV",
                          xaxis=dict(gridcolor="#161618"),
                          yaxis=dict(gridcolor="#161618"),
                          zaxis=dict(gridcolor="#161618"),
                          bgcolor="#0a0a0a"),
                paper_bgcolor="#0a0a0a",
                font=dict(color="#a1a1aa", family="IBM Plex Mono", size=11))
            st.plotly_chart(fig2, use_container_width=True)
            surf = ing.get("surface")
            if surf and surf.get("findings"):
                for f_ in surf["findings"]:
                    st.markdown(f"<div class='qt-panel'>{f_}</div>",
                               unsafe_allow_html=True)
            else:
                st.caption("Grid built, but the interpreter needs more "
                           "expiries/strikes (or delta/type columns) for "
                           "skew, term-structure or smile reads.")

    st.markdown("### 🎯 Sector & Target Scan")
    if st.button("Scan sectors & targets", use_container_width=True):
        with st.spinner(f"Scanning {len(E['universe'])}-name universe × "
                        f"verdict × tilts…"):
            scan = orch.sector_scan(E["universe"],
                                    account=broker.equity(marks),
                                    risk_pct=rp)
        if not scan:
            st.info("Not enough history on the universe symbols yet — "
                    "each needs 220+ bars.")
        else:
            n_req = scan.get("n_requested", "—")
            if scan.get("throttled"):
                st.caption(f"⏱ Rate-limited (once per 5 min) — showing the "
                          f"last scan: {scan['n_scanned']}/{n_req} names.")
            else:
                st.caption(f"Scanned {scan['n_scanned']}/{n_req} names.")
            if scan["sectors"]:
                st.markdown("**Ranked sectors**")
                st.dataframe(pd.DataFrame(scan["sectors"]),
                            use_container_width=True, hide_index=True)
            if scan["names"]:
                st.markdown("**Ranked names**")
                rows = [{"ticker": n["ticker"], "sector": n["sector"],
                        "verdict": n["verdict"], "target score": n["target_score"],
                        "entry": n["entry"], "stop": n["stop"],
                        "target": n["target"], "rr": n["rr"],
                        "avoid above": n.get("avoid_above"),
                        "avoid below": n.get("avoid_below"),
                        "why": "; ".join(n["reasons_pro"][:2] + n["tilt_reasons"])}
                       for n in scan["names"]]
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                            hide_index=True)
            else:
                st.caption("Nothing cleared a tradeable verdict today.")
            if scan["avoid"]:
                st.markdown("**Avoid**")
                st.dataframe(pd.DataFrame(scan["avoid"]),
                            use_container_width=True, hide_index=True)
            st.caption("Suggestions only — nothing here executes a trade. "
                       "Acting on any of these still goes through RUN "
                       "DECISION CYCLE's propose → RiskEngine veto → "
                       "PaperBroker → AuditLog chain.")

    st.markdown("### 🌊 Flow Confluence")
    if st.button("Scan flow confluence", use_container_width=True):
        from quant.orderflow import cvd as flow_cvd, vpin as flow_vpin
        with st.spinner("Tape (BVC/CVD/VPIN) × options positioning…"):
            df_flow = E["provider"].get_candles(chart_sym)
            fc = orch.scan_flow_confluence(chart_sym)
        if not fc:
            st.info(f"Not enough bars for {chart_sym} yet (need 40+).")
        else:
            c1f, c2f = st.columns([2, 1])
            with c1f:
                cd = flow_cvd(df_flow)
                if "error" not in cd:
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(
                        x=df_flow.index, y=df_flow["Close"], name="Price",
                        yaxis="y1", line=dict(color=ACCENT)))
                    fig3.add_trace(go.Scatter(
                        x=cd["cvd_series"].index, y=cd["cvd_series"].values,
                        name="CVD", yaxis="y2", line=dict(color="#f59e0b")))
                    fig3.update_layout(
                        height=340, margin=dict(l=6, r=6, t=24, b=6),
                        paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
                        font=dict(color="#a1a1aa", family="IBM Plex Mono", size=11),
                        yaxis=dict(title="Price", side="left", gridcolor="#161618"),
                        yaxis2=dict(title="CVD", side="right", overlaying="y",
                                   gridcolor="#161618"),
                        xaxis=dict(gridcolor="#161618"),
                        legend=dict(orientation="h", y=1.08))
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.caption(cd["error"])
            with c2f:
                vp = flow_vpin(df_flow)
                if "error" not in vp:
                    fig4 = go.Figure(go.Indicator(
                        mode="gauge+number", value=vp["percentile"],
                        title={"text": "VPIN toxicity percentile"},
                        gauge={"axis": {"range": [0, 100]},
                              "bar": {"color": "#ef4444" if vp["toxic"] else ACCENT},
                              "steps": [{"range": [0, 85], "color": "#141417"},
                                       {"range": [85, 100], "color": "#3f1212"}]}))
                    fig4.update_layout(
                        height=220, margin=dict(l=6, r=6, t=30, b=6),
                        paper_bgcolor="#0a0a0a",
                        font=dict(color="#a1a1aa", family="IBM Plex Mono", size=11))
                    st.plotly_chart(fig4, use_container_width=True)
                st.caption("Options premium imbalance (put ← → call)")
                st.progress(min(max((fc["options_score"] + 1) / 2, 0.0), 1.0))
            badge = {"CONFLUENCE LONG": ACCENT, "CONFLUENCE SHORT": "#ef4444",
                    "CONFLICT": "#f59e0b", "QUIET": "#71717a"}.get(fc["verdict"],
                                                                  "#71717a")
            st.markdown(
                f"<div class='qt-panel'><b style='color:{badge}'>{fc['verdict']}"
                f"</b> · tape {fc['tape_score']:+.2f} · options "
                f"{fc['options_score']:+.2f}<br>" +
                " · ".join(fc["tape_reasons"] + fc["options_reasons"]) +
                "</div>", unsafe_allow_html=True)

    st.markdown("### 📊 Execution Quality (P7d)")
    eq_days = st.slider("Lookback (days)", 1, 30, 7, key="eq_lookback")
    if st.button("Generate execution report", use_container_width=True):
        eqr = orch.execution_quality_report(lookback_days=eq_days)
        if "error" in eqr:
            st.info(eqr["error"])
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg slippage", f"{eqr['avg_slippage_pct']:+.3f}%")
            m2.metric("Worst fill", f"{eqr['worst_slippage_pct']:+.3f}%")
            m3.metric("Total cost drag", f"${eqr['total_cost_drag_$']:,.2f}")
            st.caption(f"{eqr['n_fills']} fill(s) over the last "
                       f"{eqr['lookback_days']} day(s)")
            if eqr["worst_fills"]:
                wf = pd.DataFrame(eqr["worst_fills"])
                wf["time"] = pd.to_datetime(wf["ts"], unit="s").dt.strftime(
                    "%m-%d %H:%M")
                st.dataframe(wf[["time", "ticker", "side", "qty",
                                "decision_price", "price", "slippage_pct"]],
                            use_container_width=True, hide_index=True)
            st.caption("PaperBroker currently applies a fixed 0.05% "
                       "slippage constant, not a market-condition model — "
                       "this report is real infrastructure over real "
                       "fills, but every number will cluster near that "
                       "fixed value until the broker's slippage model "
                       "itself becomes more realistic.")

    st.markdown("### 🎲 Portfolio Stress Test (P7g)")
    if st.button("Run Monte Carlo stress test (10,000 paths)",
                use_container_width=True):
        with st.spinner("Simulating 10,000 correlated paths of the book…"):
            stress = orch.stress_test()
        if "error" in stress:
            st.info(stress["error"])
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("P(10% DD next month)", f"{stress['p_10pct_drawdown_%']}%")
            m2.metric("Expected worst week",
                      f"{stress.get('expected_worst_week_%', '—')}%")
            m3.metric("95% VaR", f"${stress['var95_$']:,.0f}")
            budget = stress["risk_budget"]
            if budget["elevated_risk"]:
                st.warning("⚠️ Elevated risk — new-entry size cut to 50% "
                          "until the next stress test run.")
            else:
                st.caption("Risk budget normal — full new-entry sizing.")
            st.caption(f"{stress['n_paths']:,} paths · {stress['horizon_days']}d "
                       f"horizon · starting value ${stress['starting_value_$']:,.0f} "
                       f"· 95% CVaR ${stress['cvar95_$']:,.0f}")
            st.caption("Feeds into every decision cycle (auto or forced) "
                       "automatically: this result is cached and applied "
                       "to every new entry's sizing until the next time "
                       "this test runs.")

    st.markdown("### 📋 Daily Institutional Report (P7h)")
    if st.button("Generate today's report", use_container_width=True):
        with st.spinner("Assembling P&L, risk, signal quality, candidates…"):
            rep = orch.daily_report(watchlist=TRADING_WATCHLIST)
        st.success(f"Saved to `{rep['path']}`")
        st.markdown(rep["markdown"])
        st.download_button("Download markdown", rep["markdown"],
                           file_name=f"{rep['date']}.md", mime="text/markdown")
    st.caption("On-demand for now — no scheduler exists yet to genuinely "
               "auto-run this at market close (see CLAUDE.md roadmap).")

with t_audit:
    st.markdown("### Audit timeline — trigger → model → reasoning")
    tail = audit.tail(20)
    if tail:
        for r in reversed(tail):
            veto = "VETO" in r["action"]
            st.markdown(
                f"<div class='qt-audit{' veto' if veto else ''}'>"
                f"<span class='who'>{r['actor']}</span> · {r['action']}"
                f"<span class='t'>"
                f"{time.strftime('%H:%M:%S', time.localtime(r['ts']))}</span>"
                f"<br>{r['reasoning']}</div>", unsafe_allow_html=True)
    else:
        st.caption("Nothing yet — run a decision cycle.")

st.caption("QuantTrader v0.4 · QuantSignal brain inside · LSE vault contract verified from official "
           "SDK · paper-only by constitution · keys via Secrets/.env only")
