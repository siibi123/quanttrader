"""QuantTrader — pro terminal UI (v0.2). Thin shell; engine in core/data/ai."""
from __future__ import annotations

import dataclasses
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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
from core.gist_store import get_gist_store, hydrate_runtime

st.set_page_config(page_title="QuantTrader", page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded")

ACCENT = "#22c55e"

# P9 amendment (owner, 2026-08-02): the decision cycle now evaluates the
# FULL loaded universe (~550 S&P500+Nasdaq100 symbols, see get_engine()'s
# `universe`), not a small hardcoded watchlist -- RuleOrchestrator.step()
# batches candle fetches and budgets/caches the P7c regime-HMM refit per
# cycle to make that affordable (see step()'s own docstring/comments).
# The sidebar's "Chart symbol" input ONLY controls what the CHART tab
# displays; it plays no role in what gets scanned or traded.
DEFAULT_CHART_SYMBOL = "AAPL"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
html, body, .stApp, [data-testid="stAppViewContainer"] {{
  background:#080a0b !important; color:#cfd3d6;
  font-family:'IBM Plex Sans',sans-serif;
}}
header[data-testid="stHeader"] {{
  background: transparent !important;
  height: 3.5rem !important;
}}
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stToolbar"] {{ visibility:hidden; }}
div[data-testid="stStatusWidget"], .stDeployButton {{ display:none !important; }}

/* Native Streamlit reopen control stays clickable but invisible —
   we draw ONE "Open desk" chip on top of it. Two green buttons was a bug. */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"],
div[data-testid="stSidebarCollapseButton"] {{
  opacity: 0 !important;
  pointer-events: none !important;
}}

.block-container {{
  padding-top: 3.4rem; max-width: 1480px !important; padding-bottom: 3rem;
}}
.qt-nav {{ display:flex; align-items:baseline; gap:16px; padding:2px 2px 14px;
  border-bottom:1px solid rgba(255,255,255,0.06); margin-bottom:6px; }}
.qt-logo {{ font-weight:700; font-size:1.08rem; color:#f4f1ea; letter-spacing:-0.03em; }}
.qt-logo span {{ color:{ACCENT}; }}
.qt-sub {{ font-size:.62rem; letter-spacing:0.22em; color:#6b7178; font-weight:500; }}
.qt-link {{ font-size:.76rem; color:#8b9198; }}
.qt-strip {{ display:flex; border:1px solid rgba(255,255,255,0.08);
  margin:8px 0 18px; background:linear-gradient(180deg,#101315,#0c0e10);
  border-radius:8px; overflow:hidden; }}
.qt-stat {{ flex:1; text-align:left; padding:14px 16px;
  border-right:1px solid rgba(255,255,255,0.06); }}
.qt-stat:last-child {{ border-right:0; }}
.qt-stat .v {{ font-family:'IBM Plex Mono',monospace; font-weight:500;
  font-size:1.08rem; color:#f4f1ea; letter-spacing:-0.02em; }}
.qt-stat .v.g {{ color:{ACCENT}; }} .qt-stat .v.r {{ color:#f07167; }}
.qt-stat .k {{ font-size:.58rem; letter-spacing:0.16em; color:#6b7178;
  margin-top:5px; font-weight:500; }}
section[data-testid="stSidebar"] {{
  background:#0b0d0f !important;
  border-right:1px solid rgba(255,255,255,0.07);
}}
section[data-testid="stSidebar"] > div {{ padding-top: 0.4rem; }}
section[data-testid="stSidebar"] .stExpander {{
  border:1px solid rgba(255,255,255,0.08); border-radius:8px;
  background:#101315; margin-bottom:10px; }}
section[data-testid="stSidebar"] summary {{ font-size:.68rem !important;
  letter-spacing:0.16em; text-transform:uppercase; color:#e8e6df !important; }}
.stButton>button {{ border-radius:8px; border:1px solid rgba(255,255,255,0.12);
  background:#14181c; color:#d5d0c7; font-weight:500; }}
.stButton>button:hover {{ border-color:{ACCENT}; color:#fff; }}
.stButton>button[kind="primary"] {{ background:{ACCENT}; color:#06210f;
  border:0; font-weight:700; letter-spacing:0.08em; }}
.qt-panel {{ border:1px solid rgba(255,255,255,0.08); border-radius:8px;
  background:#101315; padding:14px 16px; margin-bottom:10px; }}
.qt-kicker {{ font-size:.62rem; letter-spacing:0.18em; color:#6b7178;
  text-transform:uppercase; margin-bottom:6px; font-weight:600; }}
div[data-testid="stDataFrame"] {{ font-family:'IBM Plex Mono',monospace; font-size:.8rem; }}
div[data-testid="stMetric"] {{ background:#101315; border:1px solid rgba(255,255,255,0.08);
  border-radius:8px; padding:8px 12px; }}
.qt-audit {{ border-left:2px solid {ACCENT}; padding:8px 12px; margin:5px 0;
  background:#101315; font-family:'IBM Plex Mono',monospace; font-size:.76rem;
  border-radius:0 8px 8px 0; }}
.qt-audit.veto {{ border-left-color:#f07167; }}
.qt-audit .who {{ color:{ACCENT}; font-weight:600; }}
.qt-audit.veto .who {{ color:#f07167; }}
.qt-audit .t {{ color:#5c6370; float:right; }}
h3 {{ color:#f4f1ea !important; font-size:0.92rem !important;
  letter-spacing:0.04em; font-weight:600 !important; }}
.stTabs [data-baseweb="tab-list"] {{
  gap: 4px; border-bottom: 1px solid rgba(255,255,255,0.06); }}
.stTabs [data-baseweb="tab"] {{ font-size:.72rem; letter-spacing:0.14em;
  text-transform:uppercase; background:transparent; }}
.stTabs [aria-selected="true"] {{
  color:{ACCENT} !important;
  border-bottom: 2px solid {ACCENT} !important;
}}
.stSlider label, .stSelectbox label, .stToggle label, .stTextInput label,
.stNumberInput label {{ color:#8b9198 !important; }}
</style>""", unsafe_allow_html=True)

components.html(
    """
<script>
(function () {
  const d = window.parent.document;
  if (d.getElementById("qt-open-desk")) return;
  const b = d.createElement("button");
  b.id = "qt-open-desk";
  b.type = "button";
  b.textContent = "Open desk";
  b.setAttribute("aria-label", "Open desk panel");
  Object.assign(b.style, {
    position: "fixed",
    left: "16px",
    top: "16px",
    zIndex: "2147483647",
    display: "none",
    background: "#22c55e",
    color: "#06210f",
    border: "0",
    borderRadius: "8px",
    padding: "10px 14px",
    font: "700 12px/1 IBM Plex Sans, system-ui, sans-serif",
    letterSpacing: "0.12em",
    textTransform: "uppercase",
    cursor: "pointer",
    boxShadow: "0 8px 24px rgba(0,0,0,.45)",
  });
  function target() {
    return (
      d.querySelector('[data-testid="collapsedControl"]') ||
      d.querySelector('[data-testid="stExpandSidebarButton"]') ||
      d.querySelector('[data-testid="stSidebarCollapsedControl"] button') ||
      d.querySelector('[data-testid="stSidebarCollapsedControl"]') ||
      d.querySelector('button[aria-label="Open sidebar"]') ||
      d.querySelector('button[aria-label*="sidebar" i]')
    );
  }
  b.onclick = function () {
    const el = target();
    if (el) el.click();
  };
  d.body.appendChild(b);
  setInterval(function () {
    const side = d.querySelector('section[data-testid="stSidebar"]');
    const w = side ? side.getBoundingClientRect().width : 0;
    b.style.display = w < 80 ? "block" : "none";
  }, 250);
})();
</script>
    """,
    height=0,
)

PLOT = dict(paper_bgcolor="#070809", plot_bgcolor="#070809",
            font=dict(color="#8b9198", family="IBM Plex Mono", size=11),
            xaxis=dict(gridcolor="#161a1c", rangeslider_visible=False),
            yaxis=dict(gridcolor="#161a1c", side="right"))


@st.cache_data(ttl=1800, show_spinner=False)
def _desk_study(ohlc: pd.DataFrame, symbol: str, tf: str):
    """Attribution + walk-forward + vs buy-hold for the chart name only."""
    from quant.backtest import BTConfig, run_backtest, walk_forward
    from quant.desk_read import attribution, robustness
    from quant.signals import composite
    if ohlc is None or len(ohlc) < 130:
        return None
    att = attribution(composite(ohlc))
    out = {"att": att, "wf": None, "eq": None, "bh": None,
           "rob": None, "metrics": None}
    if tf == "1h":
        out["rob"] = robustness(pd.DataFrame(), {})
        return out
    try:
        cfg = BTConfig(starting_cash=10_000)
        bt = run_backtest(ohlc, cfg)
        wf = (walk_forward(ohlc, cfg, n_folds=4)
              if len(ohlc) >= 400 else pd.DataFrame())
        out["metrics"] = bt.metrics
        out["eq"] = bt.equity
        out["bh"] = bt.bh_equity
        out["wf"] = wf
        out["rob"] = robustness(wf, bt.metrics)
    except Exception:
        out["rob"] = {"label": "N/A",
                      "line": "Study failed on this series.",
                      "activity": None, "n_pos": 0, "n": 0}
    return out


@st.cache_resource
def get_engine():
    cfg = Config()
    bus = EventBus()
    state = GlobalState(bus)
    # Restore runtime files from the private gist BEFORE AuditLog /
    # PaperBroker / StrategyRegistry read them. Streamlit Cloud wipes
    # runtime/ on every reboot; without this, every restart is a new book.
    hydrate_runtime()
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
    # symbols), loaded a single time here (get_engine is @st.cache_resource
    # -- runs once per process) and held in this `universe` list for the
    # rest of the process's life. Both the feed AND the decision cycle
    # (via scheduler.universe_fn below) read this same cached list --
    # neither ever calls load_universe() again mid-session.
    universe = load_universe()
    # priority_fn/symbols_fn/universe_fn are callables (not fixed values)
    # so sidebar edits (chart symbol, risk %, bypass toggle) take effect
    # on the next tick/cycle without restarting the feed or scheduler.
    feed = PollingFeed(
        bus, state, provider, universe, interval_s=30,
        priority_fn=lambda: sorted(set(broker.positions) |
                                   {state.get("ui.chart_symbol")
                                    or DEFAULT_CHART_SYMBOL}),
        batch_size=50)
    scheduler = TradingScheduler(
        orch,
        # small regime-read set for the daily report/morning briefing
        # only (see TradingScheduler's docstring) -- NOT what gets traded.
        symbols_fn=lambda: sorted(set(broker.positions) |
                                  {state.get("ui.chart_symbol")
                                   or DEFAULT_CHART_SYMBOL}),
        risk_pct_fn=lambda: state.get("ui.risk_pct") or 1.0,
        # P7a bypass default ON (see sidebar STRATEGY PROMOTION expander):
        # with zero signal history the gate would otherwise block every
        # entry forever, so the system trades from day one until the
        # owner turns it off.
        bypass_incubation_fn=lambda: state.get("ui.bypass_incubation", True),
        require_discount_fn=lambda: bool(state.get("ui.discount_zone", True)),
        # the decision cycle AND sector_scan's candidate ranking both
        # scan this same cached full universe.
        universe_fn=lambda: universe)
    scheduler.start()
    state.set("session", {"started": time.strftime(
        "%Y-%m-%d %H:%M UTC", time.gmtime())})
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
            "Chart symbol (type any ticker)",
            DEFAULT_CHART_SYMBOL).strip().upper() or DEFAULT_CHART_SYMBOL)
        state.set("ui.chart_symbol", chart_sym, source="ui")
        tf = st.select_slider("Timeframe", ["1h", "1d", "1wk"], value="1d")
        st.caption("This only picks what the CHART tab displays — the "
                  "decision cycle always scans the full universe below, "
                  "regardless of this symbol.")
        st.caption(f"Decision cycle universe: {len(E['universe']):,} "
                  f"symbols (S&P 500 + Nasdaq-100)")
    with st.expander("CONFIGURATION", expanded=True):
        st.caption(f"Paper capital · ${cfg.starting_cash:,.0f}")
        # Desk sizes every ticket. This number is a silent hard cap,
        # not a per-trade target — no slider on the main path.
        DESK_RISK_CAP = 1.0
        rp = float(st.session_state.get("desk_risk_cap", DESK_RISK_CAP))
        state.set("ui.risk_pct", rp, source="ui")
        st.caption(f"Desk sizes each trade (zone, weekly, models, news, "
                   f"book overlap). Hard cap {rp:.2f}% of AUM — not a "
                   f"slider. Override only if you disagree with the desk.")
        with st.expander("Override cap (optional)", expanded=False):
            rp = st.slider("Hard cap % of AUM", 0.5, 3.0, rp, 0.25,
                           key="desk_risk_cap")
            state.set("ui.risk_pct", rp, source="ui")
        if "discount_zone" not in st.session_state:
            st.session_state.discount_zone = True
        discount_zone = st.toggle(
            "Discount-zone filter (don't buy premium)",
            key="discount_zone",
            help="Stand aside when the name is expensive in its last swing. "
                 "Buy the 0.618–0.786 discount pocket. B-X flipping negative "
                 "raises the stop instead of dumping the whole position.")
        state.set("ui.discount_zone", discount_zone, source="ui")
        if discount_zone:
            st.caption("Swing mode ON · wait for discount · trail on TIGHTEN")
        else:
            st.caption("Filter OFF · original 5-gate playbook")
        deep = st.toggle("Options greeks pass (LSE)",
                         value=bool(cfg.lse_api_key),
                         disabled=not cfg.lse_api_key,
                         help="Pull the chart symbol's chain with "
                              "precomputed greeks each cycle")
        news_pass = st.toggle("News + sentiment pass (Finnhub)",
                              value=bool(cfg.news_api_key),
                              disabled=not cfg.news_api_key,
                              help="Headlines + sentiment for the chart "
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
        n_logged = counts["total"]
        if s_status == StrategyRegistry.STATUS_PAPER:
            st.caption(f"🟢 PAPER — {strat_name}")
            st.caption("Promoted: entries execute normally.")
        else:
            st.caption(f"🔬 INCUBATION — {strat_name}")
            if n_logged < MIN_SIGNALS_TO_PROMOTE:
                st.caption(f"{n_logged}/{MIN_SIGNALS_TO_PROMOTE} signals logged · "
                           "bypass locked ON so new entries still fill")
            else:
                st.caption(f"{counts['settled']}/{MIN_SIGNALS_TO_PROMOTE} settled "
                           f"signals needed · new entries held back unless bypass is on")
        force_bypass = (s_status != StrategyRegistry.STATUS_PAPER
                        and n_logged < MIN_SIGNALS_TO_PROMOTE)
        if "p7a_bypass" not in st.session_state:
            st.session_state.p7a_bypass = True
        if force_bypass:
            st.session_state.p7a_bypass = True
        bypass_gate = st.toggle(
            f"P7a gate: INCUBATION (need {MIN_SIGNALS_TO_PROMOTE} signals) "
            "[BYPASS FOR TESTING]",
            key="p7a_bypass",
            disabled=force_bypass,
            help="ON: new-entry signals go straight to risk review and "
                 "execution even while INCUBATION, so the system trades "
                 "from day one with zero history. Forced ON until "
                 f"{MIN_SIGNALS_TO_PROMOTE} signals are logged. Signals are "
                 "still logged toward promotion either way. OFF: restores "
                 "the normal P7a gate — new entries wait for promotion. "
                 "Never affects exits, which are always allowed.")
        state.set("ui.bypass_incubation", bypass_gate, source="ui")
        if force_bypass:
            st.caption(f"Bypass locked ON until {MIN_SIGNALS_TO_PROMOTE} "
                       f"signals are logged "
                       f"({n_logged}/{MIN_SIGNALS_TO_PROMOTE}).")
        elif bypass_gate and s_status != StrategyRegistry.STATUS_PAPER:
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
        run_manual = st.button("Force cycle now", use_container_width=True,
                               help="Only if Cloud slept or you want a "
                                    "scan immediately. Not required.")
    st.write("")
    render_cycle_countdown()

    # Wake cycle: when the app process comes back (Cloud sleep, first
    # open) and the market is open and the last scan is stale, run once
    # without a button. Scheduler keeps going after that.
    last_scan = state.get("decision_cycle.last_scan") or {}
    last_ts = float(last_scan.get("ts") or 0)
    stale = (time.time() - last_ts) > 300
    session_open = market_status().get("session") == "open"
    if "wake_cycle_done" not in st.session_state:
        st.session_state.wake_cycle_done = False
    auto_wake = (session_open and stale
                 and not st.session_state.wake_cycle_done)
    if auto_wake:
        st.session_state.wake_cycle_done = True
    run = bool(run_manual or auto_wake)

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
                "stop": round(p["stop"], 2) if p.get("stop") else "—",
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
    alts = state.get("desk.alternatives") or []
    if alts:
        st.markdown("<div class='qt-kicker'>Desk alternatives — not held</div>",
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{
            "ticker": n.get("ticker"), "sector": n.get("sector"),
            "verdict": n.get("verdict"), "score": n.get("target_score"),
            "entry": n.get("entry"), "stop": n.get("stop"),
            "rr": n.get("rr"),
        } for n in alts[:6]]), use_container_width=True, hide_index=True)
        st.caption("Ranked by the live sector engine. They still have to "
                   "clear playbook + RiskEngine on the next cycle — this is "
                   "the bench, not a second broker.")


# ---------------------------------------------------------------------------
# NAV + LSE-style STATS STRIP: TRADES · WIN% · PF · P&L · DD · SR
# ---------------------------------------------------------------------------
nav_l, nav_r = st.columns([4, 1])
with nav_l:
    st.markdown("""
    <div class="qt-nav">
     <div><span class="qt-logo">Quant<span>Trader</span></span>
     <span class="qt-sub">PAPER DESK</span></div>
     <span class="qt-link">Live cycle every 5 min · research auto-feeds sizing</span>
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
    with st.spinner(f"Scanning {len(E['universe']):,} symbols → "
                    f"propose → risk review → execute…"):
        # research()/scan_news() are chart-symbol-scoped display features
        # (CHART/METRICS tabs), unrelated to the universe-wide decision
        # cycle below -- see the sidebar's "This only picks what the
        # CHART tab displays" caption.
        orch.research(chart_sym)
        if news_pass and cfg.news_api_key:
            orch.scan_news(chart_sym)
        if deep and cfg.lse_api_key:
            orch.ingest_chain(chart_sym,
                              E["lse"].options_chain(chart_sym))
        if macro_pass and cfg.lse_api_key:
            orch.scan_macro()
            orch.scan_flow(chart_sym)
        new_fills = orch.step(E["universe"], risk_pct=rp,
                              bypass_incubation=bypass_gate,
                              require_discount=discount_zone)
    st.toast(f"Forced cycle complete — {len(new_fills)} fill(s) · research + "
             f"news/macro/flow in AUDIT")

# ---------------------------------------------------------------------------
# TABS — CHART | METRICS | TRADES | AUDIT
# ---------------------------------------------------------------------------
t_chart, t_metrics, t_trades, t_lab, t_audit = st.tabs(
    ["CHART", "METRICS", "TRADES", "RESEARCH", "AUDIT"])

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
            st.markdown(f"<div class='qt-panel'><b>{chart_sym}</b> · "
                        + " &nbsp;|&nbsp; ".join(line) + "</div>",
                        unsafe_allow_html=True)

        study = _desk_study(df, chart_sym, tf)
        if study and study.get("att"):
            att, rob = study["att"], study.get("rob") or {}
            crowd_col = {"BROAD": ACCENT, "NARROW": "#e3b341",
                         "COUNTERTREND": "#f07167", "MIXED": "#8b9198"}.get(
                att["crowd"], "#8b9198")
            rob_col = {"ROBUST": ACCENT, "FRAGILE": "#f07167",
                       "UNSTABLE": "#e3b341", "N/A": "#8b9198"}.get(
                rob.get("label"), "#8b9198")
            st.markdown(
                f"<div class='qt-panel'><span class='qt-kicker'>"
                f"Name study · {chart_sym}</span><br>"
                f"<b style='color:{crowd_col}'>{att['line']}</b><br>"
                f"<b style='color:{rob_col}'>{rob.get('label','N/A')}</b> — "
                f"{rob.get('line','')}"
                + (f"<br>{rob['activity']}" if rob.get("activity") else "")
                + "</div>",
                unsafe_allow_html=True)
            c_att, c_eq = st.columns((1, 1))
            with c_att:
                rows = att["rows"]
                fig_a = go.Figure(go.Bar(
                    y=[r["model"] for r in rows][::-1],
                    x=[r["contrib"] for r in rows][::-1],
                    orientation="h",
                    marker_color=[ACCENT if r["contrib"] >= 0 else "#f07167"
                                  for r in rows][::-1],
                    name="contrib"))
                fig_a.update_layout(
                    height=240, margin=dict(l=8, r=8, t=8, b=8),
                    showlegend=False, **PLOT)
                fig_a.update_xaxes(title="weighted contribution")
                st.plotly_chart(fig_a, use_container_width=True)
            with c_eq:
                if study.get("eq") is not None and study.get("bh") is not None:
                    fig_e = go.Figure()
                    fig_e.add_trace(go.Scatter(
                        x=study["eq"].index, y=study["eq"].values,
                        name="Strategy", line=dict(color=ACCENT, width=1.6)))
                    fig_e.add_trace(go.Scatter(
                        x=study["bh"].index, y=study["bh"].values,
                        name="Buy & hold",
                        line=dict(color="#8b9198", width=1.2, dash="dot")))
                    fig_e.update_layout(
                        height=240, margin=dict(l=8, r=8, t=8, b=8),
                        legend=dict(orientation="h", y=1.12), **PLOT)
                    st.plotly_chart(fig_e, use_container_width=True)
                else:
                    st.caption("Walk-forward skipped on 1h (too heavy). Switch to 1d.")
            wf = study.get("wf")
            if wf is not None and len(wf):
                show = [c for c in ("fold", "start", "end", "Sharpe",
                                    "CAGR %", "Buy&Hold CAGR %", "Trades")
                        if c in wf.columns]
                st.caption("Walk-forward folds — last fold is the one that matters.")
                st.dataframe(wf[show], use_container_width=True, hide_index=True)

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
    from quant.scorecard import book_heat, trade_stats, vs_benchmark
    st.markdown("<div class='qt-kicker'>PM scorecard — vs SPY, not vanity P&L</div>",
                unsafe_allow_html=True)
    book_ret = (eq / broker.start_equity - 1) * 100 if broker.start_equity else 0.0
    spy = state.get("benchmark.spy") or {}
    alpha = vs_benchmark(book_ret, spy.get("ret_pct"))
    ts = trade_stats(broker.fills)
    ht = book_heat(broker.positions, marks)
    heat_pct = (ht["heat_$"] / eq * 100) if eq else 0.0
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Book", f"{alpha['book_pct']:+.2f}%")
    a2.metric("SPY (session)", "—" if alpha["spy_pct"] is None
              else f"{alpha['spy_pct']:+.2f}%")
    a3.metric("Excess vs SPY", "—" if alpha["excess_pct"] is None
              else f"{alpha['excess_pct']:+.2f}%")
    a4.metric("Heat to stops", f"{heat_pct:.1f}%")
    st.caption("Excess is this process vs SPY since the desk started — "
               "not a live audited track record. Heat = $ lost if every "
               "stop hits, as % of equity (cap 6%).")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Exits", ts["n_exits"])
    b2.metric("Win %", "—" if ts["win_rate"] is None else f"{ts['win_rate']}%")
    b3.metric("Expectancy / exit", "—" if ts["expectancy_$"] is None
              else f"${ts['expectancy_$']:+.0f}")
    b4.metric("PF", "—" if ts["profit_factor"] is None else str(ts["profit_factor"]))
    render_quote_strip()
    sig_d = state.get("signals") or {}
    if sig_d:
        st.markdown("### Signals")
        # step() now scans the full universe every cycle, so state.signals
        # holds one entry per symbol WITH enough history -- most of them
        # NONE (no setup today). Show only the actionable ones (BUY/SELL)
        # so 12 real signals out of 550 scanned don't drown in noise.
        non_none = {s: v for s, v in sig_d.items() if v.get("signal") != "NONE"}
        st.caption(f"{len(non_none)} non-NONE signal(s) out of "
                  f"{len(sig_d)} symbols scanned with enough history.")
        if non_none:
            rows = [{k: _readable_cell(v) for k, v in row.items()}
                   for row in non_none.values()]
            st.dataframe(pd.DataFrame(rows),
                         use_container_width=True, hide_index=True)
    for title, key in (("Research", "research"),
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
        if broker.fills:
            fd = pd.DataFrame(broker.fills[::-1])
            fd["time"] = pd.to_datetime(fd["ts"], unit="s").dt.strftime(
                "%m-%d %H:%M")
            buys = fd[fd["side"] == "BUY"]
            sells = fd[fd["side"] == "SELL"]
            st.markdown("### Entries — why we got in")
            if len(buys):
                st.dataframe(buys[["time", "ticker", "qty", "price", "reason"]],
                             use_container_width=True, hide_index=True,
                             height=220)
            else:
                st.caption("No entries yet.")
            st.markdown("### Exits — why we got out")
            if len(sells):
                st.dataframe(sells[["time", "ticker", "qty", "price",
                                    "realized", "reason"]],
                             use_container_width=True, hide_index=True,
                             height=220)
            else:
                st.caption("No exits yet.")
        else:
            st.markdown("### Fills")
            st.caption("No fills yet — only what survives the gates AND "
                       "the veto trades.")
        gist = get_gist_store()
        label = gist.last_saved_label()
        if label:
            st.caption(label)
        else:
            st.caption("Local runtime only — add GITHUB_TOKEN in Streamlit "
                       "Secrets so trades survive Cloud restarts.")

with t_lab:
    st.markdown("<div class=\'qt-kicker\'>Live desk — feeds the cycle automatically</div>",
                unsafe_allow_html=True)
    st.caption("No buttons. Stress, sector rank, flow, execution quality and "
               "alternatives refresh on the 5-minute decision cycle. "
               "You override sliders; the desk picks the rest.")
    desk = state.get("desk.last_refresh") or {}
    if desk.get("ts"):
        st.caption("Last desk refresh " + time.strftime("%H:%M:%S", time.localtime(desk["ts"]))
                   + ((" · " + ", ".join(desk.get("ran") or [])) if desk.get("ran") else ""))

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("### Book risk")
        stress = state.get("portfolio_stress") or {}
        bk = state.get("risk.book") or {}
        if stress and "error" not in stress:
            m1, m2, m3 = st.columns(3)
            m1.metric("P(10% DD)", f"{stress.get('p_10pct_drawdown_%', '—')}%")
            m2.metric("95% VaR", f"${stress.get('var95_$', 0):,.0f}")
            m3.metric("CVaR", f"${stress.get('cvar95_$', 0):,.0f}")
            budget = stress.get("risk_budget") or {}
            if budget.get("elevated_risk"):
                st.warning("Elevated — new-entry size already cut to 50%.")
            else:
                st.caption("Risk budget normal.")
        elif bk:
            st.caption(f"Live book VaR {bk.get('var_VaR_%', '—')}% · "
                       f"avg corr {bk.get('avg_correlation', '—')}")
        else:
            st.caption("Need two open names before Monte Carlo stress runs.")

        eqr = state.get("execution_quality") or {}
        st.markdown("### Execution")
        if eqr and "error" not in eqr:
            e1, e2, e3 = st.columns(3)
            e1.metric("Avg slip", f"{eqr.get('avg_slippage_pct', 0):+.3f}%")
            e2.metric("Worst", f"{eqr.get('worst_slippage_pct', 0):+.3f}%")
            e3.metric("Drag", f"${eqr.get('total_cost_drag_$', 0):,.2f}")
        else:
            st.caption("Fills will populate slippage vs decision price.")

    with r2:
        st.markdown("### Alternatives")
        scan = state.get("sector_scan") or {}
        if scan.get("etf_leaders"):
            st.caption("ETF gate: " + ", ".join(scan["etf_leaders"])
                       + " — names outside this are sidelined.")
        alts = state.get("desk.alternatives") or []
        if alts:
            st.dataframe(pd.DataFrame([{
                "ticker": n.get("ticker"), "sector": n.get("sector"),
                "verdict": n.get("verdict"), "score": n.get("target_score"),
                "rr": n.get("rr"),
                "why": "; ".join((n.get("reasons_pro") or [])[:2]),
            } for n in alts]), use_container_width=True, hide_index=True, height=280)
            st.caption("Not held. Next cycle still has to clear playbook + veto.")
        else:
            st.caption("Universe scan will rank names here after the next cycle.")

        st.markdown("### Flow")
        flow_d = state.get("flow") or {}
        if flow_d:
            for sym, fc in list(flow_d.items())[:4]:
                if not isinstance(fc, dict):
                    continue
                badge = {"CONFLUENCE LONG": ACCENT, "CONFLUENCE SHORT": "#f07167",
                         "CONFLICT": "#e3b341", "QUIET": "#6b7178"}.get(
                    fc.get("verdict"), "#6b7178")
                st.markdown(
                    f"<div class=\'qt-panel\'><b style=\'color:{badge}\'>{sym} "
                    f"{fc.get('verdict','')}</b> · tape {fc.get('tape_score', 0):+.2f} "
                    f"· options {fc.get('options_score', 0):+.2f}</div>",
                    unsafe_allow_html=True)
        else:
            st.caption("Tape × options confluence on the open book + chart symbol.")

    surf = (state.get(f"options.{chart_sym}") or {}).get("surface") or {}
    st.markdown("### Volatility surface")
    if not cfg.lse_api_key:
        st.caption("Needs LSE_API_KEY — skipped, not a trading gate.")
    elif surf.get("findings"):
        for f_ in surf["findings"]:
            st.markdown(f"<div class=\'qt-panel\'>{f_}</div>", unsafe_allow_html=True)
    else:
        st.caption("Chain interpreter runs with the cycle when the key is set. "
                   "Empty findings = not enough strikes/expiries, not a crash.")

    brief = state.get("morning_briefing") or {}
    last_rep = state.get("daily_report") or {}
    st.markdown("### Briefing")
    st.caption("Morning 09:25 ET · close report 16:05 ET — scheduler, not a button.")
    if brief.get("date") or last_rep.get("date"):
        st.caption(f"Last morning {brief.get('date', '—')} · last close {last_rep.get('date', '—')}")

with t_audit:
    st.markdown("### Audit timeline — trigger → model → reasoning")
    last_scan = state.get("decision_cycle.last_scan")
    if last_scan:
        st.caption(f"🔎 Last decision cycle: scanning "
                  f"{last_scan['n_symbols']:,} symbols · "
                  f"{time.strftime('%H:%M:%S', time.localtime(last_scan['ts']))}")
    # a universe-scale cycle can produce dozens of records in one pass
    # (SIGNAL LOGGED/PROPOSE BUY/etc per symbol) -- a wider tail than the
    # old 4-symbol-watchlist days gives more of that context on screen.
    tail = audit.tail(50)
    if tail:
        for r in reversed(tail):
            action = str(r.get("action") or "")
            if not action:
                continue
            veto = "VETO" in action
            try:
                ts_lbl = time.strftime(
                    "%H:%M:%S", time.localtime(float(r.get("ts") or 0)))
            except Exception:
                ts_lbl = ""
            st.markdown(
                f"<div class='qt-audit{' veto' if veto else ''}'>"
                f"<span class='who'>{r.get('actor') or '?'}</span> · {action}"
                f"<span class='t'>{ts_lbl}</span>"
                f"<br>{r.get('reasoning') or ''}</div>", unsafe_allow_html=True)
    else:
        st.caption("Nothing yet — run a decision cycle.")

st.caption("QuantTrader v0.4 · QuantSignal brain inside · LSE vault contract verified from official "
           "SDK · paper-only by constitution · keys via Secrets/.env only")
