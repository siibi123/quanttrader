"""QuantTrader — UI shell. A thin window onto the engine; all logic lives
in core/, data/, ai/. The engine runs identically headless on a VPS."""
from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai.orchestrator import RuleOrchestrator, TOOL_SCHEMAS
from core.engine import AuditLog, PaperBroker, RiskEngine
from core.state import Config, EventBus, GlobalState
from data.providers import (CompositeProvider, LSEProvider, PollingFeed,
                            YahooProvider)

st.set_page_config(page_title="QuantTrader", page_icon="◆", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
.stApp { background: radial-gradient(1200px 600px at 15% -10%, #0e1b34 0%, #070b14 55%) fixed; color:#dbe4f3; font-family:'Space Grotesk',sans-serif; }
h1,h2,h3 { color:#e8eefc !important; letter-spacing:.3px; }
.qt-hero { font-size:2rem; font-weight:700; background:linear-gradient(90deg,#67e8f9,#818cf8 60%,#c084fc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.qt-card { border:1px solid #1e2a44; border-radius:14px; padding:14px 18px; background:rgba(13,20,36,.82); box-shadow:0 6px 24px rgba(0,0,0,.35); margin-bottom:10px; }
.qt-audit { border-left:3px solid #67e8f9; padding:8px 12px; margin:6px 0; background:rgba(15,23,42,.7); border-radius:0 10px 10px 0; font-family:'IBM Plex Mono',monospace; font-size:.82rem; }
.qt-veto { border-left-color:#f87171; }
div[data-testid="stMetric"] { background:rgba(13,20,36,.82); border:1px solid #1e2a44; border-radius:12px; padding:10px 14px; }
div[data-testid="stDataFrame"] { font-family:'IBM Plex Mono',monospace; }
.stButton>button { border-radius:10px; border:1px solid #2a3a5f; background:#101a30; color:#dbe4f3; }
.stButton>button[kind="primary"] { background:linear-gradient(90deg,#0e7490,#4f46e5); border:0; }
</style>""", unsafe_allow_html=True)

PLOT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dbe4f3", family="IBM Plex Mono"),
            xaxis=dict(gridcolor="#16223b"), yaxis=dict(gridcolor="#16223b"))


# ---------------------------------------------------------------------------
# Engine singleton — one engine per server process, shared across sessions
# ---------------------------------------------------------------------------
@st.cache_resource
def get_engine():
    cfg = Config()
    bus = EventBus()
    state = GlobalState(bus)
    audit = AuditLog(bus)
    lse = LSEProvider(cfg.lse_api_key, cfg.lse_base_url)
    provider = CompositeProvider([lse, YahooProvider()], state)
    broker = PaperBroker(cfg, bus, state, audit)
    risk = RiskEngine(cfg, bus, state, audit)
    orch = RuleOrchestrator(bus, state, audit, risk, broker, provider)
    feed = PollingFeed(bus, state, provider,
                       ["SPY", "QQQ", "AAPL", "NVDA"], interval_s=45)
    state.set("session", {"started": time.strftime("%Y-%m-%d %H:%M UTC",
                                                   time.gmtime())})
    return dict(cfg=cfg, bus=bus, state=state, audit=audit, lse=lse,
                provider=provider, broker=broker, risk=risk, orch=orch,
                feed=feed)


E = get_engine()
cfg, state, audit = E["cfg"], E["state"], E["audit"]
broker, risk, orch, feed = E["broker"], E["risk"], E["orch"], E["feed"]

st.markdown("<div class='qt-hero'>◆ QuantTrader</div>", unsafe_allow_html=True)
st.caption("Event-driven quant platform · risk-vetoed autonomous paper desk · "
           "every action audited with reasoning · AI-ready by contract")

# ---- top row: connection + feed + kill switch -----------------------------
c1, c2, c3, c4 = st.columns([1.3, 1.3, 1, 1])
with c1:
    lse_ok = E["lse"].working is not None
    st.metric("Data chain", "LSE → Yahoo" if lse_ok else "Yahoo (LSE unprobed)")
    if cfg.lse_api_key and not lse_ok and st.button("🧪 Probe LSE endpoints"):
        with st.spinner("Probing…"):
            E["lse"].probe()
        st.rerun()
    elif not cfg.lse_api_key:
        st.caption("Add LSE_API_KEY to .env to enable the free feed")
with c2:
    st.metric("Feed", "🟢 RUNNING" if feed.running else "⚫ STOPPED",
              f"{len(feed.symbols)} symbols · {feed.interval_s}s")
    fc1, fc2 = st.columns(2)
    if fc1.button("▶ Start feed"):
        feed.start(); st.rerun()
    if fc2.button("⏹ Stop"):
        feed.stop(); st.rerun()
with c3:
    eq = broker.equity({t: q.get("price", 0) for t, q in
                        (state.get("quotes") or {}).items()})
    st.metric("Paper equity", f"${eq:,.0f}",
              f"{(eq / broker.start_equity - 1) * 100:+.2f}%")
with c4:
    st.metric("Risk engine", "🛡️ ARMED",
              f"pos≤{cfg.max_position_pct}% · day≤-{cfg.max_daily_loss_pct}%")

quotes = state.get("quotes") or {}
if quotes:
    qcols = st.columns(min(len(quotes), 6))
    for col, (s, q) in zip(qcols, list(quotes.items())[:6]):
        col.metric(s, f"{q.get('price', 0):,.2f}",
                   f"{q.get('chg_pct', 0):+.2f}%")

st.markdown("---")
left, right = st.columns([1.15, 1])

with left:
    st.subheader("🧠 Orchestrator")
    st.caption("Deterministic rule policy v1 — proposes with written "
               "reasoning; RiskEngine holds absolute veto; broker refuses "
               "anything unapproved. The LLM socket (ANTHROPIC_API_KEY) "
               f"plugs into the same {len(TOOL_SCHEMAS)}-tool contract.")
    syms = st.text_input("Watchlist", "SPY, QQQ, AAPL, NVDA")
    rp = st.slider("Risk per position %", 0.5, 3.0, 1.0, 0.25)
    if st.button("⚡ Run decision cycle", type="primary"):
        with st.spinner("Analyzing → proposing → risk review → executing…"):
            fills = orch.step([s.strip().upper() for s in syms.split(",")
                               if s.strip()], risk_pct=rp)
        st.success(f"Cycle complete — {len(fills)} fill(s). Every decision "
                   "is in the audit timeline →")
    sigs = state.get("signals") or {}
    if sigs:
        st.dataframe(pd.DataFrame(sigs.values()), use_container_width=True,
                     hide_index=True)

    st.subheader("💼 Paper book")
    if broker.positions:
        rows = [{"ticker": t, "qty": p["qty"],
                 "avg": round(p["avg_price"], 2),
                 "mark": (quotes.get(t) or {}).get("price", "—")}
                for t, p in broker.positions.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)
    else:
        st.caption("Flat — no open positions.")
    if broker.fills:
        with st.expander(f"🧾 Fills ({len(broker.fills)})"):
            st.dataframe(pd.DataFrame(broker.fills[::-1]),
                         use_container_width=True, hide_index=True,
                         height=240)

with right:
    st.subheader("📜 Audit timeline")
    st.caption("Trigger → model → reasoning, for every action by anyone.")
    for r in reversed(audit.tail(14)):
        veto = "VETO" in r["action"]
        st.markdown(
            f"<div class='qt-audit{' qt-veto' if veto else ''}'>"
            f"<b>{r['actor']}</b> · {r['action']} "
            f"<span style='color:#64748b'>{time.strftime('%H:%M:%S', time.localtime(r['ts']))}</span>"
            f"<br>{r['reasoning']}</div>", unsafe_allow_html=True)
    if not audit.tail(1):
        st.caption("No actions yet — run a decision cycle.")

st.markdown("---")
st.caption("QuantTrader v0.1 — engine 23/23 tests · paper-only by "
           "constitution · keys via .env only · see CLAUDE.md for the "
           "build roadmap")
