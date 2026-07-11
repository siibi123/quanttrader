"""QuantTrader — pro terminal UI (v0.2). Thin shell; engine in core/data/ai."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai.orchestrator import RuleOrchestrator, TOOL_SCHEMAS
from core.engine import AuditLog, PaperBroker, RiskEngine
from core.state import Config, EventBus, GlobalState
from data.providers import (CompositeProvider, LSEProvider, PollingFeed,
                            YahooProvider)

st.set_page_config(page_title="QuantTrader", page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded")

ACCENT = "#22c55e"
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
    provider = CompositeProvider([lse, YahooProvider()], state)
    broker = PaperBroker(cfg, bus, state, audit)
    risk = RiskEngine(cfg, bus, state, audit)
    orch = RuleOrchestrator(bus, state, audit, risk, broker, provider)
    feed = PollingFeed(bus, state, provider,
                       ["SPY", "QQQ", "AAPL", "NVDA"], interval_s=45)
    state.set("session", {"started": time.strftime(
        "%Y-%m-%d %H:%M UTC", time.gmtime())})
    return dict(cfg=cfg, bus=bus, state=state, audit=audit, lse=lse,
                provider=provider, broker=broker, risk=risk, orch=orch,
                feed=feed)


E = get_engine()
cfg, state, audit = E["cfg"], E["state"], E["audit"]
broker, risk, orch, feed = E["broker"], E["risk"], E["orch"], E["feed"]
quotes = state.get("quotes") or {}

# ---------------------------------------------------------------------------
# LEFT RAIL
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='qt-logo'>◆ Quant<span>Trader</span></div>"
                "<div class='qt-sub'>AUTONOMOUS PAPER DESK</div>",
                unsafe_allow_html=True)
    st.write("")
    with st.expander("INSTRUMENT", expanded=True):
        wl = st.text_input("Watchlist", "SPY, QQQ, AAPL, NVDA",
                           label_visibility="collapsed")
        symbols = [s.strip().upper() for s in wl.split(",") if s.strip()]
        chart_sym = st.selectbox("Chart symbol", symbols or ["SPY"])
        tf = st.select_slider("Timeframe", ["1h", "1d", "1wk"], value="1d")
    with st.expander("CONFIGURATION", expanded=True):
        st.caption(f"Paper capital · ${cfg.starting_cash:,.0f}")
        rp = st.slider("Risk per position %", 0.5, 3.0, 1.0, 0.25)
        deep = st.toggle("Options greeks pass (LSE)",
                         value=bool(cfg.lse_api_key),
                         disabled=not cfg.lse_api_key,
                         help="Pull the chart symbol's chain with "
                              "precomputed greeks each cycle")
    with st.expander("RISK MANAGEMENT", expanded=True):
        st.caption(f"Position cap · {cfg.max_position_pct}% of equity")
        st.caption(f"Gross exposure · ≤{cfg.max_gross_exposure_pct}%")
        st.caption(f"Daily loss halt · −{cfg.max_daily_loss_pct}%")
        st.caption(f"VaR ceiling · {cfg.max_var_pct}%")
        st.caption("⛔ RiskEngine veto: ABSOLUTE")
    with st.expander("DATA CHAIN", expanded=False):
        if cfg.lse_api_key:
            st.caption("🟢 LSE vault (verified contract) → Yahoo failsafe")
            st.caption("WS parked for roadmap #7: "
                       f"`{E['lse'].WS_URL_ROADMAP}`")
            if st.button("Vault usage"):
                st.json(E["lse"].usage() or {"note": "no response"})
        else:
            st.caption("⚪ Yahoo only — add LSE_API_KEY in Secrets/.env")
    st.write("")
    run = st.button("▷  RUN DECISION CYCLE", type="primary",
                    use_container_width=True)
    fc1, fc2 = st.columns(2)
    if fc1.button("▶ Feed", use_container_width=True):
        feed.symbols = symbols
        feed.start()
        st.rerun()
    if fc2.button("⏹ Stop", use_container_width=True):
        feed.stop()
        st.rerun()
    st.caption(("🟢 feed running" if feed.running else "⚫ feed stopped") +
               f" · {len(symbols)} symbols · {feed.interval_s}s")

# ---------------------------------------------------------------------------
# NAV + LSE-style STATS STRIP: TRADES · WIN% · PF · P&L · DD · SR
# ---------------------------------------------------------------------------
st.markdown("""
<div class="qt-nav">
 <div><span class="qt-logo">◆ Quant<span style="color:#22c55e">Trader</span></span>
 <span class="qt-sub"> MARKET INTELLIGENCE</span></div>
 <span class="qt-link active">Terminal</span><span class="qt-link">Data</span>
 <span class="qt-link">AI Core</span><span class="qt-link">Docs</span>
</div>""", unsafe_allow_html=True)

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
        for s_ in symbols:
            orch.research(s_)
        if deep and cfg.lse_api_key:
            orch.ingest_chain(chart_sym,
                              E["lse"].options_chain(chart_sym))
        new_fills = orch.step(symbols, risk_pct=rp)
    st.toast(f"Cycle complete — {len(new_fills)} fill(s) · research + "
             f"greeks in AUDIT")

# ---------------------------------------------------------------------------
# TABS — CHART | METRICS | TRADES | AUDIT
# ---------------------------------------------------------------------------
t_chart, t_metrics, t_trades, t_lab, t_audit = st.tabs(
    ["CHART", "METRICS", "TRADES", "LAB", "AUDIT"])

with t_chart:
    iv = {"1h": ("1h", "720d"), "1d": ("1d", "2y"),
          "1wk": ("1wk", "10y")}[tf]
    df = E["provider"].get_candles(chart_sym, interval=iv[0], lookback=iv[1])
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
    else:
        st.info(f"No data for {chart_sym} — throttled or bad symbol.")

with t_metrics:
    if quotes:
        qc = st.columns(min(len(quotes), 6))
        for col, (s_, q) in zip(qc, list(quotes.items())[:6]):
            col.metric(s_, f"{q.get('price', 0):,.2f}",
                       f"{q.get('chg_pct', 0):+.2f}%")
    for title, key in (("Signals", "signals"), ("Research", "research"),
                       ("Options (greeks distilled)", "options")):
        d = state.get(key) or {}
        if d:
            st.markdown(f"### {title}")
            st.dataframe(pd.DataFrame(d.values()),
                         use_container_width=True, hide_index=True)
    if not (state.get("signals") or state.get("research")):
        st.caption("Run a decision cycle to populate.")
    st.caption(f"AI contract: {len(TOOL_SCHEMAS)} tools · LLM socket awaits "
               "ANTHROPIC_API_KEY · every call risk-reviewed.")

with t_trades:
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown("### Open book")
        if broker.positions:
            rows = [{"ticker": t, "qty": p["qty"],
                     "avg": round(p["avg_price"], 2),
                     "mark": marks.get(t, "—"),
                     "P&L $": round((marks.get(t, p["avg_price"]) -
                                     p["avg_price"]) * p["qty"], 0)}
                    for t, p in broker.positions.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)
        else:
            st.caption("Flat.")
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
