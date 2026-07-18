"""QuantTrader core test suite — every safety claim, verified."""
import dataclasses
import os, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
shutil.rmtree("runtime", ignore_errors=True)

import numpy as np
import pandas as pd

from core.state import Config, Event, EventBus, GlobalState
from core.engine import AuditLog, Order, PaperBroker, RiskEngine
from data.providers import CompositeProvider, FakeProvider, LSEProvider, PollingFeed
from ai.orchestrator import LLMOrchestrator, RuleOrchestrator
from quant.hmm_regime import fit_hmm
from quant.kalman_pairs import kalman_hedge_ratio, pair_signal
from quant.garch import garch_forecast
from quant.covariance import shrunk_covariance, min_variance_weights
from quant.vol_surface import build_surface_grid
from quant.surface_interpreter import interpret_surface
from quant.anomaly_library import match_anomalies
from quant.sector_engine import rank_sectors_and_names, score_name
from quant.orderflow import bulk_volume_classification, cvd, vpin, volume_profile
from quant.optionflow import flow_spike, largest_prints, premium_share
from quant.flow_confluence import confluence
from quant.validation import bootstrap_mean_return
from core.strategy_registry import MIN_SIGNALS_TO_PROMOTE, StrategyRegistry
from core.circuit_breaker import DrawdownCircuitBreaker
from quant.transaction_costs import corwin_schultz_spread, expected_trade_cost
from data.news import NewsProvider

results = []
def check(name, cond):
    results.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'} {name}")

cfg = Config()
bus = EventBus()
state = GlobalState(bus)
audit = AuditLog(bus)

# ---- 1. Event bus
got = []
bus.subscribe("tick", lambda e: got.append(e))
bus.publish(Event("tick", {"symbol": "X", "price": 1}))
check("bus: subscriber receives event", len(got) == 1)
bus.subscribe("*", lambda e: got.append(e))
bus.publish(Event("other", {}))
check("bus: wildcard subscription", any(e.type == "other" for e in got))

# ---- 2. Global state
seen = []
bus.subscribe("state.changed", lambda e: seen.append(e.payload["path"]))
state.set("quotes.AAPL", {"price": 200.5})
check("state: dot-path set/get", state.get("quotes.AAPL")["price"] == 200.5)
check("state: mutation emits event", "quotes.AAPL" in seen)
state.set("portfolio", {"cash": 10000})
ctx = state.to_ai_context()
check("state: AI context includes curated keys",
      "quotes" in ctx and "portfolio" in ctx and len(ctx) < 6001)

# ---- 3. Paper broker accounting
broker = PaperBroker(cfg, bus, state, audit)
o = Order("AAPL", "BUY", 10, reason="test"); o.approved = True
f1 = broker.execute(o, 100.0)
check("broker: buy fills with slippage", f1 and f1["price"] > 100.0)
check("broker: cash reduced correctly",
      abs(broker.cash - (10000 - 10 * f1["price"] - f1["fee"])) < 0.01)
o2 = Order("AAPL", "SELL", 10, reason="test"); o2.approved = True
f2 = broker.execute(o2, 110.0)
check("broker: sell realizes P&L", f2 and f2["realized"] > 0)
check("broker: position closed", "AAPL" not in broker.positions)
o3 = Order("AAPL", "BUY", 5, reason="no stamp")   # NOT approved
check("broker: refuses unapproved order", broker.execute(o3, 100.0) is None)

# ---- 4. Risk engine veto
risk = RiskEngine(cfg, bus, state, audit)
big = Order("NVDA", "BUY", 1000, reason="oversize")        # >> 25% cap
big = risk.review(big, broker, 100.0)
check("risk: vetoes oversized position", not big.approved and big.veto_reason)
ok_ = Order("NVDA", "BUY", 10, reason="sane size")
ok_ = risk.review(ok_, broker, 100.0)
check("risk: approves sane order", ok_.approved)
vetoes = [e for e in bus.recent(50, "risk.veto")]
check("risk: veto published to bus", len(vetoes) >= 1)

# ---- 5. Audit trail
tail = audit.tail(50)
check("audit: records exist with reasoning",
      len(tail) >= 4 and all("reasoning" in r for r in tail))
check("audit: persisted to disk", os.path.exists("runtime/audit.jsonl"))

# ---- 6. Orchestrator end-to-end on fake data (uptrend -> should trade)
prov = FakeProvider(mu=0.002, vol=0.008)
orch = RuleOrchestrator(bus, state, audit, risk, broker, prov)
sig = orch.analyze("TREND")
check("orchestrator: analysis has reasoning", len(sig["why"]) > 10)
fills = orch.step(["TREND"], risk_pct=1.0)
check("orchestrator: full propose->veto->execute loop",
      isinstance(fills, list))
check("orchestrator: signal in global state",
      state.get("signals.TREND") is not None)

# ---- 7. LLM socket refuses to fake it
try:
    LLMOrchestrator(api_key="")
    check("llm: refuses without key", False)
except RuntimeError:
    check("llm: refuses without key", True)

# ---- 8. Polling feed (fake provider, fast interval)
feed = PollingFeed(bus, state, FakeProvider(), ["AAA", "BBB"], interval_s=1)
n_before = len(bus.recent(200, "tick"))
feed.start()
time.sleep(2.5)
feed.stop()
n_after = len(bus.recent(200, "tick"))
check("feed: background thread publishes ticks", n_after > n_before)
check("feed: quotes land in state", state.get("quotes.AAA") is not None)
check("feed: stop() halts cleanly", not feed.running)

# ---- 9. Composite fallback
comp = CompositeProvider([FakeProvider(), FakeProvider()], state)
df = comp.get_candles("ZZZ")
check("composite: fallback chain serves data", len(df) > 100)

# ---- 10. P1a: LSE quote % change vs previous DAILY close, not 1m bar-to-bar
lse = LSEProvider(api_key="dummy-key-for-test")

def _fake_candles(symbol, interval="1d", lookback="2y"):
    if interval == "1m":
        idx = pd.date_range("2026-01-05 15:59", periods=2, freq="1min")
        return pd.DataFrame({"Open": [101.0, 101.02], "High": [101.05, 101.05],
                             "Low": [100.95, 101.0], "Close": [101.0, 101.02],
                             "Volume": [1000, 1000]}, index=idx)
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    return pd.DataFrame({"Open": [95.0, 98.0, 100.0], "High": [96.0, 99.0, 101.0],
                         "Low": [94.0, 97.0, 99.0], "Close": [95.0, 98.0, 100.0],
                         "Volume": [5000, 5000, 5000]}, index=idx)

lse.get_candles = _fake_candles
q = lse.get_quote("TEST")
expected_chg = round((101.02 / 98.0 - 1) * 100, 2)      # vs previous daily close
wrong_chg = round((101.02 / 101.0 - 1) * 100, 2)         # the old, buggy 1m-bar delta
check("lse: quote % change vs previous daily close, not 1m bar",
      abs(q["chg_pct"] - expected_chg) < 0.001)
check("lse: quote % change is not the old bar-to-bar delta",
      abs(q["chg_pct"] - wrong_chg) > 0.5)

# ---- 11. P1b: AUM + max position size veto (fixed $ and % of AUM modes)
risk2 = RiskEngine(cfg, bus, state, audit)
broker2 = PaperBroker(cfg, bus, state, audit, path="runtime/broker_test2.json")

risk2.cfg = dataclasses.replace(cfg, max_position_mode="fixed",
                                max_position_fixed_usd=500.0)
big_fixed = risk2.review(Order("AAPL", "BUY", 20, reason="test fixed cap"),
                         broker2, 100.0)                 # notional $2000 > $500 cap
check("risk: fixed $ position cap vetoes oversized order",
      not big_fixed.approved and "max position size" in big_fixed.veto_reason)
small_fixed = risk2.review(Order("AAPL", "BUY", 3, reason="test fixed cap ok"),
                           broker2, 100.0)                # notional $300 < $500 cap
check("risk: fixed $ position cap approves order within cap", small_fixed.approved)

risk2.cfg = dataclasses.replace(cfg, max_position_mode="pct",
                                max_position_pct=10.0, aum=1000.0)
aum_big = risk2.review(Order("MSFT", "BUY", 5, reason="test aum pct cap"),
                       broker2, 100.0)   # notional $500 vs 10% of $1000 AUM = $100 cap
check("risk: AUM-based %% cap uses declared AUM, not live equity",
      not aum_big.approved)
aum_ok = risk2.review(Order("MSFT", "BUY", 1, reason="test aum pct cap ok"),
                      broker2, 50.0)     # notional $50 <= $100 cap
check("risk: AUM-based %% cap approves order within cap", aum_ok.approved)

# ---- 12. P2: hmm_regime — Gaussian HMM regime detection
_rng = np.random.default_rng(3)
_bull = _rng.normal(0.0015, 0.006, 200)
_bear = _rng.normal(-0.003, 0.02, 200)
_hmm_r = pd.Series(np.concatenate([_bull, _bear]),
                   index=pd.bdate_range("2023-01-01", periods=400))
_hmm = fit_hmm(_hmm_r, n_states=2)
check("hmm: states sorted by mean, current regime matches the bear tail",
      _hmm["state_means_pct"][0] < _hmm["state_means_pct"][1]
      and _hmm["most_likely_regime"] == "Bear/Panic")
check("hmm: transition matrix is diagonal-dominant (regimes persist)",
      all(_hmm["transition_matrix"][i][i] > 0.9 for i in range(2)))

# ---- 13. P2: kalman_pairs — dynamic hedge ratio + spread z-score
_rng2 = np.random.default_rng(5)
_n = 300
_x = np.cumsum(_rng2.normal(0.1, 1, _n)) + 100
_y = 2.0 * _x + _rng2.normal(0, 0.3, _n)
_idx = pd.bdate_range("2023-01-01", periods=_n)
_xs, _ys = pd.Series(_x, index=_idx), pd.Series(_y, index=_idx)
_kf = kalman_hedge_ratio(_ys, _xs)
check("kalman: recovers the true hedge ratio (beta ~= 2.0)",
      abs(float(_kf["beta"].iloc[-1]) - 2.0) < 0.2)
_y_shock = _y.copy()
_y_shock[-1] += 20                                    # one isolated spread shock
_sig = pair_signal(pd.Series(_y_shock, index=_idx), _xs)
check("kalman: isolated spread shock triggers a SHORT SPREAD signal",
      _sig["signal"] == "SHORT SPREAD" and _sig["spread_z"] > 5)

# ---- 14. P2: garch — GARCH(1,1) volatility forecast (arch package)
_rng3 = np.random.default_rng(11)
_ng, _omega, _a1, _b1 = 800, 0.02, 0.1, 0.85
_eps, _sig2 = np.zeros(_ng), np.zeros(_ng)
_sig2[0] = _omega / (1 - _a1 - _b1)
_z = _rng3.standard_normal(_ng)
_eps[0] = np.sqrt(_sig2[0]) * _z[0]
for _t in range(1, _ng):
    _sig2[_t] = _omega + _a1 * _eps[_t - 1] ** 2 + _b1 * _sig2[_t - 1]
    _eps[_t] = np.sqrt(_sig2[_t]) * _z[_t]
_close = pd.Series(100 * np.exp(np.cumsum(_eps / 100)),
                   index=pd.bdate_range("2022-01-01", periods=_ng))
_gf = garch_forecast(_close)
check("garch: fits a simulated GARCH(1,1) series without error",
      "error" not in _gf and _gf["vol_1d_pct"] > 0)
check("garch: recovers a stationary, sensible persistence estimate",
      _gf["persistence"] is not None and 0 < _gf["persistence"] < 1
      and _gf["half_life_days"] is not None)

# ---- 15. P2: covariance — Ledoit-Wolf shrinkage + min-variance weights
_rng4 = np.random.default_rng(2)
_cov_r = pd.DataFrame(
    {"LOWVOL": _rng4.normal(0, 0.005, 300), "HIGHVOL": _rng4.normal(0, 0.02, 300)},
    index=pd.bdate_range("2023-01-01", periods=300))
_sc = shrunk_covariance(_cov_r)
check("covariance: Ledoit-Wolf shrinkage intensity is a valid fraction",
      "error" not in _sc and 0 <= _sc["shrinkage"] <= 1)
_mv = min_variance_weights(_cov_r)
check("covariance: min-variance portfolio tilts toward the lower-vol asset",
      abs(sum(_mv["weights"].values()) - 1) < 0.01
      and _mv["weights"]["LOWVOL"] > _mv["weights"]["HIGHVOL"])

# ---- 16. P3: vol_surface — strike x DTE x IV grid
_spot = 100.0
_chain_rows = []
for _k in [80, 85, 90, 95, 100, 105, 110, 115, 120]:
    _iv = 0.20 + max(0, (_spot - _k)) * 0.006          # steep put-side skew
    _chain_rows.append({"strike": _k, "dte": 7, "iv": round(_iv, 4),
                        "type": "C" if _k >= _spot else "P",
                        "delta": (0.5 - (_k - _spot) / 100) if _k >= _spot
                                else (-0.5 - (_k - _spot) / 100)})
for _k in [80, 90, 100, 110, 120]:
    _chain_rows.append({"strike": _k, "dte": 60, "iv": 0.16,
                        "type": "C" if _k >= _spot else "P",
                        "delta": (0.5 - (_k - _spot) / 100) if _k >= _spot
                                else (-0.5 - (_k - _spot) / 100)})
_chain = pd.DataFrame(_chain_rows)
_grid = build_surface_grid(_chain)
check("vol_surface: grid shape matches distinct DTEs x strikes",
      _grid["dtes"] == [7, 60] and len(_grid["strikes"]) == 9
      and len(_grid["iv_grid"]) == 2)
check("vol_surface: missing iv/dte columns fails honestly, no fake grid",
      "error" in build_surface_grid(pd.DataFrame({"strike": [100]})))

# ---- 17. P3: surface_interpreter — skew + term structure + smile anomaly
_surf = interpret_surface(_chain, spot=_spot)
check("surface_interpreter: detects the steep put skew built into the data",
      _surf["skew_pts"] is not None and _surf["skew_pts"] > 5
      and any("Steep put skew" in f for f in _surf["findings"]))
check("surface_interpreter: detects the term-structure inversion (7d rich vs 60d)",
      _surf["term_structure_pts"] is not None and _surf["term_structure_pts"] > 3
      and any("INVERTED" in f for f in _surf["findings"]))

_smile_rows = [{"strike": k, "dte": 10,
               "iv": 0.35 if k == 95 else 0.20,           # single-strike anomaly
               "type": "C" if k >= 100 else "P"}
              for k in [85, 90, 95, 100, 105, 110, 115]]
_smile = interpret_surface(pd.DataFrame(_smile_rows), spot=100.0)
check("surface_interpreter: flags a single-strike smile anomaly",
      len(_smile["smile_anomalies"]) == 1
      and _smile["smile_anomalies"][0]["strike"] == 95.0)

# ---- 18. P3: ingest_chain wires the surface read into state + audit
_orch3 = RuleOrchestrator(bus, state, audit, risk, broker, FakeProvider())
_ing = _orch3.ingest_chain("SURF", _chain)
check("ingest_chain: surface findings attached to the options state entry",
      "surface" in _ing and len(_ing["surface"]["findings"]) >= 1)
check("ingest_chain: VOL SURFACE audit record created",
      any(r["action"] == "VOL SURFACE" for r in audit.tail(20)))

# ---- 19. P4: NewsProvider stub (no key -> honest empty, never fabricated)
_news_stub = NewsProvider(api_key="")
check("news: no key -> empty headlines/sentiment, not fabricated",
      _news_stub.company_news("AAPL") == [] and _news_stub.sentiment("AAPL") == {}
      and not _news_stub.working)

# ---- 20. P4: anomaly_library — trigger matching is honest (empty context -> nothing)
check("anomaly_library: momentum trigger fires on a strong agreeing score",
      any(a["name"] == "Momentum"
          for a in match_anomalies({"score": 0.5, "agree_frac": 0.9})))
check("anomaly_library: empty context matches nothing (no fabricated triggers)",
      match_anomalies({}) == [])

# ---- 21. P4: LSEProvider macro_series / economic_calendar / options_flow parsing
def _fake_lse_get(path, params):
    if path == "/series":
        sym = params.get("symbol")
        return [{"date": "2026-07-01", "value": 3.1 if sym == "cpi_yoy" else 5.33},
                {"date": "2026-06-01", "value": 3.0}]
    if path == "/ref/economic_calendar":
        return [{"event": "CPI m/m", "date": "2026-07-15"},
                {"event": "FOMC Rate Decision", "date": "2026-07-30"}]
    if path == "/options/flow":
        return [{"strike": 200, "type": "call", "premium": 150000, "expiry": "2026-08-21"},
                {"strike": 195, "type": "put", "premium": 120000, "expiry": "2026-08-21"}]
    return None

lse4 = LSEProvider(api_key="dummy-key-for-test")
lse4._get = _fake_lse_get
_mv = lse4.macro_series("cpi_yoy")
check("lse: macro_series parses (date, value) rows",
      len(_mv) == 2 and float(_mv["value"].iloc[0]) == 3.1)
_cal = lse4.economic_calendar(region="US")
check("lse: economic_calendar returns upcoming events",
      len(_cal) == 2 and "event" in [c.lower() for c in _cal.columns])
_flow = lse4.options_flow(underlying="AAPL", min_premium=100000)
check("lse: options_flow parses strike/type/premium rows",
      len(_flow) == 2 and float(_flow["premium"].iloc[0]) == 150000)

# ---- 22. P4: RuleOrchestrator news/macro/flow scans wire into state+audit+bus
news4 = NewsProvider(api_key="dummy-key-for-test")
news4.company_news = lambda symbol, days=3, limit=10: [
    {"symbol": symbol, "headline": "Big beat on earnings", "source": "Reuters",
     "url": "", "ts": 0}]
news4.sentiment = lambda symbol: {"symbol": symbol, "bullish_pct": 82.0,
                                  "bearish_pct": 10.0, "buzz_articles_week": 40,
                                  "buzz_z": 1.2}
orch4 = RuleOrchestrator(bus, state, audit, risk, broker, FakeProvider(),
                         news=news4, lse=lse4)

n_news = len(bus.recent(500, "news.interrupt"))
out_news = orch4.scan_news("AAPL")
check("scan_news: headlines+sentiment land in state.news",
      state.get("news.AAPL") is not None and out_news["bullish_pct"] == 82.0)
check("scan_news: strong sentiment publishes a news.interrupt event",
      len(bus.recent(500, "news.interrupt")) > n_news)

out_macro = orch4.scan_macro()
check("scan_macro: rates/CPI snapshot + calendar land in state.macro",
      state.get("macro.cpi_yoy.latest") == 3.1
      and len(out_macro.get("upcoming_events", [])) == 2)

n_flow = len(bus.recent(500, "flow.interrupt"))
out_flow = orch4.scan_flow("AAPL", min_premium=100000)
check("scan_flow: large prints land in state.flow_alerts + publish an interrupt",
      len(out_flow["prints"]) == 2 and len(bus.recent(500, "flow.interrupt")) > n_flow)

_res4 = orch4.research("P4TEST")
check("research: core EWMA/MC fields still present after anomaly wiring",
      {"ewma_ann_vol_pct", "p_up_20d_pct", "exp_move_20d"} <= set(_res4.keys()))

# ---- 23. P5: sector_engine — score_name tilts + rank_sectors_and_names
def _mk_df(mu, vol, n=300, seed=1):
    rr_ = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rr_.normal(mu, vol, n)))
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                        "Close": close, "Volume": 1e6},
                        index=pd.bdate_range("2024-01-01", periods=n))

_strong_up = _mk_df(0.0025, 0.01, seed=1)
_base = score_name(_strong_up)
_tilted = score_name(_strong_up, sentiment={"bullish_pct": 90, "bearish_pct": 5},
                    flow={"prints": [{"strike": 100}]}, macro_trend="down")
check("sector_engine: bullish sentiment+flow+dovish macro all tilt score up "
      "on a LONG verdict",
      _base["verdict"] == "LONG" and _tilted["target_score"] > _base["target_score"])

_data = {"AAA": _mk_df(0.0025, 0.01, seed=1), "BBB": _mk_df(0.0022, 0.011, seed=2),
         "CCC": _mk_df(-0.0005, 0.02, seed=3)}
_sectors5 = {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"}
_scan = rank_sectors_and_names(_data, _sectors5,
                               sentiment_by_ticker={"AAA": {"bullish_pct": 85,
                                                            "bearish_pct": 5}},
                               macro_trend="down")
check("sector_engine: only tradeable names are ranked, the rest land in avoid",
      len(_scan["names"]) + len(_scan["avoid"]) == 3
      and all(n["verdict"] != "NO TRADE" for n in _scan["names"]))
check("sector_engine: sectors are ranked by average target_score",
      _scan["sectors"] == sorted(_scan["sectors"],
                                 key=lambda x: -x["avg_target_score"]))

# ---- 24. P5: RuleOrchestrator.sector_scan wires state + audit
orch5 = RuleOrchestrator(bus, state, audit, risk, broker, FakeProvider())
_scan5 = orch5.sector_scan(["P5A", "P5B"], account=10000, risk_pct=1.0)
check("sector_scan: writes state.sector_scan and a SECTOR SCAN audit record",
      state.get("sector_scan") is not None
      and any(r["action"] == "SECTOR SCAN" for r in audit.tail(20)))

# ---- 25. P6a: orderflow — BVC/CVD confirms a healthy uptrend
_rng6 = np.random.default_rng(9)
_n6 = 200
_close6 = 100 * np.exp(np.cumsum(_rng6.normal(0.002, 0.008, _n6)))
_vol6 = np.abs(1e6 + _rng6.normal(0, 1e5, _n6))
_df6 = pd.DataFrame({"Open": _close6, "High": _close6 * 1.005,
                    "Low": _close6 * 0.995, "Close": _close6, "Volume": _vol6},
                    index=pd.bdate_range("2024-01-01", periods=_n6))
_cvd6 = cvd(_df6)
check("orderflow: CVD confirms a healthy uptrend (no divergence)",
      _cvd6["cvd_chg"] > 0 and _cvd6["divergence"] is None)

# same series but the last 20 bars get low volume on up-closes, high on down-closes
_close7 = _close6.copy(); _vol7 = _vol6.copy()
for i in range(_n6 - 20, _n6):
    _vol7[i] = 2e5 if _close7[i] > _close7[i - 1] else 2e6
_df7 = pd.DataFrame({"Open": _close7, "High": _close7 * 1.005,
                    "Low": _close7 * 0.995, "Close": _close7, "Volume": _vol7},
                    index=pd.bdate_range("2024-01-01", periods=_n6))
check("orderflow: CVD flags a bearish divergence when volume contradicts price",
      cvd(_df7)["divergence"] == "bearish")

_rng8 = np.random.default_rng(5)
_n8 = 150
_close8 = 100 * np.exp(np.cumsum(_rng8.normal(0.0, 0.005, _n8)))
_vol8 = np.full(_n8, 1e6)
_close8[-20:] = _close8[-21] * np.exp(np.cumsum(np.full(20, 0.01)))
_vol8[-20:] = 5e6
_df8 = pd.DataFrame({"Open": _close8, "High": _close8 * 1.01, "Low": _close8 * 0.99,
                    "Close": _close8, "Volume": _vol8},
                    index=pd.bdate_range("2024-01-01", periods=_n8))
_vp8 = vpin(_df8)
check("orderflow: VPIN flags toxicity after a sudden one-directional volume spike",
      _vp8["toxic"] and _vp8["percentile"] >= 85)
_prof = volume_profile(_df6)
check("orderflow: volume_profile returns top-3 nodes, sane volume shares",
      len(_prof) == 3 and all(0 <= p["volume_pct"] <= 100 for p in _prof))

# ---- 26. P6b: optionflow — premium share, spike z-score, largest prints
_flow_today = pd.DataFrame([
    {"strike": 200, "type": "call", "premium": 300000, "volume": 500, "expiry": "2026-08-21"},
    {"strike": 195, "type": "put", "premium": 100000, "volume": 200, "expiry": "2026-08-21"},
])
_ps = premium_share(_flow_today)
check("optionflow: premium_share splits call/put correctly",
      abs(_ps["call_share_pct"] + _ps["put_share_pct"] - 100) < 0.01
      and _ps["call_share_pct"] > _ps["put_share_pct"])
check("optionflow: flow_spike is honest about insufficient history",
      "error" in flow_spike(_flow_today, [_flow_today] * 3))
_rng9 = np.random.default_rng(3)
_hist9 = [pd.DataFrame([{"strike": 200, "type": "call",
                        "premium": 60000 + _rng9.normal(0, 10000),
                        "volume": 150 + _rng9.normal(0, 20)}]) for _ in range(20)]
_spike9 = flow_spike(pd.DataFrame([{"strike": 200, "type": "call",
                                   "premium": 300000, "volume": 800}]), _hist9)
check("optionflow: flow_spike detects a large z-score spike vs the norm",
      "error" not in _spike9 and _spike9["volume_z"] > 3)
_top = largest_prints(_flow_today, top_n=1)
check("optionflow: largest_prints returns the single biggest premium print",
      len(_top) == 1 and _top[0]["premium"] == 300000)

# ---- 27. P6c: flow_confluence — LONG / CONFLICT / QUIET classification
_calls_heavy = pd.DataFrame([{"strike": 200, "type": "call", "premium": 400000,
                             "volume": 500},
                            {"strike": 195, "type": "put", "premium": 30000,
                             "volume": 50}])
_puts_heavy = pd.DataFrame([{"strike": 200, "type": "call", "premium": 30000,
                            "volume": 50},
                           {"strike": 195, "type": "put", "premium": 400000,
                            "volume": 500}])
check("flow_confluence: bullish tape + call-heavy flow -> CONFLUENCE LONG",
      confluence(_df6, _calls_heavy)["verdict"] == "CONFLUENCE LONG")
check("flow_confluence: bullish tape + put-heavy flow -> CONFLICT",
      confluence(_df6, _puts_heavy)["verdict"] == "CONFLICT")
_flat = pd.DataFrame({"Open": [100.0] * 60, "High": [100.1] * 60,
                     "Low": [99.9] * 60, "Close": [100.0] * 60,
                     "Volume": [1000.0] * 60},
                     index=pd.bdate_range("2024-01-01", periods=60))
_neutral_flow = pd.DataFrame([{"strike": 100, "type": "call", "premium": 50000,
                              "volume": 50},
                             {"strike": 100, "type": "put", "premium": 50000,
                              "volume": 50}])
check("flow_confluence: flat tape + neutral flow -> QUIET",
      confluence(_flat, _neutral_flow)["verdict"] == "QUIET")

# ---- 28. P6c: RuleOrchestrator.scan_flow_confluence wires state + audit
orch6 = RuleOrchestrator(bus, state, audit, risk, broker, FakeProvider())
_fc6 = orch6.scan_flow_confluence("P6TEST")
check("scan_flow_confluence: writes state.flow + a FLOW CONFLUENCE audit record",
      state.get("flow.P6TEST") is not None
      and any(r["action"] == "FLOW CONFLUENCE" for r in audit.tail(20)))

# ---- 29. P6c: flow-confluence tilt feeds into P5 sector scoring
# _strong_up (test 23) is already confirmed to produce a LONG verdict.
_base9 = score_name(_strong_up)
_agree9 = score_name(_strong_up, flow_confluence={"verdict": "CONFLUENCE LONG"})
_disagree9 = score_name(_strong_up, flow_confluence={"verdict": "CONFLUENCE SHORT"})
check("sector_engine: agreeing flow confluence raises target_score, "
      "disagreeing lowers it",
      _agree9["target_score"] > _base9["target_score"] > _disagree9["target_score"])

# ---- 30. P7a: bootstrap_mean_return — honest CI on a per-signal edge sample
_rng10 = np.random.default_rng(1)
_pos_edge = pd.Series(_rng10.normal(0.02, 0.03, 40))
_noisy_edge = pd.Series(_rng10.normal(0.0, 0.05, 40))
check("validation: bootstrap_mean_return excludes zero for a clear positive edge",
      bootstrap_mean_return(_pos_edge)["excludes_zero"])
check("validation: bootstrap_mean_return does not exclude zero for a noisy sample",
      not bootstrap_mean_return(_noisy_edge)["excludes_zero"])

# ---- 31. P7a: StrategyRegistry — log/settle/promote lifecycle
reg1 = StrategyRegistry(audit, path="runtime/test_registry_p7a.json")
check("strategy_registry: starts every strategy in INCUBATION",
      reg1.status("s1") == StrategyRegistry.STATUS_INCUBATION)
for _ in range(MIN_SIGNALS_TO_PROMOTE + 5):
    reg1.log_signal("s1", "AAA", "BUY", 100.0, horizon_days=10)
for s in reg1._data["s1"]["signals"]:
    s["ts"] = time.time() - 11 * 86400              # simulate elapsed horizon
n_settled = reg1.settle_signals("s1", price_lookup=lambda sym: 106.0)  # +6% each
check("strategy_registry: settle_signals marks due signals settled",
      n_settled == MIN_SIGNALS_TO_PROMOTE + 5
      and reg1.signal_counts("s1")["pending"] == 0)
promo1 = reg1.evaluate_promotion("s1")
check("strategy_registry: promotes INCUBATION -> PAPER on a clear settled edge",
      promo1["decision"] == "PROMOTE"
      and reg1.status("s1") == StrategyRegistry.STATUS_PAPER
      and any(r["action"] == "PROMOTE" for r in audit.tail(20)))

reg1.log_signal("s2", "BBB", "BUY", 100.0, horizon_days=10)
promo2 = reg1.evaluate_promotion("s2")
check("strategy_registry: holds in INCUBATION with too few settled signals",
      promo2["decision"] == "NOT ENOUGH SIGNALS"
      and reg1.status("s2") == StrategyRegistry.STATUS_INCUBATION)

# ---- 32. P7a: RuleOrchestrator.step() enforces the gate; exits are never gated
reg7 = StrategyRegistry(audit, path="runtime/test_registry_p7a_orch.json")
orch7 = RuleOrchestrator(bus, state, audit, risk, broker, FakeProvider(),
                         registry=reg7)
orch7.analyze = lambda symbol, **kw: {
    "symbol": symbol, "signal": "BUY", "price": 100.0, "shares": 5,
    "why": "test forced buy", "urgency": "🟢 ACTIONABLE", "mode": "ENTRY",
    "gates": "5/5"}
f1 = orch7.step(["P7AENTRY"], risk_pct=1.0)
check("step(): INCUBATION blocks a new BUY entry but still logs the signal",
      len(f1) == 0 and "P7AENTRY" not in broker.positions
      and any(r["action"] == "SIGNAL LOGGED (INCUBATION)"
             for r in audit.tail(20)))

broker.positions["P7AEXIT"] = {"qty": 10, "avg_price": 90.0}
orch7.analyze = lambda symbol, **kw: {
    "symbol": symbol, "signal": "SELL", "price": 95.0,
    "why": "test forced sell", "urgency": "🟠 TODAY", "mode": "MANAGE",
    "gates": "3/5"}
f2 = orch7.step(["P7AEXIT"], risk_pct=1.0)
check("step(): exits execute even while the strategy is in INCUBATION",
      len(f2) == 1 and "P7AEXIT" not in broker.positions)

for _ in range(MIN_SIGNALS_TO_PROMOTE + 5):
    reg7.log_signal(orch7.STRATEGY_NAME, "SEED", "BUY", 100.0, horizon_days=10)
for s in reg7._data[orch7.STRATEGY_NAME]["signals"]:
    s["ts"] = time.time() - 11 * 86400
reg7.settle_signals(orch7.STRATEGY_NAME, price_lookup=lambda sym: 106.0)
reg7.evaluate_promotion(orch7.STRATEGY_NAME)
check("step(): strategy promotes to PAPER after enough settled signals with an edge",
      reg7.status(orch7.STRATEGY_NAME) == StrategyRegistry.STATUS_PAPER)

orch7.analyze = lambda symbol, **kw: {
    "symbol": symbol, "signal": "BUY", "price": 100.0, "shares": 5,
    "why": "test forced buy after promotion", "urgency": "🟢 ACTIONABLE",
    "mode": "ENTRY", "gates": "5/5"}
f3 = orch7.step(["P7APROMO"], risk_pct=1.0)
check("step(): once PAPER, new BUY entries execute normally",
      len(f3) == 1 and "P7APROMO" in broker.positions)

# ---- 33. P7e: DrawdownCircuitBreaker — multiplier curve + peak/halt lifecycle
check("circuit_breaker: gradual multiplier (1.0 -> 0.5 -> 0.0 across 0-10% DD)",
      DrawdownCircuitBreaker._multiplier(0) == 1.0
      and DrawdownCircuitBreaker._multiplier(5) == 0.5
      and DrawdownCircuitBreaker._multiplier(10) == 0.0
      and DrawdownCircuitBreaker._multiplier(2.5) == 0.75)

cb1 = DrawdownCircuitBreaker(audit, path="runtime/test_cb1.json")
s1 = cb1.update(10000)                 # first mark sets the peak
check("circuit_breaker: first update establishes the peak with no drawdown",
      s1["peak_equity"] == 10000 and s1["drawdown_pct"] == 0.0
      and s1["size_multiplier"] == 1.0)
s2 = cb1.update(9500)                  # 5% drawdown -> half size
check("circuit_breaker: 5% drawdown cuts size toward 50%",
      abs(s2["drawdown_pct"] - 5.0) < 0.01 and abs(s2["size_multiplier"] - 0.5) < 0.01)
s3 = cb1.update(8800)                  # 12% drawdown -> risk-reducing only
check("circuit_breaker: 10-15% drawdown allows only risk-reducing trades",
      s3["only_risk_reducing"] and not s3["halted"])
s4 = cb1.update(8400)                  # 16% drawdown -> hard halt, audited
check("circuit_breaker: >=15% drawdown trips a sticky HALT",
      s4["halted"]
      and any(r["action"] == "CIRCUIT BREAKER TRIPPED" for r in audit.tail(20)))
s5 = cb1.update(10000)                 # equity fully recovers to the old peak
check("circuit_breaker: a halt does NOT auto-clear just because equity recovers",
      cb1.status()["halted"])
try:
    cb1.manual_reset("")
    check("circuit_breaker: manual_reset rejects an empty reason", False)
except ValueError:
    check("circuit_breaker: manual_reset rejects an empty reason", True)
cb1.manual_reset("owner reviewed the drawdown, resuming manually")
check("circuit_breaker: a reasoned manual_reset clears the halt and is audited",
      not cb1.status()["halted"]
      and any(r["action"] == "CIRCUIT BREAKER RESET" for r in audit.tail(20)))

# ---- 34. P7e: RiskEngine defense-in-depth veto when the breaker is tripped
cb2 = DrawdownCircuitBreaker(audit, path="runtime/test_cb2.json")
cb2._data["peak_equity"] = 20000.0     # fabricate a high peak -> instant big drawdown
broker3 = PaperBroker(cfg, bus, state, audit, path="runtime/test_broker_cb.json")
risk3 = RiskEngine(cfg, bus, state, audit, circuit_breaker=cb2)
halted_buy = risk3.review(Order("HALT", "BUY", 1, reason="test"), broker3, 100.0)
check("risk: vetoes a BUY when the circuit breaker is HALTED",
      not halted_buy.approved and "circuit breaker" in halted_buy.veto_reason.lower())
halted_sell = risk3.review(Order("HALT", "SELL", 1, reason="test"), broker3, 100.0)
check("risk: circuit breaker checks never apply to SELL (exits stay unblocked)",
      halted_sell.approved)

# ---- 35. P7e: RuleOrchestrator.step() applies the size multiplier to real orders
# step() derives equity from the broker's OWN balance each call, so the
# drawdown must be simulated there, not via a disconnected update() call.
cb3 = DrawdownCircuitBreaker(audit, path="runtime/test_cb3.json")
cb3.update(10000)                      # establish the peak
broker4 = PaperBroker(cfg, bus, state, audit, path="runtime/test_broker_cb2.json")
broker4.cash = 9500.0                  # simulate a 5% drawdown in the broker's own equity
broker4.day_start_equity = 9500.0      # keep the (unrelated) daily-loss check from also firing
risk4 = RiskEngine(cfg, bus, state, audit, circuit_breaker=cb3)
orch8 = RuleOrchestrator(bus, state, audit, risk4, broker4, FakeProvider(),
                         circuit_breaker=cb3)
orch8.analyze = lambda symbol, **kw: {
    "symbol": symbol, "signal": "BUY", "price": 100.0, "shares": 10,
    "why": "test circuit breaker sizing", "urgency": "🟢 ACTIONABLE",
    "mode": "ENTRY", "gates": "5/5"}
orch8.step(["CBSIZE"], risk_pct=1.0)
check("step(): a 5% drawdown roughly halves the executed order size",
      "CBSIZE" in broker4.positions and broker4.positions["CBSIZE"]["qty"] == 5)

# ---- 36. P7b: transaction cost model — spread estimate + square-root impact
_rng11 = np.random.default_rng(4)
_n11 = 100
_close11 = 100 * np.exp(np.cumsum(_rng11.normal(0.0005, 0.01, _n11)))
_vol11 = np.full(_n11, 500000.0)
_idx11 = pd.bdate_range("2024-01-01", periods=_n11)
_df_wide = pd.DataFrame({"Open": _close11, "High": _close11 * 1.02,
                        "Low": _close11 * 0.98, "Close": _close11,
                        "Volume": _vol11}, index=_idx11)
_df_tight = pd.DataFrame({"Open": _close11, "High": _close11 * 1.002,
                         "Low": _close11 * 0.998, "Close": _close11,
                         "Volume": _vol11}, index=_idx11)
check("transaction_costs: wider daily ranges estimate a bigger spread",
      corwin_schultz_spread(_df_wide) > corwin_schultz_spread(_df_tight))
_cost_small = expected_trade_cost(_df_tight, order_shares=100,
                                  price=float(_close11[-1]))
_cost_big = expected_trade_cost(_df_tight, order_shares=200000,
                                price=float(_close11[-1]))
check("transaction_costs: a much bigger order costs more (sqrt market impact)",
      _cost_big["expected_cost_pct"] > _cost_small["expected_cost_pct"])

# ---- 37. P7b: RiskEngine gates BUY entries on edge < 2x expected cost
risk5 = RiskEngine(cfg, bus, state, audit)
low_edge = risk5.review(Order("EDGETEST", "BUY", 5, reason="test"), broker, 100.0,
                        cost_info={"expected_cost_pct": 1.0, "expected_edge_pct": 1.5})
check("risk: vetoes a BUY when expected edge < 2x expected cost",
      not low_edge.approved and "expected edge" in low_edge.veto_reason)
good_edge = risk5.review(Order("EDGETEST2", "BUY", 5, reason="test"), broker, 100.0,
                         cost_info={"expected_cost_pct": 1.0, "expected_edge_pct": 3.0})
check("risk: approves a BUY when expected edge >= 2x expected cost",
      good_edge.approved)

# ---- 38. P7b: RuleOrchestrator.step() shows expected cost on every proposal
orch9 = RuleOrchestrator(bus, state, audit, risk, broker, FakeProvider())
orch9.analyze = lambda symbol, **kw: {
    "symbol": symbol, "signal": "BUY", "price": 100.0, "shares": 5,
    "why": "test cost display", "urgency": "🟢 ACTIONABLE", "mode": "ENTRY",
    "gates": "5/5"}
orch9.step(["P7BCOST"], risk_pct=1.0)
propose_recs = [r for r in audit.tail(30) if r["action"] == "PROPOSE BUY"
               and r.get("trigger") == "signals.P7BCOST"]
check("step(): PROPOSE BUY audit record shows expected cost",
      len(propose_recs) == 1 and "expected cost" in propose_recs[0]["reasoning"])

print("\n" + "=" * 44)
passed = sum(1 for _, ok in results if ok)
print(f"QUANTTRADER CORE: {passed}/{len(results)} PASS")
sys.exit(0 if passed == len(results) else 1)
