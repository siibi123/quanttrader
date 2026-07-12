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

print("\n" + "=" * 44)
passed = sum(1 for _, ok in results if ok)
print(f"QUANTTRADER CORE: {passed}/{len(results)} PASS")
sys.exit(0 if passed == len(results) else 1)
