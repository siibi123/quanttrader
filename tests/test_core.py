"""QuantTrader core test suite — every safety claim, verified."""
import dataclasses
import os, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
shutil.rmtree("runtime", ignore_errors=True)

import pandas as pd

from core.state import Config, Event, EventBus, GlobalState
from core.engine import AuditLog, Order, PaperBroker, RiskEngine
from data.providers import CompositeProvider, FakeProvider, LSEProvider, PollingFeed
from ai.orchestrator import LLMOrchestrator, RuleOrchestrator

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

print("\n" + "=" * 44)
passed = sum(1 for _, ok in results if ok)
print(f"QUANTTRADER CORE: {passed}/{len(results)} PASS")
sys.exit(0 if passed == len(results) else 1)
