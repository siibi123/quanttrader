"""Audit trail, Risk Engine (absolute veto), Paper Broker.

The chain of command, in one sentence:
    orchestrator PROPOSES → RiskEngine VETOES or APPROVES → PaperBroker
    EXECUTES → AuditLog RECORDS every step with the reasoning attached.
No component can trade around the RiskEngine: the broker refuses any order
that doesn't carry an approval stamp.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .state import Config, Event, EventBus, GlobalState


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

class AuditLog:
    """Every action: who, what, the data trigger, the model, the reasoning.
    Persisted as JSONL so the record survives restarts."""

    def __init__(self, bus: EventBus, path: str = "runtime/audit.jsonl"):
        self._bus = bus
        self._path = path
        self._mem: deque[dict] = deque(maxlen=1000)
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def record(self, actor: str, action: str, trigger: str = "",
               model: str = "", reasoning: str = "",
               data: dict | None = None) -> dict:
        rec = {"id": str(uuid.uuid4())[:8], "ts": time.time(),
               "actor": actor, "action": action, "trigger": trigger,
               "model": model, "reasoning": reasoning, "data": data or {}}
        with self._lock:
            self._mem.append(rec)
            try:
                with open(self._path, "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
            except Exception:
                pass
        self._bus.publish(Event("audit.record", rec, source=actor))
        return rec

    def tail(self, n: int = 25) -> list[dict]:
        with self._lock:
            return list(self._mem)[-n:]


# ---------------------------------------------------------------------------
# Risk engine — absolute veto
# ---------------------------------------------------------------------------

@dataclass
class Order:
    ticker: str
    side: str                       # "BUY" | "SELL"
    qty: int
    reason: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    approved: bool = False          # stamped ONLY by RiskEngine
    veto_reason: str = ""


class RiskEngine:
    """Holds veto power over every order and every risky state change."""

    def __init__(self, cfg: Config, bus: EventBus, state: GlobalState,
                 audit: AuditLog, circuit_breaker=None):
        self.cfg, self._bus, self._state, self._audit = cfg, bus, state, audit
        self._circuit_breaker = circuit_breaker

    # ---- live metrics -----------------------------------------------------
    def parametric_var(self, positions: dict, returns: dict[str, pd.Series],
                       equity: float, conf: float = 0.95) -> float | None:
        tks = [t for t in positions if t in returns and positions[t]["qty"] > 0]
        if not tks or equity <= 0:
            return None
        w = np.array([positions[t]["qty"] * positions[t]["avg_price"]
                      for t in tks], dtype=float)
        R = pd.DataFrame({t: returns[t] for t in tks}).dropna()
        if len(R) < 30:
            return None
        from scipy import stats
        var = float(stats.norm.ppf(conf) * np.sqrt(w @ R.cov().values @ w))
        return round(var / equity * 100, 2)          # % of equity

    # ---- the veto ----------------------------------------------------------
    def review(self, order: Order, broker: "PaperBroker",
               price: float, returns: dict[str, pd.Series] | None = None,
               cost_info: dict | None = None) -> Order:
        eq = broker.equity({order.ticker: price})
        checks: list[tuple[bool, str]] = []

        if order.side == "BUY":
            notional = order.qty * price
            pos_after = broker.position_value(order.ticker, price) + notional
            checks.append((pos_after <= eq * self.cfg.max_position_pct / 100,
                           f"position cap {self.cfg.max_position_pct}% "
                           f"(would be {pos_after / eq * 100:.1f}%)"))
            gross_after = broker.gross_exposure({order.ticker: price}) + notional
            checks.append(
                (gross_after <= eq * self.cfg.max_gross_exposure_pct / 100,
                 f"gross exposure cap {self.cfg.max_gross_exposure_pct}%"))
            checks.append((notional <= broker.cash,
                           f"cash (${broker.cash:,.0f} available)"))
            aum_basis = self.cfg.aum if self.cfg.aum > 0 else eq
            if (self.cfg.max_position_mode == "fixed"
                    and self.cfg.max_position_fixed_usd > 0):
                cap_usd = self.cfg.max_position_fixed_usd
                cap_label = f"${cap_usd:,.0f} fixed max position size"
            else:
                cap_usd = aum_basis * self.cfg.max_position_pct / 100
                cap_label = (f"{self.cfg.max_position_pct}% of AUM "
                             f"(${aum_basis:,.0f})")
            checks.append((pos_after <= cap_usd,
                           f"max position size — {cap_label} "
                           f"(position would be ${pos_after:,.0f})"))
            if cost_info:
                edge = cost_info.get("expected_edge_pct")
                cost = cost_info.get("expected_cost_pct")
                if edge is not None and cost is not None:
                    checks.append((
                        edge >= 2 * cost,
                        f"expected edge {edge:.2f}% must be >= 2x expected "
                        f"cost {cost:.2f}% (needs >= {2 * cost:.2f}%) — "
                        f"P7b transaction cost gate"))
            if self._circuit_breaker is not None:
                cb = self._circuit_breaker.update(eq)
                checks.append((
                    not cb["halted"],
                    f"drawdown circuit breaker HALTED at "
                    f"{cb['drawdown_pct']}% from peak — needs a manual "
                    f"owner reset with a reason"))
                checks.append((
                    not cb["only_risk_reducing"],
                    f"drawdown circuit breaker: {cb['drawdown_pct']}% from "
                    f"peak — only risk-reducing orders allowed"))
        daily = broker.daily_pnl_pct({order.ticker: price})
        checks.append((daily > -self.cfg.max_daily_loss_pct or
                       order.side == "SELL",
                       f"daily loss limit {self.cfg.max_daily_loss_pct}% "
                       f"(today {daily:+.2f}%) — only risk-reducing orders"))
        if returns:
            v = self.parametric_var(broker.positions_dict(), returns, eq)
            if v is not None and order.side == "BUY":
                checks.append((v <= self.cfg.max_var_pct,
                               f"portfolio VaR cap {self.cfg.max_var_pct}% "
                               f"(now {v}%)"))

        failed = [msg for ok, msg in checks if not ok]
        if failed:
            order.approved, order.veto_reason = False, "; ".join(failed)
            self._audit.record("RiskEngine", "VETO", trigger=order.reason,
                               model="limits+VaR",
                               reasoning=f"Blocked {order.side} {order.qty} "
                                         f"{order.ticker}: {order.veto_reason}",
                               data={"order_id": order.id})
            self._bus.publish(Event("risk.veto", {"order": order.__dict__}))
        else:
            order.approved = True
            self._audit.record("RiskEngine", "APPROVE", trigger=order.reason,
                               model="limits+VaR",
                               reasoning=f"{order.side} {order.qty} "
                                         f"{order.ticker} within all limits",
                               data={"order_id": order.id})
        self._state.set("risk.last_review",
                        {"order": order.ticker, "approved": order.approved,
                         "veto_reason": order.veto_reason}, source="risk")
        return order


# ---------------------------------------------------------------------------
# Paper broker
# ---------------------------------------------------------------------------

class PaperBroker:
    """Simulated execution against real prices. Refuses unapproved orders."""

    COMMISSION = 0.0005
    SLIPPAGE = 0.0005

    def __init__(self, cfg: Config, bus: EventBus, state: GlobalState,
                 audit: AuditLog, path: str = "runtime/broker.json"):
        self._bus, self._state, self._audit = bus, state, audit
        self._path = path
        self._lock = threading.RLock()
        self.cash = cfg.starting_cash
        self.start_equity = cfg.starting_cash
        self.day_start_equity = cfg.starting_cash
        self.positions: dict[str, dict] = {}     # tkr -> {qty, avg_price}
        self.fills: list[dict] = []
        self._load()

    # ---- persistence -------------------------------------------------------
    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    d = json.load(f)
                self.cash = d["cash"]
                self.start_equity = d["start_equity"]
                self.day_start_equity = d.get("day_start_equity", self.cash)
                self.positions = d["positions"]
                self.fills = d["fills"]
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump({"cash": self.cash, "start_equity": self.start_equity,
                       "day_start_equity": self.day_start_equity,
                       "positions": self.positions, "fills": self.fills},
                      f, indent=1, default=str)

    # ---- accounting --------------------------------------------------------
    def position_value(self, ticker: str, price: float) -> float:
        p = self.positions.get(ticker)
        return p["qty"] * price if p else 0.0

    def gross_exposure(self, prices: dict[str, float]) -> float:
        return sum(p["qty"] * prices.get(t, p["avg_price"])
                   for t, p in self.positions.items())

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.gross_exposure(prices)

    def daily_pnl_pct(self, prices: dict[str, float]) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return (self.equity(prices) / self.day_start_equity - 1) * 100

    def positions_dict(self) -> dict:
        return json.loads(json.dumps(self.positions))

    # ---- execution ---------------------------------------------------------
    def execute(self, order: Order, price: float) -> dict | None:
        if not order.approved:
            self._audit.record("PaperBroker", "REJECT",
                               reasoning="order lacked RiskEngine approval",
                               data={"order_id": order.id})
            return None
        with self._lock:
            px = price * (1 + self.SLIPPAGE) if order.side == "BUY" \
                else price * (1 - self.SLIPPAGE)
            fee = order.qty * px * self.COMMISSION
            if order.side == "BUY":
                cost = order.qty * px + fee
                if cost > self.cash:
                    return None
                self.cash -= cost
                p = self.positions.setdefault(order.ticker,
                                              {"qty": 0, "avg_price": 0.0})
                tot = p["qty"] * p["avg_price"] + order.qty * px
                p["qty"] += order.qty
                p["avg_price"] = tot / p["qty"]
                realized = 0.0
            else:
                p = self.positions.get(order.ticker)
                if not p or p["qty"] < order.qty:
                    return None
                self.cash += order.qty * px - fee
                realized = order.qty * (px - p["avg_price"]) - fee
                p["qty"] -= order.qty
                if p["qty"] == 0:
                    del self.positions[order.ticker]
            # P7d: slippage vs the DECISION price (pre-slippage `price`
            # arg), sign-normalized so positive always means "cost" —
            # paying more than decision price on a BUY, or receiving less
            # than decision price on a SELL.
            slip_sign = 1 if order.side == "BUY" else -1
            slippage_pct = round((px - price) / price * 100 * slip_sign, 4) \
                if price > 0 else 0.0
            fill = {"ts": time.time(), "order_id": order.id,
                    "ticker": order.ticker, "side": order.side,
                    "qty": order.qty, "price": round(px, 4),
                    "decision_price": round(price, 4),
                    "slippage_pct": slippage_pct,
                    "fee": round(fee, 4), "realized": round(realized, 2),
                    "reason": order.reason}
            self.fills.append(fill)
            self._save()
        self._bus.publish(Event("broker.fill", fill, source="broker"))
        self._state.set("portfolio",
                        {"cash": round(self.cash, 2),
                         "positions": self.positions_dict(),
                         "n_fills": len(self.fills)}, source="broker")
        self._audit.record("PaperBroker", f"FILL {order.side}",
                           trigger=order.reason, model="paper-exec",
                           reasoning=f"{order.side} {order.qty} "
                                     f"{order.ticker} @ ${px:,.2f} "
                                     f"(fee ${fee:.2f})",
                           data=fill)
        return fill
