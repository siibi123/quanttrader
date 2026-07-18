"""Drawdown circuit breakers — the position-size and trading-halt ladder
a real desk enforces on ITSELF, not just on individual orders.

  5% drawdown from peak equity  -> position sizes auto-cut, gradually
  10% drawdown                  -> only risk-reducing (SELL/exit) trades
  15% drawdown                  -> FULL STOP: no new entries at all,
                                    requires a manual owner reset with a
                                    written reason (logged to audit)

Recovery is gradual, not instant: the size multiplier is a smooth
piecewise-linear function of the CURRENT drawdown, not a step that
snaps back to 1.0 the moment equity ticks up. A trip to the 15% hard
stop is sticky — it does NOT auto-clear just because equity recovers;
it stays halted until manual_reset() is called with a reason.

Existing positions can always be exited regardless of state — this only
ever restricts NEW entries, the same standing principle as Iron Rule
#10's strategy-promotion gate: closing risk is never gated.
"""
from __future__ import annotations

import json
import os
import threading
import time

from core.engine import AuditLog

DD_CUT_START_PCT = 5.0
DD_RISK_REDUCING_ONLY_PCT = 10.0
DD_HALT_PCT = 15.0


class DrawdownCircuitBreaker:
    def __init__(self, audit: AuditLog, path: str = "runtime/circuit_breaker.json"):
        self._audit = audit
        self._path = path
        self._lock = threading.RLock()
        self._data = {"peak_equity": 0.0, "halted": False, "halted_at": None,
                     "reset_log": []}
        self._last_status: dict = {}
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    self._data.update(json.load(f))
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=1, default=str)

    @staticmethod
    def _multiplier(dd_pct: float) -> float:
        """Piecewise-linear, gradual — never a step function."""
        if dd_pct <= 0:
            return 1.0
        if dd_pct <= DD_CUT_START_PCT:
            return 1.0 - 0.5 * (dd_pct / DD_CUT_START_PCT)
        if dd_pct <= DD_RISK_REDUCING_ONLY_PCT:
            span = DD_RISK_REDUCING_ONLY_PCT - DD_CUT_START_PCT
            return 0.5 * (1 - (dd_pct - DD_CUT_START_PCT) / span)
        return 0.0

    def update(self, equity: float) -> dict:
        """Call with the live equity mark (safe to call often/idempotently
        — e.g. once per decision cycle AND again inside every RiskEngine
        review()). Updates the peak, computes current status, and trips
        the hard stop (audited) the moment drawdown first crosses 15% —
        it does not un-trip on its own afterward."""
        if equity <= 0:
            return self._last_status or self.status()
        with self._lock:
            if equity > self._data["peak_equity"]:
                self._data["peak_equity"] = equity
            peak = self._data["peak_equity"]
            dd_pct = round((peak - equity) / peak * 100, 2) if peak > 0 else 0.0

            if dd_pct >= DD_HALT_PCT and not self._data["halted"]:
                self._data["halted"] = True
                self._data["halted_at"] = time.time()
                self._audit.record(
                    "RiskEngine", "CIRCUIT BREAKER TRIPPED",
                    model=f"drawdown >= {DD_HALT_PCT}% from peak",
                    reasoning=(f"Drawdown {dd_pct}% from peak ${peak:,.0f} "
                              f"(now ${equity:,.0f}) — FULL STOP on new "
                              f"entries. Existing positions remain "
                              f"exitable. Requires a manual reset with a "
                              f"written reason to resume new entries."),
                    data={"drawdown_pct": dd_pct, "peak_equity": peak,
                         "equity": equity})
            self._save()

            halted = self._data["halted"]
            only_risk_reducing = halted or dd_pct >= DD_RISK_REDUCING_ONLY_PCT
            multiplier = 0.0 if only_risk_reducing else self._multiplier(dd_pct)
            self._last_status = {
                "peak_equity": peak, "equity": equity, "drawdown_pct": dd_pct,
                "size_multiplier": multiplier,
                "only_risk_reducing": only_risk_reducing, "halted": halted}
            return dict(self._last_status)

    def manual_reset(self, reason: str) -> dict:
        """The only way to clear a tripped halt. Requires a non-empty,
        meaningful reason — logged to audit permanently as the owner's own
        accountability trail for overriding a circuit breaker."""
        if not reason or not reason.strip():
            raise ValueError("manual_reset requires a written reason")
        with self._lock:
            self._data["halted"] = False
            self._data["reset_log"].append(
                {"ts": time.time(), "reason": reason.strip()})
            if self._last_status:
                self._last_status["halted"] = False
                self._last_status["only_risk_reducing"] = (
                    self._last_status.get("drawdown_pct", 0)
                    >= DD_RISK_REDUCING_ONLY_PCT)
            self._save()
            self._audit.record(
                "Owner", "CIRCUIT BREAKER RESET", model="manual override",
                reasoning=f"Owner manually reset the circuit breaker: "
                          f"\"{reason.strip()}\"",
                data={"reason": reason.strip()})
            return dict(self._data)

    def status(self) -> dict:
        with self._lock:
            if self._last_status:
                return dict(self._last_status)
            peak = self._data["peak_equity"]
            return {"peak_equity": peak, "equity": peak, "drawdown_pct": 0.0,
                   "size_multiplier": 1.0, "only_risk_reducing": False,
                   "halted": self._data["halted"]}
