"""Strategy Registry — the promotion gate no signal skips.

Every strategy starts in INCUBATION: its BUY/SELL signals are logged
with a forward-return horizon but never placed as a real (paper) entry.
Once a strategy has >= MIN_SIGNALS_TO_PROMOTE settled signals (their
forward-return horizon has elapsed) AND the bootstrap 90% CI on those
forward returns excludes zero, it promotes to PAPER — only then may
RuleOrchestrator.step() place real paper entries from its signals.
Deflated Sharpe (Bailey & Lopez de Prado) and the Harvey-Liu-Zhu
multiple-testing haircut are computed and recorded alongside every
promotion decision so it's provable, not asserted; the bootstrap CI is
what actually gates the promotion. Existing open positions can always
be exited regardless of status — INCUBATION blocks new entries, never
risk-reducing exits.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid

import numpy as np
import pandas as pd

from core.engine import AuditLog
from quant.validation import bootstrap_mean_return, deflated_sharpe, haircut_pvalue, permutation_test

MIN_SIGNALS_TO_PROMOTE = 30
FORWARD_HORIZON_DAYS = 10


class StrategyRegistry:
    STATUS_INCUBATION = "INCUBATION"
    STATUS_PAPER = "PAPER"

    def __init__(self, audit: AuditLog, path: str = "runtime/strategy_registry.json"):
        self._audit = audit
        self._path = path
        self._lock = threading.RLock()
        self._data: dict = {}
        self._load()

    # ---- persistence -------------------------------------------------------
    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def _save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=1, default=str)

    def _ensure(self, strategy: str) -> dict:
        return self._data.setdefault(strategy, {
            "status": self.STATUS_INCUBATION, "signals": [],
            "validation": {}, "promoted_at": None})

    # ---- lifecycle -----------------------------------------------------------
    def status(self, strategy: str) -> str:
        with self._lock:
            return self._ensure(strategy)["status"]

    def log_signal(self, strategy: str, symbol: str, direction: str,
                   price: float, horizon_days: int = FORWARD_HORIZON_DAYS,
                   regime: str | None = None) -> None:
        """direction: "BUY" or "SELL". Settled later via settle_signals().
        regime (P7c: Bull/Bear/Storm) is stored so performance can be
        broken out per regime later, even though promotion itself is
        gated on the pooled sample across all regimes."""
        with self._lock:
            st = self._ensure(strategy)
            st["signals"].append({
                "id": str(uuid.uuid4())[:8], "ts": time.time(),
                "symbol": symbol, "direction": direction, "entry_price": price,
                "horizon_days": horizon_days, "regime": regime,
                "settled": False, "forward_return": None})
            self._save()

    def settle_signals(self, strategy: str, price_lookup) -> int:
        """price_lookup(symbol) -> current price or None. Settles any
        signal whose horizon has elapsed, using today's price as the
        forward mark. Returns the count newly settled."""
        with self._lock:
            st = self._ensure(strategy)
            n = 0
            for s in st["signals"]:
                if s["settled"]:
                    continue
                age_days = (time.time() - s["ts"]) / 86400
                if age_days < s["horizon_days"]:
                    continue
                px = price_lookup(s["symbol"])
                if px is None or s["entry_price"] <= 0:
                    continue
                sign = 1 if s["direction"] == "BUY" else -1
                s["forward_return"] = float(px / s["entry_price"] - 1) * sign
                s["settled"] = True
                s["settled_ts"] = time.time()
                n += 1
            if n:
                self._save()
            return n

    def settled_returns(self, strategy: str) -> pd.Series:
        st = self._ensure(strategy)
        r = [s["forward_return"] for s in st["signals"] if s["settled"]]
        return pd.Series(r, dtype=float)

    def signal_counts(self, strategy: str) -> dict:
        st = self._ensure(strategy)
        settled = [s for s in st["signals"] if s["settled"]]
        return {"total": len(st["signals"]), "settled": len(settled),
               "pending": len(st["signals"]) - len(settled)}

    def last_validation(self, strategy: str) -> dict:
        with self._lock:
            return dict(self._ensure(strategy).get("validation") or {})

    def performance_by_regime(self, strategy: str) -> dict:
        """P7c: settled-signal stats (n, mean forward return, win rate)
        grouped by the Bull/Bear/Storm regime active when each signal
        was logged. Regimes with no settled signals are omitted rather
        than shown as a fabricated zero."""
        st = self._ensure(strategy)
        by_regime: dict[str, list[float]] = {}
        for s in st["signals"]:
            if not s["settled"]:
                continue
            by_regime.setdefault(s.get("regime") or "Unknown", []).append(
                s["forward_return"])
        out = {}
        for regime, rets in by_regime.items():
            arr = np.array(rets, dtype=float)
            out[regime] = {
                "n": len(arr),
                "mean_return_%": round(float(arr.mean()) * 100, 2),
                "win_rate_%": round(float((arr > 0).mean()) * 100, 1)}
        return out

    def evaluate_promotion(self, strategy: str, n_trials: int = 7) -> dict:
        """The mandatory gate: deflated Sharpe + HLZ haircut + permutation
        test are computed and recorded; the bootstrap 90% CI excluding
        zero is what actually promotes INCUBATION -> PAPER. n_trials
        defaults to 7 -- the composite signal's own model count
        (quant.verdict.MODELS) -- as an honest, traceable haircut basis
        rather than an arbitrary round number."""
        with self._lock:
            st = self._ensure(strategy)
            r = self.settled_returns(strategy)
            n_settled = len(r)
            result = {"strategy": strategy, "n_settled": n_settled,
                     "min_required": MIN_SIGNALS_TO_PROMOTE,
                     "prior_status": st["status"]}
            if n_settled < MIN_SIGNALS_TO_PROMOTE:
                result["decision"] = "NOT ENOUGH SIGNALS"
                st["validation"] = result
                self._save()
                return result

            horizon = st["signals"][0].get("horizon_days", FORWARD_HORIZON_DAYS)
            ann = np.sqrt(252 / max(horizon, 1))
            sharpe = float(r.mean() / r.std() * ann) if r.std() > 0 else 0.0
            tstat = float(r.mean() / (r.std() / np.sqrt(n_settled))) if r.std() > 0 else 0.0

            ds = deflated_sharpe(sharpe, n_trials=n_trials, n_obs=n_settled)
            hp = haircut_pvalue(tstat, n_tests=n_trials)
            pt = permutation_test(r)
            bc = bootstrap_mean_return(r)

            promote = bool(bc.get("excludes_zero", False))
            result.update({"sharpe_ann": round(sharpe, 2), "tstat": round(tstat, 2),
                          "deflated_sharpe": ds, "haircut": hp,
                          "permutation": pt, "bootstrap": bc,
                          "decision": "PROMOTE" if promote else "HOLD IN INCUBATION"})
            st["validation"] = result

            if promote and st["status"] == self.STATUS_INCUBATION:
                st["status"] = self.STATUS_PAPER
                st["promoted_at"] = time.time()
                self._audit.record(
                    "StrategyRegistry", "PROMOTE", trigger=strategy,
                    model="deflated Sharpe + HLZ haircut + permutation + "
                         "bootstrap CI (P7a mandatory gate)",
                    reasoning=(f"{strategy}: {n_settled} settled signals, "
                              f"bootstrap 90% CI [{bc.get('CI90_low_%')}%, "
                              f"{bc.get('CI90_high_%')}%] excludes zero -> "
                              f"promoted INCUBATION -> PAPER"),
                    data=result)
            else:
                self._audit.record(
                    "StrategyRegistry", "HOLD", trigger=strategy,
                    model="deflated Sharpe + HLZ haircut + permutation + "
                         "bootstrap CI (P7a mandatory gate)",
                    reasoning=(f"{strategy}: {n_settled} settled signals, "
                              f"remains {st['status']} ({result['decision']})"),
                    data=result)
            self._save()
            return result
