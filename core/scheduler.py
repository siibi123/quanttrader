"""Background scheduler (P8) — the autonomous cadence CLAUDE.md's roadmap
flagged as future work ("scheduler for autonomous cycles"). Quotes stay on
PollingFeed's own thread (already market-hours gated there, see
data/providers.py); this module drives the slower, wall-clock-aware
cadences on top of one APScheduler BackgroundScheduler per process:

  * decision cycle  — every 5 minutes, no-ops unless market_status() is
    "open" (checked inside the job, same pattern as every other gate in
    this codebase — e.g. quant.regime_gate)
  * morning briefing — 9:25 ET, weekdays
  * daily report      — 16:05 ET, weekdays

Every job body is wrapped so a raised exception is logged and swallowed,
never kills the scheduler thread — "every network call fails gracefully"
(Iron Rule #8) applies here too.
"""
from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from core.state import ET, market_status

log = logging.getLogger("quanttrader.scheduler")


class TradingScheduler:
    """Owns one BackgroundScheduler for the whole process. `symbols_fn`/
    `risk_pct_fn` are callables (not fixed values) so sidebar edits take
    effect on the next scheduled run without restarting the scheduler —
    the app wires them to read GlobalState.get("ui.watchlist"/"ui.risk_pct"),
    which the sidebar keeps current every rerun."""

    def __init__(self, orch, symbols_fn, risk_pct_fn=lambda: 1.0,
                bypass_incubation_fn=lambda: False, universe_fn=None):
        self._orch = orch
        self._symbols_fn = symbols_fn
        self._risk_pct_fn = risk_pct_fn
        self._bypass_incubation_fn = bypass_incubation_fn
        # universe_fn (P9 fix): the full S&P500+Nasdaq100 scan universe,
        # used ONLY for sector_scan's candidate ranking inside the daily
        # report / morning briefing -- NOT for the decision cycle, which
        # keeps trading symbols_fn()'s small watchlist (see app.py's
        # TRADING_WATCHLIST comment for why that split is deliberate).
        self._universe_fn = universe_fn or symbols_fn
        self.scheduler = BackgroundScheduler(timezone=ET)
        self._started = False

    @staticmethod
    def _safe(fn, name):
        try:
            fn()
        except Exception:
            log.exception("scheduler job %s failed", name)

    def _run_decision_cycle(self):
        if market_status()["session"] != "open":
            return
        symbols = self._symbols_fn()
        if symbols:
            self._orch.step(symbols, risk_pct=self._risk_pct_fn(),
                            bypass_incubation=self._bypass_incubation_fn())

    def _run_daily_report(self):
        self._orch.daily_report(watchlist=self._symbols_fn(),
                                scan_universe=self._universe_fn())

    def _run_morning_briefing(self):
        self._orch.morning_briefing(watchlist=self._symbols_fn(),
                                    scan_universe=self._universe_fn())

    def start(self):
        if self._started:
            return
        self.scheduler.add_job(
            lambda: self._safe(self._run_decision_cycle, "decision_cycle"),
            "interval", minutes=5, id="decision_cycle",
            max_instances=1, coalesce=True, misfire_grace_time=60)
        self.scheduler.add_job(
            lambda: self._safe(self._run_morning_briefing, "morning_briefing"),
            CronTrigger(day_of_week="mon-fri", hour=9, minute=25, timezone=ET),
            id="morning_briefing", max_instances=1, coalesce=True,
            misfire_grace_time=300)
        self.scheduler.add_job(
            lambda: self._safe(self._run_daily_report, "daily_report"),
            CronTrigger(day_of_week="mon-fri", hour=16, minute=5, timezone=ET),
            id="daily_report", max_instances=1, coalesce=True,
            misfire_grace_time=300)
        self.scheduler.start()
        self._started = True

    def shutdown(self):
        if self._started:
            self.scheduler.shutdown(wait=False)
            self._started = False

    def next_run(self, job_id: str):
        """The job's next_run_time (a tz-aware datetime) or None if the
        scheduler isn't running or the job doesn't exist -- used for the
        sidebar's live countdown."""
        if not self._started:
            return None
        job = self.scheduler.get_job(job_id)
        return job.next_run_time if job else None

    def status(self) -> dict:
        if not self._started:
            return {"running": False, "jobs": []}
        jobs = []
        for j in self.scheduler.get_jobs():
            nrt = j.next_run_time
            jobs.append({"id": j.id,
                        "next_run": nrt.strftime("%Y-%m-%d %H:%M %Z")
                        if nrt else None})
        return {"running": True, "jobs": jobs}
