"""Daily institutional report — P&L attribution, risk-limit utilization,
signal quality, and tomorrow's candidate orders, rendered as clean
markdown. Pure formatting only: the caller (RuleOrchestrator) gathers
the data from state/broker/registry; this module has no state/audit
access, keeping the same pure-function boundary the rest of quant/ uses.
"""
from __future__ import annotations


def render_report(data: dict) -> str:
    """data keys: date, equity, day_start_equity, pnl_today_$,
    pnl_today_pct, fills_today (list of fill dicts + 'strategy'),
    risk_limits (dict name -> {used, cap, pct}), signals_today
    (dict: n, buy, sell), settled_today (dict: n, win_rate_pct,
    mean_return_pct), settled_cumulative (same shape, all-time),
    suggestions (list from sector_scan's 'names'), notes (list of str
    honest caveats)."""
    d = data
    lines = [f"# QuantTrader Daily Report — {d['date']}", ""]

    lines += ["## P&L Attribution", ""]
    lines.append(f"Equity: ${d['equity']:,.2f} · Today: "
                f"{d['pnl_today_$']:+,.2f} ({d['pnl_today_pct']:+.2f}%)")
    lines.append("")
    if d["fills_today"]:
        lines.append("| Ticker | Side | Qty | Price | Realized $ | Strategy |")
        lines.append("|---|---|---|---|---|---|")
        for f in d["fills_today"]:
            lines.append(f"| {f['ticker']} | {f['side']} | {f['qty']} | "
                        f"${f['price']:.2f} | {f['realized']:+.2f} | "
                        f"{f.get('strategy', '—')} |")
    else:
        lines.append("_No fills today._")
    lines.append("")

    lines += ["## Risk Limit Utilization", ""]
    if d["risk_limits"]:
        lines.append("| Limit | Used | Cap | % of cap |")
        lines.append("|---|---|---|---|")
        for name, r in d["risk_limits"].items():
            lines.append(f"| {name} | {r['used']} | {r['cap']} | "
                        f"{r['pct']:.0f}% |")
    else:
        lines.append("_No limits to report (flat book)._")
    lines.append("")

    lines += ["## Signal Quality", ""]
    st = d["signals_today"]
    lines.append(f"{st['n']} new signal(s) logged today "
                f"({st['buy']} BUY / {st['sell']} SELL).")
    lines.append("")
    stl = d["settled_today"]
    if stl["n"]:
        lines.append(f"**Settled today:** {stl['n']} signal(s) · win rate "
                    f"{stl['win_rate_pct']}% · mean forward return "
                    f"{stl['mean_return_pct']:+.2f}%")
    else:
        lines.append("_No signals settled today._")
    cum = d.get("settled_cumulative")
    if cum and cum["n"]:
        lines.append(f"\n**Cumulative to date:** {cum['n']} settled signal(s) "
                    f"· win rate {cum['win_rate_pct']}% · mean forward "
                    f"return {cum['mean_return_pct']:+.2f}%")
    lines.append("")

    lines += ["## Tomorrow's Candidate Orders", ""]
    lines.append("_Suggestions only — this platform executes within the "
                "same decision cycle it proposes and has no pending-order "
                "queue; shown here as what the sector/target engine (P5) "
                "currently ranks highest._")
    lines.append("")
    if d["suggestions"]:
        lines.append("| Ticker | Verdict | Score | Entry | Stop | Target | Why |")
        lines.append("|---|---|---|---|---|---|---|")
        for n in d["suggestions"][:10]:
            why = "; ".join(n.get("reasons_pro", [])[:2]).replace("|", "/")
            lines.append(f"| {n['ticker']} | {n['verdict']} | "
                        f"{n['target_score']} | {n['entry']} | {n['stop']} | "
                        f"{n['target']} | {why} |")
    else:
        lines.append("_No tradeable candidates ranked as of report time._")
    lines.append("")

    if d.get("notes"):
        lines += ["## Notes", ""] + [f"- {n}" for n in d["notes"]]

    return "\n".join(lines) + "\n"


def render_morning_briefing(data: dict) -> str:
    """data keys: date, equity, circuit_breaker (status dict or None),
    correlation_regime (dict or None), regimes (dict ticker -> regime
    label), candidates (list from sector_scan's 'names'), notes (list of
    str). Same pure-formatting boundary as render_report — no state/audit
    access here, the caller (RuleOrchestrator.morning_briefing) gathers
    everything first."""
    d = data
    lines = [f"# QuantTrader Morning Briefing — {d['date']}", ""]

    lines += ["## Account", ""]
    lines.append(f"Equity: ${d['equity']:,.2f}")
    lines.append("")

    lines += ["## Risk Status", ""]
    cb = d.get("circuit_breaker")
    if cb:
        state_note = ("HALTED" if cb.get("halted") else
                      "risk-reducing only" if cb.get("only_risk_reducing")
                      else "normal")
        lines.append(f"Drawdown circuit breaker: {cb.get('drawdown_pct', 0)}% "
                    f"from peak ${cb.get('peak_equity', 0):,.0f} — {state_note}")
    else:
        lines.append("_No circuit breaker state yet (flat book)._")
    cr = d.get("correlation_regime")
    if cr:
        lines.append(f"Correlation regime: {cr['verdict']} (avg pairwise "
                    f"{cr['current_avg_correlation']})")
    lines.append("")

    lines += ["## Regime by Watchlist Symbol", ""]
    regs = d.get("regimes") or {}
    if regs:
        lines.append("| Ticker | Regime |")
        lines.append("|---|---|")
        for t, r in regs.items():
            lines.append(f"| {t} | {r} |")
    else:
        lines.append("_No regime reads yet._")
    lines.append("")

    lines += ["## Today's Candidates", ""]
    lines.append("_Suggestions only — the same sector/target scan the LAB "
                "tab uses; nothing here executes a trade._")
    lines.append("")
    if d["candidates"]:
        lines.append("| Ticker | Verdict | Score | Entry | Stop | Target |")
        lines.append("|---|---|---|---|---|---|")
        for n in d["candidates"][:10]:
            lines.append(f"| {n['ticker']} | {n['verdict']} | "
                        f"{n['target_score']} | {n['entry']} | {n['stop']} | "
                        f"{n['target']} |")
    else:
        lines.append("_No tradeable candidates ranked as of briefing time._")
    lines.append("")

    if d.get("notes"):
        lines += ["## Notes", ""] + [f"- {n}" for n in d["notes"]]

    return "\n".join(lines) + "\n"
