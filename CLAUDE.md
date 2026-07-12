# QUANTTRADER — CONSTITUTION (Claude Code reads this every session)

## Owner
Yahav — Israeli retail trader, NON-CODER. You build/test/commit/push; explain simply; never assume he can debug. Hebrew/English.

## What this is
A fresh, enterprise-grade quant platform. Event-driven engine (core/), pluggable data (data/), AI-ready orchestration (ai/), thin Streamlit shell (app.py). Deployed later to Streamlit Cloud and eventually a VPS; engine must always run headless.

## Architecture (v0.4 — 29/29 core tests passing)
- core/state.py — Config (.env only), EventBus (thread-safe pub/sub), GlobalState (dot-path store; to_ai_context() exports curated snapshot for the AI)
- core/engine.py — AuditLog (JSONL, reasoning attached to every action), RiskEngine (ABSOLUTE veto: position/gross/daily-loss/VaR caps), PaperBroker (refuses unapproved orders; slippage+fees; persisted)
- data/providers.py — DataProvider ABC; LSEProvider (probe-then-lock endpoints — their docs are JS-rendered, NEVER hardcode guessed URLs); YahooProvider fallback; FakeProvider for tests; CompositeProvider chain; PollingFeed (background thread → tick events)
- ai/orchestrator.py — TOOL_SCHEMAS (the ONLY machine surface the AI may use); RuleOrchestrator v1 (deterministic, reasoning strings, risk-reviewed); LLMOrchestrator socket (refuses without ANTHROPIC_API_KEY — no fake AI, ever)
- tests/test_core.py — run `python tests/test_core.py` before EVERY commit; extend it with every new module (currently 29 checks). NOTE: on this machine `python3` resolves to the Windows Store stub and hangs — use `python`.

## IRON RULES
1. PAPER ONLY. No real broker execution without an explicit, separate, owner-confirmed phase.
2. RiskEngine veto is sacred: nothing may execute around it; the broker's approved-stamp check stays.
3. Every action gets an AuditLog record with reasoning. If it isn't audited, it didn't happen.
4. No fake AI: LLMOrchestrator only runs with a real key; suggestions must trace to computed numbers.
5. Secrets via .env/os.getenv only. Never commit .env. Never print keys.
6. LSE contract is VERIFIED from their official SDK (github.com/londonstrategicedge/lse-data v0.14.0): REST GET {vault}/candles with x-api-key header + custom User-Agent (their CDN blocks default python UA); /options/chain carries precomputed greeks. Yahoo fallback stays forever.
7. Tests before commit; small commits; tell the owner what to check after each push; never break main.
8. 1GB-class hosting: no torch/GPU deps; cache aggressively; every network call fails gracefully.
9. Honesty in UI copy: no promised returns; paper results labeled paper.

## ROADMAP (build in order; owner picks pace)
DONE (v0.3): QuantSignal engines live in quant/ (signals, bxtrender, backtest v2, verdict, playbook, scanner, risk, validation, montecarlo, levels, advanced). Playbook+verdict drive the policy; correlation_heat+VaR guard the book each cycle.

P1 — Fixes & portfolio controls (IN PROGRESS):
(a) DONE: LSEProvider.get_quote now compares live price to the previous DAILY close, not the previous 1-minute bar.
(b) DONE: sidebar AUM (Total Portfolio Capital) + Max Position Size (fixed $ or % of AUM) inputs, written into a per-cycle Config via dataclasses.replace on RiskEngine.cfg, enforced as a new veto check.
(c) DONE: per-position "% of AUM" column in the Open Book view (TRADES tab).

P2 — Hedge-fund math in quant/: hmm_regime.py (Gaussian HMM 2-3 states), kalman_pairs.py (Kalman dynamic hedge ratio + spread z-score), GARCH(1,1) via `arch` (ask before adding the dependency), Ledoit-Wolf shrinkage portfolio optimizer. Pure functions, 2 tests each.

P3 — 3D volatility surface: Plotly Surface (strike x DTE x IV) from LSE /options/chain; surface_interpreter.py — rule-based plain-text readout (skew steepness, term-structure inversion, smile anomalies) feeding audit trail + AI context. No LLM calls, deterministic rules only.

P4 — Intelligence feeds: Finnhub news/sentiment behind NEWS_API_KEY, clean stub with no key. LSE macro endpoints (rates, CPI) into state.macro. Curated anomaly library with academic citations injected into AI context. LSE options-flow spikes as institutional flow tracker. Never fake data that has no real source.

P5 — Sector and target engine: multi-factor scoring (flow spikes, momentum, macro tilt, sentiment) producing ranked sectors/names with entry, stop, avoid zones. Every suggestion carries its computed mathematical reason. All orders still go through propose → RiskEngine veto → PaperBroker → AuditLog.

P6 — Flow Confluence Engine:
- P6a: quant/orderflow.py — Bulk Volume Classification (Easley/Lopez de Prado/O'Hara), CVD with price-divergence flag, VPIN toxicity percentile vs symbol history, volume-profile top-3 nodes. Pure functions, 3+ tests on synthetic bars.
- P6b: quant/optionflow.py — verify LSE flow endpoint from their SDK first, no guessed URLs; if none exists, use chain-delta proxies and document that honestly. Call/put premium share, volume/OI spike z-scores vs 20-day norms, largest prints.
- P6c: quant/flow_confluence.py — one score per symbol combining tape pressure + options positioning → CONFLUENCE LONG/SHORT/CONFLICT/QUIET with numbers. Wired into state.flow, audit, AI context, P5 scoring. VPIN >85th percentile adds an informational caution flag to RiskEngine reasoning only (NOT a hard veto — ask owner before changing that).
- UI: FLOW panel in LAB tab — CVD chart over price, VPIN gauge, options premium imbalance bar, confluence verdict with reasoning line.

Later: 7. WebSocket StreamingFeed — URI verified to exist (wss://data-ws.londonstrategicedge.com, parked as LSEProvider.WS_URL_ROADMAP). Build ONLY with owner sign-off after polling proves stable on the hosting; PollingFeed remains primary. Hetzner deploy guide + systemd unit; scheduler for autonomous cycles.
