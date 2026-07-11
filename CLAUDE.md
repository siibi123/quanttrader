# QUANTTRADER — CONSTITUTION (Claude Code reads this every session)

## Owner
Yahav — Israeli retail trader, NON-CODER. You build/test/commit/push; explain simply; never assume he can debug. Hebrew/English.

## What this is
A fresh, enterprise-grade quant platform. Event-driven engine (core/), pluggable data (data/), AI-ready orchestration (ai/), thin Streamlit shell (app.py). Deployed later to Streamlit Cloud and eventually a VPS; engine must always run headless.

## Architecture (v0.1 — 23/23 core tests passing)
- core/state.py — Config (.env only), EventBus (thread-safe pub/sub), GlobalState (dot-path store; to_ai_context() exports curated snapshot for the AI)
- core/engine.py — AuditLog (JSONL, reasoning attached to every action), RiskEngine (ABSOLUTE veto: position/gross/daily-loss/VaR caps), PaperBroker (refuses unapproved orders; slippage+fees; persisted)
- data/providers.py — DataProvider ABC; LSEProvider (probe-then-lock endpoints — their docs are JS-rendered, NEVER hardcode guessed URLs); YahooProvider fallback; FakeProvider for tests; CompositeProvider chain; PollingFeed (background thread → tick events)
- ai/orchestrator.py — TOOL_SCHEMAS (the ONLY machine surface the AI may use); RuleOrchestrator v1 (deterministic, reasoning strings, risk-reviewed); LLMOrchestrator socket (refuses without ANTHROPIC_API_KEY — no fake AI, ever)
- tests/test_core.py — run `python3 tests/test_core.py` before EVERY commit; extend it with every new module

## IRON RULES
1. PAPER ONLY. No real broker execution without an explicit, separate, owner-confirmed phase.
2. RiskEngine veto is sacred: nothing may execute around it; the broker's approved-stamp check stays.
3. Every action gets an AuditLog record with reasoning. If it isn't audited, it didn't happen.
4. No fake AI: LLMOrchestrator only runs with a real key; suggestions must trace to computed numbers.
5. Secrets via .env/os.getenv only. Never commit .env. Never print keys.
6. LSE endpoints must be VERIFIED (probe or a curl example from their docs pasted by owner) before relying on them; Yahoo fallback stays forever.
7. Tests before commit; small commits; tell the owner what to check after each push; never break main.
8. 1GB-class hosting: no torch/GPU deps; cache aggressively; every network call fails gracefully.
9. Honesty in UI copy: no promised returns; paper results labeled paper.

## ROADMAP (build in order; owner picks pace)
1. Port QuantSignal's proven engines (github.com/siibi123/quantsignal → quant/): composite signals, B-Xtrender, backtest v2 (blend), playbook, validation lab — as strategy plugins behind RuleOrchestrator.
2. NewsIngestionEngine: Finnhub free key (NEWS_API_KEY) → headlines→ticker/sentiment→state.news + interrupt events; stub cleanly if no key.
3. Options module via LSE (chains WITH greeks, flow) once endpoints verified; 3D IV surface + flow heatmap (Plotly) in UI.
4. LLMOrchestrator implementation when ANTHROPIC_API_KEY exists: messages+tools loop over TOOL_SCHEMAS, state.to_ai_context() as system context, every tool call risk-reviewed. Narrate-only mode first; propose-mode second.
5. Scheduler for autonomous cycles (in-process while awake; VPS systemd later). 6. Hetzner deploy guide + systemd unit. 7. WebSocket StreamingFeed ONLY if LSE docs confirm WS exists.
