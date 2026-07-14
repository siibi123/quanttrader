# QUANTTRADER — CONSTITUTION (Claude Code reads this every session)

## Owner
Yahav — Israeli retail trader, NON-CODER. You build/test/commit/push; explain simply; never assume he can debug. Hebrew/English.

## What this is
A fresh, enterprise-grade quant platform. Event-driven engine (core/), pluggable data (data/), AI-ready orchestration (ai/), thin Streamlit shell (app.py). Deployed later to Streamlit Cloud and eventually a VPS; engine must always run headless.

## Architecture (v0.4 — 59/59 core tests passing)
- core/state.py — Config (.env only), EventBus (thread-safe pub/sub), GlobalState (dot-path store; to_ai_context() exports curated snapshot for the AI)
- core/engine.py — AuditLog (JSONL, reasoning attached to every action), RiskEngine (ABSOLUTE veto: position/gross/daily-loss/VaR caps), PaperBroker (refuses unapproved orders; slippage+fees; persisted)
- data/providers.py — DataProvider ABC; LSEProvider (probe-then-lock endpoints — their docs are JS-rendered, NEVER hardcode guessed URLs); YahooProvider fallback; FakeProvider for tests; CompositeProvider chain; PollingFeed (background thread → tick events)
- ai/orchestrator.py — TOOL_SCHEMAS (the ONLY machine surface the AI may use); RuleOrchestrator v1 (deterministic, reasoning strings, risk-reviewed); LLMOrchestrator socket (refuses without ANTHROPIC_API_KEY — no fake AI, ever)
- tests/test_core.py — run `python tests/test_core.py` before EVERY commit; extend it with every new module (currently 59 checks). NOTE: on this machine `python3` resolves to the Windows Store stub and hangs — use `python`.

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

## PROGRESS (updated after every phase)
- DONE: v0.3 baseline, P1 (LSE quote fix + AUM/position-size risk controls), P2 (HMM regime, Kalman pairs, GARCH, Ledoit-Wolf), P3 (3D vol surface + surface_interpreter.py), P4 (Finnhub news/sentiment, verified LSE macro + options-flow endpoints, curated anomaly library), P5 (sector/target engine). All pushed to main except P5 (awaiting push confirmation).
- IN PROGRESS: none — P5 code complete, tests passing (59/59), about to commit/push.
- NEXT: P6 (Flow Confluence Engine — orderflow, optionflow, flow_confluence + FLOW panel). Final phase of the current roadmap.

## ROADMAP (build in order; owner picks pace)
DONE (v0.3): QuantSignal engines live in quant/ (signals, bxtrender, backtest v2, verdict, playbook, scanner, risk, validation, montecarlo, levels, advanced). Playbook+verdict drive the policy; correlation_heat+VaR guard the book each cycle.

P1 — DONE (pushed): LSEProvider.get_quote now compares live price to the previous DAILY close, not the previous 1-minute bar. Sidebar AUM (Total Portfolio Capital) + Max Position Size (fixed $ or % of AUM) inputs, written into a per-cycle Config via dataclasses.replace on RiskEngine.cfg, enforced as a new veto check. Per-position "% of AUM" column in the Open Book view (TRADES tab).

P2 — DONE (pushed): quant/hmm_regime.py (hand-rolled Gaussian HMM, Baum-Welch EM, no new dep), quant/kalman_pairs.py (hand-rolled Kalman dynamic hedge ratio + spread z-score, no new dep), quant/garch.py (GARCH(1,1) via the `arch` package — owner approved), quant/covariance.py (Ledoit-Wolf shrinkage + min-variance weights via scikit-learn — owner approved; hand-rolling the shrinkage-intensity formula was judged too risky to get right from memory). All four are standalone pure functions, not yet wired into the UI/state/orchestrator — that happens as later phases consume them (P5 sector engine is the likely first consumer). requirements.txt now includes arch>=6.3 and scikit-learn>=1.3.

P3 — DONE (pushed): quant/vol_surface.py (normalizes an options chain — tolerates a few strike/iv/dte-or-expiry column-naming conventions since the exact LSE shape isn't hardcoded-guessed — into a strike x DTE x IV grid) + Plotly Surface chart in the LAB tab ("Build vol surface" button, needs LSE_API_KEY). quant/surface_interpreter.py — rule-based, deterministic plain-text reads (skew in vol points via 25-delta or a strike proxy, term-structure inversion via near-vs-far ATM IV, single-strike smile anomalies vs a local rolling median). Wired into RuleOrchestrator.ingest_chain: findings land in state.options.{symbol}.surface, get their own "VOL SURFACE" audit record, and flow into AI context automatically (options is already a curated AI_KEYS key). No LLM calls anywhere in this phase.

P4 — DONE: data/news.py (Finnhub company-news + news-sentiment, clean empty stub with no NEWS_API_KEY — these are standard public Finnhub endpoints, not a probe-then-lock situation). LSEProvider gained macro_series() (GET /series — rates/CPI/bond yields, e.g. "cpi_yoy"/"fdtr"/"US10Y"), economic_calendar() (GET /ref/economic_calendar), and options_flow() (GET /options/flow — REAL trade prints with premium/IV/greeks, verified to exist, not a proxy). All three verified 2026-07-12 by `pip install lse-data==0.14.0` and reading the actual installed lse/client.py + lse/vault.py source — WebFetch on their GitHub gave three mutually contradictory endpoint lists across three separate fetches and was not trustworthy enough to hardcode from. quant/anomaly_library.py: 9 curated, real-citation anomalies (momentum, short-term reversal, PEAD, low-vol, size, value, January effect, turn-of-month, disposition effect) with pure-function trigger conditions — the full list is reference material for the future LLMOrchestrator's system prompt; match_anomalies() also wires the subset that matches TODAY's numbers into RuleOrchestrator.research() -> state.research.{symbol}.anomalies + an "ANOMALY MATCH" audit record now. RuleOrchestrator gained scan_news() (-> state.news, audit, news.interrupt event on strong sentiment), scan_macro() (-> state.macro, audit), scan_flow() (-> state.flow_alerts, audit, flow.interrupt event on large prints — a simple threshold alert, distinct from the fuller statistical engine P6b builds on the same feed). state.macro added to GlobalState.AI_KEYS. Sidebar gained "News + sentiment pass" and "Macro + flow pass" toggles (gated on the relevant key), wired into the RUN DECISION CYCLE button. METRICS tab shows Macro/News/Flow Alerts sections. No new dependencies.

P5 — DONE: quant/sector_engine.py — score_name() takes quant.verdict's already-computed technical conviction (momentum/agreement/edge/regime/RR) and tilts it with news sentiment, recent large option prints, and a simple macro rate-trend heuristic (hikes = headwind, cuts = tailwind for risk assets — a well-known simple rule, not a precise model); every tilt is a labeled reason, never silent. rank_sectors_and_names() ranks sectors by average tradeable-name score, then names within; NO TRADE verdicts and tilt-conflicted names (tilts fighting the technical call) go to an explicit avoid list with reasoning instead of the ranking. avoid_above/avoid_below per name reuse quant.advanced.support_resistance(). RuleOrchestrator.sector_scan() wires it all: sector comes from LSE company_profiles() (verified endpoint, GET /ref/company_profiles) when the key is set else "Unclassified" (never guessed); sentiment/flow/macro tilts are read from whatever scan_news/scan_flow/scan_macro already cached in state (sector_scan does not re-fetch those itself). Output goes to state.sector_scan (added to AI_KEYS), a SECTOR SCAN audit record, and a LAB tab "Scan sectors & targets" button (ranked sectors table, ranked names table with entry/stop/target/rr/avoid-zones/why, avoid table). Deliberately read-only/suggestion-only — nothing here places an order; acting on a suggestion still goes through the existing RUN DECISION CYCLE's propose → RiskEngine veto → PaperBroker → AuditLog chain, unchanged. No new dependencies.

P6 — Flow Confluence Engine:
- P6a: quant/orderflow.py — Bulk Volume Classification (Easley/Lopez de Prado/O'Hara), CVD with price-divergence flag, VPIN toxicity percentile vs symbol history, volume-profile top-3 nodes. Pure functions, 3+ tests on synthetic bars.
- P6b: quant/optionflow.py — verify LSE flow endpoint from their SDK first, no guessed URLs; if none exists, use chain-delta proxies and document that honestly. Call/put premium share, volume/OI spike z-scores vs 20-day norms, largest prints.
- P6c: quant/flow_confluence.py — one score per symbol combining tape pressure + options positioning → CONFLUENCE LONG/SHORT/CONFLICT/QUIET with numbers. Wired into state.flow, audit, AI context, P5 scoring. VPIN >85th percentile adds an informational caution flag to RiskEngine reasoning only (NOT a hard veto — ask owner before changing that).
- UI: FLOW panel in LAB tab — CVD chart over price, VPIN gauge, options premium imbalance bar, confluence verdict with reasoning line.

Later: 7. WebSocket StreamingFeed — URI verified to exist (wss://data-ws.londonstrategicedge.com, parked as LSEProvider.WS_URL_ROADMAP). Build ONLY with owner sign-off after polling proves stable on the hosting; PollingFeed remains primary. Hetzner deploy guide + systemd unit; scheduler for autonomous cycles.
