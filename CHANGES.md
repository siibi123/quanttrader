# QuantTrader — desk pack (2026-08-31)

Not a hedge fund. Yahoo is delayed, Streamlit Cloud sleeps, there is no
prime broker. These changes make the *desk behave* better on free data.

## Fix: Open / Close desk

The previous versions tried to hide Streamlit's sidebar with CSS/JS.
Streamlit redraws it. Result: two green buttons, Close did nothing.

Now: **Close desk does not render `st.sidebar` at all.** Streamlit then
has no left rail. **Open desk** is a real button on the main page that
puts the rail back. Values stay in `session_state`.

Upload `app.py`. Reboot Cloud. Not just refresh.

## New: free tape (`quant/macro_tape.py`)

Every decision cycle reads Yahoo `^VIX`, `^TNX`, `SPY` (5 min cache).

- VIX ≥ 22 and rising **and** (yields +20bp / 5d **or** SPY ≤ −2.5% / 5d)
  → **no new entries**. Exits still run.
- VIX rising or yields jumping alone → **half size**.
- Shown on RESEARCH as "Free tape".

## New: session gate (`quant/session_gate.py`)

No new entries in the first 15 minutes or last 10 minutes of RTH, or
when the session is not open. Daily Yahoo bars cannot time 09:31 — this
only gates the *live* cycle.

## Data hygiene

`filter_price_outliers` now also drops a close-to-close print that is
>5σ of the last 20 returns (Hampel). Vendor spikes stop looking like
signals.

## What this is not

- Not 24/7. Cloud sleeps.
- Not live IBKR.
- Not unique alpha. Public rules + vetoes.
- Paid tape (options flow, NBBO) is still optional LSE/Finnhub keys.

## Files to upload

```
app.py
ai/orchestrator.py
data/providers.py
quant/macro_tape.py      (new)
quant/session_gate.py    (new)
tests/test_core.py
```
