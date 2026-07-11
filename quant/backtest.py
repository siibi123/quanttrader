"""Backtester v2 — dual-strategy engine with institutional risk mechanics.

Strategies:
  TREND — composite-signal following. Entries gated by the 200-day SMA
          (Faber 2007). Chandelier 2.5×ATR trail, breakeven after +1R,
          time-stop on stalled trades.
  DIP   — Connors-style RSI(2) pullback buyer: short-term panic INSIDE an
          uptrend. High win rate, small wins, strict time exit.
  AUTO  — picks per ticker by Hurst exponent (trending vs mean-reverting).

Risk mechanics (applied to both):
  * next-bar-open execution (no look-ahead), commission per side
  * volatility-targeted sizing (Moreira & Muir 2017): risk scales down
    when ATR% is elevated vs its own history
  * breakeven stop once the trade is +1R
  * time stop: unprofitable after N bars -> out
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bxtrender import bxtrender
from .signals import atr, composite, rsi, sma


@dataclass
class BTConfig:
    starting_cash: float = 5000.0
    commission_pct: float = 0.001
    atr_stop_mult: float = 2.5
    risk_per_trade: float = 0.01
    mode: str = "auto"              # "auto" | "trend" | "dip" | "core" | "blend"
    allow_short: bool = False       # trend mode: short below SMA200
    fib_filter: bool = True         # dip mode: only buy inside 0.382-0.786 zone
    breakeven_r: float = 1.0        # move stop to entry after +1R
    time_stop_trend: int = 20       # bars; exit if unprofitable by then
    time_stop_dip: int = 10
    regime_filter: bool = True      # longs only above SMA200
    vol_target: bool = True         # inverse-vol position scaling
    bars_per_year: int = 252        # 1638 hourly, 52 weekly, 12 monthly


@dataclass
class BTResult:
    equity: pd.Series
    bh_equity: pd.Series
    trades: pd.DataFrame
    metrics: dict
    mode_used: str = "trend"


def _hurst_quick(close: pd.Series, max_lag: int = 80) -> float:
    p = np.log(close.dropna().values)
    if len(p) < max_lag * 2:
        max_lag = max(20, len(p) // 4)
    lags = range(2, max_lag)
    tau = np.maximum([np.std(p[l:] - p[:-l]) for l in lags], 1e-12)
    return float(np.clip(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0],
                         0.0, 1.0))


def _metrics(equity: pd.Series, trades: pd.DataFrame, bh: pd.Series,
             bpy: int = 252) -> dict:
    rets = equity.pct_change().dropna()
    n_years = max(len(equity) / bpy, 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
    sharpe = (rets.mean() / rets.std() * np.sqrt(bpy)) if rets.std() > 0 else 0.0
    downside = rets[rets < 0].std()
    sortino = (rets.mean() / downside * np.sqrt(bpy)) if downside and downside > 0 else 0.0
    dd = (equity / equity.cummax() - 1).min()
    wins = (trades["pnl"] > 0).sum() if len(trades) else 0
    scr = (trades["pnl"].abs() < trades["pnl"].abs().mean() * 0.1).sum() if len(trades) else 0
    bh_cagr = (bh.iloc[-1] / bh.iloc[0]) ** (1 / n_years) - 1
    pf = None
    if len(trades):
        gross_w = trades.loc[trades["pnl"] > 0, "pnl"].sum()
        gross_l = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
        pf = round(float(gross_w / gross_l), 2) if gross_l > 0 else None
    extra = {}
    if len(trades):
        w = trades[trades["pnl"] > 0]["pnl"]
        l = trades[trades["pnl"] < 0]["pnl"]
        extra["Avg win $"] = round(float(w.mean()), 0) if len(w) else 0
        extra["Avg loss $"] = round(float(l.mean()), 0) if len(l) else 0
        extra["Expectancy R"] = round(float(trades["R"].mean()), 2) \
            if "R" in trades else None
        extra["Avg hold (bars)"] = round(float(trades["bars"].mean()), 1) \
            if "bars" in trades else None
        signs = (trades["pnl"] > 0).astype(int).values
        mx_w = mx_l = cw = cl = 0
        for s_ in signs:
            cw = cw + 1 if s_ else 0
            cl = cl + 1 if not s_ else 0
            mx_w, mx_l = max(mx_w, cw), max(mx_l, cl)
        extra["Max consec W/L"] = f"{mx_w} / {mx_l}"
    return {
        "CAGR %": round(float(cagr) * 100, 1),
        "Buy&Hold CAGR %": round(float(bh_cagr) * 100, 1),
        "Sharpe": round(float(sharpe), 2),
        "Sortino": round(float(sortino), 2),
        "Max Drawdown %": round(float(dd) * 100, 1),
        "Trades": int(len(trades)),
        "Win Rate %": round(float(wins) / len(trades) * 100, 1) if len(trades) else 0.0,
        "Profit Factor": pf,
        "Final Equity $": round(float(equity.iloc[-1]), 0),
        "Buy&Hold Final $": round(float(bh.iloc[-1]), 0),
        **extra,
    }


def run_backtest(df: pd.DataFrame, cfg: BTConfig = BTConfig()) -> BTResult:
    mode = cfg.mode
    if mode == "auto":
        h = _hurst_quick(df["Close"])
        mode = "trend" if h >= 0.5 else "dip"

    comp = composite(df) if mode in ("trend", "blend") else None
    bx = bxtrender(df)
    a = atr(df)
    s200 = sma(df["Close"], 200)
    r2 = rsi(df["Close"], 2)

    # Fibonacci retracement zone of the rolling 126-bar swing (for dip mode):
    roll_hi = df["High"].rolling(126, min_periods=40).max()
    roll_lo = df["Low"].rolling(126, min_periods=40).min()
    rng_ = roll_hi - roll_lo
    fib_lo = roll_hi - 0.786 * rng_          # deep edge of the pocket
    fib_hi = roll_hi - 0.382 * rng_          # shallow edge

    atr_pct = (a / df["Close"])
    med = atr_pct.rolling(252, min_periods=60).median()
    vt = (med / atr_pct).clip(0.5, 1.5).fillna(1.0) if cfg.vol_target else \
        pd.Series(1.0, index=df.index)

    cash = cfg.starting_cash
    shares = 0.0                      # negative = short
    entry_price = stop = 0.0
    entry_i = 0
    be_armed = False
    entry_kind = ""                   # "trend" | "dip" (matters in blend)
    trade_lo = trade_hi = 0.0         # MAE/MFE tracking
    equity_rows, trade_rows = [], []

    o_, h_, l_, c_ = (df[k].values for k in ("Open", "High", "Low", "Close"))
    sig = comp["signal"].values if comp is not None else None
    bx_long = bx["long_osc"].values
    bx_rising = bx["t3_rising"].values
    bx_buyturn = bx["buy_turn"].values
    idx = df.index
    time_stop = cfg.time_stop_trend if mode == "trend" else cfg.time_stop_dip

    def close_pos(i, exit_px, reason):
        nonlocal cash, shares, be_armed
        exit_px = max(exit_px, 0.01)
        if shares > 0:
            proceeds = shares * exit_px * (1 - cfg.commission_pct)
            pnl = proceeds - shares * entry_price * (1 + cfg.commission_pct)
            cash += proceeds
        else:  # short cover
            qty = -shares
            cost = qty * exit_px * (1 + cfg.commission_pct)
            pnl = qty * entry_price * (1 - cfg.commission_pct) - cost
            cash += qty * entry_price * (1 - cfg.commission_pct) - cost + qty * entry_price * 0  # margin release handled via cash below
            cash = cash  # cash already holds short proceeds at entry
        r_unit_ = cfg.atr_stop_mult * a.values[max(entry_i - 1, 0)]
        mae_r = (entry_price - trade_lo) / r_unit_ if r_unit_ > 0 else 0
        mfe_r = (trade_hi - entry_price) / r_unit_ if r_unit_ > 0 else 0
        trade_rows.append({"entry_date": idx[entry_i], "exit_date": idx[i],
                           "side": "LONG" if shares > 0 else "SHORT",
                           "entry": round(entry_price, 2),
                           "exit": round(exit_px, 2),
                           "pnl": round(pnl, 2), "reason": reason,
                           "bars": int(i - entry_i),
                           "R": round(pnl / (abs(shares) * r_unit_), 2)
                           if r_unit_ > 0 and shares != 0 else 0.0,
                           "MAE_R": round(float(mae_r), 2),
                           "MFE_R": round(float(mfe_r), 2)})
        shares = 0.0
        be_armed = False

    for i in range(1, len(df)):
        o, hi, lo, c = o_[i], h_[i], l_[i], c_[i]
        prev_atr = a.values[i - 1]
        s200_ok = not np.isnan(s200.values[i - 1])
        above200 = s200_ok and c_[i - 1] > s200.values[i - 1]
        below200 = s200_ok and c_[i - 1] < s200.values[i - 1]

        # ================= CORE mode: improved buy & hold =================
        if mode == "core":
            in_market = shares > 0
            healthy = above200 and bx_long[i - 1] > 0
            if in_market and not healthy:
                close_pos(i, o, "regime exit")
            elif (not in_market) and healthy:
                shares = float((cash * 0.98) / (o * (1 + cfg.commission_pct)))
                if shares * o >= 100:
                    cash -= shares * o * (1 + cfg.commission_pct)
                    entry_price, entry_i = o, i
                else:
                    shares = 0.0
            equity_rows.append(cash + shares * c)
            continue

        # ================= exits (trend / dip) =================
        if shares != 0:
            trade_lo = min(trade_lo, lo)
            trade_hi = max(trade_hi, hi)
            bars_in = i - entry_i
            r_dist = cfg.atr_stop_mult * a.values[entry_i - 1]
            long_pos = shares > 0

            if long_pos:
                if not be_armed and hi >= entry_price + cfg.breakeven_r * r_dist:
                    stop = max(stop, entry_price); be_armed = True
                if mode == "trend" or (mode == "blend"
                                       and entry_kind == "trend"):
                    stop = max(stop, hi - cfg.atr_stop_mult * prev_atr)
                hit = lo <= stop
            else:
                if not be_armed and lo <= entry_price - cfg.breakeven_r * r_dist:
                    stop = min(stop, entry_price); be_armed = True
                if mode == "trend":
                    stop = min(stop, lo + cfg.atr_stop_mult * prev_atr)
                hit = hi >= stop

            exit_now, reason = False, ""
            if hit:
                exit_now, reason = True, "breakeven" if be_armed else "stop"
            elif mode in ("trend", "blend") and long_pos and sig[i - 1] == "SELL":
                exit_now, reason = True, "signal"
            elif mode == "trend" and not long_pos and sig[i - 1] == "BUY":
                exit_now, reason = True, "signal"
            elif long_pos and r2.values[i - 1] > 65 and (
                    mode == "dip" or
                    (mode == "blend" and entry_kind == "dip")):
                exit_now, reason = True, "target(rsi)"
            elif bars_in >= time_stop and (
                    (long_pos and c_[i - 1] < entry_price) or
                    (not long_pos and c_[i - 1] > entry_price)):
                exit_now, reason = True, "time"

            if exit_now:
                px = o
                if hit:
                    px = min(o, stop) if long_pos and o > stop else \
                         max(o, stop) if (not long_pos) and o < stop else o
                close_pos(i, px, reason)

        # ================= entries =================
        if shares == 0 and prev_atr > 0:
            stop_dist = cfg.atr_stop_mult * prev_atr
            risk_dollars = cash * cfg.risk_per_trade * vt.values[i - 1]
            size = min(risk_dollars / stop_dist,
                       cash / (o * (1 + cfg.commission_pct)))

            go_long = go_short = False
            if mode == "blend":
                trend_go = (above200 or not cfg.regime_filter) and \
                           sig[i - 1] == "BUY" and bx_long[i - 1] > 0 and \
                           bx_rising[i - 1]
                pocket_b = (not np.isnan(fib_lo.values[i - 1]) and
                            fib_lo.values[i - 1] <= c_[i - 1]
                            <= fib_hi.values[i - 1])
                dip_go = (above200 or not cfg.regime_filter) and \
                         r2.values[i - 1] < 10 and (pocket_b or bx_buyturn[i - 1])
                go_long = trend_go or dip_go
                entry_kind = "dip" if (dip_go and not trend_go) else "trend"
            elif mode == "trend":
                # B-Xtrender confirmation: long osc positive & T3 rising
                go_long = (above200 or not cfg.regime_filter) and \
                          sig[i - 1] == "BUY" and \
                          bx_long[i - 1] > 0 and bx_rising[i - 1]
                if cfg.allow_short:
                    go_short = below200 and sig[i - 1] == "SELL" and \
                               bx_long[i - 1] < 0 and not bx_rising[i - 1]
            else:  # dip
                in_pocket = True
                if cfg.fib_filter and not np.isnan(fib_lo.values[i - 1]):
                    in_pocket = fib_lo.values[i - 1] <= c_[i - 1] <= fib_hi.values[i - 1]
                go_long = (above200 or not cfg.regime_filter) and \
                          r2.values[i - 1] < 10 and \
                          (in_pocket or bx_buyturn[i - 1])

            if go_long and size * o >= 100:
                shares = float(size)
                cash -= shares * o * (1 + cfg.commission_pct)
                entry_price, entry_i = o, i
                stop = o - stop_dist
                be_armed = False
                trade_lo = trade_hi = o
            elif go_short and size * o >= 100:
                shares = -float(size)
                cash += size * o * (1 - cfg.commission_pct)   # short proceeds
                entry_price, entry_i = o, i
                stop = o + stop_dist
                be_armed = False
                trade_lo = trade_hi = o

        equity_rows.append(cash + shares * c)

    equity = pd.Series(equity_rows, index=idx[1:], name="strategy")
    bh = pd.Series(cfg.starting_cash / c_[0] * c_[1:], index=idx[1:],
                   name="buy_hold")
    trades = pd.DataFrame(trade_rows)
    return BTResult(equity, bh, trades, _metrics(equity, trades, bh, cfg.bars_per_year), mode)


def walk_forward(df: pd.DataFrame, cfg: BTConfig = BTConfig(),
                 n_folds: int = 4) -> pd.DataFrame:
    fold_len = len(df) // n_folds
    rows = []
    for k in range(n_folds):
        chunk = df.iloc[k * fold_len:(k + 1) * fold_len + 1]
        if len(chunk) < 120:
            continue
        res = run_backtest(chunk, cfg)
        row = {"fold": k + 1, "start": chunk.index[0].date(),
               "end": chunk.index[-1].date(), "mode": res.mode_used}
        row.update(res.metrics)
        rows.append(row)
    return pd.DataFrame(rows)
