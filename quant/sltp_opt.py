"""SL/TP Optimizer — grid-search stop/target combos over the composite
model's actual BUY signals, replayed bar-by-bar (TP-first win, SL-first
loss, time exit). Modes: ATR multiples or % of entry. Ranked by Sharpe /
PF / Win% / Return / MinDD / Expectancy / R:R."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .signals import atr, composite


def optimize_sltp(df: pd.DataFrame, sl_grid=None, tp_grid=None,
                  mode: str = "atr", hold: int = 20, capital: float = 10_000,
                  risk_pct: float = 1.0, rank_by: str = "sharpe",
                  max_signals: int = 120) -> pd.DataFrame:
    sl_grid = sl_grid or [1.0, 1.5, 2.0, 2.5, 3.0]
    tp_grid = tp_grid or [1.5, 2.0, 3.0, 4.0, 5.0]
    sig = composite(df)["signal"].shift(1) == "BUY"
    a = atr(df)
    idx = np.where(sig.values)[0]
    idx = idx[(idx > 20) & (idx < len(df) - 2)][-max_signals:]
    o, h, l = df["Open"].values, df["High"].values, df["Low"].values
    rows = []
    for slm in sl_grid:
        for tpm in tp_grid:
            pnl = []
            for i in idx:
                e = o[i]
                unit = a.values[i - 1] if mode == "atr" else e / 100
                sl, tp = e - slm * unit, e + tpm * unit
                res = None
                for j in range(i, min(i + hold, len(df))):
                    if l[j] <= sl:
                        res = -(e - sl) / e
                        break
                    if h[j] >= tp:
                        res = (tp - e) / e
                        break
                if res is None:
                    j2 = min(i + hold, len(df) - 1)
                    res = (df["Close"].values[j2] - e) / e
                shares = (capital * risk_pct / 100) / max(slm * unit, 1e-9)
                pnl.append(res * e * shares)
            p = np.array(pnl)
            if not len(p):
                continue
            wins, losses = p[p > 0], p[p <= 0]
            eqc = capital + np.cumsum(p)
            ddm = float(((np.maximum.accumulate(eqc) - eqc)
                         / np.maximum.accumulate(eqc)).max() * 100)
            rows.append({
                "SL": slm, "TP": tpm, "trades": len(p),
                "win_pct": round(len(wins) / len(p) * 100, 1),
                "PF": round(wins.sum() / abs(losses.sum()), 2)
                if losses.sum() < 0 else np.inf,
                "return_pct": round(p.sum() / capital * 100, 2),
                "sharpe": round(float(p.mean() / p.std()) * np.sqrt(252 / 5), 2)
                if p.std() > 0 else 0.0,
                "max_dd_pct": round(ddm, 2),
                "expect_$": round(float(p.mean()), 1),
                "rr": round(tpm / slm, 2)})
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    asc = rank_by in ("max_dd_pct",)
    key = {"sharpe": "sharpe", "pf": "PF", "win": "win_pct",
           "return": "return_pct", "min_dd": "max_dd_pct",
           "expect": "expect_$", "rr": "rr"}.get(rank_by, "sharpe")
    return out.sort_values(key, ascending=asc).reset_index(drop=True)
