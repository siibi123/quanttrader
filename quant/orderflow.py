"""Order flow — Bulk Volume Classification, CVD, VPIN, volume profile.

Built entirely from OHLCV bars (this platform has no tick data), using
Bulk Volume Classification from Easley, Lopez de Prado & O'Hara, "The
Volume Clock: Insights into the High-Frequency Paradigm" (Journal of
Portfolio Management, 2012): each bar's volume is split into buy/sell
fractions via the standardized price change through the normal CDF,
rather than classified trade-by-trade. VPIN here is the standard
BVC-based approximation over a trailing window of TIME bars — not the
textbook equal-volume-bucket construction, which needs tick data this
platform doesn't have. Documented as an approximation, not the exact
paper construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def bulk_volume_classification(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Per-bar buy/sell volume split via BVC.

    z_t = (price change) / (rolling std of price changes); buy fraction =
    Phi(z_t) (standard normal CDF), sell fraction = 1 - Phi(z_t).
    """
    dp = df["Close"].diff()
    sigma = dp.rolling(window).std()
    z = (dp / sigma.replace(0, np.nan)).fillna(0)
    buy_frac = stats.norm.cdf(z)
    out = df[["Close", "Volume"]].copy()
    out["buy_volume"] = out["Volume"] * buy_frac
    out["sell_volume"] = out["Volume"] - out["buy_volume"]
    out["imbalance"] = out["buy_volume"] - out["sell_volume"]
    return out


def cvd(df: pd.DataFrame, window: int = 20, divergence_lookback: int = 20) -> dict:
    """Cumulative Volume Delta + a price/CVD divergence flag."""
    bvc = bulk_volume_classification(df, window=window)
    cvd_series = bvc["imbalance"].cumsum()
    look = min(divergence_lookback, len(df) - 1)
    if look < 1:
        return {"error": "not enough bars for a divergence read"}
    price_chg = float(df["Close"].iloc[-1] - df["Close"].iloc[-look])
    cvd_chg = float(cvd_series.iloc[-1] - cvd_series.iloc[-look])
    divergence = None
    if price_chg > 0 and cvd_chg < 0:
        divergence = "bearish"          # price up, selling pressure underneath
    elif price_chg < 0 and cvd_chg > 0:
        divergence = "bullish"          # price down, buying pressure underneath
    return {
        "cvd_series": cvd_series,
        "cvd_latest": round(float(cvd_series.iloc[-1]), 0),
        "cvd_chg": round(cvd_chg, 0),
        "price_chg": round(price_chg, 4),
        "divergence": divergence,
    }


def vpin(df: pd.DataFrame, window: int = 20, history_window: int = 100) -> dict:
    """BVC-approximated VPIN (toxicity) + its percentile vs this symbol's
    own recent history."""
    bvc = bulk_volume_classification(df, window=window)
    imbalance_abs = bvc["imbalance"].abs()
    vpin_series = (imbalance_abs.rolling(window).sum()
                  / bvc["Volume"].rolling(window).sum().replace(0, np.nan))
    vpin_series = vpin_series.dropna()
    if len(vpin_series) < 10:
        return {"error": "need more history for a stable VPIN read"}
    current = float(vpin_series.iloc[-1])
    hist = vpin_series.iloc[-history_window:]
    pctile = float(stats.percentileofscore(hist, current))
    return {"vpin": round(current, 4), "vpin_series": vpin_series,
           "percentile": round(pctile, 1), "toxic": pctile >= 85}


def volume_profile(df: pd.DataFrame, lookback: int = 120, n_bins: int = 24,
                   top_n: int = 3) -> list[dict]:
    """Top-N highest-volume price nodes over the lookback window."""
    win = df.iloc[-lookback:]
    if len(win) < 10:
        return []
    lo, hi = float(win["Low"].min()), float(win["High"].max())
    if hi <= lo:
        return []
    bins = np.linspace(lo, hi, n_bins + 1)
    mid = (bins[:-1] + bins[1:]) / 2
    vol_per_bin = np.zeros(n_bins)
    typical = (win["High"] + win["Low"] + win["Close"]) / 3
    idx = np.clip(np.digitize(typical.values, bins) - 1, 0, n_bins - 1)
    for i, v in zip(idx, win["Volume"].values):
        vol_per_bin[i] += v
    order = np.argsort(-vol_per_bin)[:top_n]
    total = vol_per_bin.sum()
    return [{"price": round(float(mid[i]), 2),
            "volume_pct": round(float(vol_per_bin[i] / total) * 100, 1)
                         if total > 0 else 0.0}
           for i in order]
