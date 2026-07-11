"""Validation lab — the honesty engine a real desk demands.

Four tests that separate real edges from data-mined noise:

  1. DEFLATED SHARPE (Bailey & López de Prado 2014) — corrects a strategy's
     Sharpe for how many strategies were TRIED. Testing 20 models and keeping
     the best inflates Sharpe; this deflates it back to reality.

  2. MULTIPLE-TESTING p-value (Harvey, Liu & Zhu 2016, RFS) — a t-stat of 2
     is NOT significant when you mined 20 signals. Applies a Bonferroni-style
     haircut so you trust what survives.

  3. MONTE CARLO PERMUTATION — shuffle the returns 500× and ask: could this
     equity curve have happened by luck? Gives an empirical p-value on skill.

  4. BOOTSTRAP CONFIDENCE INTERVAL — resample trades 1000× for a 90% CI on
     CAGR. A wide band that straddles zero = you don't actually know if it works.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int,
                    skew: float = 0.0, kurt: float = 3.0) -> dict:
    """Bailey & López de Prado deflated Sharpe ratio."""
    if n_obs < 20:
        return {"error": "Too few observations."}
    # Expected max Sharpe from N independent random trials (order statistic)
    emc = 0.5772156649
    e_max = (np.sqrt(2 * np.log(max(n_trials, 2)))
             - (np.log(np.log(max(n_trials, 2))) + np.log(4 * np.pi))
             / (2 * np.sqrt(2 * np.log(max(n_trials, 2)))))
    # Work in per-period Sharpe (deannualize), as the theory requires.
    sr_p = sharpe / np.sqrt(252)
    sr_std = np.sqrt((1 - skew * sr_p + (kurt - 1) / 4 * sr_p ** 2) / (n_obs - 1))
    sr0 = e_max * sr_std                              # deflated benchmark (per-period)
    dsr = stats.norm.cdf((sr_p - sr0) / sr_std) if sr_std > 0 else 0.0
    return {
        "observed_sharpe": round(float(sharpe), 2),
        "deflated_benchmark_ann": round(float(sr0 * np.sqrt(252)), 2),
        "DSR_probability": round(float(dsr), 3),
        "verdict": ("✅ Likely real edge" if dsr > 0.95 else
                    "⚠️ Borderline" if dsr > 0.75 else
                    "❌ Probably data-mined noise"),
    }


def haircut_pvalue(tstat: float, n_tests: int) -> dict:
    """Harvey-Liu-Zhu style multiple-testing haircut (Bonferroni + BY)."""
    single_p = 2 * (1 - stats.norm.cdf(abs(tstat)))
    bonferroni = min(single_p * n_tests, 1.0)
    # Benjamini-Yekutieli constant
    c = sum(1.0 / i for i in range(1, n_tests + 1))
    by = min(single_p * n_tests * c / 1.0, 1.0)
    return {
        "raw_p": round(float(single_p), 4),
        "bonferroni_p": round(float(bonferroni), 4),
        "BY_p": round(float(by), 4),
        "survives_5pct": bonferroni < 0.05,
        "verdict": ("✅ Survives multiple-testing" if bonferroni < 0.05 else
                    "⚠️ Marginal" if bonferroni < 0.20 else
                    "❌ Not significant after correction"),
    }


def permutation_test(returns: pd.Series, n_perm: int = 500,
                     seed: int = 7) -> dict:
    """Could this equity curve be luck? Shuffle returns, compare final wealth."""
    r = returns.dropna().values
    if len(r) < 30:
        return {"error": "Too few returns."}
    def _sharpe(x):
        return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0

    # Sign-flip permutation: under the null of "no edge", each return's sign
    # is equally likely +/-. This tests whether the DRIFT is real, and unlike
    # reshuffling it is not invariant to a positive mean.
    actual = float(_sharpe(r))
    rng = np.random.default_rng(seed)
    perms = np.array([_sharpe(r * rng.choice([-1, 1], size=len(r)))
                      for _ in range(n_perm)])
    pval = float((perms >= actual).mean())
    return {
        "actual_sharpe": round(actual, 2),
        "perm_sharpe_95pct": round(float(np.percentile(perms, 95)), 2),
        "perm_p_value": round(pval, 3),
        "verdict": ("✅ Beats luck (p<0.05)" if pval < 0.05 else
                    "⚠️ Weak (p<0.20)" if pval < 0.20 else
                    "❌ Indistinguishable from luck"),
    }


def bootstrap_cagr(trade_pnls: pd.Series, starting: float = 5000.0,
                   n_boot: int = 1000, seed: int = 7) -> dict:
    """90% confidence interval on total return by resampling trades."""
    p = trade_pnls.dropna().values
    if len(p) < 8:
        return {"error": "Need at least 8 closed trades for a bootstrap."}
    rng = np.random.default_rng(seed)
    finals = []
    for _ in range(n_boot):
        s = rng.choice(p, size=len(p), replace=True)
        finals.append((starting + s.sum()) / starting - 1)
    lo, med, hi = np.percentile(finals, [5, 50, 95])
    return {
        "median_return_%": round(float(med) * 100, 1),
        "CI90_low_%": round(float(lo) * 100, 1),
        "CI90_high_%": round(float(hi) * 100, 1),
        "excludes_zero": lo > 0,
        "verdict": ("✅ Profitable even at the 5th percentile" if lo > 0 else
                    "⚠️ CI straddles zero — edge unproven" if hi > 0 else
                    "❌ Likely unprofitable"),
    }
