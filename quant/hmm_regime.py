"""Gaussian Hidden Markov Model — regime detection via Baum-Welch EM.

Hand-rolled (no hmmlearn dependency): forward-backward E-step in log-space,
closed-form Gaussian M-step. 2-3 states covers what a discretionary desk
actually reasons about: "calm/trending" vs "choppy/crisis" (2-state), or
add a mid regime (3-state). States are always returned sorted by mean
return so state 0 is the most bearish and the last state the most bullish.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp


def _log_gauss(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    var = max(var, 1e-12)
    return -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)


def _forward_backward(logB: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray):
    n, k = logB.shape
    log_alpha = np.zeros((n, k))
    log_alpha[0] = log_pi + logB[0]
    for t in range(1, n):
        log_alpha[t] = logB[t] + logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)
    log_beta = np.zeros((n, k))
    for t in range(n - 2, -1, -1):
        log_beta[t] = logsumexp(log_A + logB[t + 1] + log_beta[t + 1], axis=1)
    loglik = float(logsumexp(log_alpha[-1]))
    log_gamma = log_alpha + log_beta - loglik
    gamma = np.exp(log_gamma)
    log_xi = (log_alpha[:-1, :, None] + log_A[None, :, :] +
              logB[1:, None, :] + log_beta[1:, None, :] - loglik)
    xi = np.exp(log_xi)
    return gamma, xi, loglik


def fit_hmm(returns: pd.Series, n_states: int = 2, n_iter: int = 100,
           tol: float = 1e-6, seed: int = 7) -> dict:
    """Fit a Gaussian HMM to a returns series via Baum-Welch EM.

    Returns each state's mean/vol, the transition matrix, TODAY's filtered
    state probabilities, and a human regime label.
    """
    x = returns.dropna().values.astype(float)
    n = len(x)
    min_n = max(30, 10 * n_states)
    if n < min_n:
        return {"error": f"need >= {min_n} observations"}

    q = np.quantile(x, np.linspace(0, 1, n_states + 1))
    means = np.array([x[(x >= q[i]) & (x <= q[i + 1])].mean()
                      if len(x[(x >= q[i]) & (x <= q[i + 1])]) else x.mean()
                      for i in range(n_states)])
    vars_ = np.full(n_states, x.var() + 1e-8)
    pi = np.full(n_states, 1.0 / n_states)
    A = np.full((n_states, n_states), 0.05 / max(n_states - 1, 1))
    np.fill_diagonal(A, 0.95)

    prev_ll = -np.inf
    gamma = None
    for _ in range(n_iter):
        logB = np.column_stack([_log_gauss(x, means[k], vars_[k])
                                for k in range(n_states)])
        with np.errstate(divide="ignore"):
            log_pi, log_A = np.log(pi), np.log(A)
        gamma, xi, ll = _forward_backward(logB, log_pi, log_A)
        pi = gamma[0] / gamma[0].sum()
        A = xi.sum(axis=0)
        A = A / A.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)
        means = (gamma * x[:, None]).sum(axis=0) / w
        vars_ = (gamma * (x[:, None] - means) ** 2).sum(axis=0) / w
        vars_ = np.maximum(vars_, 1e-10)
        if abs(ll - prev_ll) < tol:
            prev_ll = ll
            break
        prev_ll = ll

    order = np.argsort(means)
    means, vars_ = means[order], vars_[order]
    A = A[order][:, order]
    gamma = gamma[:, order]

    labels = (["Bear/Panic", "Bull/Calm"] if n_states == 2 else
             ["Bear/Panic", "Choppy/Neutral", "Bull/Calm"] if n_states == 3 else
             [f"State {i}" for i in range(n_states)])
    current = gamma[-1]
    return {
        "n_states": n_states,
        "state_means_pct": [round(float(m) * 100, 3) for m in means],
        "state_vols_pct": [round(float(np.sqrt(v)) * 100, 3) for v in vars_],
        "transition_matrix": [[round(float(v), 3) for v in row] for row in A],
        "current_state_probs": {labels[i]: round(float(current[i]) * 100, 1)
                                for i in range(n_states)},
        "most_likely_regime": labels[int(np.argmax(current))],
        "loglik": round(prev_ll, 2),
    }
