"""Regime-conditional behavior — the P2 HMM regime detector gates
trading policy, not just informs it.

Bull  (bullish-drift HMM state): trend strategies active, full sizing.
Bear  (bearish-drift HMM state, not the highest-vol state): only
      dip-buys at extremes allowed, half sizing.
Storm (whichever HMM state has the highest fitted volatility,
      regardless of its mean — a crash can happen on the way down or as
      a blow-off top): NO NEW TRADES, existing stops tighten, an alert
      fires.

Built entirely on top of quant.hmm_regime.fit_hmm() — reuses its exact
fitted numbers (state means/vols/current probabilities), never refits
or reinterprets the underlying EM result.
"""
from __future__ import annotations

import pandas as pd

from .hmm_regime import fit_hmm

REGIME_POLICY = {
    "Bull": {"size_multiplier": 1.0, "dip_only": False,
             "new_trades_allowed": True, "tighten_stops": False},
    "Bear": {"size_multiplier": 0.5, "dip_only": True,
             "new_trades_allowed": True, "tighten_stops": False},
    "Storm": {"size_multiplier": 0.0, "dip_only": False,
              "new_trades_allowed": False, "tighten_stops": True},
}


def classify_regime(returns: pd.Series, n_states: int = 3) -> dict:
    """Fit the P2 HMM (3 states) and map them onto Bull/Bear/Storm."""
    hmm = fit_hmm(returns, n_states=n_states)
    if "error" in hmm:
        return {"error": hmm["error"], "regime": "Bull",
               "policy": REGIME_POLICY["Bull"]}

    vols, means = hmm["state_vols_pct"], hmm["state_means_pct"]
    storm_idx = int(max(range(len(vols)), key=lambda i: vols[i]))
    labels = ["Storm" if i == storm_idx else
             ("Bull" if means[i] > 0 else "Bear") for i in range(len(vols))]

    prob_values = list(hmm["current_state_probs"].values())
    current_idx = int(max(range(len(prob_values)), key=lambda i: prob_values[i]))
    regime = labels[current_idx]
    return {"regime": regime, "policy": REGIME_POLICY[regime],
           "state_labels": labels, "hmm": hmm}
