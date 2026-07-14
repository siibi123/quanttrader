"""Flow Confluence Engine — one score per symbol combining tape pressure
(quant.orderflow: CVD trend + divergence + VPIN) and options positioning
(quant.optionflow: premium share + flow spike) into a single read:
CONFLUENCE LONG, CONFLUENCE SHORT, CONFLICT, or QUIET — always with the
numbers that produced it, never a bare label. VPIN toxicity is
informational only here (a caution flag), never a directional input.
"""
from __future__ import annotations

import pandas as pd

from .orderflow import cvd as tape_cvd
from .orderflow import vpin as tape_vpin
from .optionflow import flow_spike, premium_share

DIR_THRESHOLD = 0.1        # |score| below this counts as "no direction"


def tape_pressure(df: pd.DataFrame) -> dict:
    """-1..+1 tape-pressure score from CVD trend + divergence."""
    c = tape_cvd(df)
    if "error" in c:
        return {"score": 0.0, "reasons": ["insufficient bars for CVD"],
               "vpin_percentile": None, "toxic": False}
    v = tape_vpin(df)
    look_bars = min(20, len(df) - 1)
    total_vol = float(df["Volume"].iloc[-look_bars:].sum())
    sign = 1.0 if c["cvd_chg"] > 0 else (-1.0 if c["cvd_chg"] < 0 else 0.0)
    magnitude = min(abs(c["cvd_chg"]) / (total_vol + 1e-9) * 10, 1.0)
    score = sign * magnitude
    reasons = [f"CVD {c['cvd_chg']:+,.0f} over the last {look_bars} bars"]
    if c["divergence"]:
        score *= 0.3                            # divergence undercuts the read
        reasons.append(f"{c['divergence']} price/CVD divergence — tempering the signal")
    toxic = bool(v.get("toxic"))
    if toxic:
        reasons.append(f"VPIN toxicity {v.get('percentile')}pct — elevated "
                       f"informed-trading risk")
    return {"score": round(float(score), 3), "reasons": reasons,
           "vpin_percentile": v.get("percentile"), "toxic": toxic}


def options_positioning(flow_today: pd.DataFrame,
                        flow_history: list[pd.DataFrame] | None = None) -> dict:
    """-1..+1 options-positioning score from call/put premium share, tilted
    by a volume/premium spike z-score when history is available."""
    ps = premium_share(flow_today)
    if "error" in ps:
        return {"score": 0.0, "reasons": [ps["error"]]}
    tilt = (ps["call_share_pct"] - 50) / 50               # -1..+1
    reasons = [f"{ps['call_share_pct']}% of premium in calls "
              f"({ps['put_share_pct']}% puts)"]
    score = tilt
    if flow_history:
        fs = flow_spike(flow_today, flow_history)
        if "error" not in fs:
            z = fs.get("premium_z", fs.get("volume_z", 0))
            boost = max(min(z / 10, 0.5), -0.5)
            direction = 1 if tilt >= 0 else -1
            score = max(min(score + boost * direction, 1.0), -1.0)
            reasons.append(f"flow spike z={z:+.1f} vs {fs['n_history_days']}d norm")
    return {"score": round(float(score), 3), "reasons": reasons}


def confluence(df: pd.DataFrame, flow_today: pd.DataFrame | None = None,
              flow_history: list[pd.DataFrame] | None = None) -> dict:
    """One CONFLUENCE read per symbol. QUIET if both reads are weak,
    CONFLICT if they meaningfully disagree, else LONG/SHORT."""
    tape = tape_pressure(df)
    opts = (options_positioning(flow_today, flow_history)
           if flow_today is not None and len(flow_today)
           else {"score": 0.0, "reasons": ["no options flow available"]})

    t, o = tape["score"], opts["score"]
    tape_dir = 1 if t > DIR_THRESHOLD else (-1 if t < -DIR_THRESHOLD else 0)
    opt_dir = 1 if o > DIR_THRESHOLD else (-1 if o < -DIR_THRESHOLD else 0)

    if tape_dir != 0 and opt_dir != 0 and tape_dir != opt_dir:
        verdict = "CONFLICT"
    elif tape_dir == 1 or opt_dir == 1:
        verdict = "CONFLUENCE LONG" if tape_dir >= 0 and opt_dir >= 0 else "CONFLICT"
    elif tape_dir == -1 or opt_dir == -1:
        verdict = "CONFLUENCE SHORT" if tape_dir <= 0 and opt_dir <= 0 else "CONFLICT"
    else:
        verdict = "QUIET"

    return {"verdict": verdict, "tape_score": t, "options_score": o,
           "combined_score": round((t + o) / 2, 3),
           "tape_reasons": tape["reasons"], "options_reasons": opts["reasons"],
           "vpin_percentile": tape.get("vpin_percentile"),
           "toxic_caution": tape.get("toxic", False)}
