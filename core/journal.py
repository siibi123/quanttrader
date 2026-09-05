"""Human-readable desk blotter.

Audit.jsonl stays the raw machine trail. This file is what a second
person reads in the morning: time, ticker, decision, why.
Persisted as runtime/journal.jsonl and mirrored to the gist.
"""
from __future__ import annotations

import json
import os
import threading
import time

from .gist_store import sync_runtime_file

PATH = "runtime/journal.jsonl"
_LOCK = threading.RLock()

_KEEP = (
    "VETO", "APPROVE", "FILL", "BUY", "SELL", "STORM",
    "STAND DOWN", "SIGNAL LOGGED", "PROPOSE", "HALT",
    "CIRCUIT", "DECISION CYCLE",
)

_HEAD = {
    "VETO": "נחסם",
    "APPROVE": "אושר",
    "FILL": "בוצע",
    "STAND DOWN (SESSION)": "לא נכנס — סשן",
    "STAND DOWN (TAPE)": "לא נכנס — מאקרו",
    "SIGNAL LOGGED (INCUBATION)": "סיגנל נרשם — בלי פיל (אינקובציה)",
    "SIGNAL LOGGED (STORM REGIME)": "לא נכנס — סטורם",
    "SIGNAL LOGGED (BEAR REGIME)": "לא נכנס — רק דיפ בביר",
    "SIGNAL LOGGED (CIRCUIT BREAKER)": "לא נכנס — שובר מעגל",
    "SIGNAL LOGGED (CORRELATION ALERT)": "לא נכנס — קורלציה",
    "STORM REGIME ALERT": "התראת סטורם",
    "DECISION CYCLE": "מחזור החלטה",
}


def _wanted(action: str) -> bool:
    a = (action or "").upper()
    return any(k in a for k in _KEEP)


def _ticker(rec: dict) -> str:
    data = rec.get("data") or {}
    if isinstance(data, dict):
        for k in ("symbol", "ticker"):
            if data.get(k):
                return str(data[k]).upper()
    trig = str(rec.get("trigger") or "")
    if trig.startswith("signals."):
        return trig.split(".", 1)[-1].upper()
    reason = str(rec.get("reasoning") or "")
    for tok in reason.replace(":", " ").split():
        t = tok.strip(",.;")
        if t.isupper() and 1 <= len(t) <= 5 and t.isalpha():
            return t
    return "—"


def _headline(action: str) -> str:
    for k, v in _HEAD.items():
        if k in (action or ""):
            return v
    if "VETO" in (action or "").upper():
        return "נחסם"
    if "FILL" in (action or "").upper():
        return "בוצע"
    return action or "—"


def from_audit(rec: dict) -> dict | None:
    action = str(rec.get("action") or "")
    if not _wanted(action):
        return None
    ts = float(rec.get("ts") or time.time())
    return {
        "id": rec.get("id"),
        "ts": ts,
        "when": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts)) + " UTC",
        "ticker": _ticker(rec),
        "decision": _headline(action),
        "action": action,
        "who": rec.get("actor") or "",
        "why": (rec.get("reasoning") or "").strip(),
    }


def append_from_audit(rec: dict) -> None:
    row = from_audit(rec)
    if not row:
        return
    os.makedirs(os.path.dirname(PATH) or ".", exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, default=str)
    with _LOCK:
        try:
            with open(PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            return
    sync_runtime_file(PATH, immediate=False)


def tail(n: int = 80) -> list[dict]:
    if not os.path.exists(PATH):
        return []
    out: list[dict] = []
    try:
        with open(PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict) and rec.get("decision"):
                    out.append(rec)
    except Exception:
        return []
    return out[-n:]
