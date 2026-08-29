"""GitHub Gist persistence for Streamlit Cloud.

Streamlit Cloud deletes the local `runtime/` directory on every reboot, which
used to wipe paper trades, positions and the audit trail. When GITHUB_TOKEN
is set (env or Streamlit Secrets), this module mirrors the live runtime
files to a private gist named "quanttrader-runtime" and hydrates them back
on startup.

No token → local files only (dev / tests). Failures never raise into the
trading path: a down GitHub is logged and the local write still stands.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable

import requests

from core.state import _env

GIST_DESCRIPTION = "quanttrader-runtime"
TRACKED = (
    "broker.json",
    "audit.jsonl",
    "strategy_registry.json",
    "circuit_breaker.json",
)
API = "https://api.github.com/gists"
_EMPTY = {
    "broker.json": "{}",
    "audit.jsonl": "\n",
    "strategy_registry.json": "{}",
    "circuit_breaker.json": "{}",
}


class GistStore:
    def __init__(self, token: str | None = None, gist_id: str | None = None,
                 runtime_dir: str = "runtime",
                 http_get: Callable | None = None,
                 http_post: Callable | None = None,
                 http_patch: Callable | None = None):
        self.token = (token if token is not None else _env("GITHUB_TOKEN")).strip()
        self.gist_id = (gist_id if gist_id is not None else _env("GIST_ID")).strip()
        self.runtime_dir = runtime_dir
        self.enabled = bool(self.token)
        self.last_saved_ts: float | None = None
        self.last_error: str | None = None
        self._lock = threading.RLock()
        self._pending: dict[str, str] = {}
        self._timer: threading.Timer | None = None
        self._hydrated = False
        self._get = http_get or requests.get
        self._post = http_post or requests.post
        self._patch = http_patch or requests.patch

    # ---- HTTP --------------------------------------------------------------
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "quanttrader",
        }

    def tracks(self, path: str) -> bool:
        if not self.enabled or os.path.basename(path) not in TRACKED:
            return False
        # The live store (runtime/) must never upload pytest isolation
        # files. A test-constructed store whose runtime_dir itself lives
        # under runtime/_test/ is allowed to sync its own files.
        live = os.path.abspath(self.runtime_dir)
        p = os.path.abspath(path)
        test_root = os.path.abspath(os.path.join("runtime", "_test"))
        if test_root in p and test_root not in live:
            return False
        return True

    def last_saved_label(self) -> str | None:
        if not self.enabled:
            return None
        if self.last_saved_ts:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC",
                                  time.gmtime(self.last_saved_ts))
            return f"Last saved to GitHub · {stamp}"
        if self.last_error:
            return f"GitHub save failed · {self.last_error}"
        return "GitHub persistence armed · waiting for first save"

    # ---- hydrate on startup ------------------------------------------------
    def hydrate(self) -> bool:
        """Pull gist → write local runtime files. Returns True if anything
        was restored. Safe to call more than once; subsequent calls no-op."""
        with self._lock:
            if self._hydrated or not self.enabled:
                self._hydrated = True
                return False
            try:
                gist = self._load_or_create_gist()
                if not gist:
                    self._hydrated = True
                    return False
                files = gist.get("files") or {}
                os.makedirs(self.runtime_dir, exist_ok=True)
                restored = False
                for name in TRACKED:
                    info = files.get(name) or {}
                    content = info.get("content")
                    if not content or not str(content).strip():
                        continue
                    dest = os.path.join(self.runtime_dir, name)
                    # Don't clobber a non-empty local file with gist emptiness;
                    # a local file on a VPS is the source of truth until the
                    # first successful save uploads it.
                    if os.path.exists(dest) and os.path.getsize(dest) > 2:
                        continue
                    with open(dest, "w") as f:
                        f.write(content if content.endswith("\n") or name != "audit.jsonl"
                                else content + "\n")
                    restored = True
                self._hydrated = True
                return restored
            except Exception as e:
                self.last_error = f"hydrate: {e}"
                self._hydrated = True
                return False

    def _load_or_create_gist(self) -> dict | None:
        if self.gist_id:
            r = self._get(f"{API}/{self.gist_id}", headers=self._headers(),
                          timeout=20)
            if r.status_code == 200:
                return r.json()
            self.last_error = f"GET gist {r.status_code}"
            # fall through and try to find/create rather than give up
        r = self._get(API, headers=self._headers(),
                      params={"per_page": 100}, timeout=20)
        if r.status_code != 200:
            self.last_error = f"list gists {r.status_code}"
            return None
        for g in r.json():
            if g.get("description") == GIST_DESCRIPTION:
                self.gist_id = g["id"]
                # list payload omits file contents — fetch the full gist
                full = self._get(f"{API}/{self.gist_id}",
                                 headers=self._headers(), timeout=20)
                if full.status_code == 200:
                    return full.json()
                return g
        created = self._post(
            API, headers=self._headers(), timeout=20,
            json={"description": GIST_DESCRIPTION, "public": False,
                  "files": {n: {"content": _EMPTY[n]} for n in TRACKED}},
        )
        if created.status_code not in (200, 201):
            self.last_error = f"create gist {created.status_code}"
            return None
        body = created.json()
        self.gist_id = body.get("id", "")
        return body

    # ---- save --------------------------------------------------------------
    def queue(self, path: str, immediate: bool = False) -> None:
        if not self.tracks(path):
            return
        with self._lock:
            self._pending[os.path.basename(path)] = path
            if immediate:
                self._cancel_timer()
                self._flush_locked()
            else:
                self._arm_timer()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _arm_timer(self) -> None:
        self._cancel_timer()
        t = threading.Timer(2.0, self.flush)
        t.daemon = True
        self._timer = t
        t.start()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            try:
                self._timer.cancel()
            except Exception:
                pass
            self._timer = None

    def _flush_locked(self) -> None:
        if not self.enabled or not self._pending:
            return
        if not self.gist_id:
            gist = self._load_or_create_gist()
            if not gist:
                return
        files = {}
        for name, path in list(self._pending.items()):
            try:
                if os.path.exists(path):
                    with open(path) as f:
                        content = f.read()
                else:
                    content = _EMPTY.get(name, "{}")
                if not content.strip():
                    content = _EMPTY.get(name, "\n")
                # Gist per-file cap is 10 MB. Keep a tail of the audit log
                # rather than fail the whole save.
                if len(content) > 9_000_000 and name == "audit.jsonl":
                    lines = content.splitlines()[-5000:]
                    content = "\n".join(lines) + "\n"
                files[name] = {"content": content}
            except Exception as e:
                self.last_error = f"read {name}: {e}"
        if not files:
            return
        try:
            r = self._patch(f"{API}/{self.gist_id}",
                            headers=self._headers(), timeout=25,
                            json={"files": files})
            if r.status_code in (200, 201):
                self.last_saved_ts = time.time()
                self.last_error = None
                self._pending.clear()
            else:
                self.last_error = f"PATCH gist {r.status_code}"
        except Exception as e:
            self.last_error = f"PATCH: {e}"


_STORE: GistStore | None = None
_STORE_LOCK = threading.Lock()


def get_gist_store() -> GistStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = GistStore()
        return _STORE


def reset_gist_store(store: GistStore | None = None) -> None:
    """Tests only — swap or clear the process singleton."""
    global _STORE
    with _STORE_LOCK:
        _STORE = store


def hydrate_runtime() -> bool:
    return get_gist_store().hydrate()


def sync_runtime_file(path: str, immediate: bool = False) -> None:
    try:
        get_gist_store().queue(path, immediate=immediate)
    except Exception:
        pass
