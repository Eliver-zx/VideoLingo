from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
import weakref

from core.workspace.fs_atomic import register_cleanup_hook

# Track (resolved_db_path, st_ino, birthtime_ns) for initialized SQLite databases.
_INITIALIZED_DB_INODES: set[tuple[Path, int, int]] = set()
_INIT_DB_LOCK = threading.Lock()

# Thread-local SQLite connection reuse with automatic close on thread exit.
_THREAD_LOCAL = threading.local()
_DB_GENERATIONS: dict[Path, int] = {}
_GLOBAL_DB_GEN = 0
_POOL_LOCK = threading.Lock()


def _close_conns_dict(
    conns: dict[
        Path, tuple[sqlite3.Connection, tuple[Path, int, int] | None, int, int]
    ],
) -> None:
    for entry in list(conns.values()):
        with contextlib.suppress(sqlite3.Error):
            entry[0].close()
    conns.clear()


class _ThreadConnHolder:
    def __init__(self) -> None:
        self.conns: dict[
            Path, tuple[sqlite3.Connection, tuple[Path, int, int] | None, int, int]
        ] = {}
        self._finalizer = weakref.finalize(self, _close_conns_dict, self.conns)


def _current_db_gen(resolved_db: Path) -> int:
    with _POOL_LOCK:
        return _GLOBAL_DB_GEN + _DB_GENERATIONS.get(resolved_db, 0)


# In-memory mirror of runtime task attributes keyed by resolved workspace root path.
_RUNTIME_STATE_BY_WS: dict[Path, dict[str, Any]] = {}
_RUNTIME_HYDRATED_SIG: dict[Path, tuple[Path, int, int] | None] = {}
_RUNTIME_STATE_LOCK = threading.Lock()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS task_kv (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS words (
    idx INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    punct_after TEXT NOT NULL DEFAULT '',
    start REAL NOT NULL DEFAULT 0.0,
    end REAL NOT NULL DEFAULT 0.0,
    speaker TEXT,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS terms (
    idx INTEGER PRIMARY KEY,
    src TEXT NOT NULL,
    tgt TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    note TEXT NOT NULL DEFAULT '',
    fix_source INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'task'
);
CREATE TABLE IF NOT EXISTS utterances (
    idx INTEGER PRIMARY KEY,
    id TEXT NOT NULL,
    start REAL NOT NULL DEFAULT 0.0,
    end REAL NOT NULL DEFAULT 0.0,
    gap_after REAL NOT NULL DEFAULT 0.0,
    word_start INTEGER NOT NULL DEFAULT 0,
    word_end INTEGER NOT NULL DEFAULT 0,
    speaker TEXT,
    src TEXT NOT NULL DEFAULT '',
    tgt TEXT NOT NULL DEFAULT '',
    tts_text TEXT NOT NULL DEFAULT '',
    sub_cues_json TEXT NOT NULL DEFAULT '[]',
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS translation_cache (
    idx INTEGER PRIMARY KEY,
    src TEXT NOT NULL,
    tgt TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dub_tasks (
    idx INTEGER PRIMARY KEY,
    number INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_cache (
    log_title TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    prompt TEXT NOT NULL,
    model TEXT NOT NULL,
    reasoning_effort TEXT NOT NULL,
    resp_type TEXT NOT NULL,
    resp_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (log_title, prompt_hash, model, reasoning_effort, resp_type)
);
"""


def _path_is_within(child: Path, target_resolved: Path) -> bool:
    return child == target_resolved or target_resolved in child.parents


def _purge_path_caches(target_resolved: Path) -> None:
    with _INIT_DB_LOCK:
        for entry in [
            e for e in _INITIALIZED_DB_INODES if _path_is_within(e[0], target_resolved)
        ]:
            _INITIALIZED_DB_INODES.discard(entry)
    with _RUNTIME_STATE_LOCK:
        for ws_k in [
            k for k in _RUNTIME_STATE_BY_WS if _path_is_within(k, target_resolved)
        ]:
            _RUNTIME_STATE_BY_WS.pop(ws_k, None)
        for ws_k in [
            k for k in _RUNTIME_HYDRATED_SIG if _path_is_within(k, target_resolved)
        ]:
            _RUNTIME_HYDRATED_SIG.pop(ws_k, None)
    with _POOL_LOCK:
        global _GLOBAL_DB_GEN
        _GLOBAL_DB_GEN += 1
        if target_resolved.suffix == ".sqlite":
            _DB_GENERATIONS[target_resolved] = (
                _DB_GENERATIONS.get(target_resolved, 0) + 1
            )
        else:
            for db_k in [
                k for k in _DB_GENERATIONS if _path_is_within(k, target_resolved)
            ]:
                _DB_GENERATIONS.pop(db_k, None)
    holder: _ThreadConnHolder | None = getattr(_THREAD_LOCAL, "holder", None)
    if holder is not None and holder.conns:
        to_close: list[sqlite3.Connection] = []
        for db_p in [p for p in holder.conns if _path_is_within(p, target_resolved)]:
            popped = holder.conns.pop(db_p, None)
            if popped is not None:
                to_close.append(popped[0])
        for conn in to_close:
            with contextlib.suppress(sqlite3.Error):
                conn.close()


register_cleanup_hook(_purge_path_caches)


class WorkspaceDbMixin:
    root: Path

    @property
    def task_id(self) -> str:
        return self.root.name

    @property
    def db_path(self) -> Path:
        return self.root / "project.sqlite"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def src_srt_path(self) -> Path:
        return self.root / "src.srt"

    @property
    def trans_srt_path(self) -> Path:
        return self.root / "trans.srt"

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.executescript(_SCHEMA_SQL)

    @staticmethod
    def _db_stat_signature(resolved_db: Path) -> tuple[Path, int, int] | None:
        try:
            st = resolved_db.stat()
        except OSError:
            return None
        if st.st_size <= 0:
            return None
        birthtime_ns = getattr(st, "st_birthtime_ns", None)
        if birthtime_ns is None:
            birthtime = getattr(st, "st_birthtime", None)
            if birthtime is not None:
                birthtime_ns = int(birthtime * 1e9)
            else:
                birthtime_ns = st.st_ctime_ns if os.name == "nt" else 0
        if st.st_ino == 0 and birthtime_ns == 0:
            return None
        return (resolved_db, st.st_ino, int(birthtime_ns))

    def _ensure_db_initialized(
        self, conn: sqlite3.Connection, resolved_db: Path | None = None
    ) -> tuple[Path, int, int] | None:
        resolved = resolved_db if resolved_db is not None else self.db_path.resolve()
        sig = self._db_stat_signature(resolved)
        if sig is not None and sig in _INITIALIZED_DB_INODES:
            return sig
        with _INIT_DB_LOCK:
            sig = self._db_stat_signature(resolved)
            if sig is not None and sig in _INITIALIZED_DB_INODES:
                return sig
            self._ensure_schema(conn)
            sig = self._db_stat_signature(resolved)
            if sig is not None:
                _INITIALIZED_DB_INODES.add(sig)
            return sig

    def _get_thread_conn(
        self, resolved_db: Path
    ) -> tuple[sqlite3.Connection, bool, int]:
        holder: _ThreadConnHolder | None = getattr(_THREAD_LOCAL, "holder", None)
        if holder is None:
            holder = _ThreadConnHolder()
            _THREAD_LOCAL.holder = holder
        local_map = holder.conns

        pid = os.getpid()
        gen = _current_db_gen(resolved_db)
        current_sig = self._db_stat_signature(resolved_db)
        cached = local_map.get(resolved_db)
        if cached is not None:
            conn, cached_sig, cached_pid, cached_gen = cached
            if (
                cached_pid == pid
                and cached_gen == gen
                and current_sig is not None
                and cached_sig == current_sig
            ):
                return conn, True, gen
            local_map.pop(resolved_db, None)
            with contextlib.suppress(sqlite3.Error):
                conn.close()

        conn = sqlite3.connect(str(resolved_db), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000;")
        sig = self._ensure_db_initialized(conn, resolved_db)
        if sig is not None:
            gen = _current_db_gen(resolved_db)
            local_map[resolved_db] = (conn, sig, pid, gen)
            return conn, True, gen
        return conn, False, gen

    @contextlib.contextmanager
    def _db(self, *, write: bool = False):
        if write:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.db_path.is_file():
            yield None
            return
        resolved_db = self.db_path.resolve()
        conn, is_pooled, opened_gen = self._get_thread_conn(resolved_db)
        try:
            if write:
                with conn:
                    yield conn
            else:
                yield conn
        finally:
            if not is_pooled:
                with contextlib.suppress(sqlite3.Error):
                    conn.close()
            elif _current_db_gen(resolved_db) != opened_gen:
                holder: _ThreadConnHolder | None = getattr(
                    _THREAD_LOCAL, "holder", None
                )
                if holder is not None:
                    cached = holder.conns.get(resolved_db)
                    if cached is not None and cached[0] is conn:
                        holder.conns.pop(resolved_db, None)
                with contextlib.suppress(sqlite3.Error):
                    conn.close()

    def _hydrate_runtime_bucket(self, ws_key: Path) -> dict[str, Any]:
        resolved_db = ws_key / "project.sqlite"
        db_sig = self._db_stat_signature(resolved_db)
        with _RUNTIME_STATE_LOCK:
            if (
                ws_key in _RUNTIME_HYDRATED_SIG
                and _RUNTIME_HYDRATED_SIG[ws_key] == db_sig
            ):
                return _RUNTIME_STATE_BY_WS.setdefault(ws_key, {})

        db_loaded: dict[str, Any] = {}
        if db_sig is not None:
            with self._db(write=False) as conn:
                if conn is not None:
                    rows = conn.execute(
                        "SELECT key, value_json FROM task_kv WHERE key LIKE 'runtime.%'"
                    ).fetchall()
                    for r in rows:
                        db_loaded[r["key"][len("runtime.") :]] = json.loads(
                            r["value_json"]
                        )
            db_sig = self._db_stat_signature(resolved_db)

        with _RUNTIME_STATE_LOCK:
            existing = _RUNTIME_STATE_BY_WS.get(ws_key)
            merged = dict(db_loaded)
            if existing:
                merged.update(existing)
            _RUNTIME_STATE_BY_WS[ws_key] = merged
            _RUNTIME_HYDRATED_SIG[ws_key] = db_sig
            return merged

    def set_runtime_state(self, key: str, value: Any) -> None:
        ws_key = self.root.resolve()
        self._hydrate_runtime_bucket(ws_key)
        self.set_kv(f"runtime.{key}", value)
        resolved_db = ws_key / "project.sqlite"
        db_sig = self._db_stat_signature(resolved_db)
        with _RUNTIME_STATE_LOCK:
            bucket = _RUNTIME_STATE_BY_WS.setdefault(ws_key, {})
            bucket[key] = (
                copy.deepcopy(value) if isinstance(value, (dict, list)) else value
            )
            _RUNTIME_HYDRATED_SIG[ws_key] = db_sig

    def get_runtime_state(self, key: str, default: Any = None) -> Any:
        ws_key = self.root.resolve()
        bucket = self._hydrate_runtime_bucket(ws_key)
        with _RUNTIME_STATE_LOCK:
            if key in bucket:
                val = bucket[key]
                return copy.deepcopy(val) if isinstance(val, (dict, list)) else val
        return default

    def get_all_runtime_state(self) -> dict[str, Any]:
        ws_key = self.root.resolve()
        bucket = self._hydrate_runtime_bucket(ws_key)
        with _RUNTIME_STATE_LOCK:
            if not bucket:
                return {}
            return copy.deepcopy(bucket)

    def clear_runtime_cache(self, prefix: str | None = None) -> None:
        ws_key = self.root.resolve()
        with _RUNTIME_STATE_LOCK:
            if prefix is None:
                _RUNTIME_STATE_BY_WS.pop(ws_key, None)
                _RUNTIME_HYDRATED_SIG.pop(ws_key, None)
            else:
                bucket = _RUNTIME_STATE_BY_WS.get(ws_key)
                if bucket:
                    for k in [
                        k for k in bucket if k == prefix or k.startswith(f"{prefix}.")
                    ]:
                        bucket.pop(k, None)
        if prefix is None:
            _purge_path_caches(self.db_path.resolve())
        elif self.db_path.is_file():
            with self._db(write=True) as conn:
                assert conn is not None
                conn.execute(
                    "DELETE FROM task_kv WHERE key = ? OR key LIKE ?",
                    (f"runtime.{prefix}", f"runtime.{prefix}.%"),
                )

    def save_llm_cache(
        self,
        *,
        log_title: str,
        prompt: str,
        model: str,
        reasoning_effort: str,
        resp_type: str,
        resp: Any,
    ) -> None:
        prompt_hash = hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
        norm_type = str(resp_type or "")
        norm_model = str(model or "")
        norm_effort = str(reasoning_effort or "none")
        raw_resp = json.dumps(resp, ensure_ascii=False)
        ts = time.time()
        with self._db(write=True) as conn:
            assert conn is not None
            conn.execute(
                """
                INSERT INTO llm_cache (
                    log_title, prompt_hash, prompt, model, reasoning_effort, resp_type, resp_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(log_title, prompt_hash, model, reasoning_effort, resp_type) DO UPDATE SET
                    resp_json = excluded.resp_json,
                    created_at = excluded.created_at
                """,
                (
                    str(log_title),
                    prompt_hash,
                    str(prompt),
                    norm_model,
                    norm_effort,
                    norm_type,
                    raw_resp,
                    ts,
                ),
            )

    def load_llm_cache(
        self,
        *,
        log_title: str,
        prompt: str,
        resp_type: str | None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[bool, Any]:
        if not self.db_path.is_file():
            return False, None
        prompt_hash = hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
        norm_type = str(resp_type or "")
        query = (
            "SELECT resp_json, model, reasoning_effort FROM llm_cache "
            "WHERE log_title = ? AND prompt_hash = ? AND prompt = ? AND resp_type = ? "
            "ORDER BY created_at DESC"
        )
        with self._db(write=False) as conn:
            if conn is None:
                return False, None
            rows = conn.execute(
                query, (str(log_title), prompt_hash, str(prompt), norm_type)
            ).fetchall()
            for r in rows:
                if model is not None and r["model"] and r["model"] != str(model):
                    continue
                if reasoning_effort is not None and r["reasoning_effort"] != str(
                    reasoning_effort
                ):
                    continue
                return True, json.loads(r["resp_json"])
        return False, None

    def get_kv(self, key: str, default: Any = None) -> Any:
        with self._db(write=False) as conn:
            if conn is None:
                return default
            row = conn.execute(
                "SELECT value_json FROM task_kv WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return default
            return json.loads(row["value_json"])

    def get_kv_mtime(self, key: str) -> float:
        with self._db(write=False) as conn:
            if conn is None:
                return 0.0
            row = conn.execute(
                "SELECT updated_at FROM task_kv WHERE key = ?", (key,)
            ).fetchone()
            return float(row["updated_at"]) if row is not None else 0.0

    def set_kv(self, key: str, value: Any, *, mtime: float | None = None) -> None:
        ts = float(mtime) if mtime is not None else time.time()
        raw = json.dumps(value, ensure_ascii=False)
        with self._db(write=True) as conn:
            assert conn is not None
            conn.execute(
                """
                INSERT INTO task_kv (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (key, raw, ts),
            )
        if key.startswith("runtime."):
            ws_key = self.root.resolve()
            rt_key = key[len("runtime.") :]
            with _RUNTIME_STATE_LOCK:
                if ws_key in _RUNTIME_STATE_BY_WS:
                    _RUNTIME_STATE_BY_WS[ws_key][rt_key] = (
                        copy.deepcopy(value)
                        if isinstance(value, (dict, list))
                        else value
                    )

    def delete_kv(self, *keys: str) -> None:
        if not keys or not self.db_path.is_file():
            return
        with self._db(write=True) as conn:
            assert conn is not None
            conn.executemany("DELETE FROM task_kv WHERE key = ?", [(k,) for k in keys])
        rt_keys = [k[len("runtime.") :] for k in keys if k.startswith("runtime.")]
        if rt_keys:
            ws_key = self.root.resolve()
            with _RUNTIME_STATE_LOCK:
                bucket = _RUNTIME_STATE_BY_WS.get(ws_key)
                if bucket:
                    for rk in rt_keys:
                        bucket.pop(rk, None)
