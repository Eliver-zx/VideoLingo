from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import threading
import time
from typing import Any, Callable

_CLEANUP_HOOKS: list[Callable[[Path], None]] = []
_CLEANUP_HOOKS_LOCK = threading.Lock()


def register_cleanup_hook(hook: Callable[[Path], None]) -> None:
    with _CLEANUP_HOOKS_LOCK:
        _CLEANUP_HOOKS.append(hook)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def replace_with_retry(src: str | Path, dst: str | Path, attempts: int = 5) -> None:
    """Retry os.replace on transient Windows NTFS PermissionError locks."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05 * (i + 1))


def rmtree_robust(
    path: str | Path, *, ignore_errors: bool = False, attempts: int = 5
) -> None:
    """Remove a directory tree cleanly after closing pooled SQLite handles under it."""
    target = Path(path)
    if not target.exists():
        return
    target_resolved = target.resolve()
    with _CLEANUP_HOOKS_LOCK:
        hooks = list(_CLEANUP_HOOKS)
    for hook in hooks:
        hook(target_resolved)
    gc.collect()

    def _onerror(func, err_path, _exc_info):
        with contextlib.suppress(OSError):
            os.chmod(err_path, stat.S_IWRITE)
            func(err_path)

    for i in range(attempts):
        if not target.exists():
            return
        try:
            shutil.rmtree(target, onerror=_onerror)
            return
        except OSError:
            if i == attempts - 1:
                if ignore_errors:
                    shutil.rmtree(target, ignore_errors=True)
                    return
                raise
            gc.collect()
            time.sleep(0.05 * (i + 1))


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        replace_with_retry(tmp_path, path)
    finally:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()


def file_fast_fingerprint(path: str | Path | None) -> str:
    """Compute a deterministic head+tail SHA-256 fingerprint for media or text files."""
    if not path:
        return "none"
    p = Path(path)
    if not p.is_file():
        return "missing"
    st = p.stat()
    size = st.st_size
    hasher = hashlib.sha256()
    hasher.update(f"size:{size}:".encode("utf-8"))
    chunk_size = 1024 * 1024  # 1 MiB head + tail
    with open(p, "rb") as f:
        if size <= chunk_size * 2:
            hasher.update(f.read())
        else:
            hasher.update(f.read(chunk_size))
            f.seek(size - chunk_size)
            hasher.update(f.read(chunk_size))
    return f"sha256:{hasher.hexdigest()[:24]}"


def json_content_fingerprint(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"
