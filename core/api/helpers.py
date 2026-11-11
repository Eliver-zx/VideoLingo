from __future__ import annotations

import copy
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any
from urllib.parse import unquote

from fastapi import HTTPException

from core.utils.config_utils import task_config_context
from core.workspace import TaskWorkspace, workspace_context
from core.workspace.fs_atomic import register_cleanup_hook

_ORPHAN_ACTIVE_STATES = {"running", "stopping", "paused", "queued"}

_STATIC_SUMMARY_CACHE: dict[Path, tuple[tuple[Any, ...], dict[str, Any]]] = {}
_SUMMARY_CACHE_LOCK = threading.Lock()


def _purge_summary_cache(target_resolved: Path) -> None:
    with _SUMMARY_CACHE_LOCK:
        for k in [
            p
            for p in _STATIC_SUMMARY_CACHE
            if p == target_resolved or target_resolved in p.parents
        ]:
            _STATIC_SUMMARY_CACHE.pop(k, None)


register_cleanup_hook(_purge_summary_cache)


def get_output_dir() -> Path:
    import api as _api

    return _api.OUTPUT


def get_runner():
    import api as _api

    return _api.runner


def get_pipeline_steps(stage: str = "subtitles", dubbing: bool = False):
    import api as _api

    return _api.get_steps(stage, dubbing)


def _clean_user_path(raw: str | Path) -> str:
    """Strip whitespace and surrounding quotes from Windows 'Copy as path' strings or file:/// URIs."""
    s = str(raw or "").strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    if s.lower().startswith("file:///"):
        rest = unquote(s[8:])
        if len(rest) >= 2 and rest[1] in (":", "|"):
            s = rest[0] + ":" + rest[2:]
        else:
            s = "/" + rest
    return s


def subtitle_file(raw_path: str | Path) -> Path:
    """Validate an external SRT file supplied in a task creation request."""
    from core._2_import_subtitles import read_cues

    out_dir = get_output_dir()
    path = Path(_clean_user_path(raw_path)).resolve()
    if (
        not path.is_file()
        or path.suffix.lower() != ".srt"
        or path.is_relative_to(out_dir.resolve())
    ):
        raise HTTPException(
            422, "Subtitles must be an existing SRT file outside output/"
        )
    try:
        read_cues(path.read_bytes())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return path


def _wrap_workspace_steps(
    ws: TaskWorkspace,
    steps: list[tuple[str, Any]],
    config_override: dict[str, Any] | None = None,
) -> list[tuple[str, Any]]:
    def _run_step(fn, overrides):
        with workspace_context(ws.root), task_config_context(overrides):
            return fn()

    return [(label, partial(_run_step, fn, config_override)) for label, fn in steps]


def _resolve_workspace(task_id: str) -> TaskWorkspace:
    out_dir = get_output_dir()
    if task_id == "output" and out_dir.is_dir():
        return TaskWorkspace(out_dir)
    task_dir = (out_dir / task_id).resolve()
    if task_dir.parent != out_dir.resolve() or not task_dir.is_dir():
        raise HTTPException(404, f"Task workspace not found: {task_id}")
    return TaskWorkspace(task_dir)


def _stat_mtime_ns(p: Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return 0


def _workspace_static_sig(ws: TaskWorkspace) -> tuple[Any, ...]:
    db_wal = ws.root / "project.sqlite-wal"
    return (
        _stat_mtime_ns(ws.root),
        _stat_mtime_ns(ws.manifest_path),
        _stat_mtime_ns(ws.audio_dir),
        _stat_mtime_ns(ws.audio_dir / "dub.wav"),
        _stat_mtime_ns(ws.root / "dub.mp3"),
        _stat_mtime_ns(ws.src_srt_path),
        _stat_mtime_ns(ws.trans_srt_path),
        _stat_mtime_ns(ws.root / "output_sub.mp4"),
        _stat_mtime_ns(ws.root / "output_dub.mp4"),
        _stat_mtime_ns(ws.log_dir / "dub_truncated.json"),
        _stat_mtime_ns(ws.db_path),
        _stat_mtime_ns(db_wal),
    )


def _workspace_summary(ws: TaskWorkspace) -> dict[str, Any]:
    runner = get_runner()
    ws_resolved = ws.root.resolve()
    sig = _workspace_static_sig(ws)

    cached_static: dict[str, Any] | None = None
    with _SUMMARY_CACHE_LOCK:
        hit = _STATIC_SUMMARY_CACHE.get(ws_resolved)
        if hit is not None and hit[0] == sig:
            cached_static = hit[1]

    if cached_static is not None:
        manifest = copy.deepcopy(cached_static)
    else:
        manifest = ws.load_manifest()
        files = (
            sorted(
                entry.name
                for entry in os.scandir(ws.root)
                if entry.is_file() and not entry.name.startswith(".")
            )
            if ws.root.is_dir()
            else []
        )
        for audio_sub in ("dub.wav", "raw.mp3", "vocal.mp3", "background.mp3"):
            if (ws.audio_dir / audio_sub).is_file():
                files.append(f"audio/{audio_sub}")
        truncated = []
        trunc_path = ws.log_dir / "dub_truncated.json"
        if trunc_path.is_file():
            data = json.loads(trunc_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError(
                    f"Expected list in {trunc_path}, got {type(data).__name__}"
                )
            truncated = data
        stages_done = ws.compute_stages_done()
        manifest["files"] = files
        manifest["truncated"] = truncated
        manifest["stages_done"] = stages_done
        with _SUMMARY_CACHE_LOCK:
            _STATIC_SUMMARY_CACHE[ws_resolved] = (sig, copy.deepcopy(manifest))

    if runner.active_task_id == ws.task_id:
        manifest["execution"] = {
            **manifest.get("execution", {}),
            "state": runner.state,
            "current_step": runner.current_label or None,
            "sub_step": runner.sub_label or None,
            "sub_progress": round(runner.sub_progress, 4),
            "step_index": runner.current_step,
            "total_steps": runner.total_steps,
            "progress": runner.progress,
            "pause_reason": runner.pause_message or None,
            "error": runner.error_msg or None,
        }
    elif runner.is_queued(ws.task_id):
        manifest["execution"] = {
            **manifest.get("execution", {}),
            "state": "queued",
            "current_step": "Queued",
        }
    elif (manifest.get("execution") or {}).get("state") in _ORPHAN_ACTIVE_STATES:
        ws.update_manifest(execution={"state": "stopped", "pause_reason": None})
        manifest["execution"] = {
            **manifest.get("execution", {}),
            "state": "stopped",
            "pause_reason": None,
        }
        with _SUMMARY_CACHE_LOCK:
            _STATIC_SUMMARY_CACHE.pop(ws_resolved, None)
    return manifest


def reveal_in_file_manager(target: Path, *, select_file: bool = False) -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        return
    if sys.platform == "darwin":
        subprocess.Popen(
            ["open", "-R", str(target)] if select_file else ["open", str(target)]
        )
    elif sys.platform.startswith("win"):
        if select_file:
            subprocess.Popen(["explorer", f"/select,{target}"])
        else:
            os.startfile(str(target))  # type: ignore[attr-defined]
    else:
        folder = target.parent if select_file else target
        subprocess.Popen(["xdg-open", str(folder)])
