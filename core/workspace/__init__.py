from __future__ import annotations

import contextlib
import contextvars
from datetime import datetime
from pathlib import Path
import re
import uuid

from core.utils.models import ensure_windows_ffmpeg_path
from core.workspace.fs_atomic import (
    _atomic_write_json,
    _now_iso,
    file_fast_fingerprint,
    json_content_fingerprint,
    replace_with_retry,
    rmtree_robust,
)
from core.workspace.glossary_store import (
    DEFAULT_GLOSSARY_DOMAIN,
    DOMAIN_DICT_FILENAME,
    DOMAIN_TERMS_FILENAME,
    count_domain_dict_entries,
    create_glossary_domain,
    delete_glossary_domain,
    domain_csv_known_srcs,
    domain_dict_json_path,
    domain_dir_path,
    domain_terms_csv_path,
    ensure_default_glossary_domain,
    get_glossary_domain_meta,
    list_glossary_domains,
    load_domain_glossary,
    load_global_glossary,
    promote_terms_to_domain,
    promote_terms_to_global,
    remove_terms_from_domain,
    remove_terms_from_global,
    sanitize_domain_name,
    save_domain_glossary,
    save_global_glossary,
    search_domain_dict_entries,
)
from core.workspace.media_probe import link_or_copy_media, probe_media_metadata
from core.workspace.task_workspace import (
    STAGE_ORDER,
    TASK_MANIFEST_NAME,
    TaskWorkspace,
)

ensure_windows_ffmpeg_path()

OUTPUT_ROOT = Path("output")
GLOSSARY_DIR = Path("glossary")
GLOBAL_GLOSSARY_CSV = GLOSSARY_DIR / "custom_terms.csv"

_ACTIVE_TASK_DIR: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "_ACTIVE_TASK_DIR", default=None
)


def get_active_workspace() -> TaskWorkspace:
    ctx_dir = _ACTIVE_TASK_DIR.get()
    if ctx_dir is not None:
        return TaskWorkspace(ctx_dir)
    return TaskWorkspace(OUTPUT_ROOT)


def workspace_for_path(path: str | Path | None = None) -> TaskWorkspace:
    if path is None:
        return get_active_workspace()
    s = str(path).replace("\\", "/")
    if s == "output" or s.startswith("output/"):
        return get_active_workspace()
    p = Path(path)
    ws_root = p.parent.parent if p.parent.name in {"log", "audio"} else p.parent
    return TaskWorkspace(ws_root)


@contextlib.contextmanager
def workspace_context(task_dir: str | Path | TaskWorkspace):
    resolved = task_dir.root if isinstance(task_dir, TaskWorkspace) else Path(task_dir)
    token = _ACTIVE_TASK_DIR.set(resolved)
    try:
        yield TaskWorkspace(resolved)
    finally:
        _ACTIVE_TASK_DIR.reset(token)


def create_task_workspace(
    title_hint: str = "video", base_dir: str | Path | None = None
) -> TaskWorkspace:
    base = Path(base_dir if base_dir is not None else OUTPUT_ROOT)
    base.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w\-]+", "_", title_hint).strip("_")[:32] or "video"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:6]
    task_dir = base / f"{ts}_{slug}_{short_id}"
    ws = TaskWorkspace(task_dir)
    ws.ensure_dirs()
    return ws


def list_task_workspaces(base_dir: str | Path | None = None) -> list[TaskWorkspace]:
    base = Path(base_dir if base_dir is not None else OUTPUT_ROOT)
    if not base.is_dir():
        return []
    workspaces: list[TaskWorkspace] = []
    for child in sorted(base.iterdir(), reverse=True):
        if child.is_dir() and (child / TASK_MANIFEST_NAME).is_file():
            workspaces.append(TaskWorkspace(child))
    if (base / TASK_MANIFEST_NAME).is_file() or (
        base / "input_manifest.json"
    ).is_file():
        workspaces.append(TaskWorkspace(base))
    return workspaces


__all__ = [
    "DEFAULT_GLOSSARY_DOMAIN",
    "DOMAIN_DICT_FILENAME",
    "DOMAIN_TERMS_FILENAME",
    "GLOBAL_GLOSSARY_CSV",
    "GLOSSARY_DIR",
    "OUTPUT_ROOT",
    "STAGE_ORDER",
    "TASK_MANIFEST_NAME",
    "TaskWorkspace",
    "_ACTIVE_TASK_DIR",
    "_atomic_write_json",
    "_now_iso",
    "count_domain_dict_entries",
    "create_glossary_domain",
    "create_task_workspace",
    "delete_glossary_domain",
    "domain_csv_known_srcs",
    "domain_dict_json_path",
    "domain_dir_path",
    "domain_terms_csv_path",
    "ensure_default_glossary_domain",
    "file_fast_fingerprint",
    "get_active_workspace",
    "get_glossary_domain_meta",
    "json_content_fingerprint",
    "link_or_copy_media",
    "list_glossary_domains",
    "list_task_workspaces",
    "load_domain_glossary",
    "load_global_glossary",
    "probe_media_metadata",
    "promote_terms_to_domain",
    "promote_terms_to_global",
    "remove_terms_from_domain",
    "remove_terms_from_global",
    "replace_with_retry",
    "rmtree_robust",
    "sanitize_domain_name",
    "save_domain_glossary",
    "save_global_glossary",
    "search_domain_dict_entries",
    "workspace_context",
    "workspace_for_path",
]
