from __future__ import annotations

from collections import deque
import contextlib
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import threading
from typing import Any

from core.utils.config_utils import load_key_or
from core.workspace.artifacts import WorkspaceArtifactsMixin
from core.workspace.fs_atomic import (
    _atomic_write_json,
    _now_iso,
    register_cleanup_hook,
    rmtree_robust,
)
from core.workspace.media_probe import link_or_copy_media, probe_media_metadata

TASK_MANIFEST_NAME = "task.json"

STAGE_ORDER: tuple[str, ...] = (
    "asr",
    "split",
    "summarize",
    "translate",
    "sub_split",
    "sub_video",
    "dub_tts",
    "dub_merge",
    "dub_video",
)

# Manifest cache keyed by resolved manifest_path -> ((st_ino, st_mtime_ns, st_size), data).
_MANIFEST_CACHE: dict[Path, tuple[tuple[int, int, int], dict[str, Any]]] = {}
_MANIFEST_CACHE_LOCK = threading.Lock()


def _purge_manifest_cache(target_resolved: Path) -> None:
    with _MANIFEST_CACHE_LOCK:
        for mp in [
            p
            for p in _MANIFEST_CACHE
            if p == target_resolved or target_resolved in p.parents
        ]:
            _MANIFEST_CACHE.pop(mp, None)


register_cleanup_hook(_purge_manifest_cache)


class TaskWorkspace(WorkspaceArtifactsMixin):
    """Encapsulates an isolated per-video workspace directory (`output/<task_id>/` or legacy `output/`)."""

    def __init__(self, root_dir: str | Path | None = None):
        if root_dir is None:
            import core.workspace as _ws

            root_dir = _ws.OUTPUT_ROOT
        self.root = Path(root_dir)

    @property
    def manifest_path(self) -> Path:
        return self.root / TASK_MANIFEST_NAME

    @property
    def refers_dir(self) -> Path:
        return self.audio_dir / "refers"

    @property
    def segs_dir(self) -> Path:
        return self.audio_dir / "segs"

    @property
    def tmp_dir(self) -> Path:
        return self.audio_dir / "tmp"

    @property
    def log_dir(self) -> Path:
        return self.root / "log"

    @property
    def runtime_log_path(self) -> Path:
        return self.log_dir / "runtime.jsonl"

    @property
    def gpt_log_dir(self) -> Path:
        return self.root / "gpt_log"

    @property
    def raw_audio_path(self) -> Path:
        return self.audio_dir / "raw.mp3"

    @property
    def vocal_audio_path(self) -> Path:
        return self.audio_dir / "vocal.mp3"

    @property
    def background_audio_path(self) -> Path:
        return self.audio_dir / "background.mp3"

    @property
    def dub_wav_path(self) -> Path:
        return self.audio_dir / "dub.wav"

    @property
    def normalized_dub_path(self) -> Path:
        return self.root / "normalized_dub.wav"

    @property
    def dub_srt_path(self) -> Path:
        return self.root / "dub.srt"

    @property
    def sub_video_path(self) -> Path:
        return self.root / "output_sub.mp4"

    @property
    def dub_video_path(self) -> Path:
        return self.root / "output_dub.mp4"

    def ensure_dirs(self) -> None:
        for d in (
            self.root,
            self.audio_dir,
            self.refers_dir,
            self.segs_dir,
            self.tmp_dir,
            self.log_dir,
            self.gpt_log_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # ─── Manifest (`task.json`) management ───

    def load_manifest(self) -> dict[str, Any]:
        mp = self.manifest_path
        try:
            st = mp.stat()
        except OSError:
            return self._default_manifest()
        resolved = mp.resolve()
        sig = (st.st_ino, st.st_mtime_ns, st.st_size)
        with _MANIFEST_CACHE_LOCK:
            cached = _MANIFEST_CACHE.get(resolved)
            if cached is not None and cached[0] == sig:
                return copy.deepcopy(cached[1])
        with open(mp, "r", encoding="utf-8") as f:
            data = json.load(f)
        with _MANIFEST_CACHE_LOCK:
            _MANIFEST_CACHE[resolved] = (sig, copy.deepcopy(data))
        return data

    def _default_manifest(self) -> dict[str, Any]:
        now = _now_iso()
        return {
            "schema_version": 1,
            "task_id": self.task_id,
            "title": self.task_id,
            "created_at": now,
            "updated_at": now,
            "origin": {
                "kind": "local_file",
                "source_path": None,
                "original_filename": None,
                "url": None,
                "mount_mode": None,
            },
            "media": {
                "type": "video",
                "filename": None,
                "input_subtitle": None,
                "width": 0,
                "height": 0,
                "rotation": 0,
                "display_width": 0,
                "display_height": 0,
                "duration_sec": 0.0,
                "has_audio": True,
            },
            "config_override": {},
            "execution": {
                "state": "idle",
                "desired_stage": "all",
                "dubbing_enabled": False,
                "current_step": None,
                "step_index": -1,
                "total_steps": 0,
                "progress": 0.0,
                "pause_reason": None,
                "error": None,
            },
            "fingerprints": {},
        }

    def save_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        manifest["updated_at"] = _now_iso()
        mp = self.manifest_path
        _atomic_write_json(mp, manifest)
        with contextlib.suppress(OSError):
            st = mp.stat()
            resolved = mp.resolve()
            with _MANIFEST_CACHE_LOCK:
                _MANIFEST_CACHE[resolved] = (
                    (st.st_ino, st.st_mtime_ns, st.st_size),
                    copy.deepcopy(manifest),
                )
        return manifest

    def update_manifest(self, **sections: Any) -> dict[str, Any]:
        manifest = self.load_manifest()
        for k, v in sections.items():
            if isinstance(v, dict) and isinstance(manifest.get(k), dict):
                manifest[k].update(v)
            else:
                manifest[k] = v
        return self.save_manifest(manifest)

    def init_from_local_file(
        self,
        source_path: str | Path,
        *,
        subtitle_path: str | Path | None = None,
        title: str | None = None,
        config_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Initialize workspace from a local video/audio/SRT file using zero-copy mounting."""
        self.ensure_dirs()
        src = Path(source_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Source file not found: {src}")

        ext = src.suffix.lower()
        manifest = self.load_manifest()
        manifest["title"] = title or src.stem
        manifest["origin"] = {
            "kind": "local_file",
            "source_path": str(src),
            "original_filename": src.name,
            "url": None,
            "mount_mode": None,
        }
        if config_override:
            manifest["config_override"] = dict(config_override)

        if ext == ".srt":
            dst_srt = self.root / "source.srt"
            shutil.copy2(src, dst_srt)
            manifest["origin"]["mount_mode"] = "copy"
            manifest["media"].update(
                {
                    "type": "subtitle",
                    "filename": "source.srt",
                    "input_subtitle": "source.srt",
                    "has_audio": False,
                }
            )
            self._write_legacy_input_manifest(str(dst_srt), "subtitle", str(dst_srt))
            return self.save_manifest(manifest)

        video_exts = {
            str(x).lower().lstrip(".")
            for x in load_key_or(
                "allowed_video_formats",
                ["mp4", "mov", "mkv", "avi", "webm", "flv", "wmv"],
            )
        }
        media_type = "video" if ext.lstrip(".") in video_exts else "audio"
        canonical_name = f"source{ext}"
        dst_media = self.root / canonical_name
        mount_mode = link_or_copy_media(src, dst_media)
        meta = probe_media_metadata(dst_media)

        sub_rel: str | None = None
        if subtitle_path:
            sub_src = Path(subtitle_path).resolve()
            dst_srt = self.root / "source.srt"
            shutil.copy2(sub_src, dst_srt)
            sub_rel = "source.srt"

        manifest["origin"]["mount_mode"] = mount_mode
        manifest["media"].update(
            {
                "type": media_type,
                "filename": canonical_name,
                "input_subtitle": sub_rel,
                "width": meta["width"],
                "height": meta["height"],
                "rotation": meta["rotation"],
                "display_width": meta["display_width"],
                "display_height": meta["display_height"],
                "duration_sec": meta["duration_sec"],
                "has_audio": meta["has_audio"],
                "codec_name": meta.get("codec_name", ""),
                "audio_codec": meta.get("audio_codec", ""),
                "pix_fmt": meta.get("pix_fmt", ""),
            }
        )
        self._write_legacy_input_manifest(
            str(dst_media),
            media_type,
            str(self.root / sub_rel) if sub_rel else None,
        )
        return self.save_manifest(manifest)

    def append_log(
        self,
        msg: str,
        *,
        tag: str = "llm",
        label: str = "RUN",
        warn: bool = False,
        time_str: str | None = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "time": time_str or datetime.now().strftime("%H:%M:%S"),
            "tag": tag,
            "label": label,
            "msg": msg,
            "warn": bool(warn),
        }
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.runtime_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
        return entry

    def load_logs(self, limit: int = 300) -> list[dict[str, Any]]:
        if limit <= 0 or not self.runtime_log_path.is_file():
            return []
        size = self.runtime_log_path.stat().st_size
        if size <= 0:
            return []
        chunk_bytes = max(limit * 512, 65536)
        if size <= chunk_bytes:
            text = self.runtime_log_path.read_bytes().decode("utf-8", errors="ignore")
            tail = deque(
                (ln.strip() for ln in text.split("\n") if ln.strip()),
                maxlen=limit,
            )
            return [json.loads(line) for line in tail]

        with open(self.runtime_log_path, "rb") as fb:
            window = chunk_bytes
            while True:
                offset = max(0, size - window)
                fb.seek(offset)
                raw = fb.read()
                text = raw.decode("utf-8", errors="ignore")
                lines = text.split("\n")
                if offset > 0 and lines:
                    lines = lines[1:]
                non_empty = [ln.strip() for ln in lines if ln.strip()]
                if len(non_empty) >= limit or offset == 0:
                    return [json.loads(ln) for ln in non_empty[-limit:]]
                window *= 2

    def _write_legacy_input_manifest(
        self, media_path: str, media_type: str, subtitle_path: str | None = None
    ) -> None:
        legacy: dict[str, Any] = {
            "path": media_path.replace("\\", "/"),
            "type": media_type,
        }
        if subtitle_path:
            legacy["subtitle"] = subtitle_path.replace("\\", "/")
        _atomic_write_json(self.root / "input_manifest.json", legacy)

    def resolve_media_file(self) -> tuple[str, str]:
        """Return `(media_path, media_type)` deterministically from `task.json` or legacy `input_manifest.json`."""
        if self.manifest_path.is_file():
            manifest = self.load_manifest()
            media = manifest.get("media") or {}
            fname = media.get("filename")
            mtype = media.get("type")
            if fname and mtype in {"video", "audio", "subtitle"}:
                candidate = self.root / fname
                if candidate.is_file():
                    return str(candidate).replace("\\", "/"), mtype

        legacy_manifest = self.root / "input_manifest.json"
        if legacy_manifest.is_file():
            with open(legacy_manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
            mpath = data.get("path")
            mtype = data.get("type")
            if (
                mpath
                and mtype in {"video", "audio", "subtitle"}
                and os.path.exists(mpath)
            ):
                return str(mpath).replace("\\", "/"), mtype

        raise ValueError(
            "No media file found. Please download or upload a media file first."
        )

    def resolve_subtitle_file(self) -> str | None:
        """Return user-supplied SRT path if present."""
        if self.manifest_path.is_file():
            manifest = self.load_manifest()
            sub_name = (manifest.get("media") or {}).get("input_subtitle")
            if sub_name:
                candidate = self.root / sub_name
                if candidate.is_file():
                    return str(candidate).replace("\\", "/")
        src_srt = self.root / "source.srt"
        if src_srt.is_file():
            return str(src_srt).replace("\\", "/")
        legacy_manifest = self.root / "input_manifest.json"
        if legacy_manifest.is_file():
            with open(legacy_manifest, "r", encoding="utf-8") as f:
                sub = json.load(f).get("subtitle")
            if sub and os.path.exists(sub):
                return str(sub).replace("\\", "/")
        return None

    # ─── Stage Fingerprint DAG & Cascading Invalidation ───

    def is_stage_fresh(
        self, stage: str, current_fingerprint: str, required_files: list[str | Path]
    ) -> bool:
        manifest = self.load_manifest()
        recorded = (manifest.get("fingerprints") or {}).get(stage)
        if not recorded or recorded != current_fingerprint:
            return False
        return all(Path(p).exists() for p in required_files)

    def record_stage_fingerprint(self, stage: str, fingerprint: str) -> None:
        manifest = self.load_manifest()
        fps = manifest.setdefault("fingerprints", {})
        fps[stage] = fingerprint
        self.save_manifest(manifest)

    def invalidate_from_stage(self, stage: str) -> list[str]:
        """Invalidate `stage` and all downstream stages, cleaning up stale artifacts and cached TTS audio."""
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown stage: {stage!r}")
        idx = STAGE_ORDER.index(stage)
        invalidated = list(STAGE_ORDER[idx:])
        manifest = self.load_manifest()
        fps = manifest.get("fingerprints") or {}
        for s in invalidated:
            fps.pop(s, None)
        manifest["fingerprints"] = fps
        self.save_manifest(manifest)

        for s in invalidated:
            self._clean_stage_outputs(s)
        return invalidated

    def _clean_stage_outputs(self, stage: str) -> None:
        paths_to_remove: list[Path] = []
        dirs_to_clear: list[Path] = []

        if stage == "asr":
            self.clear_runtime_cache("whisper.detected_language")
            if self.db_path.is_file():
                with self._db(write=True) as conn:
                    assert conn is not None
                    conn.execute("DELETE FROM words")
                    conn.execute("DELETE FROM task_kv WHERE key = 'words_saved'")
        elif stage == "split":
            if self.db_path.is_file():
                with self._db(write=True) as conn:
                    assert conn is not None
                    conn.execute("DELETE FROM utterances")
                    conn.execute(
                        "DELETE FROM task_kv WHERE key IN ('utterances_saved', 'utterances_revision', 'translations_saved', 'sub_cues_saved', 'sub_split_cfg', 'sub_split_rev')"
                    )
        elif stage == "summarize":
            if self.db_path.is_file():
                with self._db(write=True) as conn:
                    assert conn is not None
                    conn.execute("DELETE FROM terms")
                    conn.execute("DELETE FROM task_kv WHERE key = 'terms_meta'")
        elif stage == "translate":
            if self.db_path.is_file():
                utts = self.load_utterances()
                if utts:
                    for u in utts:
                        u["tgt"] = ""
                        u["tts_text"] = ""
                        u.pop("sub_cues", None)
                        u.pop("sub_split_config", None)
                    self.save_utterances(utts)
                with self._db(write=True) as conn:
                    assert conn is not None
                    conn.execute("DELETE FROM translation_cache")
                    conn.execute(
                        "DELETE FROM task_kv WHERE key IN ('translation_cache_fp', 'translations_saved', 'sub_cues_saved', 'sub_split_cfg', 'sub_split_rev')"
                    )
        elif stage == "sub_split":
            if self.db_path.is_file():
                utts = self.load_utterances()
                if utts:
                    for u in utts:
                        u.pop("sub_cues", None)
                        u.pop("sub_split_config", None)
                    self.save_utterances(utts)
                self.delete_kv("sub_cues_saved", "sub_split_cfg", "sub_split_rev")
            paths_to_remove.extend(
                [
                    self.src_srt_path,
                    self.trans_srt_path,
                    self.root / "src_trans.srt",
                    self.root / "trans_src.srt",
                    self.root / ".subtitle_done",
                ]
            )
        elif stage == "sub_video":
            paths_to_remove.append(self.sub_video_path)
        elif stage == "dub_tts":
            if self.db_path.is_file():
                with self._db(write=True) as conn:
                    assert conn is not None
                    conn.execute("DELETE FROM dub_tasks")
                    conn.execute(
                        "DELETE FROM task_kv WHERE key IN ('dub_tasks_saved', 'dub_chunks_saved')"
                    )
            paths_to_remove.extend(
                [
                    self.log_dir / "dub_truncated.json",
                    self.dub_srt_path,
                    self.root / ".dubbing_done",
                ]
            )
            dirs_to_clear.extend([self.tmp_dir, self.segs_dir, self.refers_dir])
        elif stage == "dub_merge":
            paths_to_remove.extend(
                [
                    self.dub_wav_path,
                    self.root / "dub.mp3",
                    self.normalized_dub_path,
                    self.dub_srt_path,
                ]
            )
        elif stage == "dub_video":
            paths_to_remove.extend(
                [
                    self.dub_video_path,
                    self.normalized_dub_path,
                    self.root / ".dubbing_done",
                ]
            )

        for p in paths_to_remove:
            if p.is_file():
                with contextlib.suppress(OSError):
                    p.unlink()
        for d in dirs_to_clear:
            if d.is_dir():
                rmtree_robust(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)
