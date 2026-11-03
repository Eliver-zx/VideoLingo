from __future__ import annotations

import contextlib
import contextvars
import csv
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import subprocess
import threading
import time
from typing import Any
import uuid

from core.utils.config_utils import load_key_or

OUTPUT_ROOT = Path("output")
GLOSSARY_DIR = Path("glossary")
GLOBAL_GLOSSARY_CSV = GLOSSARY_DIR / "custom_terms.csv"
DEFAULT_GLOSSARY_DOMAIN = "destiny2"
DOMAIN_TERMS_FILENAME = "terms.csv"
DOMAIN_DICT_FILENAME = "dict.json"
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

_ACTIVE_TASK_DIR: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "_ACTIVE_TASK_DIR", default=None
)
_DEFAULT_TASK_DIR_LOCK = threading.Lock()
_PROCESS_TASK_DIR: Path | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()


def file_fast_fingerprint(path: str | Path | None) -> str:
    """Compute a fast, deterministic content fingerprint for large media or text files."""
    if not path:
        return "none"
    p = Path(path)
    if not p.is_file():
        return "missing"
    stat = p.stat()
    size = stat.st_size
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


def link_or_copy_media(src_path: str | Path, dst_path: str | Path) -> str:
    """Mount a local media file into a task workspace without browser buffering or fragile symlinks.

    Order of preference:
    1. macOS APFS Copy-on-Write clone (`cp -c`): 0 extra disk blocks, independent inode.
    2. Same-volume hard link (`os.link`): 0 extra disk blocks; survives moving or deleting `src_path`.
    3. Cross-volume physical copy (`shutil.copy2`): used when `src_path` is on another drive (`EXDEV`).
    Never uses `os.symlink`, so moving or deleting the original file never breaks the task.
    """
    src = Path(src_path).resolve()
    dst = Path(dst_path)
    if not src.is_file():
        raise FileNotFoundError(f"Source media file does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if os.path.samefile(src, dst):
            return "samefile"
        dst.unlink()

    if platform.system() == "Darwin":
        res = subprocess.run(
            ["cp", "-c", str(src), str(dst)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if res.returncode == 0 and dst.is_file():
            return "clonefile"

    try:
        os.link(src, dst)
        return "hardlink"
    except OSError as exc:
        if exc.errno not in (
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.ENOTSUP,
            errno.EINVAL,
        ):
            raise
        shutil.copy2(src, dst)
        return "copy"


def probe_media_metadata(media_path: str | Path) -> dict[str, Any]:
    """Probe display dimensions (accounting for rotation matrix), duration, and audio presence via ffprobe."""
    p = Path(media_path)
    info: dict[str, Any] = {
        "width": 0,
        "height": 0,
        "rotation": 0,
        "display_width": 0,
        "display_height": 0,
        "duration_sec": 0.0,
        "has_audio": False,
        "has_video": False,
    }
    if not p.is_file():
        return info

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,width,height:stream_tags=rotate:stream_side_data=rotation",
        "-of",
        "json",
        str(p),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=15
        )
        data = json.loads(proc.stdout or "{}")
    except Exception:
        return info

    fmt_dur = data.get("format", {}).get("duration")
    if fmt_dur is not None:
        with contextlib.suppress(ValueError, TypeError):
            info["duration_sec"] = round(float(fmt_dur), 3)

    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "audio":
            info["has_audio"] = True
        elif codec_type == "video" and not info["has_video"]:
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            if width <= 0 or height <= 0:
                continue
            info["has_video"] = True
            info["width"] = width
            info["height"] = height
            rot = 0
            tag_rot = (stream.get("tags") or {}).get("rotate")
            if tag_rot is not None:
                with contextlib.suppress(ValueError, TypeError):
                    rot = int(round(float(tag_rot)))
            for sd in stream.get("side_data_list") or []:
                if "rotation" in sd:
                    with contextlib.suppress(ValueError, TypeError):
                        rot = int(round(float(sd["rotation"])))
            rot_norm = abs(rot) % 360
            info["rotation"] = rot
            if rot_norm in (90, 270):
                info["display_width"] = height
                info["display_height"] = width
            else:
                info["display_width"] = width
                info["display_height"] = height
    return info


class TaskWorkspace:
    """Encapsulates an isolated per-video workspace directory (`output/<task_id>/` or legacy `output/`)."""

    def __init__(self, root_dir: str | Path = OUTPUT_ROOT):
        self.root = Path(root_dir)

    @property
    def task_id(self) -> str:
        return self.root.name

    @property
    def manifest_path(self) -> Path:
        return self.root / TASK_MANIFEST_NAME

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

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
    def gpt_log_dir(self) -> Path:
        return self.root / "gpt_log"

    @property
    def db_path(self) -> Path:
        return self.root / "project.sqlite"

    @property
    def words_path(self) -> Path:
        return self.log_dir / "words.json"

    @property
    def terms_path(self) -> Path:
        return self.log_dir / "terms.json"

    @property
    def utterances_path(self) -> Path:
        return self.log_dir / "utterances.json"

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
    def src_srt_path(self) -> Path:
        return self.root / "src.srt"

    @property
    def trans_srt_path(self) -> Path:
        return self.root / "trans.srt"

    @property
    def bi_srt_path(self) -> Path:
        return self.root / "bi.srt"

    @property
    def bi_ass_path(self) -> Path:
        return self.root / "bi.ass"

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
        if self.manifest_path.is_file():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return self._default_manifest()

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
        _atomic_write_json(self.manifest_path, manifest)
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
            }
        )
        self._write_legacy_input_manifest(
            str(dst_media),
            media_type,
            str(self.root / sub_rel) if sub_rel else None,
        )
        return self.save_manifest(manifest)

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

    # ─── SQLite Store (`project.sqlite`) ───

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
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
            """
        )

    @contextlib.contextmanager
    def _db(self, *, write: bool = False):
        if write:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.db_path.is_file():
            yield None
            return
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            self._ensure_schema(conn)
            if write:
                with conn:
                    yield conn
            else:
                yield conn
        finally:
            conn.close()

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

    def delete_kv(self, *keys: str) -> None:
        if not keys or not self.db_path.is_file():
            return
        with self._db(write=True) as conn:
            assert conn is not None
            conn.executemany("DELETE FROM task_kv WHERE key = ?", [(k,) for k in keys])

    def save_words(self, words: list[dict[str, Any]]) -> None:
        normalized: list[dict[str, Any]] = []
        rows: list[tuple[Any, ...]] = []
        for idx, w in enumerate(words):
            item = {
                "idx": idx,
                "text": str(w.get("text", w.get("word", ""))).strip(),
                "punct_after": str(w.get("punct_after", "")),
                "start": round(float(w.get("start", 0.0)), 3),
                "end": round(float(w.get("end", 0.0)), 3),
                "speaker": w.get("speaker", w.get("speaker_id")),
            }
            normalized.append(item)
            rows.append(
                (
                    idx,
                    item["text"],
                    item["punct_after"],
                    item["start"],
                    item["end"],
                    item["speaker"],
                    json.dumps(item, ensure_ascii=False),
                )
            )
        ts = time.time()
        with self._db(write=True) as conn:
            assert conn is not None
            conn.execute("DELETE FROM words")
            if rows:
                conn.executemany(
                    """
                    INSERT INTO words (idx, text, punct_after, start, end, speaker, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            conn.execute(
                """
                INSERT INTO task_kv (key, value_json, updated_at)
                VALUES ('words_saved', 'true', ?)
                ON CONFLICT(key) DO UPDATE SET value_json = 'true', updated_at = excluded.updated_at
                """,
                (ts,),
            )
        if self.words_path.is_file():
            with contextlib.suppress(OSError):
                self.words_path.unlink()

    def load_words(self) -> list[dict[str, Any]]:
        with self._db(write=False) as conn:
            if conn is not None:
                rows = conn.execute(
                    "SELECT payload_json FROM words ORDER BY idx ASC"
                ).fetchall()
                if rows:
                    return [json.loads(r["payload_json"]) for r in rows]
        if self.words_path.is_file():
            with open(self.words_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                self.save_words(data)
                return data
        return []

    def has_words(self) -> bool:
        with self._db(write=False) as conn:
            if conn is not None:
                row = conn.execute("SELECT 1 FROM words LIMIT 1").fetchone()
                if row is not None:
                    return True
        return self.words_path.is_file()

    def get_words_mtime(self) -> float:
        mt = self.get_kv_mtime("words_saved")
        if mt > 0:
            return mt
        if self.words_path.is_file():
            return os.path.getmtime(self.words_path)
        return 0.0

    def save_terms(
        self,
        summary: str,
        terms: list[dict[str, Any]],
        *,
        mtime: float | None = None,
    ) -> None:
        cleaned_terms: list[dict[str, Any]] = []
        rows: list[tuple[Any, ...]] = []
        for item in terms:
            src = str(item.get("src") or "").strip()
            tgt = str(item.get("tgt") or "").strip()
            if not src or not tgt:
                continue
            raw_aliases = item.get("aliases") or []
            if isinstance(raw_aliases, str):
                raw_aliases = [
                    x.strip()
                    for x in re.split(r"[,;|/，；｜]", raw_aliases)
                    if x.strip()
                ]
            aliases = list(
                dict.fromkeys(
                    str(a).strip()
                    for a in raw_aliases
                    if str(a).strip() and str(a).strip().lower() != src.lower()
                )
            )
            cleaned = {
                "src": src,
                "tgt": tgt,
                "aliases": aliases,
                "note": str(item.get("note") or "").strip(),
                "fix_source": bool(item.get("fix_source", False)),
                "source": str(item.get("source") or "task"),
            }
            idx = len(cleaned_terms)
            cleaned_terms.append(cleaned)
            rows.append(
                (
                    idx,
                    cleaned["src"],
                    cleaned["tgt"],
                    json.dumps(cleaned["aliases"], ensure_ascii=False),
                    cleaned["note"],
                    1 if cleaned["fix_source"] else 0,
                    cleaned["source"],
                )
            )
        ts = float(mtime) if mtime is not None else time.time()
        summary_str = str(summary or "")
        with self._db(write=True) as conn:
            assert conn is not None
            conn.execute("DELETE FROM terms")
            if rows:
                conn.executemany(
                    """
                    INSERT INTO terms (idx, src, tgt, aliases_json, note, fix_source, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            conn.execute(
                """
                INSERT INTO task_kv (key, value_json, updated_at)
                VALUES ('terms_meta', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (
                    json.dumps(
                        {"summary": summary_str, "theme": summary_str},
                        ensure_ascii=False,
                    ),
                    ts,
                ),
            )
        for p in (self.terms_path, self.log_dir / "terminology.json"):
            if p.is_file():
                with contextlib.suppress(OSError):
                    p.unlink()

    def load_terms(self) -> dict[str, Any]:
        with self._db(write=False) as conn:
            if conn is not None:
                meta_row = conn.execute(
                    "SELECT value_json FROM task_kv WHERE key = 'terms_meta'"
                ).fetchone()
                term_rows = conn.execute(
                    "SELECT src, tgt, aliases_json, note, fix_source, source FROM terms ORDER BY idx ASC"
                ).fetchall()
                if meta_row is not None or term_rows:
                    summary_str = ""
                    if meta_row is not None:
                        meta = json.loads(meta_row["value_json"])
                        summary_str = str(
                            meta.get("summary") or meta.get("theme") or ""
                        )
                    norm_terms = [
                        {
                            "src": str(r["src"]),
                            "tgt": str(r["tgt"]),
                            "aliases": json.loads(r["aliases_json"] or "[]"),
                            "note": str(r["note"] or ""),
                            "fix_source": bool(r["fix_source"]),
                            "source": str(r["source"] or "task"),
                        }
                        for r in term_rows
                    ]
                    return {"summary": summary_str, "terms": norm_terms}

        for candidate in (self.terms_path, self.log_dir / "terminology.json"):
            if candidate.is_file():
                mt = os.path.getmtime(candidate)
                with open(candidate, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    summary_str = str(data.get("summary") or data.get("theme") or "")
                    raw_terms = data.get("terms") or []
                    self.save_terms(summary_str, raw_terms, mtime=mt)
                    return self.load_terms()
        return {"summary": "", "terms": []}

    def has_terms(self) -> bool:
        with self._db(write=False) as conn:
            if conn is not None:
                row = conn.execute(
                    "SELECT 1 FROM task_kv WHERE key = 'terms_meta'"
                ).fetchone()
                if row is not None:
                    return True
                t_row = conn.execute("SELECT 1 FROM terms LIMIT 1").fetchone()
                if t_row is not None:
                    return True
        return (
            self.terms_path.is_file() or (self.log_dir / "terminology.json").is_file()
        )

    def get_terms_mtime(self) -> float:
        mt = self.get_kv_mtime("terms_meta")
        if mt > 0:
            return mt
        for p in (self.terms_path, self.log_dir / "terminology.json"):
            if p.is_file():
                return os.path.getmtime(p)
        return 0.0

    def set_terms_mtime(self, mtime: float) -> None:
        meta = self.get_kv("terms_meta")
        if meta is None:
            loaded = self.load_terms()
            self.save_terms(
                loaded.get("summary", ""), loaded.get("terms", []), mtime=mtime
            )
        else:
            self.set_kv("terms_meta", meta, mtime=mtime)

    def save_utterances(
        self,
        utterances: list[dict[str, Any]],
        *,
        mtime: float | None = None,
    ) -> int:
        ts = float(mtime) if mtime is not None else time.time()
        rows: list[tuple[Any, ...]] = []
        has_any_tgt = False
        has_any_cues = False
        for idx, u in enumerate(utterances):
            item = dict(u)
            item["index"] = int(item.get("index", idx))
            utt_id = str(item.get("id") or f"u_{idx:04d}")
            item["id"] = utt_id
            start = round(float(item.get("start", 0.0)), 3)
            end = round(float(item.get("end", 0.0)), 3)
            gap_after = round(float(item.get("gap_after", 0.0)), 3)
            w_span = item.get("word_span") or [0, 0]
            w_start = int(w_span[0]) if len(w_span) >= 1 else 0
            w_end = int(w_span[1]) if len(w_span) >= 2 else w_start
            speaker = item.get("speaker", item.get("speaker_id"))
            src = str(item.get("src") or "")
            tgt = str(item.get("tgt") or "")
            tts_text = str(item.get("tts_text") or tgt)
            sub_cues = item.get("sub_cues") or []
            if tgt.strip():
                has_any_tgt = True
            if sub_cues:
                has_any_cues = True
            rows.append(
                (
                    idx,
                    utt_id,
                    start,
                    end,
                    gap_after,
                    w_start,
                    w_end,
                    speaker,
                    src,
                    tgt,
                    tts_text,
                    json.dumps(sub_cues, ensure_ascii=False),
                    json.dumps(item, ensure_ascii=False),
                )
            )
        with self._db(write=True) as conn:
            assert conn is not None
            rev_row = conn.execute(
                "SELECT value_json FROM task_kv WHERE key = 'utterances_revision'"
            ).fetchone()
            next_rev = (int(json.loads(rev_row["value_json"])) + 1) if rev_row else 1
            conn.execute("DELETE FROM utterances")
            if rows:
                conn.executemany(
                    """
                    INSERT INTO utterances (
                        idx, id, start, end, gap_after, word_start, word_end,
                        speaker, src, tgt, tts_text, sub_cues_json, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            for k, v in (
                ("utterances_saved", True),
                ("utterances_revision", next_rev),
                ("translations_saved", has_any_tgt),
                ("sub_cues_saved", has_any_cues),
            ):
                conn.execute(
                    """
                    INSERT INTO task_kv (key, value_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value_json = excluded.value_json,
                        updated_at = excluded.updated_at
                    """,
                    (k, json.dumps(v), ts),
                )
        if self.utterances_path.is_file():
            with contextlib.suppress(OSError):
                self.utterances_path.unlink()
        return next_rev

    def load_utterances(self) -> list[dict[str, Any]]:
        with self._db(write=False) as conn:
            if conn is not None:
                rows = conn.execute(
                    "SELECT payload_json FROM utterances ORDER BY idx ASC"
                ).fetchall()
                if rows:
                    return [json.loads(r["payload_json"]) for r in rows]
        if self.utterances_path.is_file():
            mt = os.path.getmtime(self.utterances_path)
            with open(self.utterances_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                self.save_utterances(data, mtime=mt)
                return data
        return []

    def has_utterances(self) -> bool:
        with self._db(write=False) as conn:
            if conn is not None:
                row = conn.execute("SELECT 1 FROM utterances LIMIT 1").fetchone()
                if row is not None:
                    return True
        return self.utterances_path.is_file()

    def has_translations(self) -> bool:
        if bool(self.get_kv("translations_saved", False)):
            return True
        utts = self.load_utterances()
        return bool(utts) and any(str(u.get("tgt") or "").strip() for u in utts)

    def has_sub_cues(self) -> bool:
        if bool(self.get_kv("sub_cues_saved", False)):
            return True
        utts = self.load_utterances()
        return bool(utts) and any(bool(u.get("sub_cues")) for u in utts)

    def get_utterances_mtime(self) -> float:
        mt = self.get_kv_mtime("utterances_saved")
        if mt > 0:
            return mt
        if self.utterances_path.is_file():
            return os.path.getmtime(self.utterances_path)
        return 0.0

    def get_utterances_revision(self) -> int:
        return int(self.get_kv("utterances_revision", 0) or 0)

    def save_translation_cache(
        self, fingerprint: str, rows: list[tuple[str, str]]
    ) -> None:
        ts = time.time()
        with self._db(write=True) as conn:
            assert conn is not None
            conn.execute("DELETE FROM translation_cache")
            if rows:
                conn.executemany(
                    "INSERT INTO translation_cache (idx, src, tgt) VALUES (?, ?, ?)",
                    [(i, str(s), str(t)) for i, (s, t) in enumerate(rows)],
                )
            conn.execute(
                """
                INSERT INTO task_kv (key, value_json, updated_at)
                VALUES ('translation_cache_fp', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (json.dumps(fingerprint), ts),
            )

    def load_translation_cache(
        self, expected_fingerprint: str, expected_sources: list[str]
    ) -> list[str] | None:
        with self._db(write=False) as conn:
            if conn is None:
                return None
            fp_row = conn.execute(
                "SELECT value_json FROM task_kv WHERE key = 'translation_cache_fp'"
            ).fetchone()
            if (
                fp_row is None
                or json.loads(fp_row["value_json"]) != expected_fingerprint
            ):
                return None
            rows = conn.execute(
                "SELECT src, tgt FROM translation_cache ORDER BY idx ASC"
            ).fetchall()
            if len(rows) != len(expected_sources):
                return None
            if [str(r["src"]).strip() for r in rows] != [
                s.strip() for s in expected_sources
            ]:
                return None
            return [str(r["tgt"]) for r in rows]

    def save_dub_tasks(self, tasks: Any) -> None:
        if hasattr(tasks, "to_dict"):
            records = tasks.to_dict(orient="records")
        else:
            records = list(tasks or [])
        rows: list[tuple[int, int, str]] = []
        has_chunks = False
        for idx, rec in enumerate(records):
            item = dict(rec)
            num = int(item.get("number", idx + 1))
            item["number"] = num
            if item.get("lines") is not None and item.get("cut_off") is not None:
                has_chunks = True
            rows.append((idx, num, json.dumps(item, ensure_ascii=False, default=str)))
        ts = time.time()
        with self._db(write=True) as conn:
            assert conn is not None
            conn.execute("DELETE FROM dub_tasks")
            if rows:
                conn.executemany(
                    "INSERT INTO dub_tasks (idx, number, payload_json) VALUES (?, ?, ?)",
                    rows,
                )
            for k, v in (
                ("dub_tasks_saved", True),
                ("dub_chunks_saved", has_chunks),
            ):
                conn.execute(
                    """
                    INSERT INTO task_kv (key, value_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value_json = excluded.value_json,
                        updated_at = excluded.updated_at
                    """,
                    (k, json.dumps(v), ts),
                )

    def load_dub_tasks(self) -> list[dict[str, Any]]:
        with self._db(write=False) as conn:
            if conn is not None:
                rows = conn.execute(
                    "SELECT payload_json FROM dub_tasks ORDER BY idx ASC"
                ).fetchall()
                if rows:
                    return [json.loads(r["payload_json"]) for r in rows]
        return []

    def load_dub_tasks_df(self):
        import pandas as pd

        records = self.load_dub_tasks()
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def has_dub_tasks(self) -> bool:
        with self._db(write=False) as conn:
            if conn is not None:
                row = conn.execute("SELECT 1 FROM dub_tasks LIMIT 1").fetchone()
                if row is not None:
                    return True
        return False

    def has_dub_chunks(self) -> bool:
        if not self.has_dub_tasks():
            return False
        if self.get_kv("dub_chunks_saved") is True:
            return True
        records = self.load_dub_tasks()
        return bool(
            records
            and all(
                r.get("lines") is not None and r.get("cut_off") is not None
                for r in records
            )
        )

    def is_step_done(self, target_path: str | Path) -> bool:
        s = str(target_path).replace("\\", "/")
        name = Path(s).name
        if name in {"cleaned_chunks.xlsx", "words.json"}:
            return self.has_words()
        if name in {"split_by_nlp.txt", "split_by_meaning.txt", "utterances.json"}:
            return self.has_utterances()
        if name in {"terminology.json", "terms.json"}:
            return self.has_terms()
        if name == "translation_results.xlsx":
            return self.has_translations()
        if name in {
            "translation_results_for_subtitles.xlsx",
            "translation_results_remerged.xlsx",
        }:
            return self.has_sub_cues()
        if name == "tts_tasks.xlsx":
            if not self.has_dub_chunks():
                return False
            dub_mtime = self.get_kv_mtime("dub_tasks_saved")
            for srt_p in (
                self.audio_dir / "trans_subs_for_audio.srt",
                self.trans_srt_path,
            ):
                if (
                    srt_p.is_file()
                    and dub_mtime > 0
                    and srt_p.stat().st_mtime > dub_mtime
                ):
                    return False
            return True
        return False

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
            if self.db_path.is_file():
                with self._db(write=True) as conn:
                    assert conn is not None
                    conn.execute("DELETE FROM words")
                    conn.execute("DELETE FROM task_kv WHERE key = 'words_saved'")
            paths_to_remove.extend(
                [
                    self.words_path,
                    self.log_dir / "cleaned_chunks.xlsx",
                ]
            )
        elif stage == "split":
            if self.db_path.is_file():
                with self._db(write=True) as conn:
                    assert conn is not None
                    conn.execute("DELETE FROM utterances")
                    conn.execute(
                        "DELETE FROM task_kv WHERE key IN ('utterances_saved', 'utterances_revision', 'translations_saved', 'sub_cues_saved', 'sub_split_cfg', 'sub_split_rev')"
                    )
            paths_to_remove.extend(
                [
                    self.utterances_path,
                    self.log_dir / "split_by_nlp.txt",
                    self.log_dir / "split_by_meaning.txt",
                ]
            )
        elif stage == "summarize":
            if self.db_path.is_file():
                with self._db(write=True) as conn:
                    assert conn is not None
                    conn.execute("DELETE FROM terms")
                    conn.execute("DELETE FROM task_kv WHERE key = 'terms_meta'")
            paths_to_remove.extend(
                [
                    self.terms_path,
                    self.log_dir / "terminology.json",
                ]
            )
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
            paths_to_remove.extend(
                [
                    self.log_dir / "translation_results.xlsx",
                    self.log_dir / "translation_raw.csv",
                ]
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
                    self.bi_srt_path,
                    self.bi_ass_path,
                    self.log_dir / "translation_results_for_subtitles.xlsx",
                    self.log_dir / "translation_results_remerged.xlsx",
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
                    self.audio_dir / "tts_tasks.xlsx",
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
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)


# ─── Domain Glossary (`glossary/<domain>/terms.csv` & `dict.json`) helpers ───

_DOMAIN_NAME_RE = re.compile(r"^[\w\-\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]{1,64}$")


def sanitize_domain_name(name: str) -> str:
    cleaned = str(name or "").strip()
    if (
        not cleaned
        or cleaned in {".", ".."}
        or "/" in cleaned
        or "\\" in cleaned
        or not _DOMAIN_NAME_RE.match(cleaned)
    ):
        raise ValueError(f"Invalid glossary domain name: {name!r}")
    return cleaned


def domain_dir_path(domain: str, base_dir: str | Path | None = None) -> Path:
    safe = sanitize_domain_name(domain)
    root = Path(base_dir if base_dir is not None else GLOSSARY_DIR)
    return root / safe


def domain_terms_csv_path(domain: str, base_dir: str | Path | None = None) -> Path:
    return domain_dir_path(domain, base_dir) / DOMAIN_TERMS_FILENAME


def domain_dict_json_path(domain: str, base_dir: str | Path | None = None) -> Path:
    return domain_dir_path(domain, base_dir) / DOMAIN_DICT_FILENAME


def ensure_default_glossary_domain(base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir if base_dir is not None else GLOSSARY_DIR)
    d_dir = root / DEFAULT_GLOSSARY_DOMAIN
    if not root.exists():
        d_dir.mkdir(parents=True, exist_ok=True)
        csv_file = d_dir / DOMAIN_TERMS_FILENAME
        save_global_glossary([], csv_path=csv_file)
    return d_dir


def list_glossary_domains(base_dir: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(base_dir if base_dir is not None else GLOSSARY_DIR)
    ensure_default_glossary_domain(root)
    if not root.is_dir():
        return []
    domains: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        try:
            sanitize_domain_name(child.name)
        except ValueError:
            continue
        domains.append(get_glossary_domain_meta(child.name, base_dir=root))
    return domains


def get_glossary_domain_meta(
    domain: str, base_dir: str | Path | None = None
) -> dict[str, Any]:
    safe = sanitize_domain_name(domain)
    d_dir = domain_dir_path(safe, base_dir)
    csv_file = d_dir / DOMAIN_TERMS_FILENAME
    dict_file = d_dir / DOMAIN_DICT_FILENAME
    terms = load_domain_glossary(safe, base_dir=base_dir) if csv_file.is_file() else []
    dict_count = 0
    has_dict = dict_file.is_file()
    if has_dict:
        dict_count = count_domain_dict_entries(dict_file)
    return {
        "domain": safe,
        "name": safe,
        "terms_count": len(terms),
        "has_dict": has_dict,
        "dict_count": dict_count,
    }


def count_domain_dict_entries(dict_path: str | Path) -> int:
    p = Path(dict_path)
    if not p.is_file():
        return 0
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return 0
    if isinstance(data, dict):
        if isinstance(data.get("entries"), dict):
            return len(data["entries"])
        if isinstance(data.get("terms"), list):
            return len(data["terms"])
        return len(data)
    if isinstance(data, list):
        return len(data)
    return 0


def create_glossary_domain(
    domain: str, base_dir: str | Path | None = None
) -> dict[str, Any]:
    safe = sanitize_domain_name(domain)
    d_dir = domain_dir_path(safe, base_dir)
    d_dir.mkdir(parents=True, exist_ok=True)
    csv_file = d_dir / DOMAIN_TERMS_FILENAME
    if not csv_file.is_file():
        save_global_glossary([], csv_path=csv_file)
    return get_glossary_domain_meta(safe, base_dir=base_dir)


def delete_glossary_domain(domain: str, base_dir: str | Path | None = None) -> bool:
    safe = sanitize_domain_name(domain)
    d_dir = domain_dir_path(safe, base_dir)
    if not d_dir.is_dir():
        return False
    shutil.rmtree(d_dir)
    return True


def load_domain_glossary(
    domain: str, base_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    if not domain or not str(domain).strip():
        return []
    csv_file = domain_terms_csv_path(domain, base_dir=base_dir)
    return load_global_glossary(csv_file)


def save_domain_glossary(
    domain: str,
    entries: list[dict[str, Any]],
    base_dir: str | Path | None = None,
) -> None:
    csv_file = domain_terms_csv_path(domain, base_dir=base_dir)
    save_global_glossary(entries, csv_path=csv_file)


def promote_terms_to_domain(
    terms: list[dict[str, Any]],
    domain: str | None = None,
    base_dir: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> int:
    if csv_path is not None:
        return promote_terms_to_global(terms, csv_path=csv_path)
    target_domain = str(
        domain
        if domain is not None
        else load_key_or("glossary_domain", DEFAULT_GLOSSARY_DOMAIN)
    ).strip()
    if not target_domain:
        return 0
    csv_file = domain_terms_csv_path(target_domain, base_dir=base_dir)
    return promote_terms_to_global(terms, csv_path=csv_file)


def _resolve_default_glossary_csv() -> Path:
    if (
        GLOBAL_GLOSSARY_CSV != Path("glossary") / "custom_terms.csv"
        or GLOBAL_GLOSSARY_CSV.is_file()
    ):
        return GLOBAL_GLOSSARY_CSV
    active_domain = str(
        load_key_or("glossary_domain", DEFAULT_GLOSSARY_DOMAIN) or ""
    ).strip()
    if active_domain:
        try:
            return domain_terms_csv_path(active_domain)
        except ValueError:
            pass
    return GLOBAL_GLOSSARY_CSV


def load_global_glossary(csv_path: str | Path | None = None) -> list[dict[str, Any]]:
    if csv_path is None:
        if (
            GLOBAL_GLOSSARY_CSV == Path("glossary") / "custom_terms.csv"
            and not GLOBAL_GLOSSARY_CSV.is_file()
        ):
            active_domain = str(
                load_key_or("glossary_domain", DEFAULT_GLOSSARY_DOMAIN) or ""
            ).strip()
            if not active_domain:
                return []
            p = _resolve_default_glossary_csv()
        else:
            p = GLOBAL_GLOSSARY_CSV
    else:
        p = Path(csv_path)
    if not p.is_file():
        return []
    entries: list[dict[str, Any]] = []
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = str(row.get("src") or row.get("Original") or "").strip()
            tgt = str(row.get("tgt") or row.get("Translation") or "").strip()
            if not src or not tgt:
                continue
            raw_aliases = str(row.get("aliases") or "").strip()
            aliases = [
                a.strip() for a in re.split(r"[,;|/，；｜]", raw_aliases) if a.strip()
            ]
            entries.append(
                {
                    "src": src,
                    "tgt": tgt,
                    "aliases": aliases,
                    "note": str(row.get("note") or row.get("Note") or "").strip(),
                    "fix_source": str(row.get("fix_source") or "").strip().lower()
                    in {"1", "true", "yes"},
                    "source": "global_kb",
                }
            )
    return entries


def save_global_glossary(
    entries: list[dict[str, Any]], csv_path: str | Path | None = None
) -> None:
    p = Path(csv_path) if csv_path is not None else _resolve_default_glossary_csv()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    fieldnames = ["src", "tgt", "aliases", "note", "fix_source"]
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in entries:
            src = str(item.get("src") or "").strip()
            tgt = str(item.get("tgt") or "").strip()
            if not src or not tgt:
                continue
            aliases = item.get("aliases") or []
            if isinstance(aliases, list):
                aliases_str = "|".join(
                    str(a).strip() for a in aliases if str(a).strip()
                )
            else:
                aliases_str = str(aliases).strip()
            writer.writerow(
                {
                    "src": src,
                    "tgt": tgt,
                    "aliases": aliases_str,
                    "note": str(item.get("note") or "").strip(),
                    "fix_source": "true" if item.get("fix_source") else "false",
                }
            )
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, p)


def promote_terms_to_global(
    terms: list[dict[str, Any]], csv_path: str | Path | None = None
) -> int:
    if (
        csv_path is None
        and GLOBAL_GLOSSARY_CSV == Path("glossary") / "custom_terms.csv"
        and not GLOBAL_GLOSSARY_CSV.is_file()
    ):
        active_domain = str(
            load_key_or("glossary_domain", DEFAULT_GLOSSARY_DOMAIN) or ""
        ).strip()
        if not active_domain:
            return 0
    existing = load_global_glossary(csv_path)
    by_key = {item["src"].lower(): item for item in existing}
    added_or_updated = 0
    for t in terms:
        src = str(t.get("src") or "").strip()
        tgt = str(t.get("tgt") or "").strip()
        if not src or not tgt:
            continue
        key = src.lower()
        prev_aliases = by_key.get(key, {}).get("aliases") or []
        curr_aliases = t.get("aliases")
        curr_list = curr_aliases if isinstance(curr_aliases, list) else []
        merged_aliases = list(dict.fromkeys([*prev_aliases, *curr_list]))
        by_key[key] = {
            "src": src,
            "tgt": tgt,
            "aliases": merged_aliases,
            "note": str(t.get("note") or by_key.get(key, {}).get("note") or "").strip(),
            "fix_source": bool(
                t.get("fix_source", by_key.get(key, {}).get("fix_source", False))
            ),
            "source": "global_kb",
        }
        added_or_updated += 1
    save_global_glossary(list(by_key.values()), csv_path)
    return added_or_updated


# ─── Active Workspace Context Management ───


def get_active_workspace() -> TaskWorkspace:
    ctx_dir = _ACTIVE_TASK_DIR.get()
    if ctx_dir is not None:
        return TaskWorkspace(ctx_dir)
    with _DEFAULT_TASK_DIR_LOCK:
        if _PROCESS_TASK_DIR is not None:
            return TaskWorkspace(_PROCESS_TASK_DIR)
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


def set_process_workspace(task_dir: str | Path | None) -> None:
    global _PROCESS_TASK_DIR
    with _DEFAULT_TASK_DIR_LOCK:
        _PROCESS_TASK_DIR = Path(task_dir) if task_dir is not None else None


@contextlib.contextmanager
def workspace_context(task_dir: str | Path | TaskWorkspace):
    global _PROCESS_TASK_DIR
    resolved = task_dir.root if isinstance(task_dir, TaskWorkspace) else Path(task_dir)
    token = _ACTIVE_TASK_DIR.set(resolved)
    with _DEFAULT_TASK_DIR_LOCK:
        prev_proc_dir = _PROCESS_TASK_DIR
        _PROCESS_TASK_DIR = resolved
    try:
        yield TaskWorkspace(resolved)
    finally:
        with _DEFAULT_TASK_DIR_LOCK:
            _PROCESS_TASK_DIR = prev_proc_dir
        _ACTIVE_TASK_DIR.reset(token)


def create_task_workspace(
    title_hint: str = "video", base_dir: str | Path = OUTPUT_ROOT
) -> TaskWorkspace:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w\-]+", "_", title_hint).strip("_")[:32] or "video"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:6]
    task_dir = base / f"{ts}_{slug}_{short_id}"
    ws = TaskWorkspace(task_dir)
    ws.ensure_dirs()
    return ws


def list_task_workspaces(base_dir: str | Path = OUTPUT_ROOT) -> list[TaskWorkspace]:
    base = Path(base_dir)
    if not base.is_dir():
        return []
    workspaces: list[TaskWorkspace] = []
    # Check subdirectories containing task.json
    for child in sorted(base.iterdir(), reverse=True):
        if child.is_dir() and (child / TASK_MANIFEST_NAME).is_file():
            workspaces.append(TaskWorkspace(child))
    # Also include root `output/` if it directly has a task.json or legacy input_manifest.json
    if (base / TASK_MANIFEST_NAME).is_file() or (
        base / "input_manifest.json"
    ).is_file():
        workspaces.append(TaskWorkspace(base))
    return workspaces
