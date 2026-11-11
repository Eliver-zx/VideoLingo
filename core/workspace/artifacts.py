from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any

from core.workspace.db_core import WorkspaceDbMixin

_ALIAS_SPLIT_RE = re.compile(r"[,;|/，；｜]")


class WorkspaceArtifactsMixin(WorkspaceDbMixin):
    def save_words(self, words: list[dict[str, Any]]) -> None:
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
        self.notify_artifact_update("words", tag="asr", label="ASR")

    def load_words(self) -> list[dict[str, Any]]:
        with self._db(write=False) as conn:
            if conn is not None:
                rows = conn.execute(
                    "SELECT payload_json FROM words ORDER BY idx ASC"
                ).fetchall()
                if rows:
                    return [json.loads(r["payload_json"]) for r in rows]
        return []

    def load_word_texts(self) -> list[str]:
        with self._db(write=False) as conn:
            if conn is not None:
                rows = conn.execute(
                    "SELECT text FROM words ORDER BY idx ASC"
                ).fetchall()
                if rows:
                    return [str(r["text"] or "") for r in rows]
        return []

    def has_words(self) -> bool:
        with self._db(write=False) as conn:
            if conn is not None:
                row = conn.execute("SELECT 1 FROM words LIMIT 1").fetchone()
                if row is not None:
                    return True
        return False

    def get_words_mtime(self) -> float:
        mt = self.get_kv_mtime("words_saved")
        return mt if mt > 0 else 0.0

    def save_terms(
        self,
        summary: str,
        terms: list[dict[str, Any]],
        *,
        mtime: float | None = None,
    ) -> int:
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
                    x.strip() for x in _ALIAS_SPLIT_RE.split(raw_aliases) if x.strip()
                ]
            aliases = list(
                dict.fromkeys(
                    str(a).strip()
                    for a in raw_aliases
                    if str(a).strip() and str(a).strip().lower() != src.lower()
                )
            )
            raw_fix = item.get("fix_source")
            source_str = str(item.get("source") or "task")
            cleaned = {
                "src": src,
                "tgt": tgt,
                "aliases": aliases,
                "note": str(item.get("note") or "").strip(),
                "fix_source": bool(raw_fix)
                if raw_fix is not None
                else bool(aliases and source_str == "task"),
                "source": source_str,
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
        self.notify_artifact_update("terms", tag="llm", label="TERMS")
        return len(cleaned_terms)

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
        return {"summary": "", "terms": []}

    def has_terms(self) -> bool:
        with self._db(write=False) as conn:
            if conn is not None:
                row = conn.execute(
                    "SELECT (EXISTS(SELECT 1 FROM task_kv WHERE key = 'terms_meta') "
                    "OR EXISTS(SELECT 1 FROM terms LIMIT 1)) AS ok"
                ).fetchone()
                if row is not None and bool(row["ok"]):
                    return True
        return False

    def get_terms_mtime(self) -> float:
        mt = self.get_kv_mtime("terms_meta")
        return mt if mt > 0 else 0.0

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
        mark_translated: bool | None = None,
    ) -> int:
        ts = float(mtime) if mtime is not None else time.time()
        rows: list[tuple[Any, ...]] = []
        has_any_tgt = False
        has_any_cues = False
        for idx, u in enumerate(utterances):
            item = dict(u)
            item["index"] = int(item.get("index", idx))
            raw_id = item.get("id")
            utt_id = (
                str(raw_id)
                if raw_id is not None and str(raw_id) != ""
                else f"u_{idx:04d}"
            )
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
            translations_saved = (
                has_any_tgt if mark_translated is None else bool(mark_translated)
            )
            for k, v in (
                ("utterances_saved", True),
                ("utterances_revision", next_rev),
                ("translations_saved", translations_saved),
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
        self.notify_artifact_update(
            "utterances",
            tag="llm" if has_any_tgt else "nlp",
            label="TRANS" if has_any_tgt else "NLP",
        )
        return next_rev

    def flush_live_utterance_slice(
        self, utterances: list[dict[str, Any]], dirty_indices: list[int]
    ) -> int:
        """Incrementally update modified utterance rows during live translation without rewriting the full table."""
        if not dirty_indices or not self.db_path.is_file():
            return self.get_utterances_revision()
        ts = time.time()
        update_rows: list[tuple[str, str, str, int]] = []
        for idx in dirty_indices:
            if 0 <= idx < len(utterances):
                item = dict(utterances[idx])
                item["index"] = int(item.get("index", idx))
                tgt = str(item.get("tgt") or "")
                tts_text = str(item.get("tts_text") or tgt)
                update_rows.append(
                    (tgt, tts_text, json.dumps(item, ensure_ascii=False), idx)
                )
        if not update_rows:
            return self.get_utterances_revision()
        with self._db(write=True) as conn:
            assert conn is not None
            rev_row = conn.execute(
                "SELECT value_json FROM task_kv WHERE key = 'utterances_revision'"
            ).fetchone()
            next_rev = (int(json.loads(rev_row["value_json"])) + 1) if rev_row else 1
            conn.executemany(
                "UPDATE utterances SET tgt = ?, tts_text = ?, payload_json = ? WHERE idx = ?",
                update_rows,
            )
            for k, v in (
                ("utterances_saved", True),
                ("utterances_revision", next_rev),
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
        self.notify_artifact_update("utterances", tag="llm", label="TRANS")
        return next_rev

    def load_utterances(self) -> list[dict[str, Any]]:
        with self._db(write=False) as conn:
            if conn is not None:
                rows = conn.execute(
                    "SELECT payload_json FROM utterances ORDER BY idx ASC"
                ).fetchall()
                if rows:
                    return [json.loads(r["payload_json"]) for r in rows]
        return []

    def has_utterances(self) -> bool:
        with self._db(write=False) as conn:
            if conn is not None:
                row = conn.execute("SELECT 1 FROM utterances LIMIT 1").fetchone()
                if row is not None:
                    return True
        return False

    def has_translations(self) -> bool:
        with self._db(write=False) as conn:
            if conn is None:
                return False
            row = conn.execute(
                "SELECT (SELECT value_json FROM task_kv WHERE key = 'translations_saved') AS kv, "
                "EXISTS(SELECT 1 FROM utterances WHERE TRIM(tgt) != '' LIMIT 1) AS has_tgt"
            ).fetchone()
            if row is None:
                return False
            if row["kv"] is not None:
                return bool(json.loads(row["kv"]))
            return bool(row["has_tgt"])

    def has_sub_cues(self) -> bool:
        with self._db(write=False) as conn:
            if conn is None:
                return False
            row = conn.execute(
                "SELECT (SELECT value_json FROM task_kv WHERE key = 'sub_cues_saved') AS kv, "
                "EXISTS(SELECT 1 FROM utterances WHERE sub_cues_json != '[]' AND sub_cues_json != '' LIMIT 1) AS has_cues"
            ).fetchone()
            if row is None:
                return False
            if row["kv"] is not None and bool(json.loads(row["kv"])):
                return True
            return bool(row["has_cues"])

    def get_utterances_mtime(self) -> float:
        mt = self.get_kv_mtime("utterances_saved")
        return mt if mt > 0 else 0.0

    def get_utterances_revision(self) -> int:
        return int(self.get_kv("utterances_revision", 0) or 0)

    def get_utterances_cache_token(self) -> tuple[int, float, float]:
        if not self.db_path.is_file():
            return (0, 0.0, 0.0)
        u_rev = 0
        u_mtime = 0.0
        w_mtime = 0.0
        with self._db(write=False) as conn:
            if conn is not None:
                rows = conn.execute(
                    "SELECT key, value_json, updated_at FROM task_kv "
                    "WHERE key IN ('utterances_revision', 'utterances_saved', 'words_saved')"
                ).fetchall()
                for r in rows:
                    k = r["key"]
                    if k == "utterances_revision":
                        u_rev = int(json.loads(r["value_json"]) or 0)
                    elif k == "utterances_saved":
                        u_mtime = float(r["updated_at"] or 0.0)
                    elif k == "words_saved":
                        w_mtime = float(r["updated_at"] or 0.0)
        return (u_rev, u_mtime, w_mtime)

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

    def compute_stages_done(self) -> dict[str, bool]:
        has_w = False
        has_u = False
        has_t = False
        has_tr = False
        with self._db(write=False) as conn:
            if conn is not None:
                row = conn.execute(
                    """
                    SELECT
                        EXISTS(SELECT 1 FROM words LIMIT 1) AS has_w,
                        EXISTS(SELECT 1 FROM utterances LIMIT 1) AS has_u,
                        (EXISTS(SELECT 1 FROM task_kv WHERE key = 'terms_meta')
                         OR EXISTS(SELECT 1 FROM terms LIMIT 1)) AS has_t,
                        (SELECT value_json FROM task_kv WHERE key = 'translations_saved') AS trans_kv,
                        EXISTS(SELECT 1 FROM utterances WHERE TRIM(tgt) != '' LIMIT 1) AS has_tgt
                    """
                ).fetchone()
                if row is not None:
                    has_w = bool(row["has_w"])
                    has_u = bool(row["has_u"])
                    has_t = bool(row["has_t"])
                    has_tr = (
                        bool(json.loads(row["trans_kv"]))
                        if row["trans_kv"] is not None
                        else bool(row["has_tgt"])
                    )
        return {
            "asr": has_w,
            "split": has_u,
            "terms": has_t,
            "translate": has_tr,
            "subtitles": (self.root / "trans.srt").is_file()
            and (self.root / "src.srt").is_file(),
            "sub_video": (self.root / "output_sub.mp4").is_file(),
            "dub_audio": (self.audio_dir / "dub.wav").is_file()
            or (self.root / "dub.mp3").is_file(),
            "dub_video": (self.root / "output_dub.mp4").is_file(),
        }

    def notify_artifact_update(
        self,
        kind: Any,
        message: str | None = None,
        *,
        tag: Any = None,
        label: str | None = None,
    ) -> None:
        from core.task_runner import TaskRunner

        runner = TaskRunner._current
        if runner is not None and runner.active_task_id == self.task_id:
            TaskRunner.notify_artifact(
                kind,
                self.compute_stages_done(),
                message=message,
                tag=tag,
                label=label,
            )

    def is_step_done(self, step: str | Path) -> bool:
        key = Path(str(step).replace("\\", "/")).name
        if key == "asr":
            return self.has_words()
        if key == "split":
            return self.has_utterances()
        if key == "summarize":
            return self.has_terms()
        if key == "translate":
            return self.has_translations()
        if key == "sub_split":
            return self.has_sub_cues()
        if key == "dub_tasks":
            if not self.has_dub_chunks():
                return False
            dub_mtime = self.get_kv_mtime("dub_tasks_saved")
            for srt_p in (self.src_srt_path, self.trans_srt_path):
                if (
                    srt_p.is_file()
                    and dub_mtime > 0
                    and srt_p.stat().st_mtime > dub_mtime
                ):
                    return False
            return True
        return False
