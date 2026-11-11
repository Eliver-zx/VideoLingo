from __future__ import annotations

import json
from pathlib import Path
import re
import threading
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from core.api.helpers import _resolve_workspace, get_runner
from core.api.schemas import UtterancesUpdateRequest
from core.utils.config_utils import load_key, task_config_context
from core.workspace import TaskWorkspace, workspace_context
from core.workspace.fs_atomic import register_cleanup_hook

router = APIRouter()

_PUNCT_STRIP_RE = re.compile(r"[,.;:?!，。；：？！\"“”]+")
_MULTI_SPACE_RE = re.compile(r"\s+")

_WORDS_TOKEN_CACHE: dict[Path, tuple[float, list[str]]] = {}
_WORDS_SEGMENT_CACHE: dict[
    Path, tuple[float, list[dict[str, Any]], list[str], list[dict[str, Any]]]
] = {}
_UTTERANCES_RESPONSE_CACHE: dict[
    Path,
    tuple[
        tuple[int, float, float],
        list[dict[str, Any]],
        int,
        bytes | None,
        bytes | None,
    ],
] = {}
_WORDS_CACHE_LOCK = threading.Lock()


def _purge_words_cache(target_resolved: Path) -> None:
    with _WORDS_CACHE_LOCK:
        for cache in (
            _WORDS_TOKEN_CACHE,
            _WORDS_SEGMENT_CACHE,
            _UTTERANCES_RESPONSE_CACHE,
        ):
            for k in [
                p for p in cache if p == target_resolved or target_resolved in p.parents
            ]:
                cache.pop(k, None)


register_cleanup_hook(_purge_words_cache)


def _clean_asr_word_token(raw_text: Any) -> str:
    tok = str(raw_text or "").strip().strip(',.;:?!，。；：？！"“”')
    return tok if tok else str(raw_text or "").strip()


def _build_asr_segments_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not words:
        return []
    asr_utts: list[dict[str, Any]] = []
    chunk_start_idx = 0
    n = len(words)
    for i, w in enumerate(words):
        is_last = i == n - 1
        gap = (
            max(0.0, float(words[i + 1].get("start", 0.0)) - float(w.get("end", 0.0)))
            if not is_last
            else 0.0
        )
        word_count = i - chunk_start_idx + 1
        if is_last or gap >= 0.4 or word_count >= 20:
            slice_words = words[chunk_start_idx : i + 1]
            token_items = []
            raw_tokens = []
            for offset, sw in enumerate(slice_words):
                clean_tok = _clean_asr_word_token(sw.get("text", sw.get("word", "")))
                if not clean_tok:
                    continue
                raw_tokens.append(clean_tok)
                token_items.append(
                    {
                        "idx": chunk_start_idx + offset,
                        "text": clean_tok,
                        "start": round(float(sw.get("start", 0.0)), 3),
                        "end": round(float(sw.get("end", 0.0)), 3),
                    }
                )
            raw_str = " ".join(raw_tokens)
            s_sec = round(float(slice_words[0].get("start", 0.0)), 3)
            e_sec = round(float(slice_words[-1].get("end", s_sec)), 3)
            asr_utts.append(
                {
                    "id": len(asr_utts),
                    "word_span": [chunk_start_idx, i + 1],
                    "start": s_sec,
                    "end": e_sec,
                    "gap_after": round(gap, 3),
                    "cut_reason": "pause"
                    if (not is_last and gap >= 0.4)
                    else ("eof" if is_last else "window"),
                    "src_raw": raw_str,
                    "src_punct": "",
                    "src": raw_str,
                    "tgt": "",
                    "words": token_items,
                    "sub_cues": [],
                }
            )
            chunk_start_idx = i + 1
    return asr_utts


def _get_cached_word_tokens(
    ws: TaskWorkspace, w_mtime: float | None = None
) -> list[str]:
    if not ws.db_path.is_file():
        return []
    resolved = ws.db_path.resolve()
    mt = ws.get_words_mtime() if w_mtime is None else w_mtime
    if mt > 0:
        with _WORDS_CACHE_LOCK:
            seg_hit = _WORDS_SEGMENT_CACHE.get(resolved)
            if seg_hit is not None and seg_hit[0] == mt:
                return seg_hit[2]
            tok_hit = _WORDS_TOKEN_CACHE.get(resolved)
            if tok_hit is not None and tok_hit[0] == mt:
                return tok_hit[1]

    raw_texts = ws.load_word_texts()
    cleaned_tokens = [_clean_asr_word_token(t) for t in raw_texts]
    if mt > 0:
        with _WORDS_CACHE_LOCK:
            _WORDS_TOKEN_CACHE[resolved] = (mt, cleaned_tokens)
    return cleaned_tokens


def _get_cached_words_and_segments(
    ws: TaskWorkspace, w_mtime: float | None = None
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    if not ws.db_path.is_file():
        return [], [], []
    resolved = ws.db_path.resolve()
    mt = ws.get_words_mtime() if w_mtime is None else w_mtime
    if mt > 0:
        with _WORDS_CACHE_LOCK:
            hit = _WORDS_SEGMENT_CACHE.get(resolved)
            if hit is not None and hit[0] == mt:
                return hit[1], hit[2], hit[3]

    words = ws.load_words()
    cleaned_tokens = [
        _clean_asr_word_token(w.get("text", w.get("word", ""))) for w in words
    ]
    segments = _build_asr_segments_from_words(words)
    if mt > 0:
        with _WORDS_CACHE_LOCK:
            _WORDS_SEGMENT_CACHE[resolved] = (mt, words, cleaned_tokens, segments)
            _WORDS_TOKEN_CACHE[resolved] = (mt, cleaned_tokens)
    return words, cleaned_tokens, segments


def _enrich_utterances(
    utts: list[dict[str, Any]], cleaned_tokens: list[str]
) -> list[dict[str, Any]]:
    word_count = len(cleaned_tokens)
    for u in utts:
        span = u.get("word_span")
        raw_from_words = ""
        if (
            word_count > 0
            and isinstance(span, list)
            and len(span) == 2
            and 0 <= int(span[0]) < int(span[1]) <= word_count
        ):
            raw_from_words = " ".join(
                tok for tok in cleaned_tokens[int(span[0]) : int(span[1])] if tok
            )
        existing_raw = str(u.get("src_raw") or "").strip()
        if raw_from_words:
            u["src_raw"] = raw_from_words
        elif (
            not existing_raw
            or existing_raw == str(u.get("src_punct") or u.get("src") or "").strip()
        ):
            base_src = (
                existing_raw or str(u.get("src_punct") or u.get("src") or "").strip()
            )
            cleaned_raw = _PUNCT_STRIP_RE.sub("", base_src)
            u["src_raw"] = _MULTI_SPACE_RE.sub(" ", cleaned_raw).strip()
        if not str(u.get("src_punct") or "").strip():
            u["src_punct"] = str(u.get("src") or "").strip()
    return utts


def _encode_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


@router.get("/api/tasks/{task_id}/utterances")
def get_task_utterances(task_id: str, include_asr: bool = True):
    ws = _resolve_workspace(task_id)
    token = ws.get_utterances_cache_token()
    u_rev, u_mtime, w_mtime = token
    resolved = ws.db_path.resolve() if ws.db_path.is_file() else None

    if resolved is not None and (u_rev > 0 or u_mtime > 0):
        with _WORDS_CACHE_LOCK:
            cached_resp = _UTTERANCES_RESPONSE_CACHE.get(resolved)
        if cached_resp is not None and cached_resp[0] == token:
            cached_utts, total_words, no_asr_bytes, with_asr_bytes = (
                cached_resp[1],
                cached_resp[2],
                cached_resp[3],
                cached_resp[4],
            )
            if not include_asr and no_asr_bytes is not None:
                return Response(content=no_asr_bytes, media_type="application/json")
            if include_asr and with_asr_bytes is not None:
                return Response(content=with_asr_bytes, media_type="application/json")
            if include_asr:
                _, _, asr_segments = _get_cached_words_and_segments(ws, w_mtime)
                with_asr_bytes = _encode_json_bytes(
                    {
                        "utterances": cached_utts,
                        "asr_segments": asr_segments,
                        "total_words": total_words,
                    }
                )
                with _WORDS_CACHE_LOCK:
                    cur = _UTTERANCES_RESPONSE_CACHE.get(resolved)
                    if cur is not None and cur[0] == token:
                        _UTTERANCES_RESPONSE_CACHE[resolved] = (
                            token,
                            cached_utts,
                            total_words,
                            cur[3],
                            with_asr_bytes,
                        )
                return Response(content=with_asr_bytes, media_type="application/json")
            no_asr_bytes = _encode_json_bytes(
                {
                    "utterances": cached_utts,
                    "asr_segments": None,
                    "total_words": total_words,
                }
            )
            with _WORDS_CACHE_LOCK:
                cur = _UTTERANCES_RESPONSE_CACHE.get(resolved)
                if cur is not None and cur[0] == token:
                    _UTTERANCES_RESPONSE_CACHE[resolved] = (
                        token,
                        cached_utts,
                        total_words,
                        no_asr_bytes,
                        cur[4],
                    )
            return Response(content=no_asr_bytes, media_type="application/json")

    utts = ws.load_utterances()
    if utts:
        if include_asr:
            words, cleaned_tokens, asr_segments = _get_cached_words_and_segments(
                ws, w_mtime
            )
            total_words = len(words)
        else:
            cleaned_tokens = _get_cached_word_tokens(ws, w_mtime)
            asr_segments = None
            total_words = len(cleaned_tokens)
        _enrich_utterances(utts, cleaned_tokens)
        body = _encode_json_bytes(
            {
                "utterances": utts,
                "asr_segments": asr_segments if include_asr else None,
                "total_words": total_words,
            }
        )
        if resolved is not None and (u_rev > 0 or u_mtime > 0):
            with _WORDS_CACHE_LOCK:
                _UTTERANCES_RESPONSE_CACHE[resolved] = (
                    token,
                    utts,
                    total_words,
                    None if include_asr else body,
                    body if include_asr else None,
                )
        return Response(content=body, media_type="application/json")
    words, _, asr_segments = _get_cached_words_and_segments(ws, w_mtime)
    body = _encode_json_bytes(
        {
            "utterances": asr_segments,
            "asr_segments": asr_segments if include_asr else None,
            "total_words": len(words),
        }
    )
    return Response(content=body, media_type="application/json")


@router.put("/api/tasks/{task_id}/utterances")
def update_task_utterances(task_id: str, request: UtterancesUpdateRequest):
    runner = get_runner()
    ws = _resolve_workspace(task_id)
    existing_utts = ws.load_utterances()
    words = ws.load_words() if ws.has_words() else []
    manifest = ws.load_manifest()
    overrides = dict(manifest.get("config_override") or {})

    def _build_split_cues(
        item_dict: dict[str, Any], s_parts: list[str], t_parts: list[str]
    ) -> list[dict[str, Any]]:
        if words:
            from core._5_split_sub import _partition_word_span_for_cues

            span = item_dict.get("word_span") or [0, len(words)]
            w_a = int(span[0]) if len(span) >= 2 else 0
            w_b = int(span[1]) if len(span) >= 2 else len(words)
            part_cues = _partition_word_span_for_cues(w_a, w_b, s_parts, t_parts, words)
            if part_cues:
                return [{**c, "trans": str(c.get("tgt") or "")} for c in part_cues]
        return [
            {
                "src": sp,
                "tgt": tp,
                "trans": tp,
                "word_span": list(item_dict.get("word_span") or [0, 0]),
                "start": float(item_dict.get("start") or 0.0),
                "end": float(item_dict.get("end") or 0.0),
            }
            for sp, tp in zip(s_parts, t_parts)
        ]

    normalized_utts = []
    with workspace_context(ws), task_config_context(overrides):
        sub_cfg = load_key("subtitle")
        max_len = float(sub_cfg["max_length"])
        tgt_mult = float(sub_cfg["target_multiplier"])
        local_splitter = None

        for idx, u in enumerate(request.utterances):
            item = dict(u)
            orig_u = existing_utts[idx] if idx < len(existing_utts) else {}
            if "src_punct" not in item and orig_u.get("src_punct"):
                item["src_punct"] = orig_u["src_punct"]
            src = str(item.get("src") or "").strip()
            tgt = str(item.get("tgt") or "").strip()
            tts_text = str(item.get("tts_text") or tgt).strip()
            item["src"] = src
            if src != str(orig_u.get("src") or "").strip() or not item.get("src_punct"):
                item["src_punct"] = src
            item["tgt"] = tgt
            item["tts_text"] = tts_text
            cues = item.get("sub_cues")
            default_cue = {
                "src": src,
                "tgt": tgt,
                "trans": tgt,
                "word_span": list(item.get("word_span") or [0, 0]),
                "start": float(item.get("start") or 0.0),
                "end": float(item.get("end") or 0.0),
            }
            if isinstance(cues, list) and len(cues) == 1:
                c0 = dict(cues[0]) if isinstance(cues[0], dict) else {}
                prev_s = str(c0.get("src") or "").strip()
                prev_t = str(c0.get("tgt", c0.get("trans", "")) or "").strip()
                if (src != prev_s or tgt != prev_t) and tgt:
                    if local_splitter is None:
                        from core._5_split_sub import _try_local_clause_split

                        local_splitter = _try_local_clause_split
                    local_split = local_splitter(src, tgt, max_len, tgt_mult)
                    if local_split is not None:
                        s_parts, t_parts, _ = local_split
                        item["sub_cues"] = _build_split_cues(item, s_parts, t_parts)
                    else:
                        item["sub_cues"] = [
                            {**default_cue, **c0, "src": src, "tgt": tgt, "trans": tgt}
                        ]
                else:
                    item["sub_cues"] = [
                        {**default_cue, **c0, "src": src, "tgt": tgt, "trans": tgt}
                    ]
            elif isinstance(cues, list) and len(cues) > 1:
                prev_src = " ".join(
                    str(c.get("src") or "").strip() for c in cues if c.get("src")
                ).strip()
                prev_tgt = " ".join(
                    str(c.get("tgt", c.get("trans", c.get("text", ""))) or "").strip()
                    for c in cues
                    if c.get("tgt") or c.get("trans") or c.get("text")
                ).strip()
                if src != prev_src or tgt != prev_tgt:
                    if local_splitter is None:
                        from core._5_split_sub import _try_local_clause_split

                        local_splitter = _try_local_clause_split
                    local_split = local_splitter(src, tgt, max_len, tgt_mult)
                    if local_split is not None:
                        s_parts, t_parts, _ = local_split
                        item["sub_cues"] = _build_split_cues(item, s_parts, t_parts)
                    else:
                        item["sub_cues"] = [default_cue]
            normalized_utts.append(item)
        ws.save_utterances(normalized_utts)
        ws.delete_kv("sub_split_cfg", "sub_split_rev")
        if ws.has_dub_tasks():
            ws.invalidate_from_stage("dub_tts")
        is_task_active = runner.active_task_id == ws.task_id and runner.is_active
        if (
            normalized_utts
            and not is_task_active
            and ws.has_words()
            and (ws.src_srt_path.is_file() or ws.trans_srt_path.is_file())
        ):
            from core._6_gen_sub import align_timestamp_main

            try:
                align_timestamp_main()
            except Exception as exc:
                raise HTTPException(
                    500, f"Failed to realign subtitle files: {exc}"
                ) from exc
    return {"saved": len(normalized_utts)}


@router.post("/api/tasks/{task_id}/subtitles/sync")
def sync_task_subtitles_from_srt(task_id: str):
    ws = _resolve_workspace(task_id)
    manifest = ws.load_manifest()
    overrides = dict(manifest.get("config_override") or {})
    from core._6_gen_sub import sync_subtitles_from_srt

    try:
        with workspace_context(ws), task_config_context(overrides):
            result = sync_subtitles_from_srt()
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return result
