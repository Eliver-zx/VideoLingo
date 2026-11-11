from __future__ import annotations

from bisect import bisect_left, bisect_right
import os
import re
import time
from typing import Any

from core.subtitle_gen.formatter import convert_to_srt_format, format_subtitle_text
from core.utils.models import _OUTPUT_DIR, resolve_task_path
from core.workspace import get_active_workspace

_SRT_TIMECODE_RE = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
)
_SRT_TAGS_RE = re.compile(r"<[^>]*>|\{\\[^}]*\}")


def _parse_srt_blocks(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        raw_bytes = f.read()
    try:
        text = raw_bytes.decode(
            "utf-16"
            if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff"))
            else "utf-8-sig"
        )
    except UnicodeDecodeError:
        text = raw_bytes.decode("utf-8", errors="replace")

    blocks: list[tuple[int, float, float, list[str]]] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _SRT_TIMECODE_RE.search(line)
        if m:
            cue_num = len(blocks) + 1
            if i > 0 and lines[i - 1].strip().isdigit():
                cue_num = int(lines[i - 1].strip())
                if blocks:
                    prev_blank = i >= 2 and not lines[i - 2].strip()
                    non_blank_prev = [ln for ln in blocks[-1][3] if ln.strip()]
                    if prev_blank or (
                        len(non_blank_prev) > 1 and cue_num <= len(blocks) + 2
                    ):
                        blocks[-1][3].pop()
            h1, m1, s1, ms1, h2, m2, s2, ms2 = m.groups()
            start = round(
                int(h1) * 3600 + int(m1) * 60 + float(f"{s1}.{ms1.ljust(3, '0')[:3]}"),
                3,
            )
            end = round(
                int(h2) * 3600 + int(m2) * 60 + float(f"{s2}.{ms2.ljust(3, '0')[:3]}"),
                3,
            )
            blocks.append((cue_num, start, end, []))
        elif blocks:
            blocks[-1][3].append(line)

    parsed: list[dict[str, Any]] = []
    for num, start, end, raw_lines in blocks:
        cleaned = " ".join(
            _SRT_TAGS_RE.sub("", ln).strip() for ln in raw_lines if ln.strip()
        ).strip()
        parsed.append(
            {
                "number": num,
                "start": start,
                "end": end,
                "timestamp": convert_to_srt_format(start, end),
                "text": cleaned,
            }
        )
    return parsed


def _utt_distance(start: float, end: float, mid: float) -> float:
    if start <= mid <= end:
        return 0.0
    return min(abs(mid - start), abs(mid - end))


def _bucket_cues_by_utterances(
    utts: list[dict[str, Any]], merged_cues: list[dict[str, Any]]
) -> list[list[dict[str, Any]]]:
    n = len(utts)
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    starts = [float(u.get("start", 0.0)) for u in utts]
    ends = [float(u.get("end", 0.0)) for u in utts]
    is_monotonic = all(starts[k] <= ends[k] for k in range(n)) and all(
        starts[k] <= starts[k + 1] and ends[k] <= ends[k + 1] for k in range(n - 1)
    )

    if is_monotonic:
        for mc in merged_cues:
            mid = (mc["start"] + mc["end"]) / 2.0
            pos = bisect_right(starts, mid)
            first_containing = bisect_left(ends, mid, 0, pos)
            if first_containing < pos:
                best_idx = first_containing
            else:
                cands: list[int] = []
                if pos > 0:
                    cands.append(bisect_left(ends, ends[pos - 1], 0, pos))
                if pos < n:
                    cands.append(pos)
                best_idx = min(
                    cands,
                    key=lambda idx: (_utt_distance(starts[idx], ends[idx], mid), idx),
                )
            buckets[best_idx].append(mc)
    else:
        for mc in merged_cues:
            mid = (mc["start"] + mc["end"]) / 2.0
            best_idx = min(
                range(n),
                key=lambda idx: _utt_distance(starts[idx], ends[idx], mid),
            )
            buckets[best_idx].append(mc)
    return buckets


def sync_subtitles_from_srt(
    out_dir: str | None = None, ws: Any = None
) -> dict[str, int]:
    """Sync edited `src.srt` and `trans.srt` from disk back into `project.sqlite` and bilingual SRT files."""
    if out_dir is None:
        out_dir = resolve_task_path(_OUTPUT_DIR)
    if ws is None:
        ws = get_active_workspace()
    src_path = os.path.join(out_dir, "src.srt")
    trans_path = os.path.join(out_dir, "trans.srt")

    has_src = os.path.isfile(src_path)
    has_trans = os.path.isfile(trans_path)
    if not has_src and not has_trans:
        raise FileNotFoundError("未找到 src.srt 或 trans.srt，无法同步字幕。")

    src_blocks = _parse_srt_blocks(src_path) if has_src else []
    trans_blocks = _parse_srt_blocks(trans_path) if has_trans else []
    if not src_blocks and not trans_blocks:
        raise ValueError("SRT 字幕文件为空或格式无效。")

    src_mtime = os.path.getmtime(src_path) if has_src else 0.0
    trans_mtime = os.path.getmtime(trans_path) if has_trans else 0.0
    prefer_src_ts = has_src and (not has_trans or src_mtime > trans_mtime + 1e-3)

    merged_cues: list[dict[str, Any]] = []
    if src_blocks and trans_blocks:
        if len(src_blocks) == len(trans_blocks):
            for sb, tb in zip(src_blocks, trans_blocks):
                ts_block = sb if prefer_src_ts else tb
                merged_cues.append(
                    {
                        "src": sb["text"],
                        "tgt": tb["text"],
                        "start": ts_block["start"],
                        "end": ts_block["end"],
                        "timestamp": ts_block["timestamp"],
                    }
                )
        else:
            src_by_num = {sb["number"]: sb for sb in src_blocks}
            for idx, tb in enumerate(trans_blocks):
                sb = src_by_num.get(tb["number"])
                if sb is None and idx < len(src_blocks):
                    sb = src_blocks[idx]
                merged_cues.append(
                    {
                        "src": sb["text"] if sb else "",
                        "tgt": tb["text"],
                        "start": tb["start"],
                        "end": tb["end"],
                        "timestamp": tb["timestamp"],
                    }
                )
    elif src_blocks:
        for sb in src_blocks:
            merged_cues.append(
                {
                    "src": sb["text"],
                    "tgt": "",
                    "start": sb["start"],
                    "end": sb["end"],
                    "timestamp": sb["timestamp"],
                }
            )
    else:
        for tb in trans_blocks:
            merged_cues.append(
                {
                    "src": "",
                    "tgt": tb["text"],
                    "start": tb["start"],
                    "end": tb["end"],
                    "timestamp": tb["timestamp"],
                }
            )

    utts = ws.load_utterances()
    existing_cue_total = sum(
        max(
            1,
            len(u.get("sub_cues") or []) if isinstance(u.get("sub_cues"), list) else 1,
        )
        for u in utts
    )

    if utts and existing_cue_total == len(merged_cues):
        cursor = 0
        for u in utts:
            raw_cues = u.get("sub_cues")
            old_cues = (
                raw_cues
                if isinstance(raw_cues, list) and raw_cues
                else [
                    {
                        "src": u.get("src", ""),
                        "tgt": u.get("tgt", ""),
                        "word_span": list(u.get("word_span") or [0, 0]),
                        "start": float(u.get("start") or 0.0),
                        "end": float(u.get("end") or 0.0),
                    }
                ]
            )
            new_cues: list[dict[str, Any]] = []
            src_changed = False
            tgt_changed = False
            for oc in old_cues:
                mc = merged_cues[cursor]
                cursor += 1
                old_s = str(oc.get("src") or "").strip()
                old_t = str(oc.get("tgt", oc.get("trans", "")) or "").strip()
                if mc["src"] != old_s:
                    src_changed = True
                if mc["tgt"] != old_t and mc["tgt"] != format_subtitle_text(old_t):
                    tgt_changed = True
                new_cues.append(
                    {
                        **oc,
                        "src": mc["src"],
                        "tgt": mc["tgt"],
                        "trans": mc["tgt"],
                        "start": mc["start"],
                        "end": mc["end"],
                    }
                )
            u["sub_cues"] = new_cues
            u["start"] = new_cues[0]["start"]
            u["end"] = new_cues[-1]["end"]
            if src_changed or not str(u.get("src") or "").strip():
                u["src"] = " ".join(c["src"] for c in new_cues if c["src"]).strip()
                u["src_punct"] = u["src"]
            if tgt_changed or not str(u.get("tgt") or "").strip():
                u["tgt"] = " ".join(c["tgt"] for c in new_cues if c["tgt"]).strip()
                u["tts_text"] = u["tgt"]
        utts_out = utts
    elif utts:
        buckets = _bucket_cues_by_utterances(utts, merged_cues)
        utts_out = []
        for u, b_cues in zip(utts, buckets):
            if not b_cues:
                continue
            item = dict(u)
            w_span = list(item.get("word_span") or [0, 0])
            new_cues = [
                {
                    "src": mc["src"],
                    "tgt": mc["tgt"],
                    "trans": mc["tgt"],
                    "word_span": w_span,
                    "start": mc["start"],
                    "end": mc["end"],
                }
                for mc in b_cues
            ]
            item["sub_cues"] = new_cues
            item["start"] = new_cues[0]["start"]
            item["end"] = new_cues[-1]["end"]
            item["src"] = " ".join(c["src"] for c in new_cues if c["src"]).strip()
            item["src_punct"] = item["src"]
            item["tgt"] = " ".join(c["tgt"] for c in new_cues if c["tgt"]).strip()
            item["tts_text"] = item["tgt"]
            utts_out.append(item)
    else:
        utts_out = [
            {
                "id": f"u_{i:04d}",
                "index": i,
                "start": mc["start"],
                "end": mc["end"],
                "gap_after": 0.0,
                "word_span": [0, 0],
                "src": mc["src"],
                "src_punct": mc["src"],
                "tgt": mc["tgt"],
                "tts_text": mc["tgt"],
                "sub_cues": [
                    {
                        "src": mc["src"],
                        "tgt": mc["tgt"],
                        "trans": mc["tgt"],
                        "word_span": [0, 0],
                        "start": mc["start"],
                        "end": mc["end"],
                    }
                ],
            }
            for i, mc in enumerate(merged_cues)
        ]

    sync_ts = max(time.time(), src_mtime, trans_mtime)
    rev = ws.save_utterances(utts_out, mtime=sync_ts)
    ws.set_kv("sub_split_rev", rev, mtime=sync_ts)
    ws.set_kv("srt_synced_mtime", max(src_mtime, trans_mtime), mtime=sync_ts)
    if ws.has_dub_tasks():
        ws.invalidate_from_stage("dub_tts")

    if trans_blocks:
        src_trans_str = "".join(
            f"{i + 1}\n{mc['timestamp']}\n{mc['src']}\n{mc['tgt']}\n\n"
            for i, mc in enumerate(merged_cues)
        ).strip()
        trans_src_str = "".join(
            f"{i + 1}\n{mc['timestamp']}\n{mc['tgt']}\n{mc['src']}\n\n"
            for i, mc in enumerate(merged_cues)
        ).strip()
        with open(os.path.join(out_dir, "src_trans.srt"), "w", encoding="utf-8") as f:
            f.write(src_trans_str)
        with open(os.path.join(out_dir, "trans_src.srt"), "w", encoding="utf-8") as f:
            f.write(trans_src_str)

    ws.notify_artifact_update(
        "subtitles",
        message=f"已从 SRT 同步更新 {len(merged_cues)} 轴字幕",
        tag="llm",
        label="SUBS",
    )
    return {"cues": len(merged_cues), "utterances": len(utts_out)}


def sync_subtitles_from_srt_if_newer(
    out_dir: str | None = None, ws: Any = None
) -> bool:
    """Auto-sync `src.srt` / `trans.srt` into SQLite if modified on disk after the last DB/SRT save."""
    if out_dir is None:
        out_dir = resolve_task_path(_OUTPUT_DIR)
    if ws is None:
        ws = get_active_workspace()
    src_path = os.path.join(out_dir, "src.srt")
    trans_path = os.path.join(out_dir, "trans.srt")
    mtimes = [os.path.getmtime(p) for p in (src_path, trans_path) if os.path.isfile(p)]
    if not mtimes:
        return False
    baseline_mtime = max(
        ws.get_utterances_mtime(),
        float(ws.get_kv("srt_synced_mtime", 0.0) or 0.0),
    )
    if max(mtimes) > baseline_mtime + 1e-3:
        sync_subtitles_from_srt(out_dir=out_dir, ws=ws)
        return True
    return False
