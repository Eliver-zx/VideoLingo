"""Verification tests for the VideoLingo Workspace, 3-JSON Contracts, Full-Pipeline Quality Upgrades, and FastAPI + SSE Web UI."""

from __future__ import annotations

from pathlib import Path
import re
from unittest.mock import patch
import wave

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from pydub import AudioSegment

import api
from core import (
    _3_1_split_nlp,
    _4_1_summarize,
    _5_split_sub,
    _10_gen_audio,
    _11_merge_audio,
    _12_dub_to_vid,
)
from core.tts_backend import tts_main as tts_main_mod
from core.utils import config_utils
from core.workspace import (
    TaskWorkspace,
    json_content_fingerprint,
    load_global_glossary,
    promote_terms_to_global,
    workspace_context,
)


def test_zero_copy_local_media_survives_source_move_and_delete(tmp_path: Path):
    """Mounting a local file via init_from_local_file must survive moving or deleting the original source file."""
    src_dir = tmp_path / "downloads"
    src_dir.mkdir()
    original = src_dir / "lecture_video.mp4"
    payload = b"fake-mp4-video-stream-" * 2048
    original.write_bytes(payload)

    ws = TaskWorkspace(tmp_path / "output" / "20261023_lecture")
    manifest = ws.init_from_local_file(original, title="Lecture Video")

    mounted = ws.root / str(manifest["media"]["filename"])
    assert mounted.is_file()
    assert not mounted.is_symlink()
    assert manifest["origin"]["mount_mode"] in ("clonefile", "hardlink", "copy")

    # Move the user's original file and then delete it completely
    moved = src_dir / "moved_lecture.mp4"
    original.rename(moved)
    moved.unlink()
    assert not original.exists() and not moved.exists()

    # Task workspace file must still exist and have identical bytes
    assert mounted.is_file()
    assert mounted.read_bytes() == payload
    resolved_path, media_type = ws.resolve_media_file()
    assert Path(resolved_path) == mounted
    assert media_type == "video"


def test_config_schema_backfill_and_thread_local_task_override(
    tmp_path: Path, monkeypatch
):
    """ensure_config_schema backfills missing keys; task_config_context isolates task overrides."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "display_language: zh-CN\nwhisper:\n  language: en\n", encoding="utf-8"
    )
    monkeypatch.setattr(config_utils, "CONFIG_PATH", str(cfg_path))

    changed = config_utils.ensure_config_schema()
    assert changed is True
    assert config_utils.load_key("burn_subtitles") is True

    # Thread-local override must not mutate disk config.yaml
    with config_utils.task_config_context(
        {"target_language": " Français ", "burn_subtitles": False}
    ):
        assert config_utils.load_key("target_language") == " Français "
        assert config_utils.load_key("burn_subtitles") is False

    assert config_utils.load_key("burn_subtitles") is True


def test_stage_fingerprint_dag_cascading_invalidation(tmp_path: Path):
    """Invalidating a stage cascades to all downstream stages and cleans audio/tmp on TTS invalidation."""
    ws = TaskWorkspace(tmp_path / "output" / "task_dag")
    ws.ensure_dirs()

    fp_translate = json_content_fingerprint(
        {"stage": "translate", "target_language": "zh"}
    )
    fp_split = json_content_fingerprint({"stage": "sub_split", "max_length": 75})
    fp_tts = json_content_fingerprint({"stage": "dub_tts", "tts_method": "azure_tts"})

    ws.record_stage_fingerprint("translate", fp_translate)
    ws.record_stage_fingerprint("sub_split", fp_split)
    ws.record_stage_fingerprint("dub_tts", fp_tts)

    # Create dummy downstream artifacts and TTS scratch files
    (ws.root / "trans.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8"
    )
    (ws.tmp_dir / "1_temp.wav").write_bytes(b"RIFF")
    (ws.segs_dir / "1.wav").write_bytes(b"RIFF")

    assert ws.is_stage_fresh("translate", fp_translate, [])
    assert ws.is_stage_fresh("sub_split", fp_split, [ws.root / "trans.srt"])
    assert ws.is_stage_fresh("dub_tts", fp_tts, [])

    invalidated = ws.invalidate_from_stage("sub_split")
    assert "sub_split" in invalidated
    assert "sub_video" in invalidated
    assert "dub_tts" in invalidated
    assert ws.is_stage_fresh("translate", fp_translate, [])
    assert not ws.is_stage_fresh("sub_split", fp_split, [])
    assert not ws.is_stage_fresh("dub_tts", fp_tts, [])
    assert not (ws.root / "trans.srt").exists()
    assert not list(ws.tmp_dir.iterdir())
    assert not list(ws.segs_dir.iterdir())


def test_acoustic_pause_prompt_internal_commas_and_long_sentence_split():
    """Step 3 injects ‖ on >=0.4s gaps, projects internal commas onto punct_after, and splits Plan B."""
    word_dicts = [
        {"idx": 0, "text": "Hello", "start": 0.0, "end": 0.4},
        {
            "idx": 1,
            "text": "world",
            "start": 0.9,
            "end": 1.3,
        },  # gap before idx 1 is 0.5s >= 0.4s
        {"idx": 2, "text": "Plan", "start": 1.4, "end": 1.7},
        {
            "idx": 3,
            "text": "B",
            "start": 1.7,
            "end": 2.0,
        },  # gap after "B" is 0.6s >= 0.3s
        {"idx": 4, "text": "works", "start": 2.6, "end": 3.0},
    ]
    captured_prompts: list[str] = []

    def fake_ask_gpt(prompt: str, **_kwargs):
        captured_prompts.append(prompt)
        return {"restored": "Hello, world! Plan B. Works."}

    with patch.object(_3_1_split_nlp, "ask_gpt", side_effect=fake_ask_gpt):
        sents = _3_1_split_nlp.restore_and_split_words(
            [w["text"] for w in word_dicts],
            language="en",
            word_dicts=word_dicts,
        )

    assert captured_prompts and "‖" in captured_prompts[0]
    assert word_dicts[0]["punct_after"] == ","
    assert word_dicts[1]["punct_after"] == "!"
    assert word_dicts[3]["punct_after"] == "."
    assert sents == ["Hello, world!", "Plan B.", "works."]


def test_many_to_one_aliases_in_glossary_and_global_promotion(tmp_path: Path):
    """Task terms aliases match multi-word ASR variants (e.g. 'pie torch' -> 'PyTorch') and promote to global CSV."""
    ws = TaskWorkspace(tmp_path / "output")
    ws.ensure_dirs()
    terms = [
        {
            "src": "PyTorch",
            "tgt": "PyTorch框架",
            "aliases": ["pie torch", "Pi Torch"],
            "note": "Deep learning library",
            "fix_source": True,
        }
    ]
    ws.save_terms("DL video", terms)

    with workspace_context(ws):
        note_prompt = _4_1_summarize.search_things_to_note_in_prompt(
            "We trained the model using pie torch yesterday."
        )
    assert note_prompt is not None
    assert "PyTorch" in note_prompt
    assert "PyTorch框架" in note_prompt

    glossary_csv = tmp_path / "custom_terms.csv"
    count = promote_terms_to_global(terms, csv_path=glossary_csv)
    assert count == 1
    loaded = load_global_glossary(glossary_csv)
    assert len(loaded) == 1
    assert loaded[0]["src"] == "PyTorch"
    assert "pie torch" in loaded[0]["aliases"]


def test_proportional_subtitle_split_reruns_on_max_length_increase(
    tmp_path: Path, monkeypatch
):
    """Step 5 records (max_length, multiplier) in project.sqlite utterances and re-runs when max_length increases."""
    root_dir = Path(__file__).resolve().parents[1]
    root_cfg = (
        root_dir / "config.example.yaml"
        if (root_dir / "config.example.yaml").is_file()
        else root_dir / "config.yaml"
    )
    (tmp_path / "config.yaml").write_text(
        root_cfg.read_text(encoding="utf-8"), encoding="utf-8"
    )
    ws = TaskWorkspace(tmp_path / "output")
    ws.ensure_dirs()
    monkeypatch.chdir(tmp_path)

    words = [
        {
            "idx": i,
            "text": f"word{i}",
            "punct_after": "," if i == 5 else "",
            "start": float(i),
            "end": float(i) + 0.4,
        }
        for i in range(12)
    ]
    ws.save_words(words)
    utterances = [
        {
            "id": 0,
            "word_span": [0, 12],
            "start": 0.0,
            "end": 11.4,
            "gap_after": 0.2,
            "src_raw": " ".join(w["text"] for w in words),
            "src": " ".join(w["text"] for w in words),
            "direct": "这是第一段非常长的中文字幕内容，这是第二段非常长的中文字幕内容。",
            "tgt": "这是第一段非常长的中文字幕内容，这是第二段非常长的中文字幕内容。",
            "tts_text": "这是第一段非常长的中文字幕内容，这是第二段非常长的中文字幕内容。",
        }
    ]
    ws.save_utterances(utterances)

    with (
        workspace_context(ws),
        config_utils.task_config_context(
            {"subtitle": {"max_length": 24, "target_multiplier": 1.0}}
        ),
    ):
        _5_split_sub.split_for_sub_main()
        saved_utts = ws.load_utterances()
        assert len(saved_utts[0]["sub_cues"]) >= 2
        assert saved_utts[0]["sub_cues"][0]["start"] == 0.0
        assert saved_utts[0]["sub_cues"][-1]["end"] == 11.4

    # Increasing max_length to 120 must re-split into a single cue instead of skipping!
    with (
        workspace_context(ws),
        config_utils.task_config_context(
            {"subtitle": {"max_length": 120, "target_multiplier": 1.2}}
        ),
    ):
        _5_split_sub.split_for_sub_main()
        saved_utts_wide = ws.load_utterances()
        assert len(saved_utts_wide[0]["sub_cues"]) == 1


def test_subtitle_style_filters_use_src_and_trans_srt_only(tmp_path: Path):
    """get_subtitle_filters references src.srt and trans.srt directly without intermediate bi.ass files."""
    from core.utils.subtitle_style import get_subtitle_filters

    src_srt = tmp_path / "src.srt"
    trans_srt = tmp_path / "trans.srt"
    src_srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nHello vertical world\n\n", encoding="utf-8"
    )
    trans_srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n你好竖屏世界\n\n", encoding="utf-8"
    )

    with config_utils.task_config_context(
        {"subtitle": {"sub_video_subtitles": "bilingual"}}
    ):
        filters = get_subtitle_filters("subtitle", str(src_srt), str(trans_srt))
    with config_utils.task_config_context(
        {"subtitle": {"sub_video_subtitles": "translation"}}
    ):
        win_filters = get_subtitle_filters(
            "subtitle",
            r"E:\RSM\python\VideoLingo\output\src.srt",
            r"E:\RSM\python\VideoLingo\output\trans.srt",
        )
    assert "src.srt" in filters
    assert "trans.srt" in filters
    assert "bi.ass" not in filters
    assert (
        r"subtitles=E\\:/RSM/python/VideoLingo/output/trans.srt:force_style="
        in win_filters
    )


def test_single_char_tts_and_edge_silence_trimming(tmp_path: Path, monkeypatch):
    """Single-character utterances ('好') call TTS backend instead of generating 100ms silence; edge silence is trimmed."""
    called_texts: list[str] = []

    def fake_openai_tts(text: str, save_as: str):
        called_texts.append(text)
        # Generate 200ms leading silence + 300ms tone + 200ms trailing silence
        sr = 16000
        t = np.linspace(0, 0.3, int(sr * 0.3), endpoint=False)
        tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        silence = np.zeros(int(sr * 0.2), dtype=np.int16)
        samples = np.concatenate([silence, tone, silence])
        with wave.open(str(save_as), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(samples.tobytes())

    monkeypatch.setattr(
        tts_main_mod, "load_key", lambda k: "openai_tts" if k == "tts_method" else None
    )
    monkeypatch.setattr(tts_main_mod, "openai_tts", fake_openai_tts)

    wav_path = tmp_path / "single_char.wav"
    tts_main_mod.tts_main("好", str(wav_path), 1, pd.DataFrame())
    assert called_texts == ["好"]
    assert wav_path.is_file()

    before_dur = len(AudioSegment.from_wav(str(wav_path))) / 1000.0
    _10_gen_audio.trim_edge_silence(str(wav_path))
    after_dur = len(AudioSegment.from_wav(str(wav_path))) / 1000.0
    assert before_dur > 0.65
    assert 0.30 <= after_dur <= 0.38


def test_lossless_dub_wav_concatenation_and_step12_stream_copy(
    tmp_path: Path, monkeypatch
):
    """Step 11 writes lossless dub.wav without cumulative drift; Step 12 reuses output_sub.mp4 with -c:v copy."""
    seg_path = tmp_path / "seg1.wav"
    AudioSegment.silent(duration=500, frame_rate=32000).export(
        str(seg_path), format="wav"
    )
    seg = _11_merge_audio.process_audio_segment(str(seg_path))
    assert seg.frame_rate == 32000
    assert seg.channels == 1

    # Verify Step 12 video stream reuse (-c:v copy) when output_sub.mp4 exists and burn modes match
    vid = tmp_path / "source.mp4"
    vid.write_bytes(b"video")
    sub_mp4 = tmp_path / "output_sub.mp4"
    sub_mp4.write_bytes(b"subbed-video")
    bg_mp3 = tmp_path / "background.mp3"
    bg_mp3.write_bytes(b"bg")
    dub_wav = tmp_path / "dub.wav"
    AudioSegment.silent(duration=1000, frame_rate=32000).export(
        str(dub_wav), format="wav"
    )
    dub_srt = tmp_path / "dub.srt"
    dub_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
    trans_srt = tmp_path / "trans.srt"
    trans_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
    ws_tmp = TaskWorkspace(tmp_path)

    monkeypatch.setattr(_12_dub_to_vid, "find_media_file", lambda: (str(vid), "video"))
    monkeypatch.setattr(_12_dub_to_vid, "_DUB_WAV_FILE", str(dub_wav))
    monkeypatch.setattr(_12_dub_to_vid, "DUB_SUB_FILE", str(dub_srt))
    monkeypatch.setattr(_12_dub_to_vid, "DUB_VIDEO", str(tmp_path / "output_dub.mp4"))
    monkeypatch.setattr(_12_dub_to_vid, "SUB_VIDEO", str(sub_mp4))
    monkeypatch.setattr(_12_dub_to_vid, "_BACKGROUND_AUDIO_FILE", str(bg_mp3))
    monkeypatch.setattr(
        _12_dub_to_vid,
        "normalize_dub_audio",
        lambda src, dst: Path(dst).write_bytes(b"norm"),
    )

    ffmpeg_calls: list[list[str]] = []
    monkeypatch.setattr(
        _12_dub_to_vid.subprocess,
        "run",
        lambda cmd, **_kw: ffmpeg_calls.append(list(cmd)),
    )

    with (
        workspace_context(ws_tmp),
        config_utils.task_config_context(
            {
                "burn_subtitles": True,
                "subtitle.dub_video_subtitles": "bilingual",
                "subtitle.sub_video_subtitles": "bilingual",
            }
        ),
    ):
        _12_dub_to_vid.merge_video_audio()

    assert ffmpeg_calls
    cmd = ffmpeg_calls[0]
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
    filter_arg = cmd[cmd.index("-filter_complex") + 1]
    assert (
        "amix=inputs=2:duration=longest:dropout_transition=0:normalize=0" in filter_arg
    )


def test_fastapi_webui_tasks_terms_and_utterances_endpoints(
    tmp_path: Path, monkeypatch
):
    """FastAPI server serves Web UI, mounts local video into task workspace, and round-trips terms & utterances."""
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr("core.workspace.OUTPUT_ROOT", output_root)
    monkeypatch.setattr(api, "OUTPUT", output_root)

    sample_video = tmp_path / "demo.mp4"
    sample_video.write_bytes(b"fake-video-content-12345")

    client = TestClient(api.app)

    # 1. Web UI root & parity controls
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert not re.findall(
        r"[\U0001F300-\U0001F9FF\u2600-\u27BF\U0001FA00-\U0001FAFF]",
        res_root.text,
    ), "Spec violation: Zero-emoji constraint failed on GET /"

    required_dom_anchors = [
        # Topbar & Shell
        "VideoLingo",
        'data-theme="light"',
        'id="tab-workbench"',
        'id="nav-task-count"',
        'id="tab-glossary"',
        'id="tab-settings"',
        'id="autosave-indicator"',
        'class="brand-logo"',
        'class="studio-shell"',
        # View 1: Workbench & Left Task Rail
        'id="view-workbench"',
        'id="btn-toggle-new-task"',
        'id="task-rail-list"',
        'id="detail-task-title"',
        'id="detail-task-badge"',
        'id="detail-domain-pill"',
        'id="btn-start-resume-task"',
        'id="btn-start-resume-label"',
        'id="btn-open-task-dir"',
        'id="btn-burn-sub-video"',
        'id="btn-run-dubbing"',
        'id="btn-preview-video-modal"',
        'id="btn-stop-task"',
        'id="btn-open-output-root"',
        # Standalone Burn & Unified Settings Form
        'id="tab-burn"',
        'id="view-burn"',
        'id="burn-stage-preview"',
        'id="settings-form"',
        # Center Master-Detail Workspace (Batch Tree + Subtitle Stream + Terms Workbench)
        'id="workspace-mode-timeline"',
        'id="batch-count-badge"',
        'id="batch-tree-list"',
        'id="subtitle-detail-pane"',
        'id="active-batch-heading"',
        'id="active-batch-tag"',
        'id="subtitle-stream-search"',
        'id="stream-row-counter"',
        'id="structured-stream-tbody"',
        'id="terms-review-panel"',
        'id="task-terms-counter"',
        'id="task-terms-search"',
        'id="btn-toggle-kb-terms"',
        'id="btn-rerun-term-extract"',
        'id="btn-save-terms-only"',
        'id="btn-save-terms-and-continue"',
        'id="terms-theme-input"',
        'id="task-terms-tbody"',
        # Bottom Dockable Terminal Console
        'id="console-dock"',
        'id="console-bar-toggle-area"',
        'id="console-latest-preview"',
        'id="btn-filter-warn-log"',
        'id="btn-copy-log"',
        'id="btn-toggle-console"',
        'id="global-log-console"',
        # New Task Modal
        'id="new-task-dialog"',
        'closedby="any"',
        'id="upload-dropzone"',
        'id="input-source-path"',
        'id="btn-pick-local"',
        'id="input-sub-path"',
        'id="nt-src-lang-dropdown"',
        'id="task-src-lang"',
        'id="task-tgt-lang"',
        'id="btn-open-glossary-center"',
        'id="nt-domain-dropdown"',
        'id="task-domain"',
        'id="nt-domain-menu-list"',
        'id="task-pause-terms-chk"',
        'id="btn-cancel-intake"',
        'id="btn-create-task"',
        # View 2: Glossary Center
        'id="view-glossary"',
        'id="domain-list-box"',
        'id="new-domain-input"',
        'id="btn-create-domain"',
        'id="btn-delete-domain"',
        'id="glossary-active-title"',
        'id="glossary-csv-search"',
        'id="btn-import-glossary"',
        'id="btn-add-domain-term"',
        'id="btn-save-domain-csv"',
        'id="domain-csv-tbody"',
        'id="dict-count-pill"',
        'id="dict-search-input"',
        'id="btn-import-dict"',
        'id="dict-search-tbody"',
        # View 3: Global Settings & ASS Live Preview
        'id="view-settings"',
        'id="cfg-sec-llm"',
        'id="cfg-sec-asr"',
        'id="cfg-sec-ass"',
        'id="cfg-sec-tts"',
        'id="cfg-sec-ytb"',
        'id="btn-fetch-models"',
        'id="btn-check-llm"',
        'id="llm-check-status"',
        'id="cfg-asr-runtime"',
        'id="asr-engine-panels"',
        'id="btn-detect-mai-region"',
        'id="btn-reset-ass-style"',
        'id="ass-burn-mode"',
        'id="ass-tr-size"',
        'id="ass-tr-color"',
        'id="ass-tr-color-picker"',
        'id="ass-tr-margin"',
        'id="ass-src-size"',
        'id="ass-src-color"',
        'id="ass-src-color-picker"',
        'id="ass-live-stage"',
        'id="ass-preview-box"',
        'id="ass-preview-tr"',
        'id="ass-preview-src"',
        'id="cfg-tts-method"',
        'data-cfg="speed_factor.min"',
        'data-cfg="speed_factor.accept"',
        'data-cfg="speed_factor.max"',
        'id="tts-engine-panels"',
        # Preview Modal
        'id="video-preview-dialog"',
        'id="btn-close-preview-modal"',
    ]
    for anchor in required_dom_anchors:
        assert anchor in res_root.text, f"Spec anchor missing in GET /: {anchor}"

    forbidden_prototype_placeholders = [
        "Destiny2_Attrition_Orbs_Guide",
        "Qwen3_ASR_Architecture_Review",
        "Apple_MLX_Optimization_Talk",
        "Attrition Orbs",
        "Multimach CCX",
        "btn-sim-srt-ready",
        "btn-sim-checkpoint-terms",
        "sk-openlux-demo-key",
        "fish_demo_key",
        'value="Attrition"',
    ]
    for asset in (
        "/",
        "/css/base.css",
        "/css/layout-rail.css",
        "/css/intake-modal.css",
        "/css/stream-grid.css",
        "/css/terms-console.css",
        "/css/burn-glossary.css",
        "/css/settings-dialog.css",
        "/js/state.js",
        "/js/dom-utils.js",
        "/js/console-log.js",
        "/js/router.js",
        "/js/pipeline-ribbon.js",
        "/js/task-rail.js",
        "/js/intake-modal.js",
        "/js/stream-table.js",
        "/js/terms-review.js",
        "/js/video-preview.js",
        "/js/glossary-data.js",
        "/js/glossary-view.js",
        "/js/ass-preview.js",
        "/js/settings-view.js",
        "/js/burn-view.js",
        "/js/sse.js",
        "/js/main.js",
    ):
        res_asset = client.get(asset)
        assert res_asset.status_code == 200, f"Static asset {asset} failed"
        for placeholder in forbidden_prototype_placeholders:
            assert placeholder not in res_asset.text, (
                f"Prototype placeholder {placeholder!r} still present in {asset}"
            )

    # 2. Mount local video task (with Chinese characters in title to verify RFC 5987 Content-Disposition)
    res_task = client.post(
        "/api/tasks",
        json={
            "source": str(sample_video),
            "title": "大模型课程 Demo",
            "autorun": False,
        },
    )
    assert res_task.status_code == 202
    task_data = res_task.json()
    task_id = task_data["task_id"]
    assert "stages_done" in task_data
    assert "truncated" in task_data

    # 2b. Subtitle ZIP download endpoint with non-ASCII title
    ws_demo = TaskWorkspace(output_root / task_id)
    (ws_demo.root / "src.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8"
    )
    (ws_demo.root / "trans.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8"
    )
    res_zip = client.get(f"/api/tasks/{task_id}/subtitles.zip")
    assert res_zip.status_code == 200
    assert res_zip.headers["content-type"] == "application/zip"
    assert "filename*=UTF-8''" in res_zip.headers["content-disposition"]

    # 2c. Inactive task control guards (pause / resume / stop on idle task must return 409)
    assert client.post(f"/api/tasks/{task_id}/pause").status_code == 409
    assert client.post(f"/api/tasks/{task_id}/resume").status_code == 409
    assert client.post(f"/api/tasks/{task_id}/stop").status_code == 409

    # 2d. Reject unsupported upload format before creating workspace directory
    existing_dirs = set(output_root.iterdir())
    res_bad_upload = client.post(
        "/api/tasks/upload",
        files={"file": ("malware.exe", b"MZ-fake-binary", "application/octet-stream")},
        data={"autorun": "false"},
    )
    assert res_bad_upload.status_code == 422
    assert set(output_root.iterdir()) == existing_dirs

    # 3. Round-trip terms
    res_put_terms = client.put(
        f"/api/tasks/{task_id}/terms",
        json={
            "summary": "AI demo",
            "terms": [
                {
                    "src": "LLM",
                    "tgt": "大语言模型",
                    "aliases": ["L L M"],
                    "note": "AI",
                    "fix_source": False,
                }
            ],
        },
    )
    assert res_put_terms.status_code == 200
    res_get_terms = client.get(f"/api/tasks/{task_id}/terms")
    assert res_get_terms.status_code == 200
    assert res_get_terms.json()["terms"][0]["tgt"] == "大语言模型"

    # 4. Round-trip utterances
    res_put_utts = client.put(
        f"/api/tasks/{task_id}/utterances",
        json={
            "utterances": [
                {
                    "id": 0,
                    "word_span": [0, 2],
                    "start": 0.0,
                    "end": 1.5,
                    "gap_after": 0.2,
                    "src_raw": "Hello world",
                    "src": "Hello world.",
                    "direct": "你好世界。",
                    "tgt": "你好，世界！",
                    "tts_text": "你好，世界！",
                }
            ]
        },
    )
    assert res_put_utts.status_code == 200
    res_get_utts = client.get(f"/api/tasks/{task_id}/utterances")
    assert res_get_utts.status_code == 200
    assert res_get_utts.json()["utterances"][0]["tgt"] == "你好，世界！"


def test_threadpool_workers_inherit_workspace_and_config_overrides(tmp_path: Path):
    """Child threads in ContextThreadPoolExecutor must inherit workspace_context and task_config_context."""
    from core.concurrency import ContextThreadPoolExecutor
    from core.utils.models import resolve_task_path

    ws = TaskWorkspace(tmp_path / "output" / "worker_task")
    ws.ensure_dirs()

    with (
        workspace_context(ws),
        config_utils.task_config_context(
            {"target_language": "Español", "max_workers": 2}
        ),
    ):
        with ContextThreadPoolExecutor(max_workers=2) as pool:
            f_path = pool.submit(resolve_task_path, "output/trans.srt")
            f_lang = pool.submit(config_utils.load_key, "target_language")
            resolved_in_child = f_path.result()
            lang_in_child = f_lang.result()

    assert Path(resolved_in_child) == (ws.root / "trans.srt")
    assert lang_in_child == "Español"


def test_multi_task_isolation_fix_source_and_inline_tts_text_flow(
    tmp_path: Path, monkeypatch
):
    """Two task workspaces stay isolated; fix_source rewrites aliases; inline src/tts_text edits flow into Step 6 & 8."""
    from core import (
        _4_2_translate,
        _6_gen_sub,
        _8_1_audio_task,
        _8_2_dub_chunks,
        pipeline,
    )

    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr("core.workspace.OUTPUT_ROOT", output_root)
    monkeypatch.setattr(api, "OUTPUT", output_root)
    monkeypatch.setattr(
        "core.workspace.GLOBAL_GLOSSARY_CSV", tmp_path / "glossary" / "custom_terms.csv"
    )

    ws_a = TaskWorkspace(output_root / "task_a")
    ws_b = TaskWorkspace(output_root / "task_b")
    ws_a.ensure_dirs()
    ws_b.ensure_dirs()

    # Populate task_a with ASR words ("we use pie torch") and fix_source=True term ("PyTorch")
    words_a = [
        {"idx": 0, "text": "We", "punct_after": "", "start": 0.0, "end": 0.5},
        {"idx": 1, "text": "use", "punct_after": "", "start": 0.5, "end": 1.0},
        {"idx": 2, "text": "pie", "punct_after": "", "start": 1.0, "end": 1.5},
        {"idx": 3, "text": "torch", "punct_after": ".", "start": 1.5, "end": 2.6},
    ]
    ws_a.save_words(words_a)
    ws_a.save_utterances(
        [
            {
                "id": 0,
                "word_span": [0, 4],
                "start": 0.0,
                "end": 2.6,
                "gap_after": 0.0,
                "src_raw": "We use pie torch",
                "src": "We use pie torch.",
                "direct": "我们使用PyTorch。",
                "tgt": "我们使用PyTorch。",
                "tts_text": "我们使用PyTorch。",
            }
        ]
    )
    ws_a.save_terms(
        "DL course",
        [
            {
                "src": "PyTorch",
                "tgt": "PyTorch",
                "aliases": ["pie torch"],
                "note": "",
                "fix_source": True,
            }
        ],
    )

    with workspace_context(ws_a):
        chunks = _4_2_translate.split_chunks_by_chars()
        assert chunks == ["We use pie torch."]
        assert ws_a.load_utterances()[0]["src"] == "We use pie torch."

    # Edit src, tgt, and tts_text via FastAPI
    client = TestClient(api.app)
    res_edit = client.put(
        "/api/tasks/task_a/utterances",
        json={
            "utterances": [
                {
                    "id": 0,
                    "word_span": [0, 4],
                    "start": 0.0,
                    "end": 2.6,
                    "gap_after": 0.0,
                    "src_raw": "We use pie torch",
                    "src": "We use PyTorch 2.0.",
                    "direct": "我们使用PyTorch 2.0。",
                    "tgt": "我们使用 PyTorch 2.0",
                    "tts_text": "我们使用派托奇二点零，非常快。",
                }
            ]
        },
    )
    assert res_edit.status_code == 200

    # review_translation() must pass inside workspace_context(ws_a) after src edit
    with workspace_context(ws_a):
        assert pipeline.review_translation() is None
        _5_split_sub.split_for_sub_main()
        _6_gen_sub.align_timestamp_main()
        # Only src.srt and trans.srt are generated; bi.srt, bi.ass, and *_subs_for_audio.srt are not generated
        assert "我们使用PyTorch 2.0" in (ws_a.root / "trans.srt").read_text(
            encoding="utf-8"
        )
        assert not (ws_a.root / "bi.srt").exists()
        assert not (ws_a.root / "bi.ass").exists()
        assert not (ws_a.audio_dir / "trans_subs_for_audio.srt").exists()
        assert not (ws_a.audio_dir / "src_subs_for_audio.srt").exists()

        # Step 8.1 and 8.2 in task_a read directly from task_a's src.srt and trans.srt without touching task_b
        sr = 16000
        with wave.open(str(ws_a.audio_dir / "raw.mp3"), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(np.zeros(sr * 3, dtype=np.int16).tobytes())
        _8_1_audio_task.gen_audio_task_main()
        _8_2_dub_chunks.gen_dub_chunks()
        df_tasks_a = ws_a.load_dub_tasks_df()
        assert "我们使用PyTorch 2.0" in str(df_tasks_a.loc[0, "lines"])
        assert not ws_b.has_dub_tasks()

    # Global glossary GET/PUT, nested audio/dub.wav file download, and DELETE task_b
    (ws_a.audio_dir / "dub.wav").write_bytes(b"RIFF-dub-wav")
    assert client.get("/api/tasks/task_a/files/audio/dub.wav").status_code == 200
    assert (
        client.put(
            "/api/glossary",
            json={"terms": [{"src": "CUDA", "tgt": "CUDA架构", "aliases": []}]},
        ).status_code
        == 200
    )
    assert client.get("/api/glossary").json()["terms"][0]["src"] == "CUDA"
    assert client.delete("/api/tasks/task_b").status_code == 200
    assert not ws_b.root.exists()


def test_sentence_initial_fix_source_and_global_glossary_cache_invalidation(
    tmp_path: Path, monkeypatch
):
    """Sentence-initial ASR error rewritten by fix_source preserves O(1) word_span timestamps;
    updating global glossary invalidates _translation_cache_fingerprint and applies fix_source."""
    import os
    from core import _4_1_summarize, _4_2_translate, _5_split_sub, _6_gen_sub
    from core.workspace import save_global_glossary

    output_root = tmp_path / "output"
    output_root.mkdir()
    global_csv = tmp_path / "glossary" / "custom_terms.csv"
    monkeypatch.setattr("core.workspace.OUTPUT_ROOT", output_root)
    monkeypatch.setattr("core.workspace.GLOBAL_GLOSSARY_CSV", global_csv)
    monkeypatch.setattr(_4_1_summarize, "GLOBAL_GLOSSARY_PATH", str(global_csv))
    monkeypatch.setattr(_4_1_summarize, "CUSTOM_TERMS_PATH", str(tmp_path / "no.xlsx"))
    monkeypatch.setattr(
        _4_1_summarize, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv")
    )
    monkeypatch.setattr(
        _4_1_summarize, "OPTIONAL_JSON_TERMS_PATH", str(tmp_path / "no.json")
    )

    ws = TaskWorkspace(output_root / "task_init")
    ws.ensure_dirs()

    # First word in ASR is "Gnats" (starts with 'g'), which fix_source rewrites to "NATS" (starts with 'n')
    words = [
        {"idx": 0, "text": "Gnats", "punct_after": "", "start": 1.25, "end": 1.75},
        {"idx": 1, "text": "streams", "punct_after": "", "start": 1.80, "end": 2.20},
        {"idx": 2, "text": "messages", "punct_after": ".", "start": 2.25, "end": 2.95},
    ]
    ws.save_words(words)
    df_words = pd.DataFrame(words)
    ws.save_utterances(
        [
            {
                "id": 0,
                "word_span": [0, 3],
                "start": 1.25,
                "end": 2.95,
                "gap_after": 0.0,
                "src_raw": "Gnats streams messages",
                "src": "Gnats streams messages.",
                "direct": "NATS流式传输消息。",
                "tgt": "NATS流式传输消息。",
                "tts_text": "NATS流式传输消息。",
            }
        ]
    )
    ws.save_terms("Messaging", [])

    with workspace_context(ws):
        fp_before = _4_2_translate._translation_cache_fingerprint()

        # Save a global glossary entry with fix_source=True
        save_global_glossary(
            [
                {
                    "src": "NATS",
                    "tgt": "NATS消息系统",
                    "aliases": ["Gnats"],
                    "note": "CNCF messaging",
                    "fix_source": True,
                }
            ],
            csv_path=global_csv,
        )
        # Ensure mtime advances so _get_term_index refreshes
        st = os.stat(global_csv)
        os.utime(global_csv, (st.st_atime, st.st_mtime + 2.0))

        fp_after = _4_2_translate._translation_cache_fingerprint()
        assert fp_after != fp_before

        # split_chunks_by_chars preserves original sentence while search_things_to_note_in_prompt matches alias
        chunks = _4_2_translate.split_chunks_by_chars()
        assert chunks == ["Gnats streams messages."]
        utts = ws.load_utterances()
        assert utts[0]["src"] == "Gnats streams messages."
        assert utts[0]["src_raw"] == "Gnats streams messages"
        note_prompt = _4_1_summarize.search_things_to_note_in_prompt(chunks[0])
        assert note_prompt is not None and "Gnats:NATS消息系统 (NATS)" in note_prompt

        _5_split_sub.split_for_sub_main()
        stamps = _6_gen_sub.get_sentence_timestamps(
            df_words, pd.DataFrame({"Source": ["Gnats streams messages."]})
        )
        assert stamps == [(1.25, 2.95)]


def test_domain_glossary_crud_import_and_task_binding(tmp_path: Path, monkeypatch):
    """Domain glossary supports CRUD, CSV/JSON/large-table import, config.yaml persistence (including no-domain ''),
    per-task domain binding (decoupled from subsequent global domain switches), and Stage 03 promote_to_domain."""
    import json
    from core import _4_1_summarize

    output_root = tmp_path / "output"
    output_root.mkdir()
    glossary_root = tmp_path / "glossary"
    monkeypatch.setattr("core.workspace.OUTPUT_ROOT", output_root)
    monkeypatch.setattr("core.workspace.GLOSSARY_DIR", glossary_root)
    monkeypatch.setattr(api, "OUTPUT", output_root)
    monkeypatch.setattr(_4_1_summarize, "CUSTOM_TERMS_PATH", str(tmp_path / "no.xlsx"))
    monkeypatch.setattr(
        _4_1_summarize, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv")
    )
    monkeypatch.setattr(
        _4_1_summarize, "OPTIONAL_JSON_TERMS_PATH", str(tmp_path / "no.json")
    )

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("target_language: '简体中文'\n", encoding="utf-8")
    monkeypatch.setattr(config_utils, "CONFIG_PATH", str(cfg_file))
    config_utils.ensure_config_schema(str(cfg_file))
    assert config_utils.load_key("glossary_domain") == "destiny2"

    client = TestClient(api.app)

    # 1. Listing domains auto-creates default 'destiny2' domain directory and terms.csv
    res_domains = client.get("/api/glossary/domains")
    assert res_domains.status_code == 200
    data_domains = res_domains.json()
    assert data_domains["active_domain"] == "destiny2"
    assert any(d["domain"] == "destiny2" for d in data_domains["domains"])
    assert (glossary_root / "destiny2" / "terms.csv").exists()

    # 2. Create a new domain 'medical', import small CSV (<=500 rows) -> terms.csv, and import JSON -> dict.json
    res_create_dom = client.post("/api/glossary/domains", json={"domain": "medical"})
    assert res_create_dom.status_code == 201
    assert res_create_dom.json()["domain"] == "medical"
    assert (glossary_root / "medical" / "terms.csv").exists()

    csv_bytes = (
        "src,tgt,aliases,note,fix_source\nMRI,核磁共振,M R I,imaging,true\n".encode(
            "utf-8"
        )
    )
    res_imp_csv = client.post(
        "/api/glossary/domains/medical/import",
        files={"file": ("med.csv", csv_bytes, "text/csv")},
    )
    assert res_imp_csv.status_code == 200
    assert res_imp_csv.json()["imported_as"] == "terms.csv"
    assert res_imp_csv.json()["terms_count"] == 1

    dict_bytes = json.dumps({"CT Scan": "CT扫描", "ECG": "心电图"}).encode("utf-8")
    res_imp_json = client.post(
        "/api/glossary/domains/medical/import",
        files={"file": ("med_kb.json", dict_bytes, "application/json")},
    )
    assert res_imp_json.status_code == 200
    assert res_imp_json.json()["imported_as"] == "dict.json"
    assert res_imp_json.json()["dict_count"] == 2
    assert (glossary_root / "medical" / "dict.json").exists()

    tsv_dict_bytes = "src\ttgt\nPET-CT\t正电子发射计算机断层显像\nUltrasound\t超声检查\nCBC\t血常规\n".encode(
        "utf-8"
    )
    res_imp_tsv_dict = client.post(
        "/api/glossary/domains/medical/import",
        data={"target": "dict"},
        files={"file": ("med_dict.tsv", tsv_dict_bytes, "text/tab-separated-values")},
    )
    assert res_imp_tsv_dict.status_code == 200
    assert res_imp_tsv_dict.json()["imported_as"] == "dict.json"
    assert res_imp_tsv_dict.json()["dict_count"] == 3

    # Importing >500 rows CSV with rich columns (note, aliases, fix_source) routes to dict.json and preserves all fields
    big_rows = ["src,tgt,aliases,note,fix_source"] + [
        f"Term{i},译名{i},Variant{i},Note{i},true" for i in range(505)
    ]
    res_imp_big = client.post(
        "/api/glossary/domains/destiny2/import",
        files={"file": ("big.csv", "\n".join(big_rows).encode("utf-8"), "text/csv")},
    )
    assert res_imp_big.status_code == 200
    assert res_imp_big.json()["imported_as"] == "dict.json"
    assert res_imp_big.json()["dict_count"] == 505
    saved_big_dict = json.loads(
        (glossary_root / "destiny2" / "dict.json").read_text(encoding="utf-8")
    )
    assert "terms" in saved_big_dict
    assert saved_big_dict["terms"][0]["note"] == "Note0"
    assert saved_big_dict["terms"][0]["aliases"] == ["Variant0"]
    assert saved_big_dict["terms"][0]["fix_source"] is True

    # Importing a GB18030/GBK encoded CSV file works without UnicodeDecodeError
    gbk_csv = "原词,译名,备注\nGjallarhorn,加拉尔号角,异域火箭筒\n".encode("gb18030")
    res_imp_gbk = client.post(
        "/api/glossary/domains/destiny2/import",
        files={"file": ("gbk_terms.csv", gbk_csv, "text/csv")},
    )
    assert res_imp_gbk.status_code == 200
    assert res_imp_gbk.json()["imported_as"] == "terms.csv"
    assert any(
        t["src"] == "Gjallarhorn" and t["tgt"] == "加拉尔号角"
        for t in res_imp_gbk.json()["terms"]
    )

    # Remove dict.json from destiny2
    res_del_dict = client.delete("/api/glossary/domains/destiny2/dict")
    assert res_del_dict.status_code == 200
    assert not (glossary_root / "destiny2" / "dict.json").exists()

    # 3. Task creation binds glossary_domain into config_override and isolates from subsequent config.yaml switches
    sample_video = tmp_path / "lecture.mp4"
    sample_video.write_bytes(b"fake-video-bytes")

    # Add a term to destiny2 domain
    client.put(
        "/api/glossary?domain=destiny2",
        json={
            "terms": [
                {
                    "src": "Vanguard",
                    "tgt": "先锋敢死队",
                    "aliases": ["van guard"],
                    "fix_source": True,
                }
            ]
        },
    )

    # Create task_d2 bound to destiny2
    res_t1 = client.post(
        "/api/tasks",
        json={
            "source": str(sample_video),
            "glossary_domain": "destiny2",
            "autorun": False,
        },
    )
    assert res_t1.status_code == 202
    t1_id = res_t1.json()["task_id"]
    assert res_t1.json()["config_override"]["glossary_domain"] == "destiny2"

    # Now switch global config.yaml to "" (无领域) and verify it persists in config.yaml
    client.put("/api/config", json={"updates": {"glossary_domain": ""}})
    assert config_utils.load_key("glossary_domain") == ""

    # Create task_none under 无领域 ("")
    sample_video2 = tmp_path / "general.mp4"
    sample_video2.write_bytes(b"fake-video-bytes-2")
    res_t2 = client.post(
        "/api/tasks",
        json={"source": str(sample_video2), "autorun": False},
    )
    assert res_t2.status_code == 202
    t2_id = res_t2.json()["task_id"]
    assert res_t2.json()["config_override"]["glossary_domain"] == ""

    ws_t1 = TaskWorkspace(output_root / t1_id)
    ws_t2 = TaskWorkspace(output_root / t2_id)

    # Verify task_d2 still loads 'destiny2' terms even though global config.yaml is now ""
    with (
        workspace_context(ws_t1),
        config_utils.task_config_context(ws_t1.load_manifest()["config_override"]),
    ):
        t1_custom = _4_1_summarize.load_custom_terms()
        assert any(t["src"] == "Vanguard" for t in t1_custom)

    # Verify task_none (bound to "") loads zero domain terms
    with (
        workspace_context(ws_t2),
        config_utils.task_config_context(ws_t2.load_manifest()["config_override"]),
    ):
        t2_custom = _4_1_summarize.load_custom_terms()
        assert t2_custom == []

    # 4. Stage 03 promote_to_domain promotes to the task's bound domain (and skips when bound to "")
    res_prom_t1 = client.put(
        f"/api/tasks/{t1_id}/terms",
        json={
            "summary": "Destiny lore",
            "terms": [
                {
                    "src": "Traveler",
                    "tgt": "旅行者",
                    "aliases": [],
                    "note": "Entity",
                    "fix_source": False,
                    "promote_to_domain": True,
                }
            ],
        },
    )
    assert res_prom_t1.status_code == 200
    assert res_prom_t1.json()["promoted_to_domain"] == 1
    d2_terms = client.get("/api/glossary?domain=destiny2").json()["terms"]
    assert any(t["src"] == "Traveler" and t["tgt"] == "旅行者" for t in d2_terms)

    # When task is bound to "", promote_to_domain does not promote anywhere
    res_prom_t2 = client.put(
        f"/api/tasks/{t2_id}/terms",
        json={
            "summary": "General video",
            "terms": [
                {
                    "src": "hemeral",
                    "tgt": "临时词",
                    "aliases": [],
                    "note": "",
                    "fix_source": False,
                    "promote_to_domain": True,
                }
            ],
        },
    )
    assert res_prom_t2.status_code == 200
    assert res_prom_t2.json()["promoted_to_domain"] == 0

    # Switching task_t2's bound domain in Stage 03 to 'medical' and promoting works
    res_switch_t2 = client.put(
        f"/api/tasks/{t2_id}/terms",
        json={
            "summary": "Switched to medical",
            "glossary_domain": "medical",
            "terms": [
                {
                    "src": "Biopsy",
                    "tgt": "活检",
                    "aliases": [],
                    "note": "",
                    "fix_source": False,
                    "promote_to_domain": True,
                }
            ],
        },
    )
    assert res_switch_t2.status_code == 200
    assert res_switch_t2.json()["glossary_domain"] == "medical"
    assert res_switch_t2.json()["promoted_to_domain"] == 1
    med_terms = client.get("/api/glossary?domain=medical").json()["terms"]
    assert any(t["src"] == "Biopsy" for t in med_terms)

    # Verify Stage 03 rerun-terms-extract-btn invalidates 'summarize' stage cleanly (and WebUI JS sends 'summarize', not 'summary')
    res_inv_sum = client.post(
        f"/api/tasks/{t2_id}/invalidate",
        json={"stage": "summarize", "autorun": False, "run_stage": "subtitles"},
    )
    assert res_inv_sum.status_code == 200
    assert "summarize" in res_inv_sum.json()["invalidated"]
    assert "translate" in res_inv_sum.json()["invalidated"]
    terms_js_res = client.get("/js/terms-review.js")
    assert terms_js_res.status_code == 200
    terms_js_text = terms_js_res.text
    assert "stage: 'summarize'" in terms_js_text
    assert "stage: 'summary'" not in terms_js_text
    assert re.search(
        r"source:\s*'manual',\s*in_domain_kb:\s*false,\s*promote:\s*false",
        terms_js_text,
    )

    # 5. Instant toggle add/remove in domain glossary while keeping new terms visible (`in_domain_kb=False`)
    ws_t1.save_terms(
        "Destiny lore",
        [
            {"src": "Traveler", "tgt": "旅行者", "note": "Pre-existing in terms.csv"},
            {"src": "Core DPS", "tgt": "核心DPS", "note": "Newly extracted term"},
        ],
    )
    ws_t1.set_kv("task_promoted_srcs", [])
    initial_get = client.get(f"/api/tasks/{t1_id}/terms").json()["terms"]
    by_src_init = {t["src"]: t for t in initial_get}
    assert by_src_init["Traveler"]["in_domain_kb"] is True
    assert by_src_init["Traveler"]["promoted"] is True
    assert by_src_init["Core DPS"]["in_domain_kb"] is False
    assert by_src_init["Core DPS"]["promoted"] is False

    # Click ON -> immediately added to destiny2/terms.csv, promoted=True, still in_domain_kb=False (visible)
    res_add = client.post(
        f"/api/tasks/{t1_id}/terms/domain",
        json={
            "action": "add",
            "terms": [
                {
                    "src": "Core DPS",
                    "tgt": "核心DPS",
                    "aliases": [],
                    "note": "Newly extracted term",
                }
            ],
        },
    )
    assert res_add.status_code == 200
    assert res_add.json()["promoted"] == 1
    assert any(
        t["src"] == "Core DPS"
        for t in client.get("/api/glossary?domain=destiny2").json()["terms"]
    )
    after_add = {
        t["src"]: t for t in client.get(f"/api/tasks/{t1_id}/terms").json()["terms"]
    }
    assert after_add["Core DPS"]["in_domain_kb"] is False
    assert after_add["Core DPS"]["promoted"] is True

    # Click OFF (undo mis-click) -> immediately removed from destiny2/terms.csv, promoted=False
    res_rem = client.post(
        f"/api/tasks/{t1_id}/terms/domain",
        json={"action": "remove", "srcs": ["Core DPS"]},
    )
    assert res_rem.status_code == 200
    assert res_rem.json()["removed"] == 1
    assert not any(
        t["src"] == "Core DPS"
        for t in client.get("/api/glossary?domain=destiny2").json()["terms"]
    )
    after_rem = {
        t["src"]: t for t in client.get(f"/api/tasks/{t1_id}/terms").json()["terms"]
    }
    assert after_rem["Core DPS"]["in_domain_kb"] is False
    assert after_rem["Core DPS"]["promoted"] is False

    # 5. Delete domain 'medical' and 'destiny2' (verify deleting default domain destiny2 does NOT recreate it)
    res_del_dom = client.delete("/api/glossary/domains/medical")
    assert res_del_dom.status_code == 200
    assert not (glossary_root / "medical").exists()

    client.put("/api/config", json={"updates": {"glossary_domain": "destiny2"}})
    res_del_d2 = client.delete("/api/glossary/domains/destiny2")
    assert res_del_d2.status_code == 200
    assert res_del_d2.json()["active_domain"] == ""
    assert not (glossary_root / "destiny2").exists()
    # Calling GET /api/glossary/domains after deleting destiny2 must NOT recreate destiny2!
    res_domains_after_del = client.get("/api/glossary/domains").json()
    assert res_domains_after_del["domains"] == []
    assert not (glossary_root / "destiny2").exists()


def test_review_fixes_large_dict_fix_source_paused_invalidate_domain_mtime_and_headerless_excel(
    tmp_path: Path, monkeypatch
):
    """Regression test for the 5 code review findings:
    1. >500 dict.json fix_source consistency in load_custom_terms() & _get_kb_index() preserves fix_source.
    2. Stage 03 invalidate while runner is paused stops the paused runner and succeeds (200 instead of 409).
    3. Switching task glossary_domain via PUT /api/tasks/{id}/terms without terms payload does not bump terms mtime.
    4. Headerless .xlsx import preserves the first data row instead of swallowing it as header.
    """
    import io
    import time
    from core import _4_1_summarize

    output_root = tmp_path / "output"
    output_root.mkdir()
    glossary_root = tmp_path / "glossary"
    monkeypatch.setattr("core.workspace.OUTPUT_ROOT", output_root)
    monkeypatch.setattr("core.workspace.GLOSSARY_DIR", glossary_root)
    monkeypatch.setattr(api, "OUTPUT", output_root)
    monkeypatch.setattr(_4_1_summarize, "CUSTOM_TERMS_PATH", str(tmp_path / "no.xlsx"))
    monkeypatch.setattr(
        _4_1_summarize, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv")
    )
    monkeypatch.setattr(
        _4_1_summarize, "OPTIONAL_JSON_TERMS_PATH", str(tmp_path / "no.json")
    )
    monkeypatch.setattr(
        _4_1_summarize, "GLOBAL_GLOSSARY_PATH", str(tmp_path / "no_global.csv")
    )

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("target_language: '简体中文'\n", encoding="utf-8")
    monkeypatch.setattr(config_utils, "CONFIG_PATH", str(cfg_file))
    config_utils.ensure_config_schema(str(cfg_file))

    client = TestClient(api.app)

    # Finding 5: Headerless .xlsx import must not drop row 0
    client.post("/api/glossary/domains", json={"domain": "excel_dom"})
    excel_buf = io.BytesIO()
    pd.DataFrame(
        [
            ["Kubernetes", "K8s容器编排", "Cloud native", "kube|k8s", "true"],
            ["Prometheus", "普罗米修斯监控", "Metrics", "prom", "false"],
        ]
    ).to_excel(excel_buf, index=False, header=False)
    res_xlsx = client.post(
        "/api/glossary/domains/excel_dom/import",
        files={
            "file": (
                "headerless.xlsx",
                excel_buf.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert res_xlsx.status_code == 200
    assert res_xlsx.json()["imported_count"] == 2
    xlsx_terms = res_xlsx.json()["terms"]
    assert len(xlsx_terms) == 2
    assert any(
        t["src"] == "Kubernetes" and t["fix_source"] is True and "kube" in t["aliases"]
        for t in xlsx_terms
    )

    # Finding 1: >500 entries in dict.json with fix_source=True must be returned by load_custom_terms()
    # AND _get_kb_index(custom_terms_list) must preserve fix_source on custom terms
    big_rows = ["src,tgt,aliases,note,fix_source"] + [
        f"BigTerm{i},大词{i},big alias {i},note {i},{'true' if i == 0 else 'false'}"
        for i in range(505)
    ]
    res_big = client.post(
        "/api/glossary/domains/excel_dom/import",
        files={"file": ("big.csv", "\n".join(big_rows).encode("utf-8"), "text/csv")},
    )
    assert res_big.status_code == 200
    assert res_big.json()["imported_as"] == "dict.json"

    ws = TaskWorkspace(output_root / "task_rev")
    ws.ensure_dirs()
    ws.update_manifest(config_override={"glossary_domain": "excel_dom"})

    with (
        workspace_context(ws),
        config_utils.task_config_context({"glossary_domain": "excel_dom"}),
    ):
        custom = _4_1_summarize.load_custom_terms()
        # >500 entries in dict.json stay in KB index only
        assert not any(t["src"] == "BigTerm0" for t in custom)
        assert not any(t["src"] == "BigTerm1" for t in custom)
        kb_idx = _4_1_summarize._get_kb_index(custom)
        assert kb_idx["entries"]["kubernetes"]["fix_source"] is True
        assert kb_idx["entries"]["bigterm0"]["fix_source"] is True
        # Toggling fix_source on custom_terms_list must invalidate _get_kb_index cache
        toggled_custom = [
            {**t, "fix_source": False} if t["src"] == "Kubernetes" else t
            for t in custom
        ]
        kb_idx_toggled = _4_1_summarize._get_kb_index(toggled_custom)
        assert kb_idx_toggled["entries"]["kubernetes"]["fix_source"] is False

    # Finding 4: Switching stage3 glossary_domain without passing terms must NOT bump terms mtime
    ws.save_terms(
        "Old summary",
        [
            {
                "src": "Kubernetes",
                "tgt": "旧译名",
                "aliases": [],
                "note": "",
                "fix_source": False,
            }
        ],
    )
    old_mtime = 1_700_000_000.0
    ws.set_terms_mtime(old_mtime)

    res_dom_only = client.put(
        "/api/tasks/task_rev/terms",
        json={"glossary_domain": "excel_dom"},
    )
    assert res_dom_only.status_code == 200
    assert ws.get_terms_mtime() == old_mtime
    # Because terms mtime was NOT bumped above excel_dom/terms.csv mtime, domain CSV ("K8s容器编排") wins!
    with (
        workspace_context(ws),
        config_utils.task_config_context({"glossary_domain": "excel_dom"}),
    ):
        idx_data = _4_1_summarize._get_term_index()
        k8s_entry = next(e for e in idx_data["terms"] if e["src"] == "Kubernetes")
        assert k8s_entry["tgt"] == "K8s容器编排"
        note_prompt = _4_1_summarize.search_things_to_note_in_prompt(
            "We deploy on kube today."
        )
        assert (
            note_prompt is not None and "kube:K8s容器编排 (Kubernetes)" in note_prompt
        )

    # Finding 2: Calling POST /api/tasks/{id}/invalidate when runner is paused at review_terminology
    def _paused_step():
        api.runner.pause("Terminology is ready for review.")
        api.runner.check_cancel()

    api.runner.start([("Pause step", _paused_step)], task_id="task_rev")
    deadline = time.time() + 2.0
    while api.runner.state != "paused" and time.time() < deadline:
        time.sleep(0.02)
    assert api.runner.state == "paused"
    assert api.runner.is_active is True

    res_inv_paused = client.post(
        "/api/tasks/task_rev/invalidate",
        json={"stage": "summarize", "autorun": False, "run_stage": "subtitles"},
    )
    assert res_inv_paused.status_code == 200
    assert "summarize" in res_inv_paused.json()["invalidated"]
    assert not api.runner.is_active
    api.runner.reset()


def test_destiny2_authoritative_dict_homonym_priority_and_noise_purging():
    """Verify Destiny 2 dict.json homonyms are resolved canonically, conversational emotes and generic UI words are purged,
    and terms.csv has no duplicate entries."""
    from core._4_1_summarize import (
        _extract_generic_kb_pairs,
        _scan_text_for_kb_terms,
        _tokenize_and_normalize,
        _KB_STOP_TOKENS,
    )

    dict_data = {
        "Titan": "泰坦",
        "Kinetic": "动能",
        "Solar": "烈日",
        "Arc": "电弧",
        "Void": "虚空",
        "Sparrows": "快雀",
        "The Last City": "圣城",
        "Vanguard Ops": "先锋行动",
        "Tomorrow's Answer": "明日答案",
        "Tomorrow's Answer (Adept)": "明日答案（专家）",
        "Surrounded": "腹背受敌",
        "Together Forever": "长相聚",
        "Worm God Incarnation": "虫神化身",
        "Pulse Rifle Scavenger": "脉冲步枪弹药回收器",
        "Unstoppable Hand Cannon": "势不可挡手炮",
        "Overload Hand Cannon": "超载手炮",
        "Unstoppable Grenade Launcher": "势不可挡榴弹发射器",
        "Snapload Finisher": "速装终结技",
        "Guiding Light": "指引之光",
        "Kinetic Dexterity": "动能武器灵巧",
        "Unflinching Kinetic Aim": "稳定动能武器瞄准",
        "Match Game": "元素匹配",
        "Raid": "突袭",
        "Salvage": "回收",
        "Shock and Awe": "震慑行动",
        "Lightning Strikes Twice": "两次闪电打击",
        "Radiant Largesse": "光辉慷慨",
        "Energized": "充能",
        "Reflection Effects": "映像效果",
        "Heartbreaker": "碎心者",
        "Praetorian Ornament": "禁卫军皮肤",
        "Event Pack": "活动包",
        "Rapid Response": "快速反应",
        "Momentum": "动量",
        "Heat of the Moment": "瞬时热情",
        "Shocker": "震荡",
        "Clash": "死斗",
        "Control": "占领",
        "Elimination": "灭绝",
        "Mayhem": "鏖战",
        "Rumble": "混战",
        "Supremacy": "霸权",
        "Survival": "生存",
        "Rift": "裂痕",
        "Doubles": "双打",
        "Nightfall: The Pyramidion": "日落：金字塔",
        "Nightfall: The Inverted Spire": "日落：倒置尖塔",
        "Nightfall: Exodus Crash": "日落：移民号的坠毁",
        "Nightfall: The Arms Dealer": "日落：军火交易商",
        "Nightfall: Savathûn's Song": "日落：萨瓦图恩之歌",
        "Iron Banner Clash": "铁旗死斗",
        "Iron Banner Control": "铁旗占领",
        "Iron Banner Supremacy": "铁旗霸权",
        "Nightfall: A Garden World": "日落：锦绣世界",
        "Nightfall: Tree of Probabilities": "日落：概率之树",
        "Thorn": "荆棘",
        "Lumina": "流明",
        "Austringer": "鹰匠",
        "Resilience": "韧性",
        "Recovery": "恢复",
    }

    # 1. Homonym Priority Verifications
    expected_homonyms = {
        "Titan": "泰坦",
        "Kinetic": "动能",
        "Solar": "烈日",
        "Arc": "电弧",
        "Void": "虚空",
        "Sparrows": "快雀",
        "The Last City": "圣城",
        "Vanguard Ops": "先锋行动",
        "Tomorrow's Answer": "明日答案",
        "Tomorrow's Answer (Adept)": "明日答案（专家）",
        "Surrounded": "腹背受敌",
        "Together Forever": "长相聚",
        "Worm God Incarnation": "虫神化身",
        "Pulse Rifle Scavenger": "脉冲步枪弹药回收器",
        "Unstoppable Hand Cannon": "势不可挡手炮",
        "Overload Hand Cannon": "超载手炮",
        "Unstoppable Grenade Launcher": "势不可挡榴弹发射器",
        "Snapload Finisher": "速装终结技",
        "Guiding Light": "指引之光",
        "Kinetic Dexterity": "动能武器灵巧",
        "Unflinching Kinetic Aim": "稳定动能武器瞄准",
        "Match Game": "元素匹配",
        "Raid": "突袭",
        "Salvage": "回收",
        "Shock and Awe": "震慑行动",
        "Lightning Strikes Twice": "两次闪电打击",
        "Radiant Largesse": "光辉慷慨",
        "Energized": "充能",
        "Reflection Effects": "映像效果",
        "Heartbreaker": "碎心者",
        "Praetorian Ornament": "禁卫军皮肤",
        "Event Pack": "活动包",
        "Rapid Response": "快速反应",
        "Momentum": "动量",
        "Heat of the Moment": "瞬时热情",
        "Shocker": "震荡",
        "Clash": "死斗",
        "Control": "占领",
        "Elimination": "灭绝",
        "Mayhem": "鏖战",
        "Rumble": "混战",
        "Supremacy": "霸权",
        "Survival": "生存",
        "Rift": "裂痕",
        "Doubles": "双打",
        "Nightfall: The Pyramidion": "日落：金字塔",
        "Nightfall: The Inverted Spire": "日落：倒置尖塔",
        "Nightfall: Exodus Crash": "日落：移民号的坠毁",
        "Nightfall: The Arms Dealer": "日落：军火交易商",
        "Nightfall: Savathûn's Song": "日落：萨瓦图恩之歌",
        "Iron Banner Clash": "铁旗死斗",
        "Iron Banner Control": "铁旗占领",
        "Iron Banner Supremacy": "铁旗霸权",
        "Nightfall: A Garden World": "日落：锦绣世界",
        "Nightfall: Tree of Probabilities": "日落：概率之树",
    }
    for term, expected_zh in expected_homonyms.items():
        assert dict_data.get(term) == expected_zh, (
            f"Homonym mismatch for {term}: got {dict_data.get(term)!r}, expected {expected_zh!r}"
        )

    dict_keys_lower = {k.lower() for k in dict_data}
    assert len(dict_data) == len(dict_keys_lower), (
        f"dict.json must not have duplicate keys differing only in casing ({len(dict_data)} vs {len(dict_keys_lower)})"
    )

    # 2. Transcript pre-scanning integration test
    kb_entries = {}
    _extract_generic_kb_pairs(dict_data, kb_entries)
    exact_lower_map = {}
    compact_map = {}
    token_inv = {}
    p3_inv = {}
    stem_ngram_map = {}
    for key, entry in kb_entries.items():
        en = entry["src"]
        exact_lower_map[key] = entry
        for q in re.findall(r'"([^"]+)"', en):
            ql = q.strip().lower()
            if len(ql) >= 3 and ql not in exact_lower_map:
                exact_lower_map[ql] = entry
        for al in entry.get("aliases") or []:
            al_low = str(al).strip().lower()
            if al_low and al_low not in exact_lower_map:
                exact_lower_map[al_low] = entry
        comp = re.sub(r"[^a-z0-9]", "", key)
        if len(comp) >= 2:
            compact_map.setdefault(comp, []).append(entry)
        toks = [
            t
            for t in re.findall(r"[a-z0-9]+", key)
            if len(t) >= 3 and t not in _KB_STOP_TOKENS
        ]
        for t in set(toks):
            if t not in token_inv:
                token_inv[t] = []
                p3_inv.setdefault(t[:3], []).append(t)
            token_inv[t].append(entry)
        _, stems = _tokenize_and_normalize(en)
        if stems and len(stems) <= 6:
            st_tuple = tuple(stems)
            if st_tuple not in stem_ngram_map:
                stem_ngram_map[st_tuple] = entry

    synthetic_kb = {
        "entries": kb_entries,
        "exact_lower_map": exact_lower_map,
        "compact_map": compact_map,
        "token_inv": token_inv,
        "p3_inv": p3_inv,
        "stem_ngram_map": stem_ngram_map,
    }

    transcript = (
        "Hello and welcome back everyone. Today we are looking at Thorn and Lumina in Destiny 2. "
        "You're welcome to leave a comment below. If you want help with your build, soon we will explore "
        "the new Void and Solar fragments. The Titan class has high Resilience and Recovery. "
        "And another thing, we need to talk about Overload Hand Cannon and Surrounded on our Austringer. "
        "You can enhance your weapons at the enclave. You need to defeat enemies quickly. "
        "If you talk to the Drifter he will give you a quest. We should open the gate right now. "
        "No ammo no problem with this build."
    )
    scanned = _scan_text_for_kb_terms(transcript, synthetic_kb)
    scanned_map = {s["src"]: s["tgt"] for s in scanned}

    assert "Thorn" in scanned_map and scanned_map["Thorn"] == "荆棘"
    assert "Lumina" in scanned_map and scanned_map["Lumina"] == "流明"
    assert (
        "Overload Hand Cannon" in scanned_map
        and scanned_map["Overload Hand Cannon"] == "超载手炮"
    )
    assert "Surrounded" in scanned_map and scanned_map["Surrounded"] == "腹背受敌"
    assert "Titan" in scanned_map and scanned_map["Titan"] == "泰坦"
    assert "Solar" in scanned_map and scanned_map["Solar"] == "烈日"
    assert "You're Welcome" not in scanned_map, (
        "Conversational phrase 'You're Welcome' must not be matched"
    )
    assert "And Another Thing" not in scanned_map, (
        "Conversational phrase 'And Another Thing' must not be matched"
    )
    assert "Enhance!" not in scanned_map, "Emote 'Enhance!' must not be matched"
    assert "Defeat enemies" not in scanned_map, (
        "Quest objective 'Defeat enemies' must not be matched"
    )
    assert "Talk to the Drifter" not in scanned_map, (
        "Quest objective 'Talk to the Drifter' must not be matched"
    )
    assert "Open the Gate" not in scanned_map, (
        "Quest objective 'Open the Gate' must not be matched"
    )
    assert "No Ammo? No Problem" not in scanned_map, (
        "Phrase 'No Ammo? No Problem' must not be matched"
    )


def test_config_stat_cache_zero_latency_sync_and_bidirectional_overrides(
    tmp_path: Path, monkeypatch
):
    """load_key caches parsed YAML via (path, st_mtime_ns, st_size), syncs immediately on update_key or external file edits, and normalizes task_config_context overrides."""
    root_dir = Path(__file__).resolve().parents[1]
    root_cfg = (
        root_dir / "config.example.yaml"
        if (root_dir / "config.example.yaml").is_file()
        else root_dir / "config.yaml"
    )
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(root_cfg.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(config_utils, "CONFIG_PATH", str(cfg_file))

    initial_lang = config_utils.load_key("target_language")
    assert isinstance(initial_lang, str) and initial_lang

    # Defensive copy: mutating returned dict does not pollute cache
    whisper_cfg = config_utils.load_key("whisper")
    orig_model = whisper_cfg["model"]
    whisper_cfg["model"] = "mutated-in-place"
    assert config_utils.load_key("whisper")["model"] == orig_model

    # Zero-latency sync on update_key / update_keys
    config_utils.update_key("target_language", "Français")
    assert config_utils.load_key("target_language") == "Français"
    config_utils.update_keys({"subtitle.max_length": 42, "burn_subtitles": False})
    assert config_utils.load_key("subtitle.max_length") == 42
    assert config_utils.load_key("subtitle")["max_length"] == 42
    assert config_utils.load_key("burn_subtitles") is False

    # Zero-latency invalidation when user edits config.yaml externally
    raw_text = cfg_file.read_text(encoding="utf-8").replace(
        "max_length: 42", "max_length: 68"
    )
    cfg_file.write_text(raw_text, encoding="utf-8")
    assert config_utils.load_key("subtitle.max_length") == 68
    assert config_utils.load_key("subtitle")["max_length"] == 68

    # Bidirectional overrides in task_config_context (nested dict vs dot-key)
    with config_utils.task_config_context(
        {"subtitle": {"max_length": 24, "target_multiplier": 1.05}}
    ):
        assert config_utils.load_key("subtitle.max_length") == 24
        assert config_utils.load_key("subtitle.target_multiplier") == 1.05
        sub_section = config_utils.load_key("subtitle")
        assert sub_section["max_length"] == 24
        assert "style" in sub_section  # sibling keys from config.yaml preserved

        # Updating a key while inside task_config_context updates active override too
        config_utils.update_key("subtitle.max_length", 30)
        assert config_utils.load_key("subtitle.max_length") == 30
        assert config_utils.load_key("subtitle")["max_length"] == 30

    # Opaque dict keys (fish_audio.voices) preserve dots in user-defined labels and replace cleanly
    with config_utils.task_config_context(
        {"fish_audio": {"voices": {"Dr. Smith (en)": "voice-123"}}}
    ):
        assert config_utils.load_key("fish_audio.voices") == {
            "Dr. Smith (en)": "voice-123"
        }
        assert config_utils.load_key("fish_audio")["voices"] == {
            "Dr. Smith (en)": "voice-123"
        }
        config_utils.update_key("fish_audio.voices", {"Mr. Jones (uk)": "voice-456"})
        assert config_utils.load_key("fish_audio.voices") == {
            "Mr. Jones (uk)": "voice-456"
        }
        assert config_utils.load_key("fish_audio")["voices"] == {
            "Mr. Jones (uk)": "voice-456"
        }

    # Nested task_config_context isolation and empty-dict replacement
    with config_utils.task_config_context({"subtitle.max_length": 50}):
        with config_utils.task_config_context({"subtitle": {}}):
            assert config_utils.load_key("subtitle") == {}
            config_utils.update_key("subtitle.max_length", 36)
            assert config_utils.load_key("subtitle.max_length") == 36
        # Exiting inner context restores outer override without pollution
        assert config_utils.load_key("subtitle.max_length") == 50


def test_stage_04_utterance_editor_sqlite_persistence_and_webui_controls(
    tmp_path: Path, monkeypatch
):
    """WebUI exposes Stage 04 interactive utterance controls and PUT /api/tasks/{id}/utterances persists into project.sqlite and syncs SRT/ASS."""
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr("core.workspace.OUTPUT_ROOT", output_root)
    monkeypatch.setattr(api, "OUTPUT", output_root)

    ws = TaskWorkspace(output_root / "task_edit")
    ws.ensure_dirs()
    ws.save_words(
        [
            {"idx": 0, "text": "Hello", "punct_after": "", "start": 0.0, "end": 0.8},
            {"idx": 1, "text": "world", "punct_after": ".", "start": 0.8, "end": 1.6},
        ]
    )
    ws.save_utterances(
        [
            {
                "id": "u_0000",
                "index": 0,
                "word_span": [0, 2],
                "start": 0.0,
                "end": 1.6,
                "gap_after": 0.0,
                "src": "Hello world.",
                "tgt": "你好世界。",
                "tts_text": "你好世界。",
                "sub_cues": [
                    {
                        "src": "Hello world.",
                        "tgt": "你好世界。",
                        "trans": "你好世界。",
                        "word_span": [0, 2],
                        "start": 0.0,
                        "end": 1.6,
                    }
                ],
            }
        ]
    )
    ws.src_srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,600\nHello world.\n\n", encoding="utf-8"
    )
    ws.trans_srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,600\n你好世界。\n\n", encoding="utf-8"
    )

    client = TestClient(api.app)
    html_res = client.get("/")
    assert html_res.status_code == 200
    assert 'id="btn-save-utterances"' not in html_res.text
    assert 'id="btn-save-utterances-and-continue"' in html_res.text

    put_res = client.put(
        "/api/tasks/task_edit/utterances",
        json={
            "utterances": [
                {
                    "id": "u_0000",
                    "index": 0,
                    "word_span": [0, 2],
                    "start": 0.0,
                    "end": 1.6,
                    "gap_after": 0.0,
                    "src": "Hello edited world.",
                    "tgt": "你好，修订后的世界。",
                    "tts_text": "大家注意，这是配音专享词。",
                    "sub_cues": [
                        {
                            "src": "Hello world.",
                            "tgt": "你好世界。",
                            "trans": "你好世界。",
                            "word_span": [0, 2],
                            "start": 0.0,
                            "end": 1.6,
                        }
                    ],
                }
            ]
        },
    )
    assert put_res.status_code == 200
    assert put_res.json()["saved"] == 1

    loaded = ws.load_utterances()
    assert len(loaded) == 1
    assert loaded[0]["src"] == "Hello edited world."
    assert loaded[0]["tgt"] == "你好，修订后的世界。"
    assert loaded[0]["tts_text"] == "大家注意，这是配音专享词。"
    assert loaded[0]["sub_cues"][0]["tgt"] == "你好，修订后的世界。"
    assert "修订后的世界" in ws.trans_srt_path.read_text(encoding="utf-8")
    assert not (ws.audio_dir / "trans_subs_for_audio.srt").exists()
    assert ws.db_path.is_file()


def test_sqlite_only_step11_and_dubbing_chunk_invalidation(tmp_path: Path):
    """Step 11 loads dub_tasks directly from project.sqlite; editing utterances invalidates stale dubbing."""
    ws = TaskWorkspace(tmp_path / "task_dub_sqlite")
    ws.ensure_dirs()
    seg_wav = ws.segs_dir / "1_0.wav"
    AudioSegment.silent(duration=500, frame_rate=32000).export(
        str(seg_wav), format="wav"
    )

    # 1. Step 8.1 writes initial dub_tasks without 'lines' / 'cut_off' -> is_step_done("dub_tasks") must be False
    ws.save_dub_tasks(
        [
            {
                "number": 1,
                "start_time": "00:00:00.000",
                "end_time": "00:00:01.000",
                "duration": 1.0,
                "text": "你好世界",
                "origin": "Hello world",
            }
        ]
    )
    assert ws.has_dub_tasks() is True
    assert ws.has_dub_chunks() is False
    assert ws.is_step_done("dub_tasks") is False

    # 2. Step 8.2 / Step 10 writes full dub_tasks with 'lines', 'cut_off', and 'new_sub_times'
    ws.save_dub_tasks(
        [
            {
                "number": 1,
                "start_time": "00:00:00.000",
                "end_time": "00:00:01.000",
                "duration": 1.0,
                "text": "你好世界",
                "origin": "Hello world",
                "cut_off": 1,
                "lines": ["你好世界"],
                "src_lines": ["Hello world"],
                "new_sub_times": [[0.0, 0.5]],
            }
        ]
    )
    assert ws.has_dub_chunks() is True
    assert ws.is_step_done("dub_tasks") is True

    # 3. Step 11 load_and_flatten_data and create_srt_subtitle work purely from project.sqlite
    with workspace_context(ws):
        df, lines, new_sub_times = _11_merge_audio.load_and_flatten_data()
        assert len(df) == 1
        assert lines == ["你好世界"]
        assert new_sub_times == [[0.0, 0.5]]
        _11_merge_audio.create_srt_subtitle()
    assert ws.dub_srt_path.is_file()
    assert "你好世界" in ws.dub_srt_path.read_text(encoding="utf-8")


def test_concurrent_workspace_isolation_runtime_state_and_sqlite_llm_cache(
    tmp_path: Path,
):
    """Concurrent ContextThreadPoolExecutor tasks isolate runtime state, use SQLite WAL + llm_cache, and never touch config.yaml."""
    from core import _8_1_audio_task, _8_2_dub_chunks
    from core.concurrency import ContextThreadPoolExecutor
    from core.utils.ask_gpt import _load_cache, _save_cache

    ws_a = TaskWorkspace(tmp_path / "task_a")
    ws_b = TaskWorkspace(tmp_path / "task_b")
    ws_a.ensure_dirs()
    ws_b.ensure_dirs()

    def worker_job(
        ws: TaskWorkspace, lang: str, target: str
    ) -> tuple[str, str, object]:
        with (
            workspace_context(ws),
            config_utils.task_config_context({"target_language": target}),
        ):
            config_utils.set_runtime_state("whisper.detected_language", lang)
            _save_cache(
                model="test-model",
                prompt=f"hello-{lang}",
                resp_content=f'{{"ans": "{lang}"}}',
                resp_type="json",
                resp={"ans": lang},
                log_title="unit_test",
            )
            with ContextThreadPoolExecutor(max_workers=2) as inner_pool:
                det = inner_pool.submit(
                    config_utils.load_key, "whisper.detected_language"
                ).result()
                tgt = inner_pool.submit(
                    config_utils.load_key, "target_language"
                ).result()
                cached = inner_pool.submit(
                    _load_cache,
                    "hello-" + lang,
                    "json",
                    "unit_test",
                ).result()
            return det, tgt, cached

    with ContextThreadPoolExecutor(max_workers=2) as outer_pool:
        fa = outer_pool.submit(worker_job, ws_a, "ja", "简体中文")
        fb = outer_pool.submit(worker_job, ws_b, "fr", "English")
        res_a = fa.result()
        res_b = fb.result()

    assert res_a == ("ja", "简体中文", {"ans": "ja"})
    assert res_b == ("fr", "English", {"ans": "fr"})
    assert ws_a.get_runtime_state("whisper.detected_language") == "ja"
    assert ws_b.get_runtime_state("whisper.detected_language") == "fr"
    assert ws_a.load_llm_cache(
        log_title="unit_test", prompt="hello-ja", resp_type="json"
    ) == (True, {"ans": "ja"})
    assert ws_b.load_llm_cache(
        log_title="unit_test", prompt="hello-fr", resp_type="json"
    ) == (True, {"ans": "fr"})

    with ws_a._db(write=False) as conn:
        assert conn is not None
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"

    # Verify Step 8.1 and 8.2 read directly from src.srt and trans.srt as single source of truth
    ws_a.src_srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n世界\n\n"
        "3\n00:00:02,000 --> 00:00:03,000\n(笑)\n\n",
        encoding="utf-8",
    )
    ws_a.trans_srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n第一行字幕\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n第二行字幕\n\n"
        "3\n00:00:02,000 --> 00:00:03,000\n（笑声）\n\n",
        encoding="utf-8",
    )
    assert not (ws_a.audio_dir / "trans_subs_for_audio.srt").exists()
    with workspace_context(ws_a):
        df_tasks = _8_1_audio_task.process_srt()
        assert len(df_tasks) == 1
        assert df_tasks.iloc[0]["text"] == "第一行字幕 第二行字幕"
        assert df_tasks.iloc[0]["origin"] == "こんにちは 世界"
        ws_a.save_dub_tasks(df_tasks.to_dict(orient="records"))
        _8_2_dub_chunks.gen_dub_chunks()
    dub_tasks = ws_a.load_dub_tasks()
    assert len(dub_tasks) == 1
    assert dub_tasks[0]["lines"] == ["第一行字幕", "第二行字幕"]
    assert dub_tasks[0]["src_lines"] == ["こんにちは", "世界"]

    # Verify invalidate_from_stage("asr") and clear_runtime_cache evict cached runtime state
    import shutil

    ws_a.invalidate_from_stage("asr")
    assert ws_a.get_runtime_state("whisper.detected_language") is None
    ws_b.clear_runtime_cache()
    shutil.rmtree(ws_b.root)
    ws_b_recreated = TaskWorkspace(tmp_path / "task_b")
    ws_b_recreated.ensure_dirs()
    assert ws_b_recreated.get_runtime_state("whisper.detected_language") is None


def test_windows_nvenc_cuda_filter_and_ntfs_resilience(tmp_path: Path, monkeypatch):
    """Verify RTX 2070 NVENC VBV args, scale_cuda+hwdownload subtitle filter wrapping, PermissionError retry, and quoted Windows paths."""
    import os
    from core.utils import subtitle_style
    from core.workspace import replace_with_retry, rmtree_robust

    # 1. NVENC encoder args (speed-first p1 preset + VBR max 6000k + yuv420p)
    monkeypatch.setattr(subtitle_style.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        subtitle_style,
        "_available_ffmpeg_encoders",
        lambda: frozenset({"h264_nvenc", "libx264"}),
    )
    nvenc_args = subtitle_style.get_optimal_video_encoder_args(use_gpu=True)
    assert nvenc_args == [
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p1",
        "-rc",
        "vbr",
        "-cq",
        "23",
        "-maxrate",
        "6000k",
        "-bufsize",
        "12000k",
        "-pix_fmt",
        "yuv420p",
    ]

    monkeypatch.setattr(subtitle_style.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        subtitle_style,
        "_available_ffmpeg_encoders",
        lambda: frozenset({"h264_videotoolbox", "libx264"}),
    )
    vt_args = subtitle_style.get_optimal_video_encoder_args(use_gpu=True)
    assert vt_args == [
        "-c:v",
        "h264_videotoolbox",
        "-q:v",
        "65",
        "-maxrate",
        "6000k",
        "-bufsize",
        "12000k",
        "-pix_fmt",
        "yuv420p",
    ]
    monkeypatch.setattr(subtitle_style.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        subtitle_style,
        "_available_ffmpeg_encoders",
        lambda: frozenset({"h264_nvenc", "libx264"}),
    )

    # 2. CUDA decode + scale_cuda NV12->YUV420P pre-conversion before libass
    monkeypatch.setattr(
        subtitle_style,
        "_available_ffmpeg_filters",
        lambda: frozenset({"scale_cuda", "hwdownload", "ass"}),
    )
    in_args, vf = subtitle_style.wrap_gpu_subtitle_filter(
        "ass='output/bi.ass'",
        use_gpu=True,
        video_meta={"rotation": 0, "codec_name": "h264", "pix_fmt": "yuv420p"},
    )
    assert in_args == [
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-thread_queue_size",
        "1024",
    ]
    assert (
        vf == "scale_cuda=format=yuv420p,hwdownload,format=yuv420p,ass='output/bi.ass'"
    )

    # Rotated videos (-autorotate CPU transpose) and RTX 2070 non-NVDEC streams (AV1, 10-bit H.264) fall back to -hwaccel auto
    for incompatible_meta in (
        {"rotation": 90, "codec_name": "h264", "pix_fmt": "yuv420p"},
        {"rotation": 0, "codec_name": "av1", "pix_fmt": "yuv420p"},
        {"rotation": 0, "codec_name": "h264", "pix_fmt": "yuv420p10le"},
    ):
        in_fallback, vf_fallback = subtitle_style.wrap_gpu_subtitle_filter(
            "ass='output/bi.ass'",
            use_gpu=True,
            video_meta=incompatible_meta,
        )
        assert in_fallback == ["-hwaccel", "auto", "-thread_queue_size", "1024"]
        assert vf_fallback == "ass='output/bi.ass'"

    # Fallback when scale_cuda is absent in FFmpeg build
    monkeypatch.setattr(
        subtitle_style,
        "_available_ffmpeg_filters",
        lambda: frozenset({"ass"}),
    )
    in_args_cpu, vf_cpu = subtitle_style.wrap_gpu_subtitle_filter(
        "ass='output/bi.ass'", use_gpu=True
    )
    assert in_args_cpu == ["-hwaccel", "auto", "-thread_queue_size", "1024"]
    assert vf_cpu == "ass='output/bi.ass'"

    # 3. replace_with_retry survives transient Windows NTFS PermissionError
    src = tmp_path / "src.tmp"
    dst = tmp_path / "dst.json"
    src.write_text("ok", encoding="utf-8")
    real_replace = os.replace
    attempts = []

    def flaky_replace(a, b):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError(5, "Access is denied")
        return real_replace(a, b)

    monkeypatch.setattr(os, "replace", flaky_replace)
    replace_with_retry(src, dst)
    assert len(attempts) == 3 and dst.read_text(encoding="utf-8") == "ok"
    monkeypatch.setattr(os, "replace", real_replace)

    # 4. Quoted Windows 'Copy as path' string normalization in api._clean_user_path and rmtree_robust
    video_file = tmp_path / "demo_clip.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    assert api._clean_user_path(f'  "{video_file}"  ') == str(video_file)
    ro_dir = tmp_path / "ro_ws"
    ro_dir.mkdir()
    ro_file = ro_dir / "locked.sqlite"
    ro_file.write_bytes(b"sqlite")
    os.chmod(ro_file, 0o444)
    rmtree_robust(ro_dir)
    assert not ro_dir.exists()


def test_sse_events_terminates_immediately_on_first_sigint():
    """First Ctrl+C (SIGINT) via _VideoLingoServer.handle_exit wakes and closes active SSE streams without hanging."""
    import asyncio
    import signal
    from types import SimpleNamespace

    api._shutdown_event.clear()
    try:

        async def _exercise():
            req = SimpleNamespace(
                is_disconnected=lambda: asyncio.sleep(0, result=False)
            )
            resp = await api.sse_events(req)  # type: ignore[arg-type]
            gen = aiter(resp.body_iterator)

            first_chunk = await asyncio.wait_for(anext(gen), timeout=1.0)
            assert '"type": "connected"' in str(first_chunk)
            assert len(api._sse_subscribers) == 1

            server = api._create_uvicorn_server(port=0)
            assert server.config.timeout_graceful_shutdown == 1
            server.handle_exit(signal.SIGINT, None)
            assert api._shutdown_event.is_set()

            remaining = []
            async for chunk in gen:
                remaining.append(chunk)
            assert remaining == []
            assert len(api._sse_subscribers) == 0

        asyncio.run(_exercise())
    finally:
        api._shutdown_event.clear()


def test_webui_build_index_matches_checked_in_html():
    from core.webui.build import INDEX_OUT, build_index_html

    assert build_index_html() == INDEX_OUT.read_text(encoding="utf-8")


def test_sqlite_thread_exit_finalizer_and_unicode_log_lines(tmp_path):
    import gc
    import threading
    import weakref
    import core.workspace.db_core as db_core

    ws = TaskWorkspace(tmp_path / "pool_task")
    ws.set_kv("probe", 1)

    holder_refs: list[weakref.ReferenceType[object]] = []
    holder_dicts: list[dict] = []

    def _worker() -> None:
        assert ws.get_kv("probe") == 1
        holder = getattr(db_core._THREAD_LOCAL, "holder", None)
        assert holder is not None and len(holder.conns) == 1
        holder_refs.append(weakref.ref(holder))
        holder_dicts.append(holder.conns)

    t = threading.Thread(target=_worker)
    t.start()
    t.join(5.0)
    gc.collect()
    assert len(holder_refs) == 1 and holder_refs[0]() is None
    assert holder_dicts == [{}]

    errors: list[Exception] = []
    stop_evt = threading.Event()

    def _reader() -> None:
        while not stop_evt.is_set():
            try:
                ws.get_kv("probe")
            except Exception as exc:
                errors.append(exc)
                break

    readers = [threading.Thread(target=_reader) for _ in range(3)]
    for r in readers:
        r.start()
    for _ in range(25):
        ws.clear_runtime_cache(None)
    stop_evt.set()
    for r in readers:
        r.join(5.0)
    assert errors == []

    ws.append_log("lineA\u2028lineB\u2029lineC\x85end")
    logs = ws.load_logs(limit=10)
    assert len(logs) == 1
    assert logs[0]["msg"] == "lineA\u2028lineB\u2029lineC\x85end"


def test_get_term_index_invalidates_when_utterances_change(tmp_path):
    import core._4_1_summarize as s41

    ws = TaskWorkspace(tmp_path / "shorthand_task")
    ws.save_terms(
        "Destiny",
        [{"src": "Song of Flame", "tgt": "烈焰之歌", "aliases": [], "note": ""}],
    )
    ws.save_utterances(
        [{"id": "u_0001", "src": "The sung requirement is easy.", "tgt": ""}]
    )
    with workspace_context(ws.root):
        assert (
            s41.search_things_to_note_in_prompt("The Song requirement is easy.") is None
        )
        ws.save_utterances(
            [{"id": "u_0001", "src": "The Song requirement is easy.", "tgt": ""}]
        )
        prompt_note = s41.search_things_to_note_in_prompt(
            "The Song requirement is easy."
        )
        assert prompt_note is not None
        assert "烈焰之歌" in prompt_note
