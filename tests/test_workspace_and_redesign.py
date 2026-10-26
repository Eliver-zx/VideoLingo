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
from core.utils.subtitle_style import build_ass_from_srt_pair
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
    """terms.json aliases match multi-word ASR variants (e.g. 'pie torch' -> 'PyTorch') and promote to global CSV."""
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
    """Step 5 records (max_length, multiplier) in utterances.json and re-runs when max_length increases."""
    root_cfg = Path(__file__).resolve().parents[1] / "config.yaml"
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
    pd.DataFrame(
        [{"Source": utterances[0]["src"], "Translation": utterances[0]["tgt"]}]
    ).to_excel(ws.log_dir / "translation_results.xlsx", index=False)

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


def test_single_pass_bi_ass_scales_portrait_9_16_coordinates(tmp_path: Path):
    """build_ass_from_srt_pair scales font size by min(W, H) / 288.0 so 9:16 portrait videos do not blow up fonts."""
    src_srt = tmp_path / "src.srt"
    trans_srt = tmp_path / "trans.srt"
    ass_out = tmp_path / "bi.ass"

    src_srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nHello vertical world\n\n", encoding="utf-8"
    )
    trans_srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n你好竖屏世界\n\n", encoding="utf-8"
    )

    build_ass_from_srt_pair(
        source_srt=src_srt,
        translation_srt=trans_srt,
        ass_path=ass_out,
        display_width=1080,
        display_height=1920,
    )
    ass_text = ass_out.read_text(encoding="utf-8")
    assert "PlayResX: 1080" in ass_text
    assert "PlayResY: 1920" in ass_text
    assert "Style: Translation," in ass_text
    assert "Style: Source," in ass_text
    assert "Dialogue: 1,0:00:00.00,0:00:02.00,Translation" in ass_text
    assert "Dialogue: 0,0:00:00.00,0:00:02.00,Source" in ass_text


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
        "VideoLingo Studio",
        "v3.0.0",
        '<meta name="color-scheme" content="light">',
        'id="sse-indicator"',
        'id="nav-tasks-btn"',
        'id="queue-count"',
        'id="ui-lang-select"',
        'id="open-glossary-btn"',
        'id="open-settings-btn"',
        'class="brand-logo"',
        'class="app-shell"',
        # Sidebar & Panels
        'class="acc-panel"',
        'id="sidebar-save-toast"',
        'id="sb-api-key"',
        'id="sb-api-base-url"',
        'id="sb-api-model"',
        'id="llm-models-datalist"',
        'id="btn-fetch-models"',
        'id="sb-llm-json"',
        'id="sb-max-workers"',
        'id="btn-check-api"',
        'id="api-check-result"',
        'id="sb-whisper-runtime"',
        'id="sb-whisper-backend"',
        'id="sb-qwen-model"',
        'id="sb-qwen-engine"',
        'id="asr-field-mai"',
        'id="btn-detect-mai-region"',
        'id="sb-trans-style"',
        'id="sb-sep-model"',
        'id="sb-sub-maxlen"',
        'id="btn-reset-maxlen"',
        'id="sb-demucs"',
        'id="sb-burn-subtitles"',
        'id="sb-ffmpeg-gpu"',
        'id="sb-pause-before"',
        'id="sb-pause-after"',
        'id="burn-style-section"',
        'id="sb-burn-mode-sub"',
        'id="sb-burn-mode-dub"',
        'class="sub-preview-stage"',
        'id="sub-preview-trans"',
        'id="sub-preview-src"',
        'id="sb-tts-provider"',
        'id="sb-tts-method"',
        'id="tts-provider-key-wrap"',
        'id="tts-sub-edge"',
        'id="tts-sub-openai"',
        'id="tts-sub-fish"',
        'id="tts-sub-sf-fish"',
        'id="tts-sub-sovits"',
        # View 1: Task Manager
        'id="view-task-manager"',
        'id="input-source-path"',
        'id="pick-local-btn"',
        'id="input-sub-path"',
        'id="quick-ytb-res"',
        'id="quick-ytb-cookies"',
        'id="upload-dropzone"',
        'id="input-file-upload"',
        'id="task-src-lang"',
        'id="task-tgt-lang"',
        'id="task-stage"',
        'id="task-autorun"',
        'id="create-task-btn"',
        'id="task-list"',
        'id="empty-state"',
        # View 2: Sticky Monitor
        'id="view-task-detail"',
        'class="sticky-monitor"',
        'id="btn-back-to-tasks"',
        'id="active-task-title"',
        'id="active-task-state-badge"',
        'id="btn-pause"',
        'id="btn-stop"',
        'id="btn-run-subtitles"',
        'id="btn-run-transcribe"',
        'id="btn-run-dubbing"',
        'id="btn-resume"',
        'id="btn-delete-task"',
        'class="monitor-body"',
        'class="monitor-video-wrap"',
        'id="monitor-placeholder"',
        'id="video-main-preview"',
        'id="active-task-meta"',
        'id="mon-src-sub"',
        'id="mon-src-dub"',
        'id="active-step-label"',
        'id="sub-video-badge"',
        'id="dub-video-badge"',
        'id="active-progress-pct"',
        'id="active-progress-bar"',
        'id="pause-banner"',
        'id="btn-save-and-resume"',
        'id="error-banner"',
        # 6-Step Waterfall
        'id="linear-waterfall"',
        'id="wf-stage-1"',
        'id="wf-sum-1"',
        'id="wf-body-1"',
        'id="audio-stems-box"',
        'id="wf-stage-2"',
        'id="wf-body-2"',
        'id="stage2-utt-count"',
        'data-stage="asr"',
        'data-stage="split"',
        'id="wf-stage-3"',
        'id="wf-body-3"',
        'id="add-term-row-btn"',
        'id="save-terms-btn"',
        'id="save-terms-rerun-btn"',
        'id="terms-tbody"',
        'id="wf-stage-4"',
        'id="wf-body-4"',
        'id="utt-search-input"',
        'id="sync-tgt-to-tts-btn"',
        'id="save-utterances-btn"',
        'id="save-utt-rerun-btn"',
        'id="utterances-tbody"',
        'id="wf-stage-5"',
        'id="btn-dl-sub-zip"',
        'id="wf-body-5"',
        'data-stage="sub_split"',
        'data-stage="sub_video"',
        'id="download-links-row"',
        'id="wf-stage-6"',
        'id="wf-body-6"',
        'id="truncated-banner"',
        'id="btn-jump-utterances"',
        'id="btn-stage6-start-dub"',
        'data-stage="dub_tts"',
        'id="btn-clean-dub"',
        # Dialogs & Modern Web
        'id="glossary-dialog"',
        'id="settings-dialog"',
        'closedby="any"',
        'class="table-wrapper"',
    ]
    for anchor in required_dom_anchors:
        assert anchor in res_root.text, f"Spec anchor missing in GET /: {anchor}"

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
    """Child threads in ThreadPoolExecutor must inherit workspace_context and task_config_context."""
    from concurrent.futures import ThreadPoolExecutor
    from core.utils.models import resolve_task_path

    ws = TaskWorkspace(tmp_path / "output" / "worker_task")
    ws.ensure_dirs()

    with (
        workspace_context(ws),
        config_utils.task_config_context(
            {"target_language": "Español", "max_workers": 2}
        ),
    ):
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_path = pool.submit(resolve_task_path, "output/log/terminology.json")
            f_lang = pool.submit(config_utils.load_key, "target_language")
            resolved_in_child = f_path.result()
            lang_in_child = f_lang.result()

    assert Path(resolved_in_child) == (ws.log_dir / "terminology.json")
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
    pd.DataFrame(words_a).to_excel(ws_a.log_dir / "cleaned_chunks.xlsx", index=False)
    (ws_a.log_dir / "split_by_meaning.txt").write_text(
        "We use pie torch.\n", encoding="utf-8"
    )
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
        assert chunks == ["We use PyTorch."]
        assert ws_a.load_utterances()[0]["src"] == "We use PyTorch."

    # Prepare translation_results.xlsx in task_a, then edit src, tgt, and tts_text via FastAPI
    pd.DataFrame(
        [{"Source": "We use PyTorch.", "Translation": "我们使用PyTorch。"}]
    ).to_excel(ws_a.log_dir / "translation_results.xlsx", index=False)
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
        # Display trans.srt has tgt ("我们使用PyTorch 2.0"), while trans_subs_for_audio.srt has tts_text!
        assert "我们使用PyTorch 2.0" in (ws_a.root / "trans.srt").read_text(
            encoding="utf-8"
        )
        assert "我们使用派托奇二点零，非常快。" in (
            ws_a.audio_dir / "trans_subs_for_audio.srt"
        ).read_text(encoding="utf-8")

        # Step 8.1 and 8.2 in task_a must read task_a's files and preserve spoken tts_text without touching task_b
        sr = 16000
        with wave.open(str(ws_a.audio_dir / "raw.mp3"), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(np.zeros(sr * 3, dtype=np.int16).tobytes())
        _8_1_audio_task.gen_audio_task_main()
        _8_2_dub_chunks.gen_dub_chunks()
        df_tasks_a = pd.read_excel(ws_a.audio_dir / "tts_tasks.xlsx")
        assert "我们使用派托奇二点零，非常快。" in str(df_tasks_a.loc[0, "lines"])
        assert not (ws_b.audio_dir / "tts_tasks.xlsx").exists()

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
