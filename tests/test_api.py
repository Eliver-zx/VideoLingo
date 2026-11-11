"""Local multi-task API contract (`/api/tasks/*`) and shared execution; no models or network requests."""

from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
import yaml

import api
from core import pipeline
from core.task_runner import TaskRunner, StopTask
from core.utils.decorator import except_handler
from core.workspace import TaskWorkspace, workspace_context


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "allowed_video_formats": ["mp4"],
                "allowed_audio_formats": ["wav", "mp3"],
                "ytb_resolution": "360",
                "target_language": "English",
                "glossary_domain": "destiny2",
                "whisper": {"language": "auto", "detected_language": "en"},
            }
        ),
        encoding="utf-8",
    )
    r = TaskRunner()
    r.add_listener(api._on_runner_event)
    monkeypatch.setattr(api, "runner", r)
    with TestClient(api.app) as client:
        yield client
    api.runner.stop(clear_queue=True)
    if api.runner._thread:
        api.runner._thread.join(3)
        assert not api.runner._thread.is_alive()


def finish(client, task_id: str | None = None):
    if api.runner._thread:
        api.runner._thread.join(3)
        assert not api.runner._thread.is_alive()
    if task_id:
        return client.get(f"/api/tasks/{task_id}").json()
    return client.get("/api/tasks").json()


def create_local_task(client, filename="sample.mp4", autorun=False, **options) -> str:
    p = Path(filename)
    if not p.exists():
        p.write_bytes(b"test media")
    response = client.post(
        "/api/tasks",
        json={"source": str(p.resolve()), "autorun": autorun, **options},
    )
    assert response.status_code == 202, response.text
    return response.json()["task_id"]


def test_legacy_root_routes_are_deleted(client):
    for route in ("/input", "/run", "/status", "/archive", "/stop", "/resume"):
        assert client.post(route, json={}).status_code in (404, 405)


def test_task_creation_validates_source_and_isolates_workspaces(client):
    assert client.post("/api/tasks", json={"source": "missing.mp4"}).status_code == 422
    t1 = create_local_task(client, "first.mp4")
    t2 = create_local_task(client, "second.mp4")
    assert t1 != t2
    assert (Path("output") / t1 / "source.mp4").read_bytes() == b"test media"
    assert (Path("output") / t2 / "source.mp4").read_bytes() == b"test media"


def test_create_task_does_not_autorun_by_default_and_manual_run_uses_desired_stage(
    client, monkeypatch
):
    Path("manual.mp4").write_bytes(b"test media")
    calls = []

    def plan(stage, dubbing):
        calls.append((stage, dubbing))
        return [("Step", lambda: None)]

    monkeypatch.setattr(api, "get_steps", plan)
    res = client.post(
        "/api/tasks",
        json={
            "source": str(Path("manual.mp4").resolve()),
            "stage": "subtitles",
            "dubbing": False,
        },
    )
    assert res.status_code == 202
    data = res.json()
    task_id = data["task_id"]
    assert data["execution"]["state"] == "idle"
    assert data["execution"]["desired_stage"] == "subtitles"
    assert data["execution"]["dubbing_enabled"] is False
    assert calls == []

    run_res = client.post(f"/api/tasks/{task_id}/run", json={})
    assert run_res.status_code == 202
    state = finish(client, task_id)
    assert calls == [("subtitles", False)]
    assert state["execution"]["state"] == "completed"

    res_up = client.post(
        "/api/tasks/upload",
        files={"file": ("clip.mp4", b"test media", "video/mp4")},
        data={"stage": "transcribe"},
    )
    assert res_up.status_code == 202
    up_data = res_up.json()
    up_id = up_data["task_id"]
    assert up_data["execution"]["state"] == "idle"
    assert up_data["execution"]["desired_stage"] == "transcribe"
    assert calls == [("subtitles", False)]

    assert client.post(f"/api/tasks/{up_id}/run", json={}).status_code == 202
    finish(client, up_id)
    assert calls == [("subtitles", False), ("transcribe", False)]


def test_url_uses_existing_downloader(client, monkeypatch):
    from core import _1_ytdlp

    calls = []
    monkeypatch.setattr(
        _1_ytdlp,
        "download_video_ytdlp",
        lambda source, **kwargs: calls.append((source, kwargs)),
    )
    res = client.post(
        "/api/tasks",
        json={"source": "https://example.com/video", "autorun": False},
    )
    assert res.status_code == 202
    task_id = res.json()["task_id"]
    state = finish(client, task_id)
    assert state["execution"]["state"] == "completed"
    assert len(calls) == 1
    assert calls[0][0] == "https://example.com/video"
    assert calls[0][1]["resolution"] == "360"


def test_run_uses_shared_plan_and_saves_task_overrides(client, monkeypatch):
    task_id = create_local_task(client)
    calls = []

    def plan(stage, dubbing):
        calls.append((stage, dubbing))
        return [
            (
                "Translate",
                lambda: (Path("output") / task_id / "trans.srt").write_text(
                    "translated", encoding="utf-8"
                ),
            )
        ]

    monkeypatch.setattr(api, "get_steps", plan)
    response = client.post(
        f"/api/tasks/{task_id}/run",
        json={
            "stage": "all",
            "dubbing": True,
            "target_language": "zh",
            "source_language": "ja",
        },
    )
    assert response.status_code == 202
    state = finish(client, task_id)
    assert calls == [("all", True)]
    assert state["execution"]["state"] == "completed"
    assert state["execution"]["progress"] == 1
    assert "trans.srt" in state["files"]
    assert state["config_override"]["target_language"] == "zh"
    assert state["config_override"]["whisper.detected_language"] == "ja"
    assert client.get(f"/api/tasks/{task_id}/files/trans.srt").text == "translated"
    assert client.get(f"/api/tasks/{task_id}/files/..%5Cconfig.yaml").status_code == 404


def test_busy_and_stopping_reject_active_task_mutations_and_drain_queue(
    client, monkeypatch
):
    t1 = create_local_task(client, "one.mp4")
    t2 = create_local_task(client, "two.mp4")
    entered, release = threading.Event(), threading.Event()
    t2_ran = threading.Event()

    def slow_work():
        entered.set()
        assert release.wait(3)
        TaskRunner.check_cancel()

    def fast_work():
        t2_ran.set()

    def plan(stage, dubbing):
        from core.workspace import get_active_workspace

        return [
            (
                "Step",
                lambda: (
                    slow_work() if get_active_workspace().task_id == t1 else fast_work()
                ),
            )
        ]

    monkeypatch.setattr(api, "get_steps", plan)
    try:
        assert client.post(f"/api/tasks/{t1}/run", json={}).status_code == 202
        assert entered.wait(1)
        # Running same task again or deleting while active is rejected with 409
        assert client.post(f"/api/tasks/{t1}/run", json={}).status_code == 409
        assert client.delete(f"/api/tasks/{t1}").status_code == 409
        # Enqueuing a second task while t1 is running succeeds and queues t2
        res_t2 = client.post(f"/api/tasks/{t2}/run", json={})
        assert res_t2.status_code == 202
        assert res_t2.json()["started_immediately"] is False
        assert client.get(f"/api/tasks/{t2}").json()["execution"]["state"] == "queued"
        assert client.post(f"/api/tasks/{t2}/run", json={}).status_code == 409
        # Stopping t1 transitions t1 to stopping, then drains and runs queued t2!
        assert client.post(f"/api/tasks/{t1}/stop").json()["state"] == "stopping"
    finally:
        release.set()
    assert t2_ran.wait(3)
    finish(client, t2)
    assert client.get(f"/api/tasks/{t1}").json()["execution"]["state"] == "stopped"
    assert client.get(f"/api/tasks/{t2}").json()["execution"]["state"] == "completed"


def test_failure_reports_step_and_can_retry(client, monkeypatch):
    task_id = create_local_task(client)

    def fail():
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(api, "get_steps", lambda *args: [("Translate", fail)])
    assert client.post(f"/api/tasks/{task_id}/run", json={}).status_code == 202
    state = finish(client, task_id)["execution"]
    assert state["state"] == "error" and state["current_step"] == "Translate"
    assert state["error"] == "synthetic failure"
    monkeypatch.setattr(api, "get_steps", lambda *args: [("Translate", lambda: None)])
    assert client.post(f"/api/tasks/{task_id}/run", json={}).status_code == 202
    assert finish(client, task_id)["execution"]["state"] == "completed"


def test_preconditions_and_schema(client):
    task_id = create_local_task(client, "sample.wav")
    assert (
        client.post(f"/api/tasks/{task_id}/run", json={"stage": "unknown"}).status_code
        == 422
    )
    assert (
        client.post(f"/api/tasks/{task_id}/run", json={"typo": True}).status_code == 422
    )
    assert (
        client.post(
            f"/api/tasks/{task_id}/run", json={"stage": "all", "dubbing": True}
        ).status_code
        == 422
    )
    assert "/api/tasks/{task_id}/run" in client.get("/openapi.json").json()["paths"]


def test_dubbing_requires_subtitles(client):
    task_id = create_local_task(client)
    assert (
        client.post(f"/api/tasks/{task_id}/run", json={"stage": "dubbing"}).status_code
        == 409
    )


def test_shared_pipeline_order_and_done_markers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def module(name):
        return SimpleNamespace(
            **{
                call.split(".")[1]: lambda call=call: calls.append(call)
                for _, group in pipeline.SUBTITLE_STEPS + pipeline.DUBBING_STEPS
                for call in group
                if name == "core." + call.split(".")[0]
            }
        )

    monkeypatch.setattr(pipeline, "import_module", module)
    for _, step in pipeline.get_steps("all", dubbing=True):
        step()
    expected = [
        call
        for _, group in pipeline.SUBTITLE_STEPS + pipeline.DUBBING_STEPS
        for call in group
    ]
    assert "_7_sub_into_vid.merge_subtitles_to_video" not in expected
    assert calls == expected
    assert Path("output/.subtitle_done").exists()
    assert Path("output/.dubbing_done").exists()
    burn_steps = pipeline.get_steps("burn")
    assert len(burn_steps) == 1
    assert burn_steps[0][0] == "Merging subtitles into the video"
    monkeypatch.setattr(
        pipeline,
        "import_module",
        lambda _: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    with pytest.raises(RuntimeError):
        pipeline.get_steps("subtitles")[0][1]()
    assert not Path("output/.subtitle_done").exists()


def test_pause_stop_and_no_following_steps():
    runner = TaskRunner()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def work():
        entered.set()
        assert release.wait(3)

    try:
        runner.start([("one", work), ("two", lambda: calls.append("two"))])
        assert entered.wait(1)
        runner.pause()
        assert runner.state == "paused"
        runner.stop()
        with pytest.raises(RuntimeError):
            runner.start([])
    finally:
        release.set()
        assert runner._thread is not None
        runner._thread.join(3)
    assert runner.state == "stopped" and not runner.is_active
    assert calls == []


def test_cancellation_is_not_retried():
    calls = []

    @except_handler("test", retry=3, delay=0)
    def cancel():
        calls.append(1)
        raise StopTask()

    with pytest.raises(StopTask):
        cancel()
    assert calls == [1]


def test_api_import_does_not_load_ui_or_models():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import api, sys; assert not {'streamlit', 'torch', 'spacy', 'qwen_asr'} & sys.modules.keys()",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_transcribe_stage_and_resume(client, monkeypatch):
    task_id = create_local_task(client)
    assert client.post(f"/api/tasks/{task_id}/resume").status_code == 409
    calls = []

    def plan(stage, dubbing):
        calls.append(stage)
        return [
            (
                "Checkpoint",
                lambda: (api.runner.pause("Review"), TaskRunner.check_cancel()),
            )
        ]

    monkeypatch.setattr(api, "get_steps", plan)
    assert (
        client.post(
            f"/api/tasks/{task_id}/run", json={"stage": "transcribe"}
        ).status_code
        == 202
    )
    for _ in range(300):
        if client.get(f"/api/tasks/{task_id}").json()["execution"]["state"] == "paused":
            break
        threading.Event().wait(0.01)
    assert (
        client.get(f"/api/tasks/{task_id}").json()["execution"]["pause_reason"]
        == "Review"
    )
    assert client.post(f"/api/tasks/{task_id}/resume").json()["state"] in (
        "running",
        "completed",
    )
    state = finish(client, task_id)["execution"]
    assert calls == ["transcribe"]
    assert state["state"] == "completed" and state["pause_reason"] is None


SRT = "1\n00:00:01,000 --> 00:00:02,000\nHello world.\n"


def test_input_of_subtitles_with_and_without_a_video(client):
    from core._1_ytdlp import find_media_file, find_subtitle_file

    Path("talk.srt").write_text(SRT, encoding="utf-8")
    subtitles = str(Path("talk.srt").resolve())

    t_vid = create_local_task(client, subtitles=subtitles)
    with workspace_context(TaskWorkspace(Path("output") / t_vid)):
        assert find_media_file() == (f"output/{t_vid}/source.mp4", "video")
        assert find_subtitle_file() == f"output/{t_vid}/source.srt"

    res_srt = client.post("/api/tasks", json={"source": subtitles, "autorun": False})
    assert res_srt.status_code == 202
    t_srt = res_srt.json()["task_id"]
    with workspace_context(TaskWorkspace(Path("output") / t_srt)):
        assert find_media_file() == (f"output/{t_srt}/source.srt", "subtitle")
    assert (
        client.post(
            f"/api/tasks/{t_srt}/run", json={"stage": "all", "dubbing": True}
        ).status_code
        == 422
    )


def test_input_refuses_a_file_without_subtitles(client):
    Path("sample.mp4").write_bytes(b"test media")
    Path("empty.srt").write_text("Hello world.", encoding="utf-8")
    Path("talk.txt").write_text(SRT, encoding="utf-8")
    Path("output").mkdir(exist_ok=True)
    Path("output/trans.srt").write_text(SRT, encoding="utf-8")
    video = str(Path("sample.mp4").resolve())
    for subtitles in ("empty.srt", "talk.txt", "missing.srt", "output/trans.srt"):
        request = {"source": video, "subtitles": subtitles, "autorun": False}
        assert client.post("/api/tasks", json=request).status_code == 422
        assert (
            client.post(
                "/api/tasks", json={"source": subtitles, "autorun": False}
            ).status_code
            == 422
        )


def test_nested_progress_scope_and_log_event_ring_buffer(tmp_path, monkeypatch):
    from core._4_2_translate import ChunkTranslateResult, _flush_chunk_translation_live
    from core.task_runner import TaskRunner, progress_scope

    monkeypatch.chdir(tmp_path)
    runner = TaskRunner(active_task_id="task_a", state="running")
    events: list[dict] = []
    runner.add_listener(events.append)
    prev_current = TaskRunner._current
    TaskRunner._current = runner
    try:
        with progress_scope(0.20, 0.80):
            TaskRunner.report_progress(
                0.0, "外层起点", "开始外层阶段", tag="asr", label="DEMUCS"
            )
            with progress_scope(0.50, 1.0):
                TaskRunner.report_progress(
                    0.50, "内层中点", "内层进度 50%", tag="asr", label="VAD"
                )
            TaskRunner.report_progress(
                1.0,
                "外层终点",
                "[green]完成外层阶段[/green]",
                tag="asr",
                label="DEMUCS",
            )
            # Deduplication when message is None and sub_step + bucket are unchanged
            TaskRunner.report_progress(1.0, "外层终点")

        prog_events = [e for e in events if e.get("type") == "task_progress"]
        assert len(prog_events) == 3
        assert [e["sub_progress"] for e in prog_events] == [0.2, 0.65, 0.8]
        assert (
            prog_events[1]["log_tag"] == "asr" and prog_events[1]["log_label"] == "VAD"
        )
        assert prog_events[2]["message"] == "完成外层阶段"

        TaskRunner.log_event(
            "[yellow]⚠️ 命中重试[/yellow]", tag="warn", label="RETRY", level="warn"
        )
        log_events = [e for e in events if e.get("type") == "task_log"]
        assert len(log_events) == 1
        assert log_events[0]["message"] == "命中重试"
        assert log_events[0]["warn"] is True
        assert any(
            item["msg"] == "命中重试" and item["warn"] for item in runner.recent_logs()
        )

        # Partial translation flush keeps has_translations() False until final save
        ws_a = TaskWorkspace(Path("output") / "task_a")
        ws_b = TaskWorkspace(Path("output") / "task_b")
        ws_a.ensure_dirs()
        ws_b.ensure_dirs()

        utts_a = [
            {"id": "u_0000", "index": 0, "src": "Hello world.", "tgt": ""},
            {"id": "u_0001", "index": 1, "src": "Second line.", "tgt": ""},
        ]
        ws_a.save_utterances(utts_a, mark_translated=False)
        assert ws_a.has_utterances() is True
        assert ws_a.has_translations() is False

        res_chunk = ChunkTranslateResult(
            0, "Hello world.", "你好世界。", ["你好世界直译。"]
        )
        ts = _flush_chunk_translation_live(
            ws_a,
            utts_a,
            [0, 1],
            [(0, ["Hello world."]), (1, ["Second line."])],
            res_chunk,
            0.0,
        )
        assert ts > 0.0
        assert ws_a.has_translations() is False
        saved_partial = ws_a.load_utterances()
        assert saved_partial[0]["tgt"] == "你好世界。"
        assert saved_partial[0]["direct"] == "你好世界直译。"
        assert saved_partial[1]["tgt"] == ""

        # Active task artifact notification includes stages_done with translate=False during partial flush
        # and does not re-log ambient pause_message when message is None
        runner.pause_message = "Waiting for terminology review before translation"
        log_count_before = len(runner.recent_logs())
        events.clear()
        ws_a.save_terms("AI Topic", [{"src": "Transformer", "tgt": "Transformer"}])
        art_events = [e for e in events if e.get("type") == "task_artifact"]
        assert len(art_events) == 1
        assert art_events[0]["task_id"] == "task_a"
        assert art_events[0]["artifact"] == "terms"
        assert art_events[0]["stages_done"]["terms"] is True
        assert art_events[0]["stages_done"]["translate"] is False
        assert len(runner.recent_logs()) == log_count_before
        runner.pause_message = ""

        # Cross-task isolation: saving on ws_b while task_a is active does not overwrite task_a stages_done
        events.clear()
        ws_b.save_terms("Other Topic", [{"src": "CNN", "tgt": "卷积网络"}])
        assert [e for e in events if e.get("type") == "task_artifact"] == []

        # Final save on ws_a marks translations complete
        utts_a[1]["tgt"] = "第二行。"
        ws_a.save_utterances(utts_a)
        assert ws_a.has_translations() is True
        assert runner.stages_done["translate"] is True
    finally:
        runner.remove_listener(events.append)
        TaskRunner._current = prev_current


def test_get_task_utterances_asr_words_fallback_and_raw_backfill(client):
    ws = TaskWorkspace(Path("output") / "asr_stage_task")
    ws.ensure_dirs()
    ws.save_words(
        [
            {"text": "are", "start": 0.0, "end": 0.2},
            {"text": "you", "start": 0.2, "end": 0.4},
            {"text": "ready", "start": 0.4, "end": 0.7},
            {"text": "for", "start": 1.5, "end": 1.7},
            {"text": "destiny", "start": 1.7, "end": 2.2},
        ]
    )

    res = client.get("/api/tasks/asr_stage_task/utterances")
    assert res.status_code == 200
    utts = res.json()["utterances"]
    assert len(utts) == 2
    assert utts[0]["src_raw"] == "are you ready"
    assert utts[0]["gap_after"] == 0.8
    assert utts[1]["src_raw"] == "for destiny"

    ws.save_utterances(
        [
            {
                "id": 0,
                "word_span": [0, 3],
                "start": 0.0,
                "end": 0.7,
                "gap_after": 0.8,
                "src": "Are you ready?",
                "tgt": "你准备好了吗？",
            }
        ]
    )
    res2 = client.get("/api/tasks/asr_stage_task/utterances")
    assert res2.status_code == 200
    utts2 = res2.json()["utterances"]
    assert len(utts2) == 1
    assert utts2[0]["src_raw"] == "are you ready"
    assert utts2[0]["src_punct"] == "Are you ready?"


def test_per_task_log_isolation_and_restart_persistence(client):
    task_a = create_local_task(client, filename="alpha.mp4")
    task_b = create_local_task(client, filename="beta.mp4")

    def step_a():
        TaskRunner.log_event("Alpha ASR complete", tag="asr", label="ASR")

    def step_b():
        TaskRunner.log_event(
            "Beta LLM warning", tag="warn", label="TRANS", level="warn"
        )

    api.runner.start([("Step A", step_a)], task_id=task_a)
    finish(client, task_a)

    api.runner.start([("Step B", step_b)], task_id=task_b)
    finish(client, task_b)

    post_res = client.post(
        f"/api/tasks/{task_a}/logs",
        json={
            "msg": "Client merge action",
            "tag": "llm",
            "label": "MERGE",
            "warn": False,
            "time": "12:00:01",
        },
    )
    assert post_res.status_code == 201

    assert any(x["msg"] == "Alpha ASR complete" for x in api.runner.recent_logs(task_a))
    assert not any(
        x["msg"] == "Beta LLM warning" for x in api.runner.recent_logs(task_a)
    )
    assert any(x["msg"] == "Beta LLM warning" for x in api.runner.recent_logs(task_b))
    assert not any(
        x["msg"] == "Alpha ASR complete" for x in api.runner.recent_logs(task_b)
    )

    ws_a = TaskWorkspace(Path("output") / task_a)
    ws_b = TaskWorkspace(Path("output") / task_b)
    assert ws_a.runtime_log_path.is_file()
    assert not (ws_a.log_dir / "runtime.log").exists()
    assert ws_b.runtime_log_path.is_file()

    # Simulate backend restart clearing in-memory runner buffers
    fresh_runner = TaskRunner()
    fresh_runner.add_listener(api._on_runner_event)
    api.runner = fresh_runner
    assert api.runner.recent_logs(task_a) == []

    logs_a = client.get(f"/api/tasks/{task_a}/logs").json()["logs"]
    logs_b = client.get(f"/api/tasks/{task_b}/logs").json()["logs"]

    msgs_a = [x["msg"] for x in logs_a]
    msgs_b = [x["msg"] for x in logs_b]
    assert "Alpha ASR complete" in msgs_a
    assert "Client merge action" in msgs_a
    assert all("Beta" not in m for m in msgs_a)

    assert "Beta LLM warning" in msgs_b
    assert any(
        x["msg"] == "Beta LLM warning" and x["warn"] is True and x["tag"] == "warn"
        for x in logs_b
    )
    assert all("Alpha" not in m and "Client merge" not in m for m in msgs_b)

    del_res = client.delete(f"/api/tasks/{task_a}")
    assert del_res.status_code == 200
    assert client.get(f"/api/tasks/{task_a}/logs").status_code == 404
    assert [
        x["msg"] for x in client.get(f"/api/tasks/{task_b}/logs").json()["logs"]
    ] == msgs_b
