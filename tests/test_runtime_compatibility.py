"""Focused offline checks; never start ML or TTS services."""

import ast
from pathlib import Path
import sys
import types

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

ROOT = Path(__file__).resolve().parents[1]


def functions(relative, names, namespace):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    tree.body = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    exec(compile(tree, relative, "exec"), namespace)
    return namespace


@pytest.fixture
def parse():
    import math

    return functions(
        "core/utils/task_literals.py",
        ["_NumpyNumbers", "parse_task_literal"],
        {"ast": ast, "math": math},
    )["parse_task_literal"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("[[np.float64(0.1), np.float32(-2.5)]]", [[0.1, -2.5]]),
        ("[[0.1, 1.5]]", [[0.1, 1.5]]),
        ("['hello', 'np.float64(1)']", ["hello", "np.float64(1)"]),
    ],
)
def test_legacy_numbers(parse, text, expected):
    assert parse(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').system('echo bad')",
        "np.load(1)",
        "np.float64(float(1))",
        'np.float64("nan")',
        "np.float64(1,2)",
        "np.float64(1e999)",
        "np.float64(True)",
    ],
)
def test_arbitrary_expressions_rejected(parse, text):
    with pytest.raises((ValueError, TypeError)):
        parse(text)


def test_old_workbook_readers(tmp_path, parse):
    import pandas as pd
    from core.workspace import TaskWorkspace

    ws = TaskWorkspace(tmp_path)
    ws.save_dub_tasks(
        [
            {
                "number": 1,
                "lines": "['Hello', 'World']",
                "new_sub_times": "[[np.float64(0.1), np.float64(1.5)], [1.5, 2.0]]",
            }
        ]
    )
    ns = functions(
        "core/_11_merge_audio.py",
        ["load_and_flatten_data", "get_audio_files"],
        {
            "pd": pd,
            "parse_task_literal": parse,
            "OUTPUT_FILE_TEMPLATE": "{0}.wav",
            "get_active_workspace": lambda: ws,
        },
    )
    table, lines, times = ns["load_and_flatten_data"]()
    assert lines == ["Hello", "World"] and times == [[0.1, 1.5], [1.5, 2.0]]
    assert ns["get_audio_files"](table) == ["1_0.wav", "1_1.wav"]


@pytest.mark.parametrize("mode", ["global", "project", "missing", "partial", "direct"])
def test_cache_resolution_offline_first(tmp_path, monkeypatch, mode):
    namespace = functions(
        "core/asr_backend/whisperX_local.py",
        ["_complete_model_directory", "resolve_whisper_model"],
        {"Path": Path, "rprint": lambda *a: None},
    )
    complete = tmp_path / ("project/large-v3" if mode == "direct" else "snapshot")
    complete.mkdir(parents=True)
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (complete / name).write_bytes(b"fixture")
    partial = tmp_path / "partial"
    partial.mkdir()
    calls = []

    def download(name, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            if mode == "missing":
                raise LocalEntryNotFoundError("missing")
            if kwargs["cache_dir"] is not None:
                if mode == "global":
                    raise LocalEntryNotFoundError("missing")
                if mode == "partial":
                    return str(partial)
        return str(complete)

    module = types.ModuleType("faster_whisper.utils")
    setattr(module, "download_model", download)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", module)
    result = namespace["resolve_whisper_model"]("large-v3", tmp_path / "project")
    assert Path(result) == complete
    assert sum(not c.get("local_files_only", False) for c in calls) == (
        mode == "missing"
    )
    if mode == "direct":
        assert not calls
