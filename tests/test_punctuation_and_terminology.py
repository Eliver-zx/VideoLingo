"""Unit tests for LLM punctuation restoration, word-preserving projection, terminology matching, and translation caching."""

from __future__ import annotations

import json
import pandas as pd
import pytest

from core.task_runner import StopTask
from core.workspace import TaskWorkspace, workspace_context


# ------------------------------------------------------------------
# 1. _3_1_split_nlp: Word-preserving punctuation projection & overlap windows
# ------------------------------------------------------------------


def test_project_sentence_ends_preserves_words_even_when_llm_mutates_tokens(
    monkeypatch,
):
    """Even if LLM changes/inserts/deletes words in restored_text, original words are never mutated
    and _6_gen_sub.get_sentence_timestamps aligns 100% without raising ValueError."""
    import core._3_1_split_nlp as split_nlp
    from core._6_gen_sub import get_sentence_timestamps

    raw_words = [
        "welcome",
        "to",
        "banji",
        "studio",
        "today",
        "we",
        "are",
        "going",
        "to",
        "build",
        "a",
        "new",
        "system",
        "and",
        "it",
        "runs",
        "very",
        "fast",
    ]
    df_words = pd.DataFrame(
        {
            "text": raw_words,
            "start": [i * 0.5 for i in range(len(raw_words))],
            "end": [i * 0.5 + 0.4 for i in range(len(raw_words))],
        }
    )

    # Simulate LLM fixing typo 'banji' -> 'Bungie', changing 'we are' -> "we're", and inserting periods
    mutated_llm_restored = (
        "Welcome to Bungie studio today. "
        "We're going to build a new system. "
        "And it runs very fast!"
    )

    monkeypatch.setattr(split_nlp, "get_source_language", lambda: "en")
    monkeypatch.setattr(split_nlp, "load_key_or", lambda k, d=None: 2)
    monkeypatch.setattr(
        split_nlp,
        "ask_gpt",
        lambda *args, **kwargs: {"restored": mutated_llm_restored},
    )

    sentences = split_nlp.restore_and_split_words(raw_words, language="en")

    assert sentences == [
        "welcome to banji studio today.",
        "we are going to build a new system.",
        "and it runs very fast!",
    ]

    # Verify _6_gen_sub.get_sentence_timestamps succeeds with 0 character drift
    stamps = get_sentence_timestamps(df_words, pd.DataFrame({"Source": sentences}))
    assert len(stamps) == 3
    assert stamps[0] == (0.0, 4 * 0.5 + 0.4)
    assert stamps[1] == (5 * 0.5, 12 * 0.5 + 0.4)
    assert stamps[2] == (13 * 0.5, 17 * 0.5 + 0.4)


def test_overlapping_windows_do_not_force_sentence_breaks_at_chunk_boundaries(
    monkeypatch,
):
    """A sentence crossing the 400-word window boundary stays intact unless the LLM places a period."""
    import core._3_1_split_nlp as split_nlp

    # 25 words with chunk_words=12, overlap_words=4 -> 3 windows: [0..12), [8..20), [16..25)
    # Place only one period after word_9 ("end1") and one at the end ("word_24").
    words = [f"w{i}" for i in range(25)]

    def fake_ask_gpt(prompt, **kwargs):
        # Put a period only after w9 and w24, nowhere near the window boundaries (12 or 20)
        text = prompt.split("<transcript>")[1].split("</transcript>")[0].strip()
        text = text.replace("w9", "w9.").replace("w24", "w24.")
        return {"restored": text}

    monkeypatch.setattr(split_nlp, "ask_gpt", fake_ask_gpt)
    monkeypatch.setattr(split_nlp, "load_key_or", lambda k, d=None: 2)

    sentences = split_nlp.restore_and_split_words(
        words, language="en", chunk_words=12, overlap_words=4
    )

    assert len(sentences) == 2
    assert sentences[0] == " ".join(words[:10]) + "."
    assert sentences[1] == " ".join(words[10:]) + "."


def test_split_by_spacy_writes_both_nlp_and_meaning_files(monkeypatch, tmp_path):
    """split_by_spacy persists words and utterances into project.sqlite."""
    import core._3_1_split_nlp as split_nlp

    ws = TaskWorkspace(tmp_path)
    ws.save_words(
        [
            {"idx": i, "text": w, "start": float(i), "end": float(i) + 0.4}
            for i, w in enumerate(
                ['"Hello"', '"world"', '"this"', '"is"', '"a"', '"test"']
            )
        ]
    )

    monkeypatch.setattr(split_nlp, "get_source_language", lambda: "en")
    monkeypatch.setattr(split_nlp, "load_key_or", lambda k, d=None: 1)
    monkeypatch.setattr(
        split_nlp,
        "ask_gpt",
        lambda *args, **kwargs: {"restored": "Hello world. this is a test."},
    )

    with workspace_context(ws):
        split_nlp.split_by_spacy()

    assert [u["src"] for u in ws.load_utterances()] == [
        "Hello world.",
        "this is a test.",
    ]


def test_decimals_abbreviations_and_standalone_punct_do_not_break_sentences(
    monkeypatch,
):
    """Decimals (3.5), abbreviations (Mr., U.S., node.js), and standalone punct tokens
    do not cause mid-sentence splits or dangling leading punctuation."""
    import core._3_1_split_nlp as split_nlp

    words = [
        "Mr.",
        "Smith",
        "paid",
        "3.5",
        "dollars",
        "in",
        "the",
        "U.S.",
        "today",
        ".",
        "Next",
        "line",
    ]
    restored = "Mr. Smith paid 3.5 dollars in the U.S. today. Next line."

    monkeypatch.setattr(split_nlp, "ask_gpt", lambda *a, **kw: {"restored": restored})
    monkeypatch.setattr(split_nlp, "load_key_or", lambda k, d=None: 1)

    sentences = split_nlp.restore_and_split_words(words, language="en")
    assert sentences == [
        "Mr. Smith paid 3.5 dollars in the U.S. today.",
        "Next line.",
    ]


def test_split_nlp_propagates_stop_task(monkeypatch):
    """StopTask raised inside punctuation restoration must never be swallowed by fallback."""
    import core._3_1_split_nlp as split_nlp

    def raise_stop(*args, **kwargs):
        raise StopTask()

    monkeypatch.setattr(split_nlp, "ask_gpt", raise_stop)
    monkeypatch.setattr(split_nlp, "load_key_or", lambda k, d=None: 1)

    with pytest.raises(StopTask):
        split_nlp.restore_and_split_words(["hello", "world"], language="en")


# ------------------------------------------------------------------
# 2. _4_1_summarize: N-gram fuzzy matching, mtime cache invalidation, & 15-min chunking
# ------------------------------------------------------------------


def test_search_things_to_note_matches_exact_stem_and_asr_fuzzy_variants_in_long_chunk(
    monkeypatch, tmp_path
):
    """N-gram fuzzy matching finds ASR phonetic/spelling variants (e.g., 'Bungee', 'raycastin')
    even inside a 1500-char multi-sentence translation chunk."""
    import core._4_1_summarize as summarize

    ws = TaskWorkspace(tmp_path)
    custom_xlsx = tmp_path / "custom_terms.xlsx"
    pd.DataFrame(
        {
            "src": ["Bungie", "Raycasting"],
            "tgt": ["Bungie 工作室", "光线投射"],
            "note": ["游戏开发商", "渲染技术"],
            "aliases": ["Bungee", "raycastin"],
        }
    ).to_excel(custom_xlsx, index=False)

    ws.save_terms(
        "Game dev talk",
        [
            {"src": "Destiny", "tgt": "命运", "note": "游戏名"},
            {"src": "shader pipeline", "tgt": "着色器管线", "note": "图形学"},
        ],
    )

    monkeypatch.setattr(summarize, "CUSTOM_TERMS_PATH", str(custom_xlsx))
    monkeypatch.setattr(
        summarize, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "none.csv")
    )
    monkeypatch.setattr(
        summarize, "OPTIONAL_JSON_TERMS_PATH", str(tmp_path / "none.json")
    )

    filler = (
        "This is a normal conversational sentence about software engineering and graphics rendering. "
        * 12
    )
    chunk = (
        f"{filler}\n"
        "When we worked at Bungee we optimized all the shader pipelines using raycastin in real time.\n"
        f"{filler}"
    )
    assert len(chunk) > 1200

    with workspace_context(ws):
        prompt = summarize.search_things_to_note_in_prompt(chunk)
    assert prompt is not None
    assert "Bungee:Bungie 工作室 (Bungie)" in prompt
    assert "shader pipeline:着色器管线" in prompt
    assert "raycastin:光线投射 (Raycasting)" in prompt
    assert "Destiny" not in prompt


def test_short_terms_and_lookalikes_do_not_trigger_false_positives(
    monkeypatch, tmp_path
):
    """Short acronyms (AI, UI, RAM, ID, OS) must not match inside everyday words ('said', 'building',
    'program'), and lookalikes ('faster' vs 'raster', 'severe' vs 'server') must not false-match."""
    import core._4_1_summarize as summarize

    ws = TaskWorkspace(tmp_path)
    ws.save_terms(
        "Tech",
        [
            {"src": "AI", "tgt": "人工智能", "note": ""},
            {"src": "UI", "tgt": "界面", "note": ""},
            {"src": "RAM", "tgt": "内存", "note": ""},
            {"src": "ID", "tgt": "标识", "note": ""},
            {"src": "raster", "tgt": "光栅", "note": ""},
            {"src": "server", "tgt": "服务器", "note": ""},
            {"src": "model", "tgt": "模型", "note": ""},
        ],
    )
    monkeypatch.setattr(summarize, "CUSTOM_TERMS_PATH", str(tmp_path / "no.xlsx"))
    monkeypatch.setattr(summarize, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv"))
    monkeypatch.setattr(
        summarize, "OPTIONAL_JSON_TERMS_PATH", str(tmp_path / "no.json")
    )

    sentence = "He said that building this program in dark mode runs faster despite a severe issue."
    with workspace_context(ws):
        assert summarize.search_things_to_note_in_prompt(sentence) is None


def test_term_index_cache_invalidates_when_terminology_json_is_edited(
    monkeypatch, tmp_path
):
    """When user edits task terms during pause_before_translate, mtime changes and cache reloads."""
    import core._4_1_summarize as summarize

    ws = TaskWorkspace(tmp_path)
    monkeypatch.setattr(summarize, "CUSTOM_TERMS_PATH", str(tmp_path / "no.xlsx"))
    monkeypatch.setattr(summarize, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv"))
    monkeypatch.setattr(
        summarize, "OPTIONAL_JSON_TERMS_PATH", str(tmp_path / "no.json")
    )

    ws.save_terms("t", [{"src": "OpenGL", "tgt": "旧译名", "note": "n1"}])
    with workspace_context(ws):
        first_prompt = summarize.search_things_to_note_in_prompt("We use OpenGL here.")
        assert first_prompt is not None and "OpenGL:旧译名" in first_prompt

        # Simulate user editing task terms with updated mtime
        ws.save_terms("t", [{"src": "OpenGL", "tgt": "新译名_已修改", "note": "n2"}])
        ws.set_terms_mtime(ws.get_terms_mtime() + 2.0)

        second_prompt = summarize.search_things_to_note_in_prompt("We use OpenGL here.")
        assert second_prompt is not None and "OpenGL:新译名_已修改" in second_prompt


def test_get_summary_merges_chunks_and_custom_terms_consistently(monkeypatch, tmp_path):
    """Multi-chunk extraction runs concurrently and unifies cross-chunk variants with earlier chunk winning."""
    import core._4_1_summarize as summarize

    ws = TaskWorkspace(tmp_path)
    custom_xlsx = tmp_path / "custom_terms.xlsx"

    ws.save_utterances(
        [
            {"id": 0, "src": "Sentence one in chunk zero."},
            {"id": 1, "src": "Sentence two in chunk one."},
        ]
    )
    pd.DataFrame(
        {"src": ["Vulkan"], "tgt": ["Vulkan 接口"], "note": ["图形API"]}
    ).to_excel(custom_xlsx, index=False)

    monkeypatch.setattr(summarize, "CUSTOM_TERMS_PATH", str(custom_xlsx))
    monkeypatch.setattr(summarize, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv"))
    monkeypatch.setattr(
        summarize, "OPTIONAL_JSON_TERMS_PATH", str(tmp_path / "no.json")
    )
    monkeypatch.setattr(summarize, "SUMMARY_CHUNK_WEIGHTED_CHARS", 25)
    monkeypatch.setattr(summarize, "load_key_or", lambda k, d=None: 2)

    def fake_ask_gpt(prompt, **kwargs):
        if "chunk zero" in prompt:
            return {
                "theme": "Chunk 0 theme.",
                "terms": [
                    {"src": "Nanite", "tgt": "Nanite 虚拟微多边形", "note": "UE5特性"},
                    {"src": "Vulkan", "tgt": "错误覆盖", "note": "不应覆盖自定义术语"},
                ],
            }
        return {
            "theme": "Chunk 1 theme.",
            "terms": [
                {
                    "src": "Nanites",
                    "tgt": "不一致的译名",
                    "note": "复数形式应统一到Chunk 0",
                },
                {"src": "Lumen", "tgt": "Lumen 全局光照", "note": "光照系统"},
            ],
        }

    monkeypatch.setattr(summarize, "ask_gpt", fake_ask_gpt)

    with workspace_context(ws):
        getattr(summarize.get_summary, "__wrapped__")()

    saved = ws.load_terms()
    assert "Chunk 0 theme." in saved["summary"]
    terms_by_src = {t["src"]: t["tgt"] for t in saved["terms"]}
    assert terms_by_src["Vulkan"] == "Vulkan 接口"
    assert terms_by_src["Nanite"] == "Nanite 虚拟微多边形"
    assert terms_by_src["Nanites"] == "Nanite 虚拟微多边形"
    assert terms_by_src["Lumen"] == "Lumen 全局光照"

    with workspace_context(ws):
        note_prompt = summarize.search_things_to_note_in_prompt(
            "We render with Vulkan and Nanite in real time."
        )
    assert note_prompt is not None
    assert "Vulkan:Vulkan 接口" in note_prompt
    assert "Nanite:Nanite 虚拟微多边形" in note_prompt


# ------------------------------------------------------------------
# 3. _4_2_translate: Raw translation cache reuse & fingerprint invalidation
# ------------------------------------------------------------------


def test_translate_all_reuses_raw_cache_and_invalidates_on_terminology_change(
    monkeypatch, tmp_path
):
    """If project.sqlite utterances match fingerprint, LLM is skipped;
    if task terms change, cache is invalidated."""
    import core._4_2_translate as translate

    ws = TaskWorkspace(tmp_path)
    ws.save_words(
        [
            {"idx": 0, "text": "Hello", "start": 0.0, "end": 0.4},
            {"idx": 1, "text": "world.", "start": 0.5, "end": 0.9},
            {"idx": 2, "text": "How", "start": 1.0, "end": 1.4},
            {"idx": 3, "text": "are", "start": 1.5, "end": 1.9},
            {"idx": 4, "text": "you?", "start": 2.0, "end": 2.5},
        ]
    )
    ws.save_terms("test", [])

    monkeypatch.setattr(
        translate, "load_key", lambda k: 10.0 if k == "min_trim_duration" else 2
    )
    monkeypatch.setattr(translate, "load_key_or", lambda k, d=None: "zh")
    monkeypatch.setattr(
        translate,
        "translate_chunk",
        lambda *args, **kwargs: pytest.fail(
            "Should have reused SQLite utterance translation cache without calling LLM"
        ),
    )
    monkeypatch.setattr(
        translate,
        "align_timestamp",
        lambda df_text, df_trans, *a, **kw: df_trans.assign(duration=[1.0, 1.5]),
    )

    with workspace_context(ws):
        fp = translate._translation_cache_fingerprint()
        ws.save_utterances(
            [
                {"id": 0, "src": "Hello world."},
                {"id": 1, "src": "How are you?"},
            ]
        )
        ws.save_translation_cache(
            fp,
            [
                ("Hello world.", "你好，世界。"),
                ("How are you?", "你好吗？"),
            ],
        )

        getattr(translate.translate_all, "__wrapped__")()

        utts = ws.load_utterances()
        assert [u["src"] for u in utts] == ["Hello world.", "How are you?"]
        assert [u["tgt"] for u in utts] == ["你好，世界。", "你好吗？"]

        # Now modify terms in workspace -> fingerprint changes -> cache must return None
        ws.save_terms("updated", [{"src": "world", "tgt": "天下", "note": ""}])
        ws.set_terms_mtime(ws.get_terms_mtime() + 2.0)
        new_fp = translate._translation_cache_fingerprint()
        assert new_fp != fp
        assert (
            translate._load_cached_raw_translation(
                ["Hello world.", "How are you?"], fingerprint=new_fp
            )
            is None
        )
        # Empty fingerprint must be rejected
        assert (
            translate._load_cached_raw_translation(
                ["Hello world.", "How are you?"], fingerprint=""
            )
            is None
        )


def test_generate_shared_prompt_formats_list_and_string_context_cleanly():
    """generate_shared_prompt joins list inputs with newlines and turns None into empty string instead of 'None'."""
    from core.prompts import generate_shared_prompt

    with_lists = generate_shared_prompt(
        ["Prev line 1.", "Prev line 2."],
        ["Next line 1.", "Next line 2."],
        "Theme summary",
        "Term notes",
    )
    assert (
        "<previous_context>\nPrev line 1.\nPrev line 2.\n</previous_context>"
        in with_lists
    )
    assert (
        "<subsequent_context>\nNext line 1.\nNext line 2.\n</subsequent_context>"
        in with_lists
    )
    assert "### 术语库-词典" in with_lists
    assert "<terminology>\nTerm notes\n</terminology>" in with_lists
    assert "['Prev line 1.'" not in with_lists

    with_strings = generate_shared_prompt("prev", "next", "summary", "notes")
    assert "<previous_content>" not in with_strings
    assert "<previous_context>\nprev\n</previous_context>" in with_strings
    assert "<subsequent_context>\nnext\n</subsequent_context>" in with_strings

    with_nones = generate_shared_prompt(None, None, None, None)
    assert "<previous_context>\n\n</previous_context>" in with_nones
    assert "<subsequent_context>\n\n</subsequent_context>" in with_nones
    assert "None" not in with_nones


def test_valid_translate_result_rejects_empty_strings_and_populates_missing_origin(
    monkeypatch,
):
    """valid_translate_result rejects non-dicts and empty/whitespace translations, and translate_lines populates missing origin keys."""
    import core.translate_lines as tl

    assert (
        tl.valid_translate_result([{"1": {"direct": "ok"}}], ["1"], ["direct"])[
            "status"
        ]
        == "error"
    )
    assert (
        tl.valid_translate_result({"1": {"direct": ""}}, ["1"], ["direct"])["status"]
        == "error"
    )
    assert (
        tl.valid_translate_result({"1": {"direct": "   \n "}}, ["1"], ["direct"])[
            "status"
        ]
        == "error"
    )
    assert (
        tl.valid_translate_result({"1": {"direct": 123}}, ["1"], ["direct"])["status"]
        == "error"
    )
    assert (
        tl.valid_translate_result({"1": {"direct": "有效译文"}}, ["1"], ["direct"])[
            "status"
        ]
        == "success"
    )

    def fake_ask_gpt(prompt, **kwargs):
        if kwargs.get("log_title") == "translate_faithfulness":
            # LLM omitted "origin" key
            return {"1": {"trans": "第一行直译"}, "2": {"trans": "第二行直译"}}
        return {
            "1": {"reflect": "OK", "free": "第一行直译"},
            "2": {"reflect": "Adjusted phrasing", "free": "第二行润色"},
        }

    monkeypatch.setattr(tl, "ask_gpt", fake_ask_gpt)
    monkeypatch.setattr(
        tl, "load_key", lambda k: True if k == "reflect_translate" else "简体中文"
    )
    monkeypatch.setattr(tl, "check_cancel", lambda: None)

    trans, orig = tl.translate_lines(
        "Line one.\nLine two.", ["Prev line."], ["Next line."], "notes", "summary", 0
    )
    assert trans == "第一行直译\n第二行润色"
    assert orig == "Line one.\nLine two."


def test_translate_chunk_scans_previous_content_and_chunk_for_terms(monkeypatch):
    """translate_chunk passes ONLY the current chunk (not previous/after context) to terminology search."""
    import core._4_2_translate as translate

    captured = {}

    def fake_search(text):
        captured["search_scope"] = text
        return "matched terms"

    def fake_translate_lines(chunk, prev, after, notes, theme, idx):
        captured["prev"] = prev
        captured["after"] = after
        captured["notes"] = notes
        return "译文", chunk

    monkeypatch.setattr(translate, "search_things_to_note_in_prompt", fake_search)
    monkeypatch.setattr(translate, "translate_lines", fake_translate_lines)

    chunks = [
        "L1\nL2\nL3 mentioning Raycasting\nL4\nL5",
        "Current chunk referring to it.",
        "N1\nN2\nN3\nN4",
    ]
    translate.translate_chunk(chunks[1], chunks, "theme", 1)

    assert captured["prev"] == ["L2", "L3 mentioning Raycasting", "L4", "L5"]
    assert captured["after"] == ["N1", "N2", "N3"]
    assert captured["search_scope"] == "Current chunk referring to it."
    assert "L3 mentioning Raycasting" not in captured["search_scope"]


def test_translation_prompts_contain_asr_recovery_and_keep_if_flawless_in_english(
    monkeypatch,
):
    """Both translation prompts contain 224-Forme glossary instructions, inflection alignment, and Keep-if-Flawless."""
    import core.prompts as prompts

    monkeypatch.setattr(
        prompts, "load_key", lambda k: "简体中文" if k == "target_language" else True
    )
    monkeypatch.setattr(prompts, "load_key_or", lambda k, d="": "")
    monkeypatch.setattr(prompts, "get_source_language", lambda: "English")

    shared = prompts.generate_shared_prompt(["Prev"], ["Next"], "Summary", "Notes")
    faith_prompt = prompts.get_prompt_faithfulness("Hello world.", shared)
    express_prompt = prompts.get_prompt_expressiveness(
        {"1": {"origin": "Hello world.", "direct": "你好世界。"}},
        "Hello world.",
        shared,
    )

    assert "### 术语库-词典" in faith_prompt
    assert "Strict Terminology Adherence" in faith_prompt
    assert "Principle of Necessary Enhancement (Keep-if-Flawless)" in express_prompt
    assert "Senior Subtitle Quality Auditor & Polisher" in express_prompt


def test_symbol_terms_do_not_pollute_stem_ngram_map_and_aliases_appear_in_prompt(
    monkeypatch, tmp_path
):
    """Symbol terms (.NET, C#, C++) must not strip symbols to match plain 'net' or 'c';
    matched terms with aliases must include '(also transcribed as: ...)' in Points to Note."""
    import core._4_1_summarize as summarize

    ws = TaskWorkspace(tmp_path)
    ws.save_terms(
        "Software engineering",
        [
            {"src": ".NET", "tgt": ".NET 框架", "note": "微软平台"},
            {"src": "C#", "tgt": "C# 语言", "note": "编程语言"},
            {"src": "C++", "tgt": "C++ 语言", "note": "系统语言"},
            {
                "src": "Kubernetes",
                "tgt": "K8s 容器编排",
                "aliases": ["cooper netties"],
                "note": "云原生系统",
            },
        ],
    )
    monkeypatch.setattr(
        summarize, "GLOBAL_GLOSSARY_PATH", str(tmp_path / "no_global.csv")
    )
    monkeypatch.setattr(summarize, "CUSTOM_TERMS_PATH", str(tmp_path / "no.xlsx"))
    monkeypatch.setattr(summarize, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv"))
    monkeypatch.setattr(
        summarize, "OPTIONAL_JSON_TERMS_PATH", str(tmp_path / "no.json")
    )

    with workspace_context(ws):
        # Everyday 'net' and 'c' must NOT trigger .NET, C#, or C++
        assert (
            summarize.search_things_to_note_in_prompt(
                "The fishing net caught a fish in section c with a net profit."
            )
            is None
        )

        # Exact symbol match must trigger .NET and C++
        sym_prompt = summarize.search_things_to_note_in_prompt(
            "We migrated the backend from .NET to C++ yesterday."
        )
        assert sym_prompt is not None
        assert ".NET:.NET 框架" in sym_prompt
        assert "C++:C++ 语言" in sym_prompt
        assert "C#:" not in sym_prompt

        # Alias match must surface the calibrated alias:tgt (src) line in Points to Note
        alias_prompt = summarize.search_things_to_note_in_prompt(
            "We deployed the cluster on cooper netties in production."
        )
        assert alias_prompt is not None
        assert "cooper netties:K8s 容器编排 (Kubernetes)" in alias_prompt
        assert "aliases:{" not in alias_prompt


def test_merge_terminology_results_unifies_llm_variants_into_canonical_aliases():
    """When LLM extracts an ASR variant matching a custom term's alias or returns aliases,
    _merge_terminology_results merges them into the canonical custom term."""
    from core._4_1_summarize import _merge_terminology_results

    custom_terms = [
        {
            "src": "PyTorch",
            "tgt": "PyTorch 框架",
            "aliases": ["pie torch"],
            "note": "深度学习库",
            "fix_source": True,
            "source": "global_kb",
        }
    ]
    chunk_results = [
        (
            0,
            {
                "theme": "Deep learning tutorial.",
                "terms": [
                    {
                        "src": "pie torch",
                        "tgt": "PyTorch",
                        "aliases": ["pi torch"],
                        "note": "ASR variant of PyTorch",
                    },
                    {
                        "src": "TensorRT",
                        "tgt": "TensorRT 推理引擎",
                        "aliases": ["tensor artie"],
                        "note": "NVIDIA SDK",
                    },
                ],
            },
        )
    ]
    merged = _merge_terminology_results(
        chunk_results,
        [
            *custom_terms,
            {
                "src": "UnmentionedGlossaryTerm",
                "tgt": "未出现的旧词条",
                "aliases": [],
                "note": "不应出现在任务术语校对列表",
                "source": "domain_kb",
            },
        ],
    )
    by_src = {t["src"]: t for t in merged["terms"]}
    assert "UnmentionedGlossaryTerm" not in by_src
    assert "pie torch" not in by_src
    assert "PyTorch" in by_src
    assert by_src["PyTorch"]["tgt"] == "PyTorch 框架"
    assert "pie torch" in by_src["PyTorch"]["aliases"]
    assert "pi torch" in by_src["PyTorch"]["aliases"]
    assert by_src["TensorRT"]["aliases"] == ["tensor artie"]
    assert by_src["PyTorch"]["source"] == "task"
    assert by_src["TensorRT"]["source"] == "task"


def test_get_task_terms_api_hides_preexisting_glossary_sources(monkeypatch, tmp_path):
    """GET /api/tasks/{task_id}/terms returns the complete task terminology table including matched authoritative terms."""
    from fastapi.testclient import TestClient
    import api

    out_root = tmp_path / "output"
    out_root.mkdir()
    monkeypatch.setattr("core.workspace.OUTPUT_ROOT", out_root)
    monkeypatch.setattr(api, "OUTPUT", out_root)

    ws = TaskWorkspace(out_root / "task_terms_filter")
    ws.ensure_dirs()
    ws.save_terms(
        "Weapon review",
        [
            {"src": "OldDomainTerm", "tgt": "旧领域词", "source": "domain_kb"},
            {"src": "OldGlobalTerm", "tgt": "旧全局词", "source": "global_kb"},
            {"src": "NewScannedTerm", "tgt": "新扫出的词", "source": "task"},
        ],
    )

    client = TestClient(api.app)
    res = client.get("/api/tasks/task_terms_filter/terms")
    assert res.status_code == 200
    returned_srcs = [t["src"] for t in res.json()["terms"]]
    assert returned_srcs == ["OldDomainTerm", "OldGlobalTerm", "NewScannedTerm"]


def test_multiword_fuzzy_guards_and_longest_span_suppression(monkeypatch, tmp_path):
    """Multi-word fuzzy matching must not false-match 'rate weapons' -> 'heat weapons' or 'and compendium' -> 'data compendium',
    and Longest-Span Suppression must suppress 'Compendium' when covered by 'Data Compendium'."""
    import core._4_1_summarize as summarize

    ws = TaskWorkspace(tmp_path)
    ws.save_terms(
        "Destiny 2 weapon guide",
        [
            {"src": "Heat weapons", "tgt": "热量武器", "note": "武器机制"},
            {"src": "Data Compendium", "tgt": "数据汇编", "note": "社区表格"},
            {"src": "Compendium", "tgt": "汇编", "note": "简称"},
        ],
    )
    monkeypatch.setattr(
        summarize, "GLOBAL_GLOSSARY_PATH", str(tmp_path / "no_global.csv")
    )
    monkeypatch.setattr(summarize, "CUSTOM_TERMS_PATH", str(tmp_path / "no.xlsx"))
    monkeypatch.setattr(summarize, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv"))
    monkeypatch.setattr(
        summarize, "OPTIONAL_JSON_TERMS_PATH", str(tmp_path / "no.json")
    )

    with workspace_context(ws):
        # "rate weapons" and "and compendium" must NOT fuzzy-match "Heat weapons" or "Data Compendium"
        false_pos = summarize.search_things_to_note_in_prompt(
            "High fire rate weapons are strong here."
        )
        assert false_pos is None

        # When "Data Compendium" matches, nested "Compendium" is suppressed so only "Data Compendium" appears
        span_prompt = summarize.search_things_to_note_in_prompt(
            "You can check the Destiny 2 Data Compendium for exact numbers."
        )
        assert span_prompt is not None
        assert "Data Compendium:数据汇编" in span_prompt
        assert "\nCompendium:汇编" not in f"\n{span_prompt}"


def test_generic_kb_rare_token_candidate_retrieval_and_deterministic_merge(
    monkeypatch, tmp_path
):
    """Schema-agnostic KB indexing extracts entries from nested JSON, retrieves rare-token KB candidates
    for partial/shorthand mentions ('Multimach'), scans dotted acronyms ('DFA' -> 'D.F.A.'), and merges
    transcript surface forms into canonical aliases."""
    import core._4_1_summarize as summarize

    kb_json = tmp_path / "term-map.json"
    kb_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "categories": {
                    "weapons": [
                        {"en": "Multimach CCX", "zh-chs": "多机系统CCX"},
                        {"en": "D.F.A.", "zh-chs": "死从天降"},
                        {"en": "Indebted Kindness", "zh-chs": "蒙恩"},
                        {"en": "Hand Cannons", "zh-chs": "手炮"},
                    ],
                    "misc": {
                        "What If": "假若",
                        "All": "所有",
                        "Story": "剧情",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(summarize, "OPTIONAL_JSON_TERMS_PATH", str(kb_json))
    monkeypatch.setattr(summarize, "load_key_or", lambda k, d=None: "简体中文")

    kb_data = summarize._get_kb_index([])
    rare_cands = summarize._retrieve_rare_token_candidates(
        "Using Multimach in the Crucible.", kb_data, set()
    )
    assert any(
        c["src"] == "Multimach CCX" and c["tgt"] == "多机系统CCX" for c in rare_cands
    )

    # Direct KB scan finds "Hand Cannons" and dotted acronym "DFA" -> "D.F.A." while skipping stopword phrases
    transcript = "What if we compare all Hand Cannons against the Multimoc and DFA in the story today?"
    scanned = summarize._scan_text_for_kb_terms(transcript, kb_data)
    scanned_by_src = {s["src"]: s for s in scanned}
    assert "Hand Cannons" in scanned_by_src
    assert "D.F.A." in scanned_by_src
    assert scanned_by_src["D.F.A."]["aliases"] == ["DFA"]
    assert "Story" in scanned_by_src
    assert "What If" not in scanned_by_src
    assert "All" not in scanned_by_src

    chunk_results = [
        (
            0,
            {
                "theme": "Weapon comparison.",
                "terms": [
                    {
                        "src": "Multimach CCX",
                        "tgt": "多机系统CCX",
                        "aliases": ["Multimach", "Multimoc"],
                        "note": "SMG",
                    },
                    {
                        "src": "D.F.A.",
                        "tgt": "死从天降",
                        "aliases": [],
                    },
                    {
                        "src": "Hand Cannons",
                        "tgt": "手炮",
                        "aliases": ["all Hand Cannons"],
                    },
                ],
            },
        )
    ]
    merged = summarize._merge_terminology_results(
        chunk_results, [], kb_data=kb_data, full_text=transcript
    )
    by_src = {t["src"]: t for t in merged["terms"]}
    assert by_src["Multimach CCX"]["tgt"] == "多机系统CCX"
    assert "Multimach" in by_src["Multimach CCX"]["aliases"]
    assert "Multimoc" in by_src["Multimach CCX"]["aliases"]
    assert by_src["D.F.A."]["tgt"] == "死从天降"
    assert "DFA" in by_src["D.F.A."]["aliases"]
    assert by_src["Hand Cannons"]["tgt"] == "手炮"
    assert by_src["Hand Cannons"]["aliases"] == []
    assert "Story" not in by_src

    ws = TaskWorkspace(tmp_path)
    ws.save_terms(merged.get("theme", ""), merged["terms"])
    monkeypatch.setattr(summarize, "CUSTOM_TERMS_PATH", str(tmp_path / "no.xlsx"))
    monkeypatch.setattr(summarize, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv"))
    with workspace_context(ws):
        note_prompt = summarize.search_things_to_note_in_prompt(
            "Using Multimoc and DFA today."
        )
    assert note_prompt is not None
    assert "Multimoc:多机系统CCX (Multimach CCX)" in note_prompt
    assert "DFA:死从天降 (D.F.A.)" in note_prompt


def test_stitch_incomplete_clause_spans_merges_fronted_and_trailing_subordinate_clauses():
    """Fronted adverbial clauses ('While X.') and trailing subordinate clauses ('because Y.')
    with short acoustic gaps are stitched into complete sentences."""
    import core._3_1_split_nlp as split_nlp

    words = [
        "While",
        "this",
        "gun",
        "is",
        "good",
        "It",
        "lacks",
        "range",
        "Because",
        "the",
        "barrel",
        "is",
        "short",
    ]
    punct_by_word = {4: ".", 7: ".", 12: "."}
    word_gaps = [0.05] * len(words)
    spans = [(0, 4), (5, 7), (8, 12)]
    stitched = split_nlp._stitch_incomplete_clause_spans(
        spans, words, punct_by_word, word_gaps
    )
    assert stitched == [(0, 12)]
    assert punct_by_word[4] == ","
    assert punct_by_word[7] == ","


def test_translate_lines_replaces_ok_placeholder_in_free_with_direct(monkeypatch):
    """When LLM outputs 'OK' in 'free' during expressiveness pass, translate_lines
    substitutes the faithful 'direct' translation instead of leaking 'OK' into subtitles."""
    import core.translate_lines as tl

    monkeypatch.setattr(
        tl, "load_key", lambda k: True if k == "reflect_translate" else "简体中文"
    )
    monkeypatch.setattr("core.prompts.load_key", lambda k: "简体中文")
    monkeypatch.setattr("core.prompts.get_source_language", lambda: "en")
    monkeypatch.setattr("core.prompts.load_key_or", lambda k, d="": d)

    def fake_ask_gpt(prompt, **kwargs):
        if kwargs.get("log_title") == "translate_faithfulness":
            return {
                "1": {"origin": "Hello world.", "trans": "你好，世界。"},
                "2": {"origin": "Use Multimach CCX.", "trans": "使用多机系统CCX。"},
            }
        return {
            "1": {
                "origin": "Hello world.",
                "direct": "你好，世界。",
                "reflect": "OK",
                "free": "OK",
            },
            "2": {
                "origin": "Use Multimach CCX.",
                "direct": "使用多机系统CCX。",
                "reflect": "微调语序",
                "free": "推荐使用多机系统CCX。",
            },
        }

    monkeypatch.setattr(tl, "ask_gpt", fake_ask_gpt)
    result, _ = tl.translate_lines(
        "Hello world.\nUse Multimach CCX.", None, None, None, "theme", 0
    )
    assert result.splitlines() == ["你好，世界。", "推荐使用多机系统CCX。"]


def test_split_align_subs_resolves_punctuated_clauses_locally_and_batches_remaining(
    monkeypatch,
):
    """Step 5 splits multi-clause translated sentences in O(1) locally without LLM calls when
    clause boundaries already satisfy max_length, and batches remaining punctuation-less long
    lines into a single batch LLM call."""
    import core._5_split_sub as s5

    settings = {
        "subtitle": {"max_length": 75, "target_multiplier": 1.3},
        "max_workers": 2,
    }
    monkeypatch.setattr(s5, "load_key", lambda k: settings[k])
    monkeypatch.setattr(
        s5,
        "load_key_or",
        lambda k, d=None: "简体中文" if k == "target_language" else 2,
    )
    monkeypatch.setattr("core.prompts.get_source_language", lambda: "en")

    batch_calls = []

    def fake_ask_gpt(prompt, **kwargs):
        batch_calls.append(kwargs.get("log_title"))
        return {
            "2": "这些重型榴弹发射器完全不遵循所有常规武器羸弱能量球[br]通常预期的标准弹匣命中百分比计算公式。"
        }

    monkeypatch.setattr(s5, "ask_gpt", fake_ask_gpt)

    src_in = [
        "Hey everyone, so while I was doing the primary tier list, we went through all the attrition orbs classics, like AOKT and Multimach.",
        "Now, special weapons are a little bit underwhelming when it comes to your attrition orbs.",
        "So, heavy GLs do not obey the standard mag percentage formula that is typically expected of all regular weapons for attrition orbs.",
    ]
    tr_in = [
        "大家好，在做主武器分级榜的时候，我们回顾了所有经典的羸弱能量球武器，比如AOKT和多机系统CCX。",
        "特殊武器在羸弱能量球方面有点差强人意。",
        "这些重型榴弹发射器完全不遵循所有常规武器羸弱能量球通常预期的标准弹匣命中百分比计算公式。",
    ]

    src_out, tr_out, remerged = s5.split_align_subs(src_in, tr_in)
    # Line 0 has internal clause commas and is split locally without calling LLM;
    # Line 1 has 19 Chinese chars (<=32) with no internal commas and stays intact;
    # Line 2 has 43 Chinese chars (>32) with no internal commas and triggers 1 batch LLM call.
    assert batch_calls == ["split_by_meaning_batch"]
    assert "特殊武器在羸弱能量球方面有点差强人意。" in tr_out
    assert len(src_out) == len(tr_out) >= 5
    assert all(s5.calc_len(t) * 1.3 <= 75 for t in tr_out)
    assert remerged == tr_in


def test_prompts_are_domain_and_language_agnostic_and_merge_tiny_clauses_preserves_parallel_clauses(
    monkeypatch,
):
    """Prompts contain no hardcoded Chinese characters when target_language is non-Chinese,
    and _merge_tiny_clauses merges sentence-initial lead-ins while preserving mid-sentence
    parallel short clauses."""

    import core._5_split_sub as s5
    import core.prompts as prompts

    monkeypatch.setattr(
        prompts, "load_key", lambda k: "Español" if k == "target_language" else False
    )
    monkeypatch.setattr(prompts, "load_key_or", lambda k, d="": "")
    monkeypatch.setattr(prompts, "get_source_language", lambda: "English")

    shared = prompts.generate_shared_prompt(["Prev"], ["Next"], "Summary", "Notes")
    faith_prompt = prompts.get_prompt_faithfulness("Hello world.", shared)

    assert "### 术语库-词典" in faith_prompt
    assert (
        "<<请一步到位翻译为精准、自然地道、符合口语字幕节奏的Español最终译文>>"
        in faith_prompt
    )

    # Sentence-initial lead-in ('但遗憾的是，' = 5 CJK chars, weighted 10.75) merges into following clause
    merged_lead = s5._merge_tiny_clauses(
        ["但遗憾的是，", "数据汇编里的公式并不完全准确。"],
        min_weighted_len=7.5,
        lead_in_weighted_len=13.0,
        max_merged_weighted_len=62.5,
    )
    assert merged_lead == ["但遗憾的是，数据汇编里的公式并不完全准确。"]

    # Mid-sentence parallel 5-char clauses stay separate instead of collapsing into one
    merged_parallel = s5._merge_tiny_clauses(
        ["这套动作讲究快准狠，", "稳如泰山，", "一击必杀，", "绝不拖泥带水。"],
        min_weighted_len=7.5,
        lead_in_weighted_len=13.0,
        max_merged_weighted_len=62.5,
    )
    assert merged_parallel == [
        "这套动作讲究快准狠，",
        "稳如泰山，",
        "一击必杀，",
        "绝不拖泥带水。",
    ]


def test_prompt_validators_and_json_escaping_contracts(monkeypatch):
    """Verify prompt contract hardening:
    - get_align_prompt properly escapes double quotes in JSON schema and input
    - get_prompt_expressiveness contains no Chinese characters when target_language is Español
    - _valid_summary rejects empty src/tgt and _merge_terminology_results coerces note=None to ''
    - _restore_window_punctuation valid_punct rejects truncated/unaligned LLM responses."""

    import core._3_1_split_nlp as s31
    import core._4_1_summarize as s41
    import core.prompts as prompts

    monkeypatch.setattr(
        prompts, "load_key", lambda k: "Español" if k == "target_language" else True
    )
    monkeypatch.setattr(prompts, "load_key_or", lambda k, d="": "")
    monkeypatch.setattr(prompts, "get_source_language", lambda: "English")

    # 1. get_prompt_expressiveness includes 224-Forme glossary header and Spanish target directive
    shared = prompts.generate_shared_prompt(["Prev"], ["Next"], "Summary", "Notes")
    express_prompt = prompts.get_prompt_expressiveness(
        {"1": {"origin": "Hello world.", "direct": "Hola mundo."}},
        "Hello world.",
        shared,
    )
    assert "### 术语库-词典" in express_prompt
    assert "Español" in express_prompt

    # 2. get_align_prompt properly escapes quotes inside JSON block
    align_p = prompts.get_align_prompt(
        'He said "hello" to me.',
        'Él me dijo "hola".',
        'He said "hello"\nto me.',
    )
    json_block = align_p.split("```json")[1].split("```")[0].strip()
    parsed_schema = json.loads(json_block)
    assert parsed_schema["align"][0]["src_part_1"] == 'He said "hello"'

    # 3. _valid_summary rejects empty src/tgt, and _merge_terminology_results coerces note=None to ''
    assert (
        s41._valid_summary({"terms": [{"src": "", "tgt": "X", "note": "n"}]})["status"]
        == "error"
    )
    assert (
        s41._valid_summary({"terms": [{"src": "A", "tgt": "B", "note": None}]})[
            "status"
        ]
        == "success"
    )
    merged = s41._merge_terminology_results(
        [(0, {"theme": "T", "terms": [{"src": "Foo", "tgt": "Bar", "note": None}]})],
        [],
    )
    assert merged["terms"][0]["note"] == ""

    # 4. _restore_window_punctuation valid_punct rejects truncated responses (< 0.80 alignment)
    captured_valid = {}

    def fake_ask_gpt(prompt, **kwargs):
        captured_valid["fn"] = kwargs["valid_def"]
        return {"restored": "one two three four five."}

    monkeypatch.setattr(s31, "ask_gpt", fake_ask_gpt)
    s31._restore_window_punctuation(
        0, ["one", "two", "three", "four", "five"], " ", "en"
    )
    valid_fn = captured_valid["fn"]
    assert valid_fn({"restored": "one."})["status"] == "error"
    assert valid_fn({"restored": "one, two, three, four, five."})["status"] == "success"


def test_review_edge_cases_word_gaps_partial_batch_and_standalone_punct(
    monkeypatch, tmp_path
):
    """Verify the 4 code-review edge cases:
    1. restore_and_split_words updates caller_dicts in-place when standalone punctuation tokens merge,
       and _build_utterances_from_sentences handles CJK fallback by char length rather than .split().
    2. _batch_split_sentences_llm preserves valid batch entries when one entry misses [br] and only
       falls back for the missing entry.
    3. split_for_sub_main constructs line_word_gaps from project.sqlite utterances + words and passes them
       to split_align_subs."""
    import core._3_1_split_nlp as s31
    import core._5_split_sub as s5

    # 1. Standalone punctuation token merge updates caller_dicts in-place
    caller_dicts = [
        {"idx": 0, "text": "你好", "start": 0.0, "end": 0.5, "punct_after": ""},
        {"idx": 1, "text": "。", "start": 0.5, "end": 0.6, "punct_after": ""},
        {"idx": 2, "text": "世界", "start": 0.8, "end": 1.3, "punct_after": ""},
        {"idx": 3, "text": "！", "start": 1.3, "end": 1.4, "punct_after": ""},
    ]
    monkeypatch.setattr(
        s31,
        "ask_gpt",
        lambda prompt, **kw: {"restored": "你好。世界！"},
    )
    sents = s31.restore_and_split_words(
        [d["text"] for d in caller_dicts], language="zh", word_dicts=caller_dicts
    )
    assert sents == ["你好。", "世界！"]
    assert len(caller_dicts) == 2
    assert caller_dicts[0]["end"] == 0.6
    assert caller_dicts[0]["punct_after"] == "。"
    assert caller_dicts[1]["end"] == 1.4
    assert caller_dicts[1]["punct_after"] == "！"

    # CJK fallback when sentence text is normalized differently from raw word_dicts
    cjk_words = [
        {"idx": 0, "text": "甲", "start": 0.0, "end": 0.2},
        {"idx": 1, "text": "乙", "start": 0.2, "end": 0.4},
        {"idx": 2, "text": "丙", "start": 0.4, "end": 0.6},
    ]
    utts = s31._build_utterances_from_sentences(["丁戊己。"], cjk_words, joiner="")
    assert utts[0]["word_span"] == [0, 3]

    # 2. _batch_split_sentences_llm keeps valid batch lines when another line in the batch lacks [br]
    monkeypatch.setattr(s5, "load_key_or", lambda k, d=None: 1)
    monkeypatch.setattr("core.prompts.get_source_language", lambda: "zh")
    single_calls = []

    def fake_batch_gpt(prompt, **kw):
        return {
            "0": "第一句前半部分[br]第一句后半部分",
            "1": "第二句漏掉了切分标记",
        }

    def fake_single_split(text, num_parts=2, language=None):
        single_calls.append(text)
        return "第二句前半部分\n第二句后半部分"

    monkeypatch.setattr(s5, "ask_gpt", fake_batch_gpt)
    monkeypatch.setattr(s5, "split_sentence", fake_single_split)
    batch_res = s5._batch_split_sentences_llm(
        [
            (0, "第一句前半部分第一句后半部分", 2),
            (1, "第二句漏掉了切分标记", 2),
        ],
        language="zh",
    )
    assert batch_res[0] == "第一句前半部分\n第一句后半部分"
    assert batch_res[1] == "第二句前半部分\n第二句后半部分"
    assert single_calls == ["第二句漏掉了切分标记"]

    # 3. split_for_sub_main passes line_word_gaps constructed from project.sqlite utterances + words
    ws = TaskWorkspace(tmp_path / "ws")
    ws.save_words(
        [
            {"idx": i, "text": f"w{i}", "start": float(i), "end": float(i) + 0.4}
            for i in range(16)
        ]
    )
    ws.save_utterances(
        [
            {
                "id": 0,
                "word_span": [0, 16],
                "start": 0.0,
                "end": 15.4,
                "src": "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen",
                "tgt": "这是一段包含自然逗号的中文长句，后半部分用来验证主流程声学停顿传递。",
            }
        ]
    )

    captured_gaps = []
    orig_split_align = s5.split_align_subs

    def spy_split_align(src_lines, tr_lines, line_word_gaps=None):
        captured_gaps.append(line_word_gaps)
        return orig_split_align(src_lines, tr_lines, line_word_gaps=line_word_gaps)

    monkeypatch.setattr(s5, "find_subtitle_file", lambda: None)
    monkeypatch.setattr(
        s5,
        "load_key",
        lambda k: (
            {"max_length": 75, "target_multiplier": 1.2} if k == "subtitle" else 2
        ),
    )
    monkeypatch.setattr(s5, "split_align_subs", spy_split_align)
    with workspace_context(ws):
        s5.split_for_sub_main()
    assert captured_gaps and captured_gaps[0] is not None
    assert len(captured_gaps[0][0]) == 15


def test_end_to_end_alias_matching_across_split_translate_and_sub_cues(
    monkeypatch, tmp_path
):
    """Verify multi-word alias protection in Step 3, alias-based terminology injection in Step 4.2
    without mutating source text, and token-aligned cue partitioning & timestamps in Step 5 & 6."""
    import core._3_1_split_nlp as s31
    import core._4_1_summarize as s41
    import core._4_2_translate as s42
    import core._5_split_sub as s5
    import core._6_gen_sub as s6

    ws = TaskWorkspace(tmp_path / "alias_ws")
    ws.ensure_dirs()
    monkeypatch.setattr(s41, "GLOBAL_GLOSSARY_PATH", str(tmp_path / "no_global.csv"))
    monkeypatch.setattr(s41, "CUSTOM_TERMS_PATH", str(tmp_path / "no.xlsx"))
    monkeypatch.setattr(s41, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv"))
    monkeypatch.setattr(s41, "OPTIONAL_JSON_TERMS_PATH", str(tmp_path / "no.json"))

    # 1. Step 3 protects multi-word aliases from mid-alias splits and projects LLM alias rewrites onto the end word
    raw_words = [
        {"idx": 0, "text": "Gnats", "start": 0.0, "end": 0.4},
        {"idx": 1, "text": "runs", "start": 0.5, "end": 0.8},
        {"idx": 2, "text": "pie", "start": 0.9, "end": 1.2},
        {"idx": 3, "text": "torch", "start": 1.2, "end": 1.6},
        {"idx": 4, "text": "for", "start": 1.7, "end": 1.9},
        {"idx": 5, "text": "training", "start": 1.9, "end": 2.4},
        {"idx": 6, "text": "and", "start": 2.9, "end": 3.1},
        {"idx": 7, "text": "we", "start": 3.1, "end": 3.3},
        {"idx": 8, "text": "deploy", "start": 3.3, "end": 3.7},
        {"idx": 9, "text": "cooper", "start": 3.7, "end": 4.1},
        {"idx": 10, "text": "netties", "start": 4.1, "end": 4.6},
    ]
    ws.save_words(raw_words)
    terms = [
        {
            "src": "NATS",
            "tgt": "NATS",
            "aliases": ["Gnats"],
            "note": "",
        },
        {
            "src": "PyTorch",
            "tgt": "PyTorch",
            "aliases": ["pie torch"],
            "note": "",
        },
        {
            "src": "Kubernetes",
            "tgt": "Kubernetes",
            "aliases": ["cooper netties"],
            "note": "",
        },
    ]
    ws.save_terms("Infra", terms)

    # Even if LLM puts a spurious period inside 'pie. torch' and rewrites 'cooper netties' -> 'Kubernetes.',
    # Step 3 keeps the multi-word alias intact and builds exact word_span [0, 11]
    monkeypatch.setattr(s31, "get_source_language", lambda: "en")
    monkeypatch.setattr(s31, "load_key_or", lambda k, d=None: 1)
    monkeypatch.setattr(
        s31,
        "ask_gpt",
        lambda *a, **kw: {
            "restored": "Gnats runs pie. torch for training, and we deploy Kubernetes."
        },
    )

    with workspace_context(ws):
        s31.split_by_spacy()

        utts = ws.load_utterances()
        assert len(utts) == 1
        assert utts[0]["word_span"] == [0, 11]
        assert (
            utts[0]["src"]
            == "Gnats runs pie torch for training, and we deploy cooper netties."
        )

        # 2. Step 4.2 preserves original sentence text and injects canonical terms via alias matching
        chunks = s42.split_chunks_by_chars()
        assert chunks == [
            "Gnats runs pie torch for training, and we deploy cooper netties."
        ]
        note_prompt = s41.search_things_to_note_in_prompt(chunks[0])
        assert note_prompt is not None
        assert "Gnats:NATS (NATS)" in note_prompt
        assert "pie torch:PyTorch (PyTorch)" in note_prompt
        assert "cooper netties:Kubernetes (Kubernetes)" in note_prompt

        # 3. Step 5 _partition_word_span_for_cues partitions [0, 11) at exact word index 6 ("training,")
        words_loaded = ws.load_words()
        cues = s5._partition_word_span_for_cues(
            0,
            11,
            [
                "Gnats runs pie torch for training,",
                "and we deploy cooper netties.",
            ],
            ["NATS使用PyTorch训练，", "并且我们部署Kubernetes。"],
            words_loaded,
        )
        assert [c["word_span"] for c in cues] == [[0, 6], [6, 11]]
        assert cues[0]["start"] == 0.0 and cues[0]["end"] == 2.4
        assert cues[1]["start"] == 2.9 and cues[1]["end"] == 4.6

        utts_with_cues = ws.load_utterances()
        utts_with_cues[0]["sub_cues"] = cues
        ws.save_utterances(utts_with_cues)

        # 4. Step 6 get_sentence_timestamps resolves sub_cues timestamps directly from word_span
        df_words_full = pd.DataFrame(words_loaded)
        stamps = s6.get_sentence_timestamps(
            df_words_full,
            pd.DataFrame(
                {
                    "Source": [
                        "Gnats runs pie torch for training,",
                        "and we deploy cooper netties.",
                    ]
                }
            ),
        )
        assert stamps == [(0.0, 2.4), (2.9, 4.6)]


def test_review_regressions_code_switched_thai_decimals_and_src_punct_preservation(
    monkeypatch, tmp_path
):
    """Verify the 7 independent review fixes:
    1. Code-switched Chinese+English without spaces around full-width punctuation ('有个meeting。call me')
       and Thai unspaced script with combining tone marks align (>= 0.80) and split properly.
    2. Sentence-ending decimals/acronyms ('It costs 3.5. Next sentence.', 'in the U.S. They live...') split properly.
    3. 55-word complex sentences are not prematurely split at 45 words, and 'That is...' independent sentences
       are not merged by _stitch_incomplete_clause_spans, while 'So while...' fronted clauses merge.
    4. _partition_word_span_for_cues advances char_cursor after an alias-aligned cue when >=3 cues repeat phrases.
    5. _merge_standalone_punct_tokens guards against 0.0 timestamps overwriting real word timestamps.
    6. PUT /api/tasks/{task_id}/utterances preserves src_punct when only tgt is edited.
    7. get_sentence_timestamps caps fuzzy distance on <=4-char short sentences."""
    from fastapi.testclient import TestClient
    import api
    import core._3_1_split_nlp as s31
    import core._5_split_sub as s5

    monkeypatch.setattr(s31, "load_key_or", lambda k, d=None: 1)

    # 1a. Code-switched CJK + English with full-width period between two Latin words
    monkeypatch.setattr(
        s31,
        "ask_gpt",
        lambda *a, **kw: {"restored": "有个meeting。call me Lisa就好。"},
    )
    zh_en_sents = s31.restore_and_split_words(
        ["有", "个", "meeting", "call", "me", "Lisa", "就", "好"], language="zh"
    )
    assert zh_en_sents == ["有个meeting。", "call me Lisa就好。"]

    # 1b. Thai unspaced script with combining tone marks passes _verify_word_alignment and splits
    thai_words = ["สวัสดี", "ครับ", "วันนี้", "อากาศ", "ดี"]
    thai_restored = "สวัสดีครับ. วันนี้อากาศดี."
    assert s31._verify_word_alignment(thai_words, thai_restored) >= 0.95
    monkeypatch.setattr(s31, "ask_gpt", lambda *a, **kw: {"restored": thai_restored})
    assert s31.restore_and_split_words(thai_words, language="th") == [
        "สวัสดีครับ.",
        "วันนี้อากาศดี.",
    ]

    # 2. Sentence-ending decimals and uppercase-followed acronyms split cleanly
    monkeypatch.setattr(
        s31,
        "ask_gpt",
        lambda *a, **kw: {
            "restored": "It costs 3.5. We live in the U.S. They live in Canada."
        },
    )
    dec_sents = s31.restore_and_split_words(
        [
            "It",
            "costs",
            "3.5",
            "We",
            "live",
            "in",
            "the",
            "U.S.",
            "They",
            "live",
            "in",
            "Canada",
        ],
        language="en",
    )
    assert dec_sents == [
        "It costs 3.5.",
        "We live in the U.S.",
        "They live in Canada.",
    ]

    # 3. 'That is a great point.' stays independent; 'So while this gun is good.' merges with main clause
    words_that = ["That", "is", "a", "great", "point", "We", "should", "move", "on"]
    p_that = {4: ".", 8: "."}
    assert s31._stitch_incomplete_clause_spans(
        [(0, 4), (5, 8)], words_that, p_that, [0.1] * 9
    ) == [(0, 4), (5, 8)]

    words_so_while = [
        "So",
        "while",
        "this",
        "gun",
        "is",
        "good",
        "It",
        "lacks",
        "range",
    ]
    p_sw = {5: ".", 8: "."}
    assert s31._stitch_incomplete_clause_spans(
        [(0, 5), (6, 8)], words_so_while, p_sw, [0.1] * 9
    ) == [(0, 8)]

    # 55-word complex sentence stays intact (< LONG_SENTENCE_MAX_WORDS=120)
    w_55 = [f"word{i}" for i in range(55)]
    assert s31._split_long_span_at_clauses(
        0, 54, w_55, {25: ",", 54: "."}, None, [0.5] * 55
    ) == [(0, 54)]

    # 4. _partition_word_span_for_cues advances char_cursor across 3 cues when cue 0 has an alias replacement
    words_3cues = [
        {"idx": i, "text": w, "start": float(i), "end": float(i) + 0.4}
        for i, w in enumerate(
            ["we", "use", "pie", "torch", "now", "we", "use", "it", "daily"]
        )
    ]
    cues_3 = s5._partition_word_span_for_cues(
        0,
        9,
        ["we use PyTorch now", "we use", "it daily"],
        ["我们现在用PyTorch", "我们用", "每天用"],
        words_3cues,
    )
    assert [c["word_span"] for c in cues_3] == [[0, 5], [5, 7], [7, 9]]

    # 5. _merge_standalone_punct_tokens does not overwrite real word end with 0.0 from unaligned punct
    _, merged_d = s31._merge_standalone_punct_tokens(
        ["hello", "."],
        [
            {"text": "hello", "start": 1.0, "end": 1.8},
            {"text": ".", "start": 0.0, "end": 0.0},
        ],
    )
    assert merged_d is not None and merged_d[0]["end"] == 1.8

    # 6. PUT /api/tasks/{task_id}/utterances preserves src_punct when only tgt is edited
    out_root = tmp_path / "output"
    out_root.mkdir()
    monkeypatch.setattr("core.workspace.OUTPUT_ROOT", out_root)
    monkeypatch.setattr(api, "OUTPUT", out_root)
    ws = TaskWorkspace(out_root / "t_punct")
    ws.ensure_dirs()
    ws.save_utterances(
        [
            {
                "id": 0,
                "word_span": [0, 3],
                "start": 0.0,
                "end": 1.5,
                "src_raw": "Gnats runs fast",
                "src_punct": "Gnats runs fast.",
                "src": "NATS runs fast.",
                "tgt": "NATS运行很快。",
            }
        ]
    )
    client = TestClient(api.app)
    res = client.put(
        "/api/tasks/t_punct/utterances",
        json={
            "utterances": [
                {
                    "id": 0,
                    "word_span": [0, 3],
                    "start": 0.0,
                    "end": 1.5,
                    "src_raw": "Gnats runs fast",
                    "src": "NATS runs fast.",
                    "tgt": "NATS运行非常快！",
                }
            ]
        },
    )
    assert res.status_code == 200
    saved_u = ws.load_utterances()[0]
    assert saved_u["tgt"] == "NATS运行非常快！"
    assert saved_u["src_punct"] == "Gnats runs fast."

    # 7. PUT /api/tasks/{task_id}/utterances with local clause split populates word_span on sub_cues and realigns .srt
    ws_split = TaskWorkspace(out_root / "t_local_split")
    ws_split.ensure_dirs()
    ws_split.save_words(
        [
            {"idx": 0, "text": "Destiny", "start": 0.0, "end": 0.4},
            {"idx": 1, "text": "2", "start": 0.4, "end": 0.6},
            {"idx": 2, "text": "is", "start": 0.6, "end": 0.8},
            {"idx": 3, "text": "an", "start": 0.8, "end": 1.0},
            {"idx": 4, "text": "intimidating", "start": 1.0, "end": 1.6},
            {"idx": 5, "text": "game,", "start": 1.6, "end": 2.0},
            {"idx": 6, "text": "and", "start": 2.1, "end": 2.3},
            {"idx": 7, "text": "there", "start": 2.3, "end": 2.5},
            {"idx": 8, "text": "is", "start": 2.5, "end": 2.7},
            {"idx": 9, "text": "a", "start": 2.7, "end": 2.8},
            {"idx": 10, "text": "lot", "start": 2.8, "end": 3.1},
            {"idx": 11, "text": "to", "start": 3.1, "end": 3.3},
            {"idx": 12, "text": "learn.", "start": 3.3, "end": 3.9},
        ]
    )
    ws_split.src_srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:03,900\nDestiny 2 is an intimidating game, and there is a lot to learn.\n\n",
        encoding="utf-8",
    )
    ws_split.save_utterances(
        [
            {
                "id": 0,
                "word_span": [0, 13],
                "start": 0.0,
                "end": 3.9,
                "src": "Destiny 2 is an intimidating game, and there is a lot to learn.",
                "tgt": "旧短译文。",
                "sub_cues": [
                    {
                        "src": "Destiny 2 is an intimidating game, and there is a lot to learn.",
                        "tgt": "旧短译文。",
                        "trans": "旧短译文。",
                        "word_span": [0, 13],
                        "start": 0.0,
                        "end": 3.9,
                    }
                ],
            }
        ]
    )
    res_split = client.put(
        "/api/tasks/t_local_split/utterances",
        json={
            "utterances": [
                {
                    "id": 0,
                    "word_span": [0, 13],
                    "start": 0.0,
                    "end": 3.9,
                    "src": "Destiny 2 is a very intimidating game, and there is so much to learn.",
                    "tgt": "命运2确实是一款让人望而生畏的入门游戏，而且游戏里有很多东西需要学习。",
                    "sub_cues": [
                        {
                            "src": "Destiny 2 is an intimidating game, and there is a lot to learn.",
                            "tgt": "旧短译文。",
                            "trans": "旧短译文。",
                            "word_span": [0, 13],
                            "start": 0.0,
                            "end": 3.9,
                        }
                    ],
                }
            ]
        },
    )
    assert res_split.status_code == 200
    saved_split_u = ws_split.load_utterances()[0]
    assert len(saved_split_u["sub_cues"]) == 2
    assert all(
        isinstance(c.get("word_span"), list) and len(c["word_span"]) == 2
        for c in saved_split_u["sub_cues"]
    )
    assert saved_split_u["sub_cues"][0]["start"] == 0.0
    assert saved_split_u["sub_cues"][1]["end"] == 3.9


def test_inflections_and_asr_aliases_match_search_without_source_replacement(
    monkeypatch, tmp_path
):
    """Singular/plural, possessives ('s), -ed/-ing inflections, and ASR phonetic aliases all match
    search_things_to_note_in_prompt without mutating source text."""
    import core._4_1_summarize as s41

    ws = TaskWorkspace(tmp_path / "inflection_ws")
    ws.ensure_dirs()
    monkeypatch.setattr(s41, "GLOBAL_GLOSSARY_PATH", str(tmp_path / "no_global.csv"))
    monkeypatch.setattr(s41, "CUSTOM_TERMS_PATH", str(tmp_path / "no.xlsx"))
    monkeypatch.setattr(s41, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv"))
    monkeypatch.setattr(s41, "OPTIONAL_JSON_TERMS_PATH", str(tmp_path / "no.json"))

    # 1. _normalize_aliases strips grammatical inflections while keeping genuine ASR phonetic aliases
    assert s41._normalize_aliases(["Wolves", "wolf's", "gally", "Gally"], "Wolf") == [
        "gally",
        "Gally",
    ]
    assert s41._normalize_aliases(
        ["weakened", "weakening", "weakens", "weeken"], "Weaken"
    ) == ["weeken"]
    assert s41._normalize_aliases(["glaives", "Glaive's"], "Glaive") == []
    assert s41._normalize_aliases(["buffed", "buffing", "buffs"], "Buff") == []
    assert (
        s41._normalize_aliases(
            ["Agers Scepter", "Ager’s Scepters", "ager scepter"], "Ager's Scepter"
        )
        == []
    )
    assert s41._normalize_aliases(["savathun"], "Savathûn") == ["savathun"]

    terms = [
        {"src": "Wolf", "tgt": "群狼", "aliases": ["Wolves"], "source": "task"},
        {"src": "Glaive", "tgt": "偃月", "aliases": ["glaives"], "source": "task"},
        {"src": "Buff", "tgt": "增益", "aliases": ["buffed"], "source": "task"},
        {
            "src": "Weaken",
            "tgt": "削弱",
            "aliases": ["weakened", "weakening"],
            "source": "task",
        },
        {
            "src": "Scorch",
            "tgt": "灼烧",
            "aliases": ["scorched", "scorching"],
            "source": "task",
        },
        {"src": "Jolt", "tgt": "颠簸", "aliases": ["jolted"], "source": "task"},
        {
            "src": "Ignite",
            "tgt": "点燃",
            "aliases": ["ignited", "igniting"],
            "source": "task",
        },
        {
            "src": "Ager's Scepter",
            "tgt": "阿格尔的权杖",
            "aliases": ["Agers Scepter"],
            "source": "task",
        },
        {
            "src": "Gjallarhorn",
            "tgt": "加拉尔号角",
            "aliases": ["gally"],
            "source": "task",
        },
        {
            "src": "Savathûn",
            "tgt": "萨瓦图恩",
            "aliases": ["sabathun"],
            "source": "task",
        },
        {
            "src": "Khvostov 7G-0X",
            "tgt": "赫沃斯托夫 7G-0X",
            "aliases": ["Kavostov", "Khvostov"],
            "source": "task",
        },
        {
            "src": "Well of Radiance",
            "tgt": "光焰之井",
            "aliases": ["Well"],
            "source": "task",
        },
        {
            "src": "D.A.R.C.I.",
            "tgt": "达尔西",
            "aliases": [],
            "source": "task",
        },
    ]
    ws.save_terms("Destiny 2", terms)

    src_line = (
        "Well, the Wolves and glaives are buffed, weakened, scorched, jolted, and igniting "
        "while using Agers Scepter, Kavostov, Khvostov 7G-0X, DARCI, and gally against sabathun."
    )

    # 2. _get_kb_index + _scan_text_for_kb_terms matches dotted acronyms ('DARCI' -> 'D.A.R.C.I.') and KB aliases ('gally')
    kb_idx = s41._get_kb_index(terms)
    kb_scanned_srcs = {
        h["src"]
        for h in s41._scan_text_for_kb_terms(
            "Equipping DARCI and gally with glaives.", kb_idx
        )
    }
    assert {"D.A.R.C.I.", "Gjallarhorn", "Glaive"} <= kb_scanned_srcs

    # 3. search_things_to_note_in_prompt matches all canonical entries from inflected words and calibrated aliases in src_line
    with workspace_context(ws):
        note_prompt = s41.search_things_to_note_in_prompt(src_line)
    assert note_prompt is not None
    for expected_line in (
        "Wolf:群狼",
        "Glaive:偃月",
        "Buff:增益",
        "Weaken:削弱",
        "Scorch:灼烧",
        "Jolt:颠簸",
        "Ager's Scepter:阿格尔的权杖",
        "gally:加拉尔号角 (Gjallarhorn)",
        "sabathun:萨瓦图恩 (Savathûn)",
        "Khvostov 7G-0X:赫沃斯托夫 7G-0X",
        "Kavostov:赫沃斯托夫 7G-0X (Khvostov 7G-0X)",
        ":达尔西 (D.A.R.C.I.)",
    ):
        assert expected_line in note_prompt
    assert ":点燃" in note_prompt and (
        "Ignite:" in note_prompt or "Ignition:" in note_prompt
    )


def test_book_marks_replaced_with_quotes_and_clause_stitching_guard():
    """Verify book title marks and quotes are stripped cleanly, dashes/colons normalize to commas, and complete sentences before 'While X, Y.' are not comma-stitched."""
    import core._3_1_split_nlp as s31
    import core._5_split_sub as s5
    import core._6_gen_sub as s6

    # 1. Book title marks and quotes stripped cleanly
    src_line = " Destiny 2 can definitely be an intimidating game to get into, and there is a lot that the game never tells you."
    tr_line = "《命运2》确实是一款让人望而生畏的入门游戏，而且游戏里有很多东西从来不会告诉你。"
    res = s5._try_local_clause_split(src_line, tr_line, 60, 1.5)
    assert res is not None
    split_src, split_tr, _ = res
    assert len(split_tr) == 2
    assert split_tr[0].startswith("命运2")
    assert "《" not in "".join(split_tr) and "》" not in "".join(split_tr)
    assert (
        s6.clean_translation("《命运2》回归玩家〈指南〉“测试”")
        == "命运2回归玩家指南测试"
    )

    # 2. _stitch_incomplete_clause_spans does not stitch a complete sentence with a following "While X, Y." two-clause sentence
    words = [
        "Right",
        "now",
        "the",
        "Portal",
        "is",
        "the",
        "core",
        "of",
        "Destiny",
        "2",
        "while",
        "that",
        "might",
        "change",
        "soon",
        "it",
        "is",
        "worth",
        "understanding",
    ]
    punct = {9: ".", 14: ",", 18: "."}
    gaps = [0.05] * len(words)
    spans = [(0, 9), (10, 18)]
    stitched = s31._stitch_incomplete_clause_spans(spans, words, punct, gaps)
    assert stitched == [(0, 9), (10, 18)]
    assert punct[9] == "."


def test_rerun_idempotency_compound_shorthand_retrieval_and_punctuation_cleanup(
    monkeypatch, tmp_path
):
    """Verify Stage 03 re-run preserves aliases against src_punct, compound/shorthand/fuzzy terms
    match in search_things_to_note_in_prompt, and dashes/colons/quotes are normalized while dunhao splits long clauses."""
    import core._4_1_summarize as s41
    import core._5_split_sub as s5
    import core._6_gen_sub as s6
    import core.translate_lines as tl

    # 1. Compound split/join and alias injection as both canonical src:tgt and direct alias:tgt
    ws = TaskWorkspace(tmp_path)
    ws.save_terms(
        "Destiny 2 DPS",
        [
            {
                "src": "Cloudstrike",
                "tgt": "雷云风暴",
                "aliases": [],
                "source": "domain_kb",
            },
            {
                "src": "Gear Shift",
                "tgt": "换挡",
                "aliases": [],
                "source": "domain_kb",
            },
            {
                "src": "Cuirass of the Falling Star",
                "tgt": "陨星胸甲",
                "aliases": ["Curus", "Curious"],
                "source": "task",
            },
            {
                "src": "One Thousand Voices",
                "tgt": "千语",
                "aliases": ["1K"],
                "source": "task",
            },
            {
                "src": "Fireteam Ops",
                "tgt": "火力战队行动",
                "aliases": ["ops"],
                "source": "task",
            },
            {
                "src": "Arena Ops",
                "tgt": "竞技场行动",
                "aliases": ["ops"],
                "source": "task",
            },
            {
                "src": "Solo Ops",
                "tgt": "单人行动",
                "aliases": ["ops"],
                "source": "task",
            },
        ],
    )
    monkeypatch.setattr(s41, "GLOBAL_GLOSSARY_PATH", str(tmp_path / "no_g.csv"))
    monkeypatch.setattr(s41, "CUSTOM_TERMS_PATH", str(tmp_path / "no.xlsx"))
    monkeypatch.setattr(s41, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv"))
    monkeypatch.setattr(s41, "OPTIONAL_JSON_TERMS_PATH", str(tmp_path / "no.json"))

    with workspace_context(ws):
        prompt = s41.search_things_to_note_in_prompt(
            "Using Cloud Strike, gearshift, Curious, and 1K in ops on this boss."
        )
    assert prompt is not None
    for expected in (
        "Cloud Strike:雷云风暴 (Cloudstrike)",
        "gearshift:换挡 (Gear Shift)",
        "Curious:陨星胸甲 (Cuirass of the Falling Star)",
        "1K:千语 (One Thousand Voices)",
        "ops:火力战队行动 (Fireteam Ops)",
        "ops:竞技场行动 (Arena Ops)",
        "ops:单人行动 (Solo Ops)",
    ):
        assert expected in prompt
    assert "Curus" not in prompt
    assert "aliases:{" not in prompt

    from core.prompts import get_summary_prompt

    summary_prompt = get_summary_prompt(
        "Curious is great.",
        custom_terms_json={
            "terms": [
                {
                    "src": "Cuirass of the Falling Star",
                    "tgt": "陨星胸甲",
                    "aliases": ["Curious"],
                    "src_matched": False,
                }
            ]
        },
        previously_extracted_terms=[
            {
                "src": "Cuirass of the Falling Star",
                "tgt": "陨星胸甲",
                "aliases": ["Curious"],
                "src_matched": False,
            }
        ],
    )
    assert "Curious:陨星胸甲 (Cuirass of the Falling Star)" in summary_prompt
    assert "Cuirass of the Falling Star:陨星胸甲" in summary_prompt

    # 3. Punctuation cleanup & dunhao splitting
    dirty = "“冰影泰坦”——输出机制：依靠冰晶、碎裂伤害；以及「高额爆发」！"
    cleaned_tl = tl._normalize_book_marks(dirty)
    cleaned_s6 = s6.clean_translation(dirty)
    for bad in ("“", "”", "——", "—", "：", "；", "「", "」"):
        assert bad not in cleaned_tl
        assert bad not in cleaned_s6
    assert "、" in cleaned_tl

    long_dunhao = "我们在实战里主要依靠神圣裁决、牵引器火炮、虚空猎人的暗影箭以及光焰之井来提供全队辅助增益。"
    split_res = s5._try_local_clause_split(
        "We rely on Divinity, Tractor Cannon, Tether, and Well of Radiance for team buffs.",
        long_dunhao,
        75,
        1.2,
    )
    assert split_res is not None
    _, tr_parts, _ = split_res
    assert len(tr_parts) >= 2

    # 4. Unseen 2-word KB terms ("Raiden Flux", "Second Chance") must NOT trigger on half-word matches ("raiden", "Second")
    with workspace_context(ws):
        kb_data = s41._get_kb_index(
            [
                {"src": "Raiden Flux", "tgt": "雷电通流"},
                {"src": "Second Chance", "tgt": "第二次机会"},
                {"src": "Shards of Galanor", "tgt": "加拉诺碎片"},
                {"src": "Aggregate Charge", "tgt": "聚合充能"},
            ]
        )
        hits = s41._scan_text_for_kb_terms(
            "Second, against most raiden dungeon bosses, we used Shards of Galanor and Aggregate Charge first.",
            kb_data,
        )
        hit_map = {h["src"]: h for h in hits}
        assert "Raiden Flux" not in hit_map
        assert "Second Chance" not in hit_map
        assert "Shards of Galanor" in hit_map
        assert "Aggregate Charge" in hit_map


def test_sync_subtitles_from_srt_updates_db_and_bilingual_srts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing src.srt / trans.srt externally and syncing updates SQLite utterances/sub_cues and bilingual SRTs."""
    from fastapi.testclient import TestClient

    import api
    from core import _6_gen_sub as s6
    from core.workspace import TaskWorkspace, workspace_context

    monkeypatch.setattr("core.workspace.OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(api, "OUTPUT", tmp_path)
    ws = TaskWorkspace(tmp_path / "sync_srt_task")
    ws.save_manifest({"task_id": "sync_srt_task", "title": "Sync SRT"})
    ws.save_utterances(
        [
            {
                "id": "u_0000",
                "index": 0,
                "start": 0.0,
                "end": 2.5,
                "word_span": [0, 4],
                "src": "Old source line one.",
                "src_punct": "Old source line one.",
                "tgt": "旧译文第一行",
                "tts_text": "旧译文第一行",
                "sub_cues": [
                    {
                        "src": "Old source line one.",
                        "tgt": "旧译文第一行",
                        "trans": "旧译文第一行",
                        "word_span": [0, 4],
                        "start": 0.0,
                        "end": 2.5,
                    }
                ],
            },
            {
                "id": "u_0001",
                "index": 1,
                "start": 2.5,
                "end": 6.0,
                "word_span": [4, 10],
                "src": "First half, second half.",
                "src_punct": "First half, second half.",
                "tgt": "前半句，后半句",
                "tts_text": "前半句，后半句",
                "sub_cues": [
                    {
                        "src": "First half,",
                        "tgt": "前半句",
                        "trans": "前半句",
                        "word_span": [4, 7],
                        "start": 2.5,
                        "end": 4.0,
                    },
                    {
                        "src": "second half.",
                        "tgt": "后半句",
                        "trans": "后半句",
                        "word_span": [7, 10],
                        "start": 4.0,
                        "end": 6.0,
                    },
                ],
            },
        ]
    )

    (ws.root / "src.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,500\nEdited source line one.\n\n"
        "2\n00:00:02,500 --> 00:00:04,200\nEdited first half,\n\n"
        "3\n00:00:04,200 --> 00:00:06,000\nsecond half.\n",
        encoding="utf-8",
    )
    (ws.root / "trans.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,500\n校对后译文第一行\n\n"
        "2\n00:00:02,500 --> 00:00:04,200\n校对前半句\n\n"
        "3\n00:00:04,200 --> 00:00:06,000\n校对后半句\n",
        encoding="utf-8",
    )

    client = TestClient(api.app)
    res = client.post("/api/tasks/sync_srt_task/subtitles/sync")
    assert res.status_code == 200
    assert res.json() == {"cues": 3, "utterances": 2}

    utts = ws.load_utterances()
    assert len(utts) == 2
    assert utts[0]["src"] == "Edited source line one."
    assert utts[0]["tgt"] == "校对后译文第一行"
    assert utts[0]["sub_cues"][0]["tgt"] == "校对后译文第一行"
    assert utts[1]["sub_cues"][0]["src"] == "Edited first half,"
    assert utts[1]["sub_cues"][0]["tgt"] == "校对前半句"
    assert utts[1]["sub_cues"][0]["end"] == 4.2
    assert utts[1]["sub_cues"][1]["tgt"] == "校对后半句"
    assert "校对后译文第一行" in (ws.root / "src_trans.srt").read_text(encoding="utf-8")
    assert "校对后半句" in (ws.root / "trans_src.srt").read_text(encoding="utf-8")

    with workspace_context(ws):
        assert s6.sync_subtitles_from_srt_if_newer() is False


def test_targeted_asr_candidate_retrieval_em_dash_and_domain_dict_filtering(
    monkeypatch, tmp_path
):
    import core._4_1_summarize as s41
    import core.translate_lines as tl
    import core.workspace as ws_mod

    glossary_dir = tmp_path / "glossary"
    d2_dir = glossary_dir / "destiny2"
    d2_dir.mkdir(parents=True)
    (d2_dir / "terms.csv").write_text(
        "src,tgt,aliases,note,fix_source\n"
        "Arc Compounding,电弧复利,,神器模组,false\n"
        "Star-Eater Scales,化星者之鳞,starless,猎人异域腿甲,false\n",
        encoding="utf-8",
    )
    big_dict = {f"SyntheticWeapon{i}": f"合成武器{i}" for i in range(505)}
    big_dict.update(
        {
            "Cull's Shadow": "库尔之影",
            "Praedyth": "普莱蒂斯",
            "Praedyth's Revenge": "普莱蒂斯的复仇",
            "Hezen Vengeance": "赫즌复仇",
            "Temporal Clause": "时间子句",
            "Thread of Ascent": "飞升丝线",
            "Surge Protector": "电涌保护器",
            "Set Bonus: Sage Protector": "套装加成：圣贤保护者",
            "The Queenbreaker": "弑君者",
            "Mask of Bakris": "巴克里斯面具",
            "Books of Sorrow": "悲伤之书",
            "Geodetic HSm": "测地线HSm",
            "Witherhoard": "枯骨鳞片",
            "Lucent Blade": "光明之刃",
            "Winterbite": "寒冬之咬",
            "Triton Vice": "海神之恶",
            "Reprisal": "报复",
            "Impetus": "冲动",
            "Keep": "保留",
            "Mind": "心智",
            "Long": "漫长",
        }
    )
    (d2_dir / "dict.json").write_text(
        json.dumps(big_dict, ensure_ascii=False), encoding="utf-8"
    )

    monkeypatch.setattr(ws_mod, "GLOSSARY_DIR", glossary_dir)
    monkeypatch.setattr(s41, "GLOBAL_GLOSSARY_PATH", str(tmp_path / "no_global.csv"))
    monkeypatch.setattr(s41, "CUSTOM_TERMS_PATH", str(tmp_path / "no.xlsx"))
    monkeypatch.setattr(s41, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv"))
    monkeypatch.setattr(s41, "OPTIONAL_JSON_TERMS_PATH", str(tmp_path / "no.json"))
    monkeypatch.setattr(
        s41,
        "load_key_or",
        lambda k, d=None: "destiny2" if k == "glossary_domain" else d,
    )

    custom_terms = s41.load_custom_terms()
    kb_data = s41._get_kb_index(custom_terms)

    # 1. Em-dash, slash, colon subterm ("Sage Protector"), and "The "-stripped alias ("Queenbreaker")
    scanned = s41._scan_text_for_kb_terms(
        "Doing so by any means—Arc compounding, Sage Protector, or Queenbreaker—boosts damage.",
        kb_data,
    )
    scanned_srcs = {t["src"] for t in scanned}
    assert "Arc Compounding" in scanned_srcs
    assert "Sage Protector" in scanned_srcs
    assert "The Queenbreaker" in scanned_srcs

    # 2. Targeted 3-channel + UnifiedNgramKBIndex ASR candidate retrieval recalls phrase/split/OOV corruptions
    #    while guarding 1-word OOV sub-token matches ("Bacchus" != "Books of Sorrow", "Hazem" != "Geodetic HSm")
    asr_chunk = (
        "Keep in mind for a long time we used cold shadow, pray death's revenge, "
        "praydith's revenge, hesin vengeance, Hazem, Bacchus, witherhorn, temporal claws, threat of ascent, and sage protector."
    )
    cands = s41._retrieve_rare_token_candidates(asr_chunk, kb_data, set())
    cand_by_src = {c["src"]: c for c in cands}
    for expected_src in (
        "Cull's Shadow",
        "Praedyth's Revenge",
        "Hezen Vengeance",
        "Mask of Bakris",
        "Witherhoard",
        "Temporal Clause",
        "Thread of Ascent",
        "Surge Protector",
    ):
        assert expected_src in cand_by_src, (
            f"Missing {expected_src} from {list(cand_by_src)}"
        )
    assert "cold shadow" in [a.lower() for a in cand_by_src["Cull's Shadow"]["aliases"]]
    assert "temporal claws" in [
        a.lower() for a in cand_by_src["Temporal Clause"]["aliases"]
    ]
    assert "bacchus" in [a.lower() for a in cand_by_src["Mask of Bakris"]["aliases"]]
    for noise_word in (
        "Keep",
        "Mind",
        "Long",
        "Reprisal",
        "Impetus",
        "Books of Sorrow",
        "Geodetic HSm",
    ):
        assert noise_word not in cand_by_src

    # 3. _merge_terminology_results (with Pass 2 disambiguation) grounds Pass-1 & un-echoed pre-filter spans
    monkeypatch.setattr(
        s41,
        "ask_gpt",
        lambda prompt, **kw: {
            "decisions": [
                {"span": "Lucien Blades", "canonical_src": "Lucent Blade"},
                {"span": "witherhorn", "canonical_src": "Witherhoard"},
                {"span": "pray death", "canonical_src": "Praedyth"},
            ]
        },
    )
    merged = s41._merge_terminology_results(
        [
            (
                0,
                {
                    "theme": "DPS testing",
                    "terms": [
                        {"src": "temporal claws", "tgt": "时间利爪"},
                        {"src": "Cull's Shadow", "tgt": "库尔之影"},
                        {"src": "Lucien Blades", "tgt": "卢西安之刃"},
                        {"src": "Praedyth's Revenge", "tgt": "普莱蒂斯的复仇"},
                    ],
                },
            )
        ],
        custom_terms,
        kb_data=kb_data,
        full_text=asr_chunk + "\nWe also tested Lucien Blades.",
        run_llm_disambiguation=True,
    )
    merged_by_src = {t["src"]: t for t in merged["terms"]}
    assert "Temporal Clause" in merged_by_src
    assert merged_by_src["Temporal Clause"]["tgt"] == "时间子句"
    assert "temporal claws" in [
        a.lower() for a in merged_by_src["Temporal Clause"]["aliases"]
    ]
    assert "cold shadow" in [
        a.lower() for a in merged_by_src["Cull's Shadow"]["aliases"]
    ]
    assert "Lucent Blade" in merged_by_src
    assert "lucien blades" in [
        a.lower() for a in merged_by_src["Lucent Blade"]["aliases"]
    ]
    assert "Witherhoard" in merged_by_src
    assert "witherhorn" in [a.lower() for a in merged_by_src["Witherhoard"]["aliases"]]
    assert "pray death revenge" in [
        a.lower() for a in merged_by_src["Praedyth's Revenge"]["aliases"]
    ]

    # 4. _get_term_index filters out unconfirmed single-word common English domain_dict entries when task_terms exist
    ws = TaskWorkspace(tmp_path / "task_ws")
    ws.save_terms("DPS testing", merged["terms"])
    with workspace_context(ws):
        prompt_reprisal = s41.search_things_to_note_in_prompt(
            "The reprisal and impetus of cold shadow with Winterbite and Triton Vice on a starless setup."
        )
    assert prompt_reprisal is not None
    assert "Reprisal:" not in prompt_reprisal
    assert "Impetus:" not in prompt_reprisal
    assert "cold shadow:库尔之影 (Cull's Shadow)" in prompt_reprisal
    assert "Winterbite:寒冬之咬" in prompt_reprisal
    assert "Triton Vice:海神之恶" in prompt_reprisal
    assert (
        "starless:化星者之鳞 (Star-Eater Scales) [without Star-Eater Scales; 猎人异域腿甲]"
        in prompt_reprisal
    )

    # 5. valid_translate_result rejects leaked lowercase English source words in CJK translations
    err_res = tl.valid_translate_result(
        {"1": {"trans": "等我们 inevitably 讲到弩的输出循环时"}},
        ["1"],
        ["trans"],
        origin_lines=["When we inevitably talk about crossbow rotations."],
        allowed_en_text="",
    )
    assert err_res["status"] == "error"
    assert "inevitably" in err_res["message"]

    ok_res = tl.valid_translate_result(
        {"1": {"trans": "等我们讲到 Vulkan 的 DPS 输出循环时"}},
        ["1"],
        ["trans"],
        origin_lines=["When we inevitably talk about Vulkan DPS rotations."],
        allowed_en_text="Vulkan:Vulkan 接口",
    )
    assert ok_res["status"] == "success"
