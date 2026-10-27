"""Unit tests for LLM punctuation restoration, word-preserving projection, terminology matching, and translation caching."""

from __future__ import annotations

import json
import os
import pandas as pd
import pytest

from core.task_runner import StopTask


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
    """split_by_spacy writes to both _3_1_SPLIT_BY_NLP and _3_2_SPLIT_BY_MEANING."""
    import core._3_1_split_nlp as split_nlp

    cleaned_xlsx = tmp_path / "cleaned_chunks.xlsx"
    nlp_txt = tmp_path / "split_by_nlp.txt"
    meaning_txt = tmp_path / "split_by_meaning.txt"

    pd.DataFrame(
        {"text": ['"Hello"', '"world"', '"this"', '"is"', '"a"', '"test"']}
    ).to_excel(cleaned_xlsx, index=False)

    monkeypatch.setattr(split_nlp, "_2_CLEANED_CHUNKS", str(cleaned_xlsx))
    monkeypatch.setattr(split_nlp, "_3_1_SPLIT_BY_NLP", str(nlp_txt))
    monkeypatch.setattr(split_nlp, "_3_2_SPLIT_BY_MEANING", str(meaning_txt))
    monkeypatch.setattr(split_nlp, "get_source_language", lambda: "en")
    monkeypatch.setattr(split_nlp, "load_key_or", lambda k, d=None: 1)
    monkeypatch.setattr(
        split_nlp,
        "ask_gpt",
        lambda *args, **kwargs: {"restored": "Hello world. this is a test."},
    )

    # Call split_by_spacy directly (it checks both _3_1_SPLIT_BY_NLP and _3_2_SPLIT_BY_MEANING)
    split_nlp.split_by_spacy()

    assert nlp_txt.read_text(encoding="utf-8").splitlines() == [
        "Hello world.",
        "this is a test.",
    ]
    assert meaning_txt.read_text(encoding="utf-8") == nlp_txt.read_text(
        encoding="utf-8"
    )


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

    term_json = tmp_path / "terminology.json"
    custom_xlsx = tmp_path / "custom_terms.xlsx"
    pd.DataFrame(
        {
            "src": ["Bungie", "Raycasting"],
            "tgt": ["Bungie 工作室", "光线投射"],
            "note": ["游戏开发商", "渲染技术"],
        }
    ).to_excel(custom_xlsx, index=False)

    term_json.write_text(
        json.dumps(
            {
                "theme": "Game dev talk",
                "terms": [
                    {"src": "Destiny", "tgt": "命运", "note": "游戏名"},
                    {"src": "shader pipeline", "tgt": "着色器管线", "note": "图形学"},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(summarize, "_4_1_TERMINOLOGY", str(term_json))
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

    prompt = summarize.search_things_to_note_in_prompt(chunk)
    assert prompt is not None
    assert '"Bungie": "Bungie 工作室"' in prompt
    assert '"shader pipeline": "着色器管线"' in prompt
    assert '"Raycasting": "光线投射"' in prompt
    assert '"Destiny"' not in prompt


def test_short_terms_and_lookalikes_do_not_trigger_false_positives(
    monkeypatch, tmp_path
):
    """Short acronyms (AI, UI, RAM, ID, OS) must not match inside everyday words ('said', 'building',
    'program'), and lookalikes ('faster' vs 'raster', 'severe' vs 'server') must not false-match."""
    import core._4_1_summarize as summarize

    term_json = tmp_path / "terminology.json"
    term_json.write_text(
        json.dumps(
            {
                "theme": "Tech",
                "terms": [
                    {"src": "AI", "tgt": "人工智能", "note": ""},
                    {"src": "UI", "tgt": "界面", "note": ""},
                    {"src": "RAM", "tgt": "内存", "note": ""},
                    {"src": "ID", "tgt": "标识", "note": ""},
                    {"src": "raster", "tgt": "光栅", "note": ""},
                    {"src": "server", "tgt": "服务器", "note": ""},
                    {"src": "model", "tgt": "模型", "note": ""},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(summarize, "_4_1_TERMINOLOGY", str(term_json))
    monkeypatch.setattr(summarize, "CUSTOM_TERMS_PATH", str(tmp_path / "no.xlsx"))
    monkeypatch.setattr(summarize, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv"))
    monkeypatch.setattr(
        summarize, "OPTIONAL_JSON_TERMS_PATH", str(tmp_path / "no.json")
    )

    sentence = "He said that building this program in dark mode runs faster despite a severe issue."
    assert summarize.search_things_to_note_in_prompt(sentence) is None


def test_term_index_cache_invalidates_when_terminology_json_is_edited(
    monkeypatch, tmp_path
):
    """When user edits terminology.json during pause_before_translate, mtime changes and cache reloads."""
    import core._4_1_summarize as summarize

    term_json = tmp_path / "terminology.json"
    monkeypatch.setattr(summarize, "_4_1_TERMINOLOGY", str(term_json))
    monkeypatch.setattr(summarize, "CUSTOM_TERMS_PATH", str(tmp_path / "no.xlsx"))
    monkeypatch.setattr(summarize, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv"))
    monkeypatch.setattr(
        summarize, "OPTIONAL_JSON_TERMS_PATH", str(tmp_path / "no.json")
    )

    term_json.write_text(
        json.dumps(
            {"theme": "t", "terms": [{"src": "OpenGL", "tgt": "旧译名", "note": "n1"}]}
        ),
        encoding="utf-8",
    )
    first_prompt = summarize.search_things_to_note_in_prompt("We use OpenGL here.")
    assert first_prompt is not None and '"OpenGL": "旧译名"' in first_prompt

    # Simulate user editing terminology.json with updated mtime
    term_json.write_text(
        json.dumps(
            {
                "theme": "t",
                "terms": [{"src": "OpenGL", "tgt": "新译名_已修改", "note": "n2"}],
            }
        ),
        encoding="utf-8",
    )
    st = os.stat(term_json)
    os.utime(term_json, (st.st_atime, st.st_mtime + 2.0))

    second_prompt = summarize.search_things_to_note_in_prompt("We use OpenGL here.")
    assert second_prompt is not None and '"OpenGL": "新译名_已修改"' in second_prompt


def test_get_summary_merges_chunks_and_custom_terms_consistently(monkeypatch, tmp_path):
    """Multi-chunk extraction runs concurrently and unifies cross-chunk variants with earlier chunk winning."""
    import core._4_1_summarize as summarize

    meaning_txt = tmp_path / "split_by_meaning.txt"
    term_json = tmp_path / "terminology.json"
    custom_xlsx = tmp_path / "custom_terms.xlsx"

    meaning_txt.write_text(
        "Sentence one in chunk zero.\nSentence two in chunk one.\n", encoding="utf-8"
    )
    pd.DataFrame(
        {"src": ["Vulkan"], "tgt": ["Vulkan 接口"], "note": ["图形API"]}
    ).to_excel(custom_xlsx, index=False)

    monkeypatch.setattr(summarize, "_3_2_SPLIT_BY_MEANING", str(meaning_txt))
    monkeypatch.setattr(summarize, "_4_1_TERMINOLOGY", str(term_json))
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

    getattr(summarize.get_summary, "__wrapped__")()

    saved = json.loads(term_json.read_text(encoding="utf-8"))
    assert "Chunk 0 theme." in saved["theme"]
    assert saved["topic"] == saved["theme"]
    terms_by_src = {t["src"]: t["tgt"] for t in saved["terms"]}
    assert terms_by_src["Vulkan"] == "Vulkan 接口"
    assert terms_by_src["Nanite"] == "Nanite 虚拟微多边形"
    assert terms_by_src["Nanites"] == "Nanite 虚拟微多边形"
    assert terms_by_src["Lumen"] == "Lumen 全局光照"


# ------------------------------------------------------------------
# 3. _4_2_translate: Raw translation cache reuse & fingerprint invalidation
# ------------------------------------------------------------------


def test_translate_all_reuses_raw_cache_and_invalidates_on_terminology_change(
    monkeypatch, tmp_path
):
    """If translation_raw.csv matches _3_2_SPLIT_BY_MEANING and fingerprint, LLM is skipped;
    if terminology.json changes, cache is invalidated."""
    import core._4_2_translate as translate

    meaning_txt = tmp_path / "split_by_meaning.txt"
    term_json = tmp_path / "terminology.json"
    raw_csv = tmp_path / "translation_raw.csv"
    cleaned_xlsx = tmp_path / "cleaned_chunks.xlsx"
    out_xlsx = tmp_path / "translation_results.xlsx"

    meaning_txt.write_text("Hello world.\nHow are you?\n", encoding="utf-8")
    term_json.write_text('{"theme": "test", "terms": []}', encoding="utf-8")

    monkeypatch.setattr(translate, "_3_2_SPLIT_BY_MEANING", str(meaning_txt))
    monkeypatch.setattr(translate, "_4_1_TERMINOLOGY", str(term_json))
    monkeypatch.setattr(translate, "_4_2_TRANSLATION_RAW", str(raw_csv))
    monkeypatch.setattr(translate, "_2_CLEANED_CHUNKS", str(cleaned_xlsx))
    monkeypatch.setattr(translate, "_4_2_TRANSLATION", str(out_xlsx))
    monkeypatch.setattr(
        translate, "load_key", lambda k: 10.0 if k == "min_trim_duration" else 2
    )
    monkeypatch.setattr(translate, "load_key_or", lambda k, d=None: "zh")

    fp = translate._translation_cache_fingerprint()
    pd.DataFrame(
        {
            "Source": ["Hello world.", "How are you?"],
            "Translation": ["你好，世界。", "你好吗？"],
            "Fingerprint": [fp, fp],
        }
    ).to_csv(raw_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "text": ["Hello", "world.", "How", "are", "you?"],
            "start": [0.0, 0.5, 1.0, 1.5, 2.0],
            "end": [0.4, 0.9, 1.4, 1.9, 2.5],
        }
    ).to_excel(cleaned_xlsx, index=False)

    monkeypatch.setattr(
        translate,
        "translate_chunk",
        lambda *args, **kwargs: pytest.fail(
            "Should have reused translation_raw.csv without calling LLM"
        ),
    )
    monkeypatch.setattr(
        translate,
        "align_timestamp",
        lambda df_text, df_trans, *a, **kw: df_trans.assign(duration=[1.0, 1.5]),
    )

    getattr(translate.translate_all, "__wrapped__")()

    result = pd.read_excel(out_xlsx)
    assert result["Source"].tolist() == ["Hello world.", "How are you?"]
    assert result["Translation"].tolist() == ["你好，世界。", "你好吗？"]

    # Now modify terminology.json -> fingerprint changes -> cache must return None
    term_json.write_text(
        '{"theme": "updated", "terms": [{"src": "world", "tgt": "天下", "note": ""}]}',
        encoding="utf-8",
    )
    new_fp = translate._translation_cache_fingerprint()
    assert new_fp != fp
    assert (
        translate._load_cached_raw_translation(
            ["Hello world.", "How are you?"], fingerprint=new_fp
        )
        is None
    )
    # Legacy cache without Fingerprint column must be rejected when fingerprint is required
    pd.DataFrame(
        {
            "Source": ["Hello world.", "How are you?"],
            "Translation": ["旧缓存1", "旧缓存2"],
        }
    ).to_csv(raw_csv, index=False, encoding="utf-8-sig")
    assert (
        translate._load_cached_raw_translation(
            ["Hello world.", "How are you?"], fingerprint=new_fp
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
        "<previous_content>\nPrev line 1.\nPrev line 2.\n</previous_content>"
        in with_lists
    )
    assert (
        "<subsequent_content>\nNext line 1.\nNext line 2.\n</subsequent_content>"
        in with_lists
    )
    assert "['Prev line 1.'" not in with_lists

    with_strings = generate_shared_prompt("prev", "next", "summary", "notes")
    assert "<previous_content>\nprev\n</previous_content>" in with_strings
    assert "<subsequent_content>\nnext\n</subsequent_content>" in with_strings

    with_nones = generate_shared_prompt(None, None, None, None)
    assert "<previous_content>\n\n</previous_content>" in with_nones
    assert "<subsequent_content>\n\n</subsequent_content>" in with_nones
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
            return {"1": {"direct": "第一行直译"}, "2": {"direct": "第二行直译"}}
        return {
            "1": {"reflect": "OK", "free": "第一行直译"},
            "2": {"reflect": "Adjusted phrasing", "free": "第二行润色"},
        }

    monkeypatch.setattr(tl, "ask_gpt", fake_ask_gpt)
    monkeypatch.setattr(
        tl, "load_key", lambda k: True if k == "reflect_translate" else "简体中文"
    )
    monkeypatch.setattr(tl, "check_cancel", lambda: None)
    monkeypatch.setattr(tl.console, "print", lambda *a, **kw: None)

    trans, orig = tl.translate_lines(
        "Line one.\nLine two.", ["Prev line."], ["Next line."], "notes", "summary", 0
    )
    assert trans == "第一行直译\n第二行润色"
    assert orig == "Line one.\nLine two."


def test_translate_chunk_scans_previous_content_and_chunk_for_terms(monkeypatch):
    """translate_chunk passes both previous_content (last 4 lines) and current chunk to terminology search."""
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
    assert "L3 mentioning Raycasting" in captured["search_scope"]
    assert "Current chunk referring to it." in captured["search_scope"]


def test_translation_prompts_contain_asr_recovery_and_keep_if_flawless_in_english(
    monkeypatch,
):
    """Both translation prompts contain English instructions for ASR phonetic recovery and Keep-if-Flawless."""
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

    assert "ASR Homophone & Phonetic Error Recovery" in faith_prompt
    assert "Strict Terminology & Clean Proper Nouns" in faith_prompt
    assert "Principle of Necessary Enhancement (Keep-if-Flawless)" in express_prompt
    assert "Anti-Translationese" in express_prompt


def test_symbol_terms_do_not_pollute_stem_ngram_map_and_aliases_appear_in_prompt(
    monkeypatch, tmp_path
):
    """Symbol terms (.NET, C#, C++) must not strip symbols to match plain 'net' or 'c';
    matched terms with aliases must include '(also transcribed as: ...)' in Points to Note."""
    import core._4_1_summarize as summarize

    term_json = tmp_path / "terminology.json"
    term_json.write_text(
        json.dumps(
            {
                "theme": "Software engineering",
                "terms": [
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
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(summarize, "_4_1_TERMINOLOGY", str(term_json))
    monkeypatch.setattr(
        summarize, "GLOBAL_GLOSSARY_PATH", str(tmp_path / "no_global.csv")
    )
    monkeypatch.setattr(summarize, "CUSTOM_TERMS_PATH", str(tmp_path / "no.xlsx"))
    monkeypatch.setattr(summarize, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv"))
    monkeypatch.setattr(
        summarize, "OPTIONAL_JSON_TERMS_PATH", str(tmp_path / "no.json")
    )

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
    assert '".NET"' in sym_prompt
    assert '"C++"' in sym_prompt
    assert '"C#"' not in sym_prompt

    # Alias match must surface the alias in Points to Note so LLM links 'cooper netties' -> 'Kubernetes'
    alias_prompt = summarize.search_things_to_note_in_prompt(
        "We deployed the cluster on cooper netties in production."
    )
    assert alias_prompt is not None
    assert '"Kubernetes"' in alias_prompt
    assert "cooper netties" in alias_prompt
    assert "also transcribed as" in alias_prompt


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
    merged = _merge_terminology_results(chunk_results, custom_terms)
    by_src = {t["src"]: t for t in merged["terms"]}
    assert "pie torch" not in by_src
    assert "PyTorch" in by_src
    assert by_src["PyTorch"]["tgt"] == "PyTorch 框架"
    assert "pie torch" in by_src["PyTorch"]["aliases"]
    assert "pi torch" in by_src["PyTorch"]["aliases"]
    assert by_src["TensorRT"]["aliases"] == ["tensor artie"]


def test_multiword_fuzzy_guards_and_longest_span_suppression(monkeypatch, tmp_path):
    """Multi-word fuzzy matching must not false-match 'rate weapons' -> 'heat weapons' or 'and compendium' -> 'data compendium',
    and Longest-Span Suppression must suppress 'Compendium' when covered by 'Data Compendium'."""
    import core._4_1_summarize as summarize

    term_json = tmp_path / "terminology.json"
    term_json.write_text(
        json.dumps(
            {
                "theme": "Destiny 2 weapon guide",
                "terms": [
                    {"src": "Heat weapons", "tgt": "热量武器", "note": "武器机制"},
                    {"src": "Data Compendium", "tgt": "数据汇编", "note": "社区表格"},
                    {"src": "Compendium", "tgt": "汇编", "note": "简称"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(summarize, "_4_1_TERMINOLOGY", str(term_json))
    monkeypatch.setattr(
        summarize, "GLOBAL_GLOSSARY_PATH", str(tmp_path / "no_global.csv")
    )
    monkeypatch.setattr(summarize, "CUSTOM_TERMS_PATH", str(tmp_path / "no.xlsx"))
    monkeypatch.setattr(summarize, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv"))
    monkeypatch.setattr(
        summarize, "OPTIONAL_JSON_TERMS_PATH", str(tmp_path / "no.json")
    )

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
    assert '"Data Compendium": "数据汇编"' in span_prompt
    assert '"Compendium": "汇编"' not in span_prompt


def test_generic_kb_candidate_retrieval_and_llm_reduce_consolidation(
    monkeypatch, tmp_path
):
    """Schema-agnostic KB indexing extracts entries from nested JSON, fuzzy-matches ASR variants ('Multimoc', 'DFA')
    to official KB entries ('Multimach CCX', 'D.F.A.'), scans direct KB hits, and preserves transcript surface forms in aliases."""
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
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(summarize, "OPTIONAL_JSON_TERMS_PATH", str(kb_json))
    monkeypatch.setattr(summarize, "load_key_or", lambda k, d=None: "简体中文")
    monkeypatch.setattr("core.prompts.get_source_language", lambda: "en")
    monkeypatch.setattr("core.prompts.load_key", lambda k: "简体中文")

    kb_data = summarize._get_kb_index([])
    cands_multi = summarize._retrieve_kb_candidates_for_term(
        {"src": "Multimach", "aliases": ["Multimoc"]}, kb_data
    )
    assert any(
        c["src"] == "Multimach CCX" and c["tgt"] == "多机系统CCX" for c in cands_multi
    )

    cands_dfa = summarize._retrieve_kb_candidates_for_term({"src": "DFA"}, kb_data)
    assert any(c["src"] == "D.F.A." and c["tgt"] == "死从天降" for c in cands_dfa)

    # Direct KB scan finds "Hand Cannons" in transcript while skipping conversational stopword phrases "what if" and "all"
    transcript = (
        "What if we compare all Hand Cannons against the Multimoc and DFA today?"
    )
    scanned = summarize._scan_text_for_kb_terms(transcript, kb_data)
    scanned_srcs = {s["src"] for s in scanned}
    assert "Hand Cannons" in scanned_srcs
    assert "What If" not in scanned_srcs
    assert "All" not in scanned_srcs

    # LLM Reduce consolidation adopts official KB translations and keeps original surface forms in aliases
    chunk_results = [
        (
            0,
            {
                "theme": "Weapon comparison.",
                "terms": [
                    {
                        "src": "Multimach",
                        "tgt": "多级机枪",
                        "aliases": ["Multimoc"],
                        "note": "SMG",
                    },
                    {"src": "DFA", "tgt": "DFA手炮", "note": "Hand cannon"},
                ],
            },
        )
    ]
    merged = summarize._merge_terminology_results(chunk_results, [])

    def fake_consolidate_gpt(prompt, **kwargs):
        assert "Multimach CCX" in prompt
        assert "多机系统CCX" in prompt
        assert "Hand Cannons" in prompt
        return {
            "terms": [
                {
                    "src": "Multimach CCX",
                    "tgt": "多机系统CCX",
                    "aliases": ["Multimach", "Multimoc"],
                    "note": "Iron Banner SMG",
                },
                {
                    "src": "D.F.A.",
                    "tgt": "死从天降",
                    "aliases": ["DFA"],
                    "note": "Nightfall Hand Cannon",
                },
                {
                    "src": "Hand Cannons",
                    "tgt": "手炮",
                    "aliases": [],
                    "note": "Weapon archetype",
                },
            ]
        }

    monkeypatch.setattr(summarize, "ask_gpt", fake_consolidate_gpt)
    consolidated = summarize._consolidate_terms_with_llm(
        merged, chunk_results, [], kb_data=kb_data, full_text=transcript
    )
    by_src = {t["src"]: t for t in consolidated["terms"]}
    assert by_src["Multimach CCX"]["tgt"] == "多机系统CCX"
    assert "Multimach" in by_src["Multimach CCX"]["aliases"]
    assert "Multimoc" in by_src["Multimach CCX"]["aliases"]
    assert by_src["D.F.A."]["tgt"] == "死从天降"
    assert "DFA" in by_src["D.F.A."]["aliases"]
    assert by_src["Hand Cannons"]["tgt"] == "手炮"

    # Also verify _get_term_index() does not deadlock when OPTIONAL_JSON_TERMS_PATH exists
    term_json = tmp_path / "terminology.json"
    term_json.write_text(json.dumps(consolidated), encoding="utf-8")
    monkeypatch.setattr(summarize, "_4_1_TERMINOLOGY", str(term_json))
    monkeypatch.setattr(summarize, "CUSTOM_TERMS_PATH", str(tmp_path / "no.xlsx"))
    monkeypatch.setattr(summarize, "OPTIONAL_CSV_TERMS_PATH", str(tmp_path / "no.csv"))
    note_prompt = summarize.search_things_to_note_in_prompt(
        "Using Multimoc and DFA today."
    )
    assert note_prompt is not None
    assert "多机系统CCX" in note_prompt
    assert "死从天降" in note_prompt


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
                "1": {"origin": "Hello world.", "direct": "你好，世界。"},
                "2": {"origin": "Use Multimach CCX.", "direct": "使用多机系统CCX。"},
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
    import re

    import core._5_split_sub as s5
    import core.prompts as prompts

    monkeypatch.setattr(
        prompts, "load_key", lambda k: "Español" if k == "target_language" else False
    )
    monkeypatch.setattr(prompts, "load_key_or", lambda k, d="": "")
    monkeypatch.setattr(prompts, "get_source_language", lambda: "English")

    shared = prompts.generate_shared_prompt(["Prev"], ["Next"], "Summary", "Notes")
    faith_prompt = prompts.get_prompt_faithfulness("Hello world.", shared)
    consolidate_prompt = prompts.get_consolidate_terms_prompt(
        [{"src": "API", "tgt": "API"}],
        theme="Tech",
        target_language="Español",
    )

    assert not re.search(r"[\u4e00-\u9fff]", faith_prompt)
    assert not re.search(r"[\u4e00-\u9fff]", consolidate_prompt)

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
    import re

    import core._3_1_split_nlp as s31
    import core._4_1_summarize as s41
    import core.prompts as prompts

    monkeypatch.setattr(
        prompts, "load_key", lambda k: "Español" if k == "target_language" else True
    )
    monkeypatch.setattr(prompts, "load_key_or", lambda k, d="": "")
    monkeypatch.setattr(prompts, "get_source_language", lambda: "English")

    # 1. get_prompt_expressiveness has zero hardcoded Chinese when target is Español
    shared = prompts.generate_shared_prompt(["Prev"], ["Next"], "Summary", "Notes")
    express_prompt = prompts.get_prompt_expressiveness(
        {"1": {"origin": "Hello world.", "direct": "Hola mundo."}},
        "Hello world.",
        shared,
    )
    assert not re.search(r"[\u4e00-\u9fff]", express_prompt)

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
