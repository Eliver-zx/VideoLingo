"""Unit tests for target-first subtitle splitting, timeline alignment, and cloud ASR backends."""
import pandas as pd
import pytest

from core._5_split_sub import align_source_by_ratio, calc_len, split_align_subs
from core._6_gen_sub import align_timestamp, clean_translation, format_subtitle_text, get_sentence_timestamps
from core._3_2_split_meaning import find_split_positions, mechanical_split
from core.asr_backend import assembly_asr, deepgram_asr, replicate_asr, transcription_cache, zero_asr


# ------------------------------------------------------------------
# 1. Target-language 32-char threshold & target-first subtitle splitting
# ------------------------------------------------------------------

def test_32_chinese_chars_threshold(monkeypatch):
    import core._5_split_sub as module

    # With max_length=75 and target_multiplier=1.3:
    # 32 chars * 1.75 * 1.3 = 72.8 <= 75 (not split)
    # 33 chars * 1.75 * 1.3 = 75.075 > 75 (triggers split)
    assert calc_len("汉" * 32) * 1.3 <= 75
    assert calc_len("汉" * 33) * 1.3 > 75

    settings = {"subtitle": {"max_length": 75, "target_multiplier": 1.3}, "max_workers": 1}
    monkeypatch.setattr(module, "load_key", lambda key: settings[key])
    monkeypatch.setattr(module, "load_key_or", lambda key, default=None: "简体中文" if key == "target_language" else default)

    called_sentences = []

    def fake_split(sentence, num_parts=2, language=None):
        called_sentences.append((sentence, language))
        mid = len(sentence) // 2
        return sentence[:mid] + "\n" + sentence[mid:]

    monkeypatch.setattr(module, "split_sentence", fake_split)

    line_32 = "这" * 32
    line_33 = "今天我们来深入讨论一下人工智能技术，它将如何彻底改变每个人的生活"
    assert len(line_33) == 32  # wait: 32 chars
    line_34 = "今天我们来深入讨论一下人工智能技术，以及它将如何彻底改变每个人的生活"
    assert len(line_34) == 34

    src_in = [
        "Short source sentence here.",
        "Today we will deeply discuss artificial intelligence technology, and how it will completely change everyone's life.",
    ]
    tr_in = [line_32, line_34]

    src_out, tr_out, remerged = split_align_subs(src_in, tr_in)

    assert len(called_sentences) == 1
    assert called_sentences[0][1] == "简体中文"
    assert len(src_out) == 3 and len(tr_out) == 3
    assert src_out[0] == src_in[0]
    assert tr_out[0] == line_32
    # Source should snap to the comma + conjunction boundary ("technology, / and how...")
    assert src_out[1].endswith("technology,")
    assert src_out[2].startswith("and how")
    assert remerged[1] == line_34


def test_align_source_by_ratio_keeps_single_word_unsplit():
    # When source has fewer words than target parts, never produce an empty string or drop half the line
    src, tr, remerged = align_source_by_ratio("Unbelievable!", "真是不敢相信会发生这样的事情", "真是不敢相信\n会发生这样的事情")
    assert src == ["Unbelievable!"]
    assert tr == ["真是不敢相信会发生这样的事情"]
    assert remerged == "真是不敢相信会发生这样的事情"


def test_chinese_mechanical_and_position_split_with_english_source(monkeypatch):
    import core._3_2_split_meaning as meaning

    # Even when source language is English ('en'), passing Chinese target_language splits Chinese properly
    monkeypatch.setattr(meaning, "load_key", lambda key: "en")
    monkeypatch.setattr(meaning, "get_source_language", lambda: "en")

    zh_no_punct = "今天我们来深入讨论一下人工智能技术以及它将如何彻底改变每个人的生活"
    parts = mechanical_split(zh_no_punct, 2, language="简体中文").split("\n")
    assert len(parts) == 2
    assert "".join(parts) == zh_no_punct

    zh_sentence = "今天我们来聊一聊人工智能，以及它的未来"
    pos = find_split_positions(zh_sentence, "今天我们来聊一聊人工智能，[br]以及它的未来", language="简体中文")
    assert pos == [len("今天我们来聊一聊人工智能，")]


# ------------------------------------------------------------------
# 2. Timeline fuzzy matching, overlap fixing, and CJK formatting
# ------------------------------------------------------------------

def test_word_boundary_fuzzy_match_in_timestamps():
    df_words = pd.DataFrame({
        "text": ["People", "gathered", "on", "mass", "today."],
        "start": [0.0, 0.5, 1.0, 1.2, 1.8],
        "end": [0.4, 0.9, 1.1, 1.7, 2.3],
    })
    # Slight 2-char difference ("en masse" vs "on mass")
    df_sentences = pd.DataFrame({
        "Source": ["People gathered", "en masse today."],
    })
    stamps = get_sentence_timestamps(df_words, df_sentences)
    assert stamps == [(0.0, 0.9), (1.0, 2.3)]


def test_overlap_timestamps_are_resolved_and_formatted(tmp_path):
    df_words = pd.DataFrame({
        "text": ["First", "part", "Second", "part"],
        "start": [0.0, 1.0, 1.5, 2.5],
        "end": [0.9, 2.0, 2.4, 3.5],  # word 1 ends at 2.0, word 2 starts at 1.5 (overlap!)
    })
    df_translate = pd.DataFrame({
        "Source": ["First part", "Second part"],
        "Translation": ["这是 Python 3 的测试，第一部分。", "这是 AI 模型的，第二部分！"],
    })
    df_translate["Translation"] = df_translate["Translation"].apply(clean_translation)
    out = align_timestamp(df_words, df_translate, [("trans.srt", ["Translation"])], str(tmp_path), for_display=True)

    assert out.loc[0, "timestamp"] == "00:00:00,000 --> 00:00:01,750"
    assert out.loc[1, "timestamp"] == "00:00:01,750 --> 00:00:03,500"
    assert out.loc[0, "Translation"] == "这是Python 3的测试 第一部分"
    assert out.loc[1, "Translation"] == "这是AI模型的 第二部分"
    assert format_subtitle_text("测试 123 中文 ABC 标点，句号。") == "测试123中文ABC标点 句号"


# ------------------------------------------------------------------
# 3. Cloud ASR Backends & Cache Key Integration
# ------------------------------------------------------------------

def test_deepgram_response_conversion():
    raw = {
        "results": {
            "channels": [{
                "detected_language": "en",
                "alternatives": [{
                    "transcript": "Hello world.",
                    "words": [
                        {"word": "hello", "punctuated_word": "Hello", "start": 0.1, "end": 0.5},
                        {"word": "world", "punctuated_word": "world.", "start": 0.6, "end": 1.0},
                    ],
                }],
            }],
            "utterances": [{
                "start": 0.1,
                "end": 1.0,
                "transcript": "Hello world.",
            }],
        }
    }
    res = deepgram_asr.deepgram2whisper(raw, offset=10.0, fallback_lang="auto")
    assert res["language"] == "en"
    assert len(res["segments"]) == 1
    assert res["segments"][0]["start"] == 10.1
    assert res["segments"][0]["end"] == 11.0
    assert [w["word"] for w in res["segments"][0]["words"]] == ["Hello", "world."]
    assert transcription_cache.valid_result(res)


def test_assemblyai_response_conversion():
    raw = {
        "language_code": "en",
        "words": [
            {"text": "Hello", "start": 100, "end": 500},
            {"text": "world.", "start": 600, "end": 1000},
            {"text": "Next", "start": 1200, "end": 1500},
        ],
    }
    res = assembly_asr.convert_assemblyai_response(raw, start_time_offset=5.0, fallback_lang="auto")
    assert res["language"] == "en"
    assert len(res["segments"]) == 2
    assert res["segments"][0]["start"] == 5.1
    assert res["segments"][0]["end"] == 6.0
    assert res["segments"][1]["text"] == "Next"
    assert transcription_cache.valid_result(res)


def test_gladia_zero_response_conversion():
    raw = {
        "transcription": {
            "languages": ["zh"],
            "utterances": [{
                "start": 1.0,
                "end": 2.5,
                "text": "你好世界",
                "words": [
                    {"word": "你好", "start": 1.0, "end": 1.7, "confidence": 0.99},
                    {"word": "世界", "start": 1.8, "end": 2.5, "confidence": 0.98},
                ],
            }],
        }
    }
    res = zero_asr.zero2whisper(raw, offset=20.0, fallback_lang="auto")
    assert res["language"] == "zh"
    assert res["segments"][0]["start"] == 21.0
    assert res["segments"][0]["words"][0]["start"] == 21.0
    assert transcription_cache.valid_result(res)


def test_replicate_response_normalization():
    raw = {
        "detected_language": "ja",
        "word_timestamps": [
            {"word": "こんにちは", "start": 0.5, "end": 1.2},
            {"word": "世界", "start": 1.3, "end": 2.0},
        ],
    }
    res = replicate_asr.normalize_replicate_output(raw, offset=3.0, fallback_lang="auto")
    assert res["language"] == "ja"
    assert res["segments"][0]["start"] == 3.5
    assert res["segments"][0]["end"] == 5.0
    assert transcription_cache.valid_result(res)


def test_cloud_asr_cache_keys_and_credential_checks(tmp_path, monkeypatch):
    media = tmp_path / "audio.mp3"
    media.write_bytes(b"dummy audio bytes")

    k_dg1 = transcription_cache.cache_key(
        str(media), {"runtime": "deepgram", "deepgram_model": "nova-3", "language": "en"}, False
    )
    k_dg2 = transcription_cache.cache_key(
        str(media), {"runtime": "deepgram", "deepgram_model": "nova-2", "language": "en"}, False
    )
    k_zero = transcription_cache.cache_key(
        str(media), {"runtime": "zero", "language": "en"}, False
    )
    assert k_dg1 != k_dg2
    assert k_dg1 != k_zero

    monkeypatch.setattr(deepgram_asr, "load_key_or", lambda k, d=None: "")
    with pytest.raises(ValueError, match="whisper.deepgram_api_key"):
        deepgram_asr.configured_credentials()


# ------------------------------------------------------------------
# 4. Pipeline Simplification & Edge-Case Regression Tests
# ------------------------------------------------------------------

def test_split_sentences_by_meaning_preserves_full_sentences(tmp_path, monkeypatch):
    import core._3_2_split_meaning as meaning

    nlp_txt = tmp_path / "split_by_nlp.txt"
    meaning_txt = tmp_path / "split_by_meaning.txt"
    monkeypatch.setattr(meaning, "_3_1_SPLIT_BY_NLP", str(nlp_txt))
    monkeypatch.setattr(meaning, "_3_2_SPLIT_BY_MEANING", str(meaning_txt))

    with pytest.raises(FileNotFoundError):
        meaning.split_sentences_by_meaning()

    nlp_txt.write_text("First complete sentence here.\nSecond complete sentence here.\n", encoding="utf-8")
    meaning.split_sentences_by_meaning()

    assert meaning_txt.read_text(encoding="utf-8") == nlp_txt.read_text(encoding="utf-8")


def test_split_for_sub_main_multi_pass_remerged_no_none_and_mtime_skip(tmp_path, monkeypatch):
    import os
    import time
    import core._5_split_sub as module

    trans_xlsx = tmp_path / "translation_results.xlsx"
    split_xlsx = tmp_path / "translation_results_for_subtitles.xlsx"
    remerged_xlsx = tmp_path / "translation_results_remerged.xlsx"

    orig_sources = ["word " * 25, "short line"]
    orig_translations = ["这是一段非常非常长的需要切分很多次的字幕文本内容测试" * 2, "短字幕"]
    pd.DataFrame({
        "Source": orig_sources,
        "Translation": orig_translations,
    }).to_excel(trans_xlsx, index=False)

    monkeypatch.setattr(module, "_4_2_TRANSLATION", str(trans_xlsx))
    monkeypatch.setattr(module, "_5_SPLIT_SUB", str(split_xlsx))
    monkeypatch.setattr(module, "_5_REMERGED", str(remerged_xlsx))
    monkeypatch.setattr(module, "find_subtitle_file", lambda: None)

    settings = {"subtitle": {"max_length": 40, "target_multiplier": 1.3}, "max_workers": 1}
    monkeypatch.setattr(module, "load_key", lambda key: settings[key])
    monkeypatch.setattr(module, "load_key_or", lambda key, default=None: "简体中文" if key == "target_language" else default)

    call_count = {"n": 0}

    def fake_split(sentence, num_parts=2, language=None):
        call_count["n"] += 1
        mid = len(sentence) // 2
        return sentence[:mid] + "\n" + sentence[mid:]

    monkeypatch.setattr(module, "split_sentence", fake_split)

    module.split_for_sub_main()
    first_calls = call_count["n"]
    assert first_calls >= 2  # Multi-pass split occurred

    df_remerged = pd.read_excel(remerged_xlsx)
    assert not df_remerged.isna().any().any()
    assert df_remerged["Source"].tolist() == orig_sources
    assert df_remerged["Translation"].tolist() == orig_translations

    # Second run when outputs are up-to-date and within max_length should skip
    module.split_for_sub_main()
    assert call_count["n"] == first_calls

    # Tightening subtitle.max_length invalidates the skip even without touching _4_2_TRANSLATION mtime
    settings["subtitle"]["max_length"] = 20
    module.split_for_sub_main()
    after_tighten_calls = call_count["n"]
    assert after_tighten_calls > first_calls

    # Touching _4_2_TRANSLATION with a newer mtime also invalidates the skip
    future_time = os.path.getmtime(split_xlsx) + 10.0
    os.utime(trans_xlsx, (future_time, future_time))
    time.sleep(0.01)
    module.split_for_sub_main()
    assert call_count["n"] > after_tighten_calls


def test_align_timestamp_recomputes_duration_after_gap_and_overlap_smoothing():
    import core._6_gen_sub as gen_sub

    df_text = pd.DataFrame({
        "text": ["hello", "world", "next", "line"],
        "start": [0.0, 0.4, 1.4, 2.0],
        "end": [0.4, 1.0, 1.9, 3.0],
    })
    df_translate = pd.DataFrame({
        "Source": ["hello world", "next line"],
        "Translation": ["你好世界", "下一行"],
    })
    # Gap between line 0 (ends 1.0) and line 1 (starts 1.4) is 0.4s (< 1.0s),
    # so line 0 end timestamp is extended to 1.4s -> duration must become 1.4s, not 1.0s.
    res = gen_sub.align_timestamp(df_text, df_translate, [], output_dir=None, for_display=False)
    assert res.loc[0, "duration"] == pytest.approx(1.4)
    assert res.loc[1, "duration"] == pytest.approx(1.6)


