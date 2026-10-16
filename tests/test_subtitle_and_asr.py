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
