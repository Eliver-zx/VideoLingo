"""Map ForcedAligner tokens back onto ASR transcripts while preserving punctuation."""

from collections.abc import Iterable
from typing import Any
import unicodedata

from core.asr_backend.qwen_asr.config import MAX_WORD_LENGTH


def _is_token_char(ch: str) -> bool:
    # Qwen3ForceAlignProcessor keeps letters, numbers and apostrophes only.
    return ch == "'" or unicodedata.category(ch)[0] in "LN"


def attach_words(
    text: str,
    items: Iterable[tuple[str, float, float]],
    offset: float = 0.0,
    limit: float | None = None,
) -> list[dict[str, Any]]:
    """Map aligner tokens back onto the ASR text so words keep punctuation.

    Aligner tokens are ordered subsequences of the text with punctuation removed
    ("U.S.," -> "US"); each word takes its token's span plus adjacent punctuation,
    e.g. "Hello, world." -> ["Hello,", "world."], "你好，世界。" -> ["你", "好，", "世", "界。"].
    Times are clamped to [0, limit] (the clip length): text longer than the audio, e.g. a
    window the model translated, made the aligner return times past the end of the audio.
    """

    def clamp(value: float) -> float:
        value = max(0.0, float(value))
        return min(value, limit) if limit is not None else value

    # Nagisa normalizes Japanese tokens (e.g. ＡＩ -> AI, ｶﾞ -> ガ).
    # Match compatibility-decomposed characters, retaining their original offsets.
    # NFKD also lets a composed kana match a half-width base + voiced mark.
    normalized: list[str] = []
    original_ends: list[int] = []
    for index, char in enumerate(text):
        part = unicodedata.normalize("NFKD", char)
        normalized.extend(part)
        original_ends.extend([index + 1] * len(part))
    words: list[dict[str, Any]] = []
    pos, cursor, n = 0, 0, len(text)
    for token, start, end in items:
        if not token:
            continue
        token_norm = unicodedata.normalize("NFKD", token)
        max_scan = min(len(normalized), cursor + len(token_norm) + 12)
        k = cursor
        for ch in token_norm:
            while k < max_scan and normalized[k] != ch:
                k += 1
            if k >= max_scan:
                break
            k += 1
        else:
            cursor = k
            j = original_ends[k - 1]
            if j <= pos and words:
                # One compatibility character can expand to several aligner tokens
                # (㍿ -> 株式 + 会社). Emit it once, covering all their timestamps.
                words[-1]["end"] = max(words[-1]["end"], round(offset + clamp(end), 3))
                continue
            while j < n and not text[j].isspace() and not _is_token_char(text[j]):
                j += 1
            surface = text[pos:j].strip()
            pos = j
            if len(surface) > MAX_WORD_LENGTH:
                surface = token
            c_start, c_end = clamp(start), clamp(end)
            words.append(
                {
                    "word": surface,
                    "start": round(offset + c_start, 3),
                    "end": round(offset + max(c_start, c_end), 3),
                }
            )
            continue
        # Token not found in order (should not happen); keep its bare text without consuming the transcript.
        c_start, c_end = clamp(start), clamp(end)
        words.append(
            {
                "word": token,
                "start": round(offset + c_start, 3),
                "end": round(offset + max(c_start, c_end), 3),
            }
        )
    if words and pos < n:
        # Trailing punctuation left after the last token (e.g. closing quotes).
        tail = text[pos:].strip()
        if tail and len(words[-1]["word"] + tail) <= MAX_WORD_LENGTH:
            words[-1]["word"] += tail
    return words
