from collections.abc import Callable
import difflib
from typing import Any

from core.nlp_split.tokenizer import (
    ACOUSTIC_LONG_PAUSE_GAP,
    ACOUSTIC_PAUSE_PROMPT_GAP,
    CLAUSE_PUNCT_CHARS,
    LONG_PAUSE_MARKER,
    PAUSE_MARKER,
    SENTENCE_END_CHARS,
    _TRAILING_QUOTE_CHARS,
    _clean_word,
    _existing_sentence_ends,
    _is_true_sentence_end_on_word,
    _split_atomic_tokens,
)
from core.prompts import get_punctuation_prompt
from core.utils import join_words, log_event

TokenizedWords = tuple[list[str], list[int], list[bool]]
TokenizedRestored = tuple[list[str], list[str], list[str]]
MatchingBlocks = list[difflib.Match]


def _extract_trailing_punct(token: str) -> str:
    """Extract the strongest trailing punctuation mark from ``token``."""
    stripped = str(token).strip().rstrip(_TRAILING_QUOTE_CHARS)
    if not stripped:
        return ""
    tail_punct = ""
    i = len(stripped) - 1
    while i >= 0 and (
        stripped[i] in SENTENCE_END_CHARS or stripped[i] in CLAUSE_PUNCT_CHARS
    ):
        ch = stripped[i]
        if ch in SENTENCE_END_CHARS:
            tail_punct = ch
            break
        if not tail_punct:
            tail_punct = ch
        i -= 1
    return tail_punct


def _tokenize_words_for_alignment(words: list[str]) -> TokenizedWords:
    """Expand ``words`` into atomic clean tokens with ``(clean_tok, word_idx, is_word_tail)``."""
    clean_toks: list[str] = []
    word_indices: list[int] = []
    is_tails: list[bool] = []
    for w_idx, w in enumerate(words):
        sub_matches = _split_atomic_tokens(str(w))
        valid_subs = [(cw, m) for m in sub_matches if (cw := _clean_word(m))]
        if not valid_subs:
            cw = _clean_word(w)
            if cw:
                valid_subs = [(cw, str(w))]
        for sub_i, (cw, _) in enumerate(valid_subs):
            clean_toks.append(cw)
            word_indices.append(w_idx)
            is_tails.append(sub_i == len(valid_subs) - 1)
    return clean_toks, word_indices, is_tails


def _tokenize_restored_text(restored_text: str) -> TokenizedRestored:
    """Tokenize ``restored_text`` into ``(clean_tokens, trailing_puncts, raw_tokens)``."""
    clean_toks: list[str] = []
    puncts: list[str] = []
    raw_toks: list[str] = []
    for raw_tok in _split_atomic_tokens(str(restored_text)):
        p_ch = _extract_trailing_punct(raw_tok)
        cw = _clean_word(raw_tok)
        if cw:
            clean_toks.append(cw)
            puncts.append(p_ch)
            raw_toks.append(raw_tok)
        elif p_ch and puncts:
            if p_ch in SENTENCE_END_CHARS or not puncts[-1]:
                puncts[-1] = p_ch
            raw_toks[-1] += raw_tok
    return clean_toks, puncts, raw_toks


def _align_token_sequences(
    src_toks: list[str], res_toks: list[str]
) -> tuple[float, MatchingBlocks]:
    if not src_toks:
        return 1.0, []
    if not res_toks:
        return 0.0, []
    matcher = difflib.SequenceMatcher(None, src_toks, res_toks, autojunk=False)
    blocks = list(matcher.get_matching_blocks())
    matched = sum(size for _, _, size in blocks)
    return matched / len(src_toks), blocks


def _verify_word_alignment(win_words: list[str], restored_text: str) -> float:
    """Return token-level similarity ratio in [0.0, 1.0] between ``win_words`` and ``restored_text``."""
    src_toks, _, _ = _tokenize_words_for_alignment(win_words)
    res_toks, _, _ = _tokenize_restored_text(restored_text)
    ratio, _ = _align_token_sequences(src_toks, res_toks)
    return ratio


def _opcodes_from_matching_blocks(
    matching_blocks: MatchingBlocks,
) -> list[tuple[str, int, int, int, int]]:
    i = j = 0
    opcodes: list[tuple[str, int, int, int, int]] = []
    for ai, bj, size in matching_blocks:
        tag = ""
        if i < ai and j < bj:
            tag = "replace"
        elif i < ai:
            tag = "delete"
        elif j < bj:
            tag = "insert"
        if tag:
            opcodes.append((tag, i, ai, j, bj))
        i, j = ai + size, bj + size
        if size:
            opcodes.append(("equal", ai, i, bj, j))
    return opcodes


def _project_window_punctuation(
    win_words: list[str],
    restored_text: str,
    word_gaps: list[float] | None = None,
    *,
    src_tokenized: TokenizedWords | None = None,
    res_tokenized: TokenizedRestored | None = None,
    matching_blocks: MatchingBlocks | None = None,
) -> tuple[set[int], dict[int, str]]:
    """Project punctuation from ``restored_text`` onto ``win_words`` using token-level alignment."""
    if not win_words or not restored_text:
        return set(), {}

    src_toks, tok_to_word, _ = (
        src_tokenized
        if src_tokenized is not None
        else _tokenize_words_for_alignment(win_words)
    )
    res_toks, res_puncts, res_raws = (
        res_tokenized
        if res_tokenized is not None
        else _tokenize_restored_text(restored_text)
    )
    if not src_toks or not res_toks:
        return set(), {}

    blocks = (
        matching_blocks
        if matching_blocks is not None
        else list(
            difflib.SequenceMatcher(
                None, src_toks, res_toks, autojunk=False
            ).get_matching_blocks()
        )
    )
    res_to_src_tok: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in _opcodes_from_matching_blocks(blocks):
        if tag == "equal":
            for offset in range(j2 - j1):
                res_to_src_tok[j1 + offset] = i1 + offset
        elif tag == "replace" and (i2 > i1) and (j2 > j1):
            src_span = i2 - i1
            res_span = j2 - j1
            if src_span == res_span:
                for offset in range(res_span):
                    res_to_src_tok[j1 + offset] = i1 + offset
            else:
                for r_off in range(res_span - 1):
                    mapped = i1 + min(
                        src_span - 1, int((r_off + 1) * src_span / res_span) - 1
                    )
                    res_to_src_tok[j1 + r_off] = max(i1, mapped)
                # Multi-word alias/contraction replacement always anchors its final token to i2 - 1
                res_to_src_tok[j2 - 1] = i2 - 1

    end_indices: set[int] = set()
    punct_by_word: dict[int, str] = {}
    last_w_idx = len(win_words) - 1

    for r_idx, p_ch in enumerate(res_puncts):
        if not p_ch or r_idx not in res_to_src_tok:
            continue
        s_tok_idx = res_to_src_tok[r_idx]
        w_idx = tok_to_word[s_tok_idx]
        # Avoid splitting inside a multi-character word when a replace opcode hits an internal subtoken
        if s_tok_idx + 1 < len(tok_to_word) and tok_to_word[s_tok_idx + 1] == w_idx:
            continue

        prev_p = punct_by_word.get(w_idx, "")
        if p_ch in SENTENCE_END_CHARS or not prev_p:
            punct_by_word[w_idx] = p_ch

        gap_w = (
            word_gaps[w_idx]
            if (word_gaps is not None and w_idx < len(word_gaps))
            else 0.0
        )
        next_res_raw = (
            res_raws[r_idx + 1]
            if r_idx + 1 < len(res_raws)
            else (win_words[w_idx + 1] if w_idx + 1 < len(win_words) else None)
        )
        if p_ch in SENTENCE_END_CHARS and _is_true_sentence_end_on_word(
            win_words[w_idx],
            p_ch,
            gap_after=gap_w,
            next_word=next_res_raw,
        ):
            if w_idx == last_w_idx:
                raw_last = str(win_words[w_idx]).strip().rstrip(_TRAILING_QUOTE_CHARS)
                had_orig_end = bool(
                    raw_last
                    and raw_last[-1] in SENTENCE_END_CHARS
                    and _is_true_sentence_end_on_word(
                        win_words[w_idx], raw_last[-1], gap_after=gap_w
                    )
                )
                if not had_orig_end and gap_w < ACOUSTIC_PAUSE_PROMPT_GAP:
                    continue
            end_indices.add(w_idx)

    return end_indices, punct_by_word


def _strip_prompt_punct(word_str: str) -> str:
    """Strip unreliable raw ASR punctuation while keeping decimal and abbreviation dots."""
    s = str(word_str).strip()
    if not s:
        return s
    while s and (s[-1] in SENTENCE_END_CHARS or s[-1] in CLAUSE_PUNCT_CHARS):
        if s[-1] == "." and not _is_true_sentence_end_on_word(s, "."):
            break
        s = s[:-1]
    return s or str(word_str).strip()


def _build_window_prompt_text(
    win_words: list[str],
    joiner: str,
    word_gaps: list[float] | None = None,
) -> str:
    """Join ``win_words`` with tiered acoustic pause markers (`‖` >= 0.4s, `‖‖` >= 1.0s)."""
    pieces: list[str] = []
    n = len(win_words)
    for i, w in enumerate(win_words):
        pieces.append(_strip_prompt_punct(w))
        if word_gaps is not None and i < n - 1 and i < len(word_gaps):
            gap = word_gaps[i]
            if gap >= ACOUSTIC_LONG_PAUSE_GAP:
                pieces.append(LONG_PAUSE_MARKER)
            elif gap >= ACOUSTIC_PAUSE_PROMPT_GAP:
                pieces.append(PAUSE_MARKER)
    return join_words(pieces, joiner)


def _restore_window_punctuation(
    win_idx: int,
    win_words: list[str],
    joiner: str,
    language: str,
    word_gaps: list[float] | None = None,
    *,
    ask_gpt_fn: Callable[..., Any] | None = None,
) -> tuple[set[int], dict[int, str]]:
    """Call LLM to restore punctuation for one window and return ``(end_indices, punct_by_word)``."""
    if ask_gpt_fn is None:
        from core import _3_1_split_nlp

        ask_gpt_fn = _3_1_split_nlp.ask_gpt

    raw_text = _build_window_prompt_text(win_words, joiner, word_gaps=word_gaps)
    prompt = get_punctuation_prompt(raw_text, language=language)
    src_tokenized = _tokenize_words_for_alignment(win_words)
    validated_cache: dict[str, tuple[TokenizedRestored, MatchingBlocks]] = {}

    def valid_punct(resp: Any) -> dict[str, str]:
        if not isinstance(resp, dict) or "restored" not in resp:
            return {"status": "error", "message": "Missing `restored` key in JSON"}
        restored_val = str(resp.get("restored", "")).strip()
        if not restored_val:
            return {"status": "error", "message": "`restored` text is empty"}
        cleaned_restored = (
            restored_val.replace(LONG_PAUSE_MARKER, "")
            .replace(PAUSE_MARKER, "")
            .strip()
        )
        res_tokenized = _tokenize_restored_text(cleaned_restored)
        ratio, blocks = _align_token_sequences(src_tokenized[0], res_tokenized[0])
        if len(win_words) >= 5 and ratio < 0.80:
            return {
                "status": "error",
                "message": f"Restored text alignment ratio too low ({ratio:.2f} < 0.80)",
            }
        validated_cache[cleaned_restored] = (res_tokenized, blocks)
        return {"status": "success", "message": "ok"}

    try:
        resp = ask_gpt_fn(
            prompt,
            resp_type="json",
            valid_def=valid_punct,
            log_title=f"punct_restore_{win_idx}",
        )
        restored_text = (
            str(resp["restored"])
            .replace(LONG_PAUSE_MARKER, "")
            .replace(PAUSE_MARKER, "")
        )
        cached = validated_cache.get(restored_text.strip())
        return _project_window_punctuation(
            win_words,
            restored_text,
            word_gaps=word_gaps,
            src_tokenized=src_tokenized,
            res_tokenized=cached[0] if cached is not None else None,
            matching_blocks=cached[1] if cached is not None else None,
        )
    except Exception as exc:
        if type(exc).__name__ == "StopTask":
            raise
        log_event(
            f"标点恢复窗口 #{win_idx + 1} 降级为原标点 ({exc})",
            tag="warn",
            label="NLP",
            level="warn",
        )
        ends = _existing_sentence_ends(win_words, joiner=joiner, word_gaps=word_gaps)
        fallback_punct: dict[int, str] = {}
        for i, w in enumerate(win_words):
            s = str(w).rstrip(_TRAILING_QUOTE_CHARS)
            if s and (s[-1] in SENTENCE_END_CHARS or s[-1] in CLAUSE_PUNCT_CHARS):
                fallback_punct[i] = s[-1]
        return ends, fallback_punct


def _resolve_utterance_spans(
    sentences: list[str],
    word_dicts: list[dict[str, Any]],
    spans: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Resolve ``(w_start, w_end)`` spans from precomputed spans or character alignment."""
    n_words = len(word_dicts)
    resolved = spans
    if resolved is None and isinstance(word_dicts[0].get("_sentence_spans"), list):
        cand = word_dicts[0].pop("_sentence_spans", None)
        if isinstance(cand, list) and len(cand) == len(sentences):
            resolved = cand

    if resolved is not None and len(resolved) == len(sentences):
        return resolved

    full_clean = ""
    char_to_word: list[int] = []
    for w_idx, wd in enumerate(word_dicts):
        cw = _clean_word(wd.get("text", wd.get("word", "")))
        full_clean += cw
        char_to_word.extend([w_idx] * len(cw))

    aligned: list[tuple[int, int]] = []
    cursor = 0
    for u_idx, sent in enumerate(sentences):
        sc = _clean_word(sent)
        if sc and full_clean:
            pos = full_clean.find(sc, cursor)
            if pos != -1:
                w_start = char_to_word[pos]
                w_end = char_to_word[pos + len(sc) - 1] + 1
                cursor = pos + len(sc)
            else:
                w_start = char_to_word[min(cursor, len(char_to_word) - 1)]
                end_char = min(len(char_to_word) - 1, cursor + max(1, len(sc)) - 1)
                w_end = max(w_start + 1, char_to_word[end_char] + 1)
                cursor = min(len(char_to_word), end_char + 1)
        else:
            w_start = min(u_idx, n_words - 1)
            w_end = min(n_words, w_start + 1)
        aligned.append((w_start, w_end))
    return aligned
