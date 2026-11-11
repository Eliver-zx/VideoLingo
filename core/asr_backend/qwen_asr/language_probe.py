"""Language probes, repetition/degeneration checks, script validation, and window transcription."""

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
import math
import re
import unicodedata

import numpy as np

from core.asr_backend.qwen_asr import config as cfg

ProbeFn = Callable[..., tuple[str | None, str]]
_REPEATED_UNIT = re.compile(r"(.{1,40}?)\1{3,}", re.S)
_FAST_BLOCKS = (
    (0x3040, 0x30FF, "Kana"),
    (0x00C0, 0x024F, "Latin"),
    (0x0400, 0x052F, "Cyrillic"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0370, 0x03E1, "Greek"),
    (0x03F0, 0x03FF, "Greek"),
)


def probe_ranges(
    length: int,
    sr: int = cfg.SAMPLE_RATE,
    probe_seconds: float = cfg.PROBE_SECONDS,
    count: int = cfg.PROBES_PER_WINDOW,
) -> list[tuple[int, int]]:
    """Up to `count` probe clips of probe_seconds spread evenly over a window of `length` samples."""
    probe = int(probe_seconds * sr)
    if length <= probe * 3 // 2:
        return [(0, length)]
    count = max(1, min(count, length // probe))
    ranges: list[tuple[int, int]] = []
    for i in range(count):
        center = int((i + 0.5) * length / count)
        a = min(max(0, center - probe // 2), length - probe)
        ranges.append((a, a + probe))
    return ranges


def _normalized(text: str | None) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def is_repetitive(text: str | None) -> bool:
    """A unit of 1-40 letters/digits repeated 4+ times in a row covers half the transcript."""
    norm = _normalized(text)
    if len(norm) < cfg.REPEAT_MIN_CHARS:
        return False
    longest = max(
        (m.end() - m.start() for m in _REPEATED_UNIT.finditer(norm)), default=0
    )
    return longest >= cfg.REPEAT_COVERAGE * len(norm)


def degeneration(text: str | None, heard: Iterable[str] = ()) -> str | None:
    """Why a window transcript looks degenerate, or None."""
    if is_repetitive(text):
        return "the same phrase repeats over most of the transcript"
    reference = sum(len(_normalized(t)) for t in heard)
    length = len(_normalized(text))
    if reference >= cfg.PROBE_MIN_CHARS and length < cfg.PROBE_COVERAGE * reference:
        return f"only {length} letters/digits, while shorter probe clips of it had {reference}"
    return None


def _vote(probes: Sequence[tuple[str, str]]) -> str | None:
    """Most frequent probe language; ties go to the language with more recognized text."""
    if not probes:
        return None
    counts: Counter[str] = Counter()
    amount: Counter[str] = Counter()
    for name, text in probes:
        counts[name] += 1
        amount[name] += len(_normalized(text))
    return max(counts, key=lambda name: (counts[name], amount[name]))


def token_budget(seconds: float) -> int:
    return min(
        cfg.MAX_NEW_TOKENS,
        cfg.TOKEN_MARGIN + math.ceil(seconds * cfg.TOKENS_PER_SECOND),
    )


def probe_token_budget(seconds: float) -> int:
    return min(
        token_budget(seconds),
        cfg.PROBE_TOKEN_MARGIN + math.ceil(seconds * cfg.PROBE_TOKENS_PER_SECOND),
    )


def probe_window(
    run: ProbeFn,
    raw: np.ndarray,
    a: int,
    b: int,
    seconds: float = cfg.PROBE_SECONDS,
    count: int = cfg.PROBES_PER_WINDOW,
) -> tuple[list[tuple[str, str]], int]:
    """Auto-language probe clips of raw[a:b] -> ([(language, text)] usable for voting, rejected count)."""
    probes: list[tuple[str, str]] = []
    rejected = 0
    for pa, pb in probe_ranges(b - a, probe_seconds=seconds, count=count):
        lang, text = run(
            raw[a + pa : a + pb], None, probe_token_budget((pb - pa) / cfg.SAMPLE_RATE)
        )
        name = cfg.primary_language(lang)
        if not text:
            continue
        if name and not is_repetitive(text):
            probes.append((name, text))
        else:
            rejected += 1
    return probes, rejected


def plan_languages(
    run: ProbeFn,
    raw: np.ndarray,
    windows: Sequence[tuple[int, int]],
    progress_cb: Callable[..., None] | None = None,
) -> list[tuple[str | None, list[tuple[str, str]]]]:
    """auto: [(language or None, [(language, probe text)])] per window from probe votes."""
    f = cfg._facade()
    per_window: list[list[tuple[str, str]]] = []
    rejected = 0
    total_w = max(1, len(windows))
    for w_idx, (a, b) in enumerate(windows, 1):
        if progress_cb is not None:
            progress_cb(
                0.05 + 0.10 * ((w_idx - 1) / total_w),
                f"语言探测 {w_idx}/{total_w}",
                f"🌐 语言探测 [{w_idx}/{total_w}] ({a / cfg.SAMPLE_RATE:.1f}s–{b / cfg.SAMPLE_RATE:.1f}s)...",
            )
        probes, bad = probe_window(run, raw, a, b)
        per_window.append(probes)
        rejected += bad
    overall = _vote([probe for probes in per_window for probe in probes])
    if overall is None and rejected:
        f.log_event(
            f"首次语言探测 ({rejected} 段) 未收敛，正在改用 {f.REPROBE_SECONDS}s 短切片重新探测",
            tag="warn",
            label="ASR",
            level="warn",
        )
        per_window = [
            probe_window(run, raw, a, b, f.REPROBE_SECONDS, cfg.REPROBES_PER_WINDOW)[0]
            for a, b in windows
        ]
        overall = _vote([probe for probes in per_window for probe in probes])
        if overall is None:
            raise ValueError(
                "Qwen3-ASR could not determine the language: every probe clip looped or "
                "returned an unsupported language. Set the recognition language explicitly."
            )
    plan: list[tuple[str | None, list[tuple[str, str]]]] = []
    for (a, b), probes in zip(windows, per_window):
        language = _vote(probes) or overall
        votes = (
            ", ".join(f"{k}×{n}" for k, n in Counter(k for k, _ in probes).items())
            or "none"
        )
        msg = f"🌐 {a / cfg.SAMPLE_RATE:.1f}-{b / cfg.SAMPLE_RATE:.1f}s language: {language or 'auto'} (probes: {votes})"
        if progress_cb is not None:
            progress_cb(0.15, f"语言确定: {language or 'auto'}", msg)
        else:
            f.log_event(msg, tag="asr", label="ASR")
        plan.append((language, probes))
    return plan


def missed_speech(text: str | None, spoken: Iterable[str]) -> str | None:
    """Transcript far shorter than what auto-language probes heard (any language), or None."""
    ref, length = sum(len(_normalized(t)) for t in spoken), len(_normalized(text))
    if ref >= cfg.PROBE_MIN_CHARS and length < cfg.CROSS_LANGUAGE_COVERAGE * ref:
        return f"only {length} letters/digits, while auto-detected probe clips of it had {ref}"
    return None


def script_counts(text: str | None) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not text:
        return counts
    if text.isascii():
        latin = sum(1 for ch in text if ch.isalpha())
        if latin:
            counts["Latin"] = latin
        return counts

    latin = han = hangul = 0
    for char in text:
        cp = ord(char)
        if cp < 0x80:
            if 0x41 <= cp <= 0x5A or 0x61 <= cp <= 0x7A:
                latin += 1
        elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            han += 1
        elif 0xAC00 <= cp <= 0xD7A3:
            hangul += 1
        elif char.isalpha():
            for lo, hi, block in _FAST_BLOCKS:
                if lo <= cp <= hi:
                    counts[block] += 1
                    break
            else:
                try:
                    name = unicodedata.name(char)
                except ValueError:
                    continue
                script = next(
                    (s for p, s in cfg._SCRIPT_PREFIXES if name.startswith(p)), "Other"
                )
                counts[script] += 1

    for key, n in (("Latin", latin), ("Han", han), ("Hangul", hangul)):
        if n:
            counts[key] += n
    return counts


def script_mismatch(
    text: str | None,
    iso: str,
    min_weight: float = cfg.SCRIPT_MIN_WEIGHT,
    min_share: float = cfg.SCRIPT_MIN_SHARE,
    ja_min_kana: float = cfg.JA_MIN_KANA_SHARE,
    zh_max_kana: float = cfg.ZH_MAX_KANA_SHARE,
) -> str | None:
    """Why a transcript's writing system does not fit the selected language, or None (text only)."""
    counts = script_counts(text)
    weighted = {
        script: n * cfg._SCRIPT_WEIGHT.get(script, 1) for script, n in counts.items()
    }
    total = sum(weighted.values())
    if total < min_weight:
        return None
    expected = cfg.EXPECTED_SCRIPTS.get(iso, {"Latin"})
    share = sum(weighted.get(script, 0) for script in expected) / total
    dominant = max(weighted, key=lambda k: weighted[k])
    if share < min_share:
        return f"{1 - share:.0%} of the letters are not in the expected script ({dominant} text)"
    cjk = counts["Han"] + counts["Kana"]
    if cjk * 3 >= min_weight:
        kana = counts["Kana"] / cjk
        if iso == "ja" and kana < ja_min_kana:
            return f"only {kana:.0%} kana among the CJK characters (Chinese text)"
        if iso in ("zh", "yue") and kana > zh_max_kana:
            return f"{kana:.0%} kana among the CJK characters (Japanese text)"
    return None


def probe_mismatch(
    probes: Sequence[tuple[str, str]], language: str | None
) -> str | None:
    """The language auto-detected probes heard instead of the selected one, or None."""
    reference = sum(len(_normalized(text)) for _, text in probes)
    if reference < cfg.PROBE_MIN_CHARS or any(
        cfg.iso_language(name) == cfg.iso_language(language) for name, _ in probes
    ):
        return None
    return _vote(probes)


def language_mismatch_error(span: str, language: str | None, reason: str) -> ValueError:
    return ValueError(
        f"Qwen3-ASR output for {span}: the selected recognition language ({language}) may not match the audio "
        f"({reason}). Use auto, or switch the Qwen3-ASR model size."
    )


def transcribe_window(
    run: ProbeFn,
    raw: np.ndarray,
    a: int,
    b: int,
    language: str | None,
    probes: Sequence[tuple[str, str]] | None = None,
    offset: float = 0.0,
    forced: bool = False,
) -> list[tuple[int, int, str | None, str]]:
    """[(start, end, language, text)] for one window; retries shorter windows if degenerate."""
    f = cfg._facade()
    heard = [text for name, text in probes or () if name == language]
    seconds = (b - a) / f.SAMPLE_RATE
    listened = list(probes) if probes is not None else None

    def listen() -> list[tuple[str, str]]:
        nonlocal listened
        if listened is None:
            listened = probe_window(run, raw, a, b)[0]
        return listened

    def check(text: str) -> str | None:
        reason = degeneration(text, heard)
        if reason or len(_normalized(text)) >= cfg.SPARSE_CHARS_PER_SECOND * seconds:
            return reason
        return missed_speech(text, [t for _, t in listen()])

    lang, text = run(raw[a:b], language)
    reason = check(text)
    if not reason:
        return [(a, b, lang, text)]
    span = f"{offset + a / f.SAMPLE_RATE:.1f}-{offset + b / f.SAMPLE_RATE:.1f}s"
    if forced:
        heard_instead = probe_mismatch(listen(), language)
        if heard_instead:
            raise language_mismatch_error(
                span, language, f"{reason}; the probe clips sound like {heard_instead}"
            )
    f.log_event(
        f"Qwen3-ASR 窗口 {span} 输出退化 ({reason})，拆分为 {f.RETRY_WINDOW_SECONDS}s 子窗口重试",
        tag="warn",
        label="ASR",
        level="warn",
    )
    pieces: list[tuple[int, int, str | None, str]] = []
    parts = f.split_windows(
        raw[a:b],
        window_seconds=f.RETRY_WINDOW_SECONDS,
        search_seconds=f.RETRY_SEARCH_SECONDS,
    )
    for sa, sb in parts:
        if sb - sa >= int(f.MIN_WINDOW_SECONDS * f.SAMPLE_RATE):
            sub_lang, sub_text = run(raw[a + sa : a + sb], language)
            pieces.append((a + sa, a + sb, sub_lang, sub_text))
    reason = next(
        (degeneration(t) for *_, t in pieces if degeneration(t)), None
    ) or check(" ".join(t for *_, t in pieces))
    if reason:
        advice = (
            f"The selected recognition language ({language}) may not match the audio: try auto, or the other Qwen3-ASR model size."
            if forced
            else "Set the recognition language explicitly instead of auto, try the other Qwen3-ASR model size, or check the audio."
        )
        raise ValueError(
            f"Qwen3-ASR output for {span} is still degenerate after retrying ({reason}). {advice}"
        )
    return pieces
