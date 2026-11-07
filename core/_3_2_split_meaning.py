from difflib import SequenceMatcher
import re
from core.prompts import get_split_prompt
from core.task_runner import StopTask
from core.utils import (  # noqa: F401
    ask_gpt,
    check_file_exists,
    get_joiner,
    get_source_language as _default_get_source_language,
    join_words,
    load_key,
    log_event,
    report_progress,
)
from core.utils.config_utils import get_text_joiner

get_source_language = _default_get_source_language


def _resolve_split_joiner(text: str, language: str | None = None) -> str:
    if language is not None:
        return get_text_joiner(text)
    if get_source_language is not _default_get_source_language:
        return get_joiner(get_source_language())
    try:
        whisper_language = load_key("whisper.language")
        lang = (
            load_key("whisper.detected_language")
            if whisper_language == "auto"
            else whisper_language
        )
        return get_joiner(lang)
    except Exception:
        return get_joiner(get_source_language())


def find_split_positions(original, modified, language=None):
    split_positions = []
    parts = modified.split("[br]")
    start = 0
    joiner = _resolve_split_joiner(original, language)

    for i in range(len(parts) - 1):
        max_similarity = 0
        best_split = None

        for j in range(start + 1, len(original)):
            original_left = original[start:j]
            modified_left = join_words(parts[i].split(), joiner)

            left_similarity = SequenceMatcher(
                None, original_left, modified_left
            ).ratio()

            if left_similarity > max_similarity:
                max_similarity = left_similarity
                best_split = j

        if max_similarity < 0.9:
            log_event(
                f"切分点匹配相似度偏低 ({max_similarity:.2f})",
                tag="warn",
                label="NLP",
                level="warn",
            )
        if best_split is not None:
            split_positions.append(best_split)
            start = best_split
        else:
            log_event(
                f"未能为第 {i + 1} 段找到合适切分点",
                tag="warn",
                label="NLP",
                level="warn",
            )

    return split_positions


BR_TAG = re.compile(
    r"\[\s*/?\s*br\s*/?\s*\]|<\s*/?\s*br\s*/?\s*>", re.IGNORECASE
)  # [br] <br> <br/> [BR] ...
BREAK_MARKS = ",;:，、；：。！？!?"


def pick_split(response_data):
    """The split the model chose, with [br] tags; the other one if the choice is unusable."""
    if not isinstance(response_data, dict):
        return None
    direct_split = BR_TAG.sub("[br]", str(response_data.get("split", "")))
    direct_parts = direct_split.split("[br]")
    if len(direct_parts) > 1 and all(part.strip() for part in direct_parts):
        return direct_split
    choice = re.search(r"[12]", str(response_data.get("choice", "")))
    order = [choice.group()] if choice else []
    for number in order + [n for n in ("1", "2") if n not in order]:
        split = BR_TAG.sub("[br]", str(response_data.get(f"split{number}", "")))
        parts = split.split("[br]")
        if len(parts) > 1 and all(part.strip() for part in parts):
            return split
    return None


def mechanical_split(sentence, num_parts, language=None):
    """Split into roughly equal parts without the LLM: at punctuation, else at a space."""
    lines, rest = [], sentence.strip()
    joiner = _resolve_split_joiner(sentence, language)
    for parts_left in range(num_parts, 1, -1):
        target, window = len(rest) // parts_left, max(1, len(rest) // (parts_left * 3))
        nearby = range(max(1, target - window), min(len(rest) - 1, target + window) + 1)
        cuts = [i for i in nearby if rest[i - 1] in BREAK_MARKS] or [
            i for i in nearby if rest[i].isspace()
        ]
        if not cuts and joiner == " ":
            break  # never cut inside a word
        cut = min(cuts, key=lambda i: abs(i - target)) if cuts else target
        lines.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    return "\n".join(line for line in lines + [rest] if line)


def split_sentence(
    sentence, num_parts, word_limit=20, index=-1, retry_attempt=0, language=None
):
    """Split a long sentence using GPT and return the result as a string."""
    split_prompt = (
        get_split_prompt(sentence, num_parts, word_limit)
        if language is None
        else get_split_prompt(sentence, num_parts, word_limit, language=language)
    )

    def valid_split(response_data):
        if pick_split(response_data) is None:
            return {"status": "error", "message": "Split failed, no [br] found"}
        return {"status": "success", "message": "Split completed"}

    try:
        response_data = ask_gpt(
            split_prompt + " " * retry_attempt,
            resp_type="json",
            valid_def=valid_split,
            log_title="split_by_meaning",
        )
    except StopTask:
        raise
    except Exception as e:
        log_event(
            f"LLM 长句切分失败 ({e})，回退为标点机械切分",
            tag="warn",
            label="NLP",
            level="warn",
        )
        return mechanical_split(sentence, num_parts, language=language)
    best_split = pick_split(response_data)
    if best_split is None:
        return mechanical_split(sentence, num_parts, language=language)
    split_points = find_split_positions(sentence, best_split, language=language)
    for i, split_point in enumerate(split_points):
        if i == 0:
            best_split = sentence[:split_point] + "\n" + sentence[split_point:]
        else:
            parts = best_split.split("\n")
            last_part = parts[-1]
            parts[-1] = (
                last_part[: split_point - split_points[i - 1]]
                + "\n"
                + last_part[split_point - split_points[i - 1] :]
            )
            best_split = "\n".join(parts)
    if index != -1:
        log_event(f"已完成第 {index} 句语义切分", tag="nlp", label="NLP")
    return best_split


@check_file_exists("split")
def split_sentences_by_meaning():
    """Keep complete sentences from _3_1_split_nlp intact for translation;
    short subtitle line splitting is performed after translation in _5_split_sub."""
    from core.workspace import get_active_workspace

    ws = get_active_workspace()
    if not ws.has_utterances():
        raise FileNotFoundError(
            f"Missing utterances in workspace database: {ws.db_path}"
        )
    report_progress(
        1.0,
        "句子准备就绪",
        "源文句子整理完成，准备进入术语提取与翻译",
        tag="nlp",
        label="NLP",
    )


if __name__ == "__main__":
    split_sentences_by_meaning()
