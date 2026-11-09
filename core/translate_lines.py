import re

from core._4_1_summarize import _is_standard_english
from core.prompts import (
    generate_shared_prompt,
    get_prompt_faithfulness,
    get_prompt_expressiveness,
)
from core.task_runner import StopTask
from core.utils import ask_gpt, check_cancel, load_key, log_event


_QUOTE_BOOK_MARK_TABLE = str.maketrans({c: None for c in "《》〈〉“”‘’「」『』"})
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_LOWER_EN_WORD_RE = re.compile(r"(?<![A-Za-z0-9_.-])[a-z]{3,}(?![A-Za-z0-9_.-])")
_ANY_EN_WORD_RE = re.compile(r"[A-Za-z]+")
_ALLOWED_LOANWORDS: frozenset[str] = frozenset(
    (
        "dps pve pvp dot aoe ads rng hud ui ux ai ar vr fps rpg mmo npc dlc ost cg op ed mv up "
        "meta solo boss buff nerf debuff"
    ).split()
)


def _normalize_book_marks(text: str) -> str:
    s = str(text).translate(_QUOTE_BOOK_MARK_TABLE)
    if _CJK_CHAR_RE.search(s):
        s = s.replace('"', "")
        s = re.sub(r"(?:——+|—+|--+|[：；;])+", "，", s)
        s = re.sub(r"(?<!\d):(?!\d)", "，", s)
        s = re.sub(r"，+", "，", s)
        s = re.sub(r"，([。？！、])", r"\1", s)
        s = re.sub(r"([，。？！、])，", r"\1", s)
        s = re.sub(r"^，+|，+$", "", s)
    return s


class TranslateResultTuple(tuple):
    """2-tuple (translate_result, lines) carrying `.direct_lines` without global state."""

    direct_lines: list[str]

    def __new__(cls, translate_result: str, lines: str, direct_lines: list[str]):
        obj = super().__new__(cls, (translate_result, lines))
        obj.direct_lines = direct_lines
        return obj


def valid_translate_result(
    result: object,
    required_keys: list,
    required_sub_keys: list,
    origin_lines: list[str] | None = None,
    allowed_en_text: str = "",
):
    if not isinstance(result, dict):
        return {
            "status": "error",
            "message": f"Expected a JSON object, got {type(result).__name__}",
        }
    # Check for the required key
    if not all(key in result for key in required_keys):
        return {
            "status": "error",
            "message": f"Missing required key(s): {', '.join(set(required_keys) - set(result.keys()))}",
        }

    allowed_words = (
        {w.lower() for w in _ANY_EN_WORD_RE.findall(allowed_en_text)}
        if allowed_en_text
        else set()
    ) | _ALLOWED_LOANWORDS

    # Check for required sub-keys in all items; keys the model added on its own are ignored
    for key in required_keys:
        if not isinstance(result[key], dict):
            return {"status": "error", "message": f"Item {key} is not an object"}
        if not all(sub_key in result[key] for sub_key in required_sub_keys):
            return {
                "status": "error",
                "message": f"Missing required sub-key(s) in item {key}: {', '.join(set(required_sub_keys) - set(result[key].keys()))}",
            }
        idx = int(key) - 1 if str(key).isdigit() else -1
        orig_words = (
            {w.lower() for w in _ANY_EN_WORD_RE.findall(origin_lines[idx])}
            if (origin_lines is not None and 0 <= idx < len(origin_lines))
            else set()
        )
        for sub_key in required_sub_keys:
            val = result[key][sub_key]
            if not isinstance(val, str) or not val.strip():
                return {
                    "status": "error",
                    "message": f"Empty or non-string value for '{sub_key}' in item {key}",
                }
            if orig_words and _CJK_CHAR_RE.search(val):
                for w in _LOWER_EN_WORD_RE.findall(val):
                    if (
                        w in orig_words
                        and w not in allowed_words
                        and _is_standard_english(w)
                    ):
                        return {
                            "status": "error",
                            "message": f"Item {key} contains untranslated source word '{w}' in '{sub_key}'",
                        }

    return {"status": "success", "message": "Translation completed"}


def translate_lines(
    lines,
    previous_content_prompt,
    after_cotent_prompt,
    things_to_note_prompt,
    summary_prompt,
    index=0,
):
    check_cancel()
    shared_prompt = generate_shared_prompt(
        previous_content_prompt,
        after_cotent_prompt,
        summary_prompt,
        things_to_note_prompt,
    )
    line_list = lines.split("\n")
    allowed_en_text = f"{things_to_note_prompt or ''}\n{summary_prompt or ''}"

    # Retry translation if the length of the original text and the translated text are not the same, or if the specified key is missing
    def retry_translation(prompt, length, step_name):
        def valid_faith(response_data):
            return valid_translate_result(
                response_data,
                [str(i) for i in range(1, length + 1)],
                ["trans"],
                origin_lines=line_list,
                allowed_en_text=allowed_en_text,
            )

        def valid_express(response_data):
            return valid_translate_result(
                response_data,
                [str(i) for i in range(1, length + 1)],
                ["free"],
                origin_lines=line_list,
                allowed_en_text=allowed_en_text,
            )

        for retry in range(3):
            if step_name == "faithfulness":
                result = ask_gpt(
                    prompt + retry * " ",
                    resp_type="json",
                    valid_def=valid_faith,
                    log_title=f"translate_{step_name}",
                )
            else:
                result = ask_gpt(
                    prompt + retry * " ",
                    resp_type="json",
                    valid_def=valid_express,
                    log_title=f"translate_{step_name}",
                )
            result = {str(i): result[str(i)] for i in range(1, length + 1)}
            if len(lines.split("\n")) == len(result):
                return result
            if retry != 2:
                log_event(
                    f"翻译批次 #{index + 1:02d} ({step_name}) 行数校验未通过，正在重试 ({retry + 1}/2)",
                    tag="warn",
                    label="TRANS",
                    level="warn",
                )
        raise ValueError(
            f"{step_name.capitalize()} translation of block {index} failed after 3 retries. Please check `output/gpt_log/error.json` for more details."
        )

    ## Step 1: Faithful to the Original Text
    prompt1 = get_prompt_faithfulness(lines, shared_prompt)
    faith_result = retry_translation(prompt1, len(line_list), "faithfulness")

    for idx, i in enumerate(faith_result):
        faith_result[i].setdefault(
            "origin", line_list[idx] if idx < len(line_list) else ""
        )
        normalized = _normalize_book_marks(faith_result[i]["trans"].replace("\n", " "))
        faith_result[i]["trans"] = normalized
        faith_result[i]["direct"] = normalized
    direct_lines = [faith_result[i]["trans"].strip() for i in faith_result]

    # If reflect_translate is False or not set, use faithful translation directly
    reflect_translate = load_key("reflect_translate")
    if not reflect_translate:
        translate_result = "\n".join(direct_lines)
        return TranslateResultTuple(translate_result, lines, direct_lines)

    ## Step 2: Express Smoothly
    prompt2 = get_prompt_expressiveness(faith_result, lines, shared_prompt)
    try:
        express_result = retry_translation(
            prompt2, len(lines.split("\n")), "expressiveness"
        )
    except StopTask:
        raise
    except Exception as e:
        # The polish step is optional: keep the faithful translation instead of stopping the video
        log_event(
            f"翻译批次 #{index + 1:02d} 意译润色失败 ({e})，回退至直译结果",
            tag="warn",
            label="TRANS",
            level="warn",
        )
        express_result = {
            key: {"free": faith_result[key]["direct"]} for key in faith_result
        }

    for key in express_result:
        free_val = _normalize_book_marks(
            express_result[key]["free"].replace("\n", " ").strip()
        )
        if free_val.upper() in ("OK", "SAME", "UNCHANGED", "KEEP"):
            express_result[key]["free"] = faith_result[key]["direct"].strip()
        else:
            express_result[key]["free"] = free_val

    translate_result = "\n".join([express_result[i]["free"] for i in express_result])

    if len(lines.split("\n")) != len(translate_result.split("\n")):
        log_event(
            f"翻译批次 #{index + 1:02d} 行数不匹配，请检查 translate_expressiveness.json",
            tag="warn",
            label="TRANS",
            level="error",
        )
        raise ValueError(f"Origin ···{lines}···,\nbut got ···{translate_result}···")

    return TranslateResultTuple(translate_result, lines, direct_lines)


if __name__ == "__main__":
    # test e.g.
    lines = """All of you know Andrew Ng as a famous computer science professor at Stanford.
He was really early on in the development of neural networks with GPUs.
Of course, a creator of Coursera and popular courses like deeplearning.ai.
Also the founder and creator and early lead of Google Brain."""
    previous_content_prompt = None
    after_cotent_prompt = None
    things_to_note_prompt = None
    summary_prompt = None
    translate_lines(
        lines,
        previous_content_prompt,
        after_cotent_prompt,
        things_to_note_prompt,
        summary_prompt,
    )
