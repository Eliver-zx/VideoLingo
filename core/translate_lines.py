from core.prompts import (
    generate_shared_prompt,
    get_prompt_faithfulness,
    get_prompt_expressiveness,
)
from core.task_runner import StopTask
from core.utils import ask_gpt, check_cancel, load_key, log_event


_BOOK_MARK_TABLE = str.maketrans({"《": "“", "》": "”", "〈": "‘", "〉": "’"})


def _normalize_book_marks(text: str) -> str:
    return str(text).translate(_BOOK_MARK_TABLE)


class TranslateResultTuple(tuple):
    """2-tuple (translate_result, lines) carrying `.direct_lines` without global state."""

    direct_lines: list[str]

    def __new__(cls, translate_result: str, lines: str, direct_lines: list[str]):
        obj = super().__new__(cls, (translate_result, lines))
        obj.direct_lines = direct_lines
        return obj


def valid_translate_result(
    result: object, required_keys: list, required_sub_keys: list
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

    # Check for required sub-keys in all items; keys the model added on its own are ignored
    for key in required_keys:
        if not isinstance(result[key], dict):
            return {"status": "error", "message": f"Item {key} is not an object"}
        if not all(sub_key in result[key] for sub_key in required_sub_keys):
            return {
                "status": "error",
                "message": f"Missing required sub-key(s) in item {key}: {', '.join(set(required_sub_keys) - set(result[key].keys()))}",
            }
        for sub_key in required_sub_keys:
            val = result[key][sub_key]
            if not isinstance(val, str) or not val.strip():
                return {
                    "status": "error",
                    "message": f"Empty or non-string value for '{sub_key}' in item {key}",
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

    # Retry translation if the length of the original text and the translated text are not the same, or if the specified key is missing
    def retry_translation(prompt, length, step_name):
        def valid_faith(response_data):
            return valid_translate_result(
                response_data, [str(i) for i in range(1, length + 1)], ["direct"]
            )

        def valid_express(response_data):
            return valid_translate_result(
                response_data, [str(i) for i in range(1, length + 1)], ["free"]
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
    line_list = lines.split("\n")
    prompt1 = get_prompt_faithfulness(lines, shared_prompt)
    faith_result = retry_translation(prompt1, len(line_list), "faithfulness")

    for idx, i in enumerate(faith_result):
        faith_result[i].setdefault(
            "origin", line_list[idx] if idx < len(line_list) else ""
        )
        faith_result[i]["direct"] = _normalize_book_marks(
            faith_result[i]["direct"].replace("\n", " ")
        )
    direct_lines = [faith_result[i]["direct"].strip() for i in faith_result]

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
