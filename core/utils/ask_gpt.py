import os
import re
import json
import time
import ipaddress
from threading import Lock
from urllib.parse import urlparse
import json_repair
from openai import OpenAI
from core.utils.config_utils import load_key, load_key_or, is_placeholder_key
from core.utils.models import resolve_task_path
from rich import print as rprint
from core.utils.decorator import except_handler

# ------------
# cache gpt response
# ------------

LOCK = Lock()
GPT_LOG_FOLDER = "output/gpt_log"
_CACHE_MISS = object()


def _resolve_log_folder() -> str:
    if GPT_LOG_FOLDER == "output/gpt_log":
        return resolve_task_path(GPT_LOG_FOLDER)
    return GPT_LOG_FOLDER


def normalize_reasoning_effort(effort) -> str:
    val = str(effort or "").strip().lower()
    if val in ("low", "medium", "high"):
        return val
    return "none"


def build_reasoning_params(model: str, effort=None) -> dict:
    """Build provider-aware thinking/reasoning parameters for chat.completions.create."""
    norm_effort = normalize_reasoning_effort(effort)
    m = str(model or "").strip().lower()
    if "deepseek" in m:
        if "reasoner" in m or "-r1" in m or m.endswith("r1"):
            return {"reasoning_effort": norm_effort} if norm_effort != "none" else {}
        thinking_type = "disabled" if norm_effort == "none" else "enabled"
        return {"extra_body": {"thinking": {"type": thinking_type}}}
    if norm_effort != "none":
        return {"reasoning_effort": norm_effort}
    if any(k in m for k in ("claude", "haiku", "sonnet", "opus", "2.5-pro")):
        return {}
    if any(k in m for k in ("gemini", "gpt-5", "gpt-6", "luna", "o1", "o3", "o4")):
        return {"reasoning_effort": "none"}
    return {}


def _save_cache(
    model,
    prompt,
    resp_content,
    resp_type,
    resp,
    message=None,
    log_title="default",
    reasoning_effort="none",
):
    if log_title in ("None", None):
        return
    norm_effort = normalize_reasoning_effort(reasoning_effort)
    if GPT_LOG_FOLDER == "output/gpt_log" and log_title != "error":
        from core.workspace import get_active_workspace

        ws = get_active_workspace()
        ws.save_llm_cache(
            log_title=str(log_title),
            prompt=str(prompt),
            model=str(model or ""),
            reasoning_effort=norm_effort,
            resp_type=str(resp_type or ""),
            resp=resp,
        )
        return
    with LOCK:
        logs = []
        folder = _resolve_log_folder()
        file = os.path.join(folder, f"{log_title}.json")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if os.path.exists(file):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        logs = loaded
            except (OSError, ValueError):
                logs = []
        logs.append(
            {
                "model": model,
                "reasoning_effort": norm_effort,
                "prompt": prompt,
                "resp_content": resp_content,
                "resp_type": resp_type,
                "resp": resp,
                "message": message,
            }
        )
        tmp_file = f"{file}.tmp.{os.getpid()}"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)
        for attempt in range(5):
            try:
                os.replace(tmp_file, file)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))


def _load_cache(prompt, resp_type, log_title, model=None, reasoning_effort=None):
    if log_title in ("None", None):
        return False
    target_effort = (
        normalize_reasoning_effort(reasoning_effort)
        if reasoning_effort is not None
        else None
    )
    if GPT_LOG_FOLDER == "output/gpt_log":
        from core.workspace import get_active_workspace

        ws = get_active_workspace()
        hit, val = ws.load_llm_cache(
            log_title=str(log_title),
            prompt=str(prompt),
            resp_type=resp_type,
            model=model,
            reasoning_effort=target_effort,
        )
        if hit:
            return val
    with LOCK:
        folder = _resolve_log_folder()
        file = os.path.join(folder, f"{log_title}.json")
        if os.path.exists(file):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    items = json.load(f)
            except (OSError, ValueError):
                return False
            if isinstance(items, list):
                for item in reversed(items):
                    if not isinstance(item, dict):
                        continue
                    if (
                        item.get("prompt") == prompt
                        and item.get("resp_type") == resp_type
                    ):
                        cached_model = item.get("model")
                        if (
                            model is not None
                            and cached_model is not None
                            and cached_model != model
                        ):
                            continue
                        if target_effort is not None:
                            cached_effort = normalize_reasoning_effort(
                                item.get("reasoning_effort")
                            )
                            if cached_effort != target_effort:
                                continue
                        return item.get("resp")
        return False


# ------------
# api key
# ------------

LOCAL_API_KEY = (
    "not-needed"  # the OpenAI client refuses an empty key; local servers ignore it
)


def is_local_endpoint(base_url):
    """Ollama, LM Studio, vLLM... on this machine or the private network need no API key."""
    host = (
        urlparse(base_url if "//" in str(base_url) else f"//{base_url}").hostname or ""
    ).lower()
    if host in ("localhost", "host.docker.internal") or host.endswith(
        (".local", ".localhost")
    ):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_unspecified


def get_api_key():
    key = load_key("api.key")
    if key and not is_placeholder_key(key):
        return key
    if is_local_endpoint(load_key("api.base_url")):
        return LOCAL_API_KEY
    raise ValueError("API key is not set")


# ------------
# base url
# ------------


def normalize_base_url(base_url):
    """Accept the forms people paste: with or without /v1, or the full chat completions URL."""
    base_url = str(base_url or "").strip()
    if "ark" in base_url:
        return "https://ark.cn-beijing.volces.com/api/v3"  # huoshan base url
    base_url = re.sub(r"/chat/completions/?$", "", base_url.rstrip("/"))
    if "v1" not in base_url and not re.search(
        r"/v\d+$", base_url
    ):  # e.g. .../api/paas/v4
        base_url = base_url.strip("/") + "/v1"
    return base_url


# ------------
# parse json response
# ------------


def parse_json_response(resp_content):
    """Read the JSON object of a reply that may carry reasoning or prose around it."""
    text = re.sub(
        r"<think>.*?</think>", "", resp_content or "", flags=re.DOTALL | re.IGNORECASE
    )
    text = re.split(r"</think>", text, flags=re.IGNORECASE)[
        -1
    ]  # opening tag cut off by the server
    text = re.sub(
        r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE
    )  # reasoning never closed
    blocks = re.findall(
        r"```(?:json)?[ \t]*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    )
    candidates = [block for block in reversed(blocks) if "{" in block] + [text]
    resp = None
    for candidate in candidates:
        resp = json_repair.loads(candidate)
        if isinstance(
            resp, list
        ):  # prose with several braces: the answer is the last object
            resp = next(
                (item for item in reversed(resp) if isinstance(item, dict)), resp
            )
        if isinstance(resp, dict) and resp:
            return resp
    raise ValueError(
        f"❎ API response is not a JSON object: {str(resp_content)[:200]!r}"
    )


# ------------
# ask gpt once
# ------------


@except_handler("GPT request failed", retry=5)
def ask_gpt(prompt, resp_type=None, valid_def=None, log_title="default"):
    api_key = get_api_key()
    model = load_key("api.model")
    reasoning_effort = normalize_reasoning_effort(
        load_key_or("api.reasoning_effort", "none")
    )
    cached = _load_cache(
        prompt,
        resp_type,
        log_title,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    if cached is not False and cached is not None:
        if valid_def is None or valid_def(cached).get("status") == "success":
            rprint("use cache response")
            return cached

    base_url = normalize_base_url(load_key("api.base_url"))
    client = OpenAI(api_key=api_key, base_url=base_url)
    response_format = (
        {"type": "json_object"}
        if resp_type == "json" and load_key("api.llm_support_json")
        else None
    )

    messages = [{"role": "user", "content": prompt}]

    params = dict(
        model=model,
        messages=messages,
        timeout=300,
    )
    if response_format is not None:
        params["response_format"] = response_format
    params.update(build_reasoning_params(model, reasoning_effort))
    resp_raw = client.chat.completions.create(**params)

    resp_content = resp_raw.choices[0].message.content
    if resp_type == "json":
        try:
            resp = parse_json_response(resp_content)
        except ValueError as e:
            _save_cache(
                model,
                prompt,
                resp_content,
                resp_type,
                None,
                log_title="error",
                message=str(e),
                reasoning_effort=reasoning_effort,
            )
            raise
    else:
        resp = resp_content

    if valid_def:
        valid_resp = valid_def(resp)
        if valid_resp["status"] != "success":
            _save_cache(
                model,
                prompt,
                resp_content,
                resp_type,
                resp,
                log_title="error",
                message=valid_resp["message"],
                reasoning_effort=reasoning_effort,
            )
            raise ValueError(f"❎ API response error: {valid_resp['message']}")

    _save_cache(
        model,
        prompt,
        resp_content,
        resp_type,
        resp,
        log_title=log_title,
        reasoning_effort=reasoning_effort,
    )
    return resp
