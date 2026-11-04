"""Load processing modules only when used (API startup needs no models)."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import (
        _1_ytdlp,
        _2_asr,
        _3_1_split_nlp,
        _3_2_split_meaning,
        _4_1_summarize,
        _4_2_translate,
        _5_split_sub,
        _6_gen_sub,
        _7_sub_into_vid,
        _8_1_audio_task,
        _8_2_dub_chunks,
        _9_refer_audio,
        _10_gen_audio,
        _11_merge_audio,
        _12_dub_to_vid,
    )
    from .utils import ask_gpt, load_key, set_runtime_state, update_key


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    if name.startswith("_"):
        value = import_module(f".{name}", __name__)
    else:
        value = getattr(import_module(".utils", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "ask_gpt",
    "load_key",
    "update_key",
    "set_runtime_state",
    "_1_ytdlp",
    "_2_asr",
    "_3_1_split_nlp",
    "_3_2_split_meaning",
    "_4_1_summarize",
    "_4_2_translate",
    "_5_split_sub",
    "_6_gen_sub",
    "_7_sub_into_vid",
    "_8_1_audio_task",
    "_8_2_dub_chunks",
    "_9_refer_audio",
    "_10_gen_audio",
    "_11_merge_audio",
    "_12_dub_to_vid",
]
