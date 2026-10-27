import os
import re
from pydub import AudioSegment

from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.gpt_sovits_tts import gpt_sovits_tts_for_videolingo
from core.tts_backend.sf_fishtts import siliconflow_fish_tts_for_videolingo
from core.tts_backend.openai_tts import openai_tts
from core.tts_backend.fish_audio_tts import fish_audio_tts
from core.tts_backend.edge_tts import edge_tts
from core.tts_backend.sf_cosyvoice2 import cosyvoice_tts_for_videolingo
from core.tts_backend.custom_tts import custom_tts
from core.prompts import get_correct_text_prompt
from core.tts_backend._302_f5tts import f5_tts_for_videolingo
from core.utils import ask_gpt, load_key, rprint


def clean_text_for_tts(text):
    """Remove problematic characters for TTS"""
    chars_to_remove = ["&", "®", "™", "©"]
    for char in chars_to_remove:
        text = text.replace(char, "")
    return text.strip()


def tts_main(text, save_as, number, task_df):
    text = clean_text_for_tts(text)
    # Check if text is empty or punctuation-only
    cleaned_text = re.sub(
        r"[^\w\s\u0300-\u036f\u0900-\u097f\u0e00-\u0e7f]", "", text
    ).strip()
    if not cleaned_text:
        silence = AudioSegment.silent(duration=100)  # 100ms = 0.1s
        silence.export(save_as, format="wav")
        rprint(f"Created silent audio for empty text: {save_as}")
        return

    # Skip if file exists
    if os.path.exists(save_as):
        return

    print(f"Generating <{text}...>")
    TTS_METHOD = load_key("tts_method")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt >= max_retries - 1:
                print("Asking GPT to correct text...")

                def _valid_tts_text(resp):
                    if (
                        not isinstance(resp, dict)
                        or not isinstance(resp.get("text"), str)
                        or not resp["text"].strip()
                    ):
                        return {
                            "status": "error",
                            "message": "Response must contain non-empty 'text' string",
                        }
                    return {"status": "success", "message": "ok"}

                correct_text = ask_gpt(
                    get_correct_text_prompt(text),
                    resp_type="json",
                    valid_def=_valid_tts_text,
                    log_title="tts_correct_text",
                )
                if (
                    isinstance(correct_text, dict)
                    and isinstance(correct_text.get("text"), str)
                    and correct_text["text"].strip()
                ):
                    text = correct_text["text"].strip()
            if TTS_METHOD == "openai_tts":
                openai_tts(text, save_as)
            elif TTS_METHOD == "gpt_sovits":
                gpt_sovits_tts_for_videolingo(text, save_as, number, task_df)
            elif TTS_METHOD == "fish_audio":
                fish_audio_tts(text, save_as, number, task_df)
            elif TTS_METHOD == "sf_fish_tts":
                siliconflow_fish_tts_for_videolingo(text, save_as, number, task_df)
            elif TTS_METHOD == "edge_tts":
                edge_tts(text, save_as)
            elif TTS_METHOD == "custom_tts":
                custom_tts(text, save_as)
            elif TTS_METHOD == "sf_cosyvoice2":
                cosyvoice_tts_for_videolingo(text, save_as, number, task_df)
            elif TTS_METHOD == "f5tts":
                f5_tts_for_videolingo(text, save_as, number, task_df)

            # Check generated audio duration
            duration = get_audio_duration(save_as)
            if duration > 0:
                break
            else:
                if os.path.exists(save_as):
                    os.remove(save_as)
                if attempt == max_retries - 1:
                    print(f"Warning: Generated audio duration is 0 for text: {text}")
                    # Create silent audio file
                    silence = AudioSegment.silent(duration=100)  # 100ms silence
                    silence.export(save_as, format="wav")
                    return
                print(f"Attempt {attempt + 1} failed, retrying...")
        except NotImplementedError:
            raise
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(
                    f"Failed to generate audio after {max_retries} attempts: {str(e)}"
                )
            print(f"Attempt {attempt + 1} failed, retrying...")
