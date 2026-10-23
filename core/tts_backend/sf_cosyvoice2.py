from openai import OpenAI
from pathlib import Path
import base64
import io
from pydub import AudioSegment
from core.utils import except_handler, load_key
from core.utils.models import resolve_task_path

# The API answers 400 to a large reference: send mono 24 kHz and no more than 15 s
MAX_REFERENCE_SECONDS = 15
REFERENCE_SAMPLE_RATE = 24000

def prepare_reference(wav_file_path, prompt_text):
    """Returns the reference as base64 WAV within the limits, and the text that is spoken in it."""
    audio = AudioSegment.from_file(wav_file_path)
    audio = audio.set_channels(1).set_frame_rate(REFERENCE_SAMPLE_RATE).set_sample_width(2)
    prompt_text = str(prompt_text)
    limit = MAX_REFERENCE_SECONDS * 1000
    if len(audio) > limit:
        # Keep the share of the text that goes with the audio that is kept
        kept = prompt_text[:int(len(prompt_text) * limit / len(audio))]
        if ' ' in kept and len(kept) < len(prompt_text) and not prompt_text[len(kept)].isspace():
            kept = kept.rsplit(' ', 1)[0]
        prompt_text = kept.strip() or prompt_text
        audio = audio[:limit]
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return base64.b64encode(buffer.getvalue()).decode('utf-8'), prompt_text

@except_handler("Failed to generate audio using SiliconFlow TTS")
def cosyvoice_tts_for_videolingo(text, save_as, number, task_df):
    prompt_text = task_df.loc[task_df['number'] == number, 'origin'].values[0]
    API_KEY = load_key("sf_cosyvoice2.api_key")
    # 设置参考音频路径
    ref_audio_path = Path(resolve_task_path(f"output/audio/refers/{number}.wav")).resolve()
    
    # 如果参考音频不存在，尝试提取或回退到第一个可用参考音频并同步对应的参考文本
    if not ref_audio_path.exists():
        first_num = 1 if (task_df is not None and not task_df.empty and 1 in task_df['number'].values) else (
            int(task_df.iloc[0]['number']) if task_df is not None and not task_df.empty else 1
        )
        fallback_path = Path(resolve_task_path(f"output/audio/refers/{first_num}.wav")).resolve()
        if not fallback_path.exists():
            try:
                from core._9_refer_audio import extract_refer_audio_main
                print(f"参考音频文件不存在，尝试提取: {ref_audio_path}")
                extract_refer_audio_main()
            except Exception as e:
                print(f"提取参考音频失败: {str(e)}")
                raise
        if not ref_audio_path.exists():
            ref_audio_path = fallback_path
            if task_df is not None and not task_df.empty and first_num in task_df['number'].values:
                prompt_text = task_df.loc[task_df['number'] == first_num, 'origin'].values[0]

    reference_base64, prompt_text = prepare_reference(ref_audio_path, prompt_text)
    client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")

    save_path = Path(save_as)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        model="FunAudioLLM/CosyVoice2-0.5B",
        voice="",
        input=text,
        response_format="wav",
        extra_body={"references": [{"audio": f"data:audio/wav;base64,{reference_base64}", "text": prompt_text}]}
    ) as response:
        response.stream_to_file(save_path)
    
    print(f"音频已成功保存至: {save_path}")
    return True