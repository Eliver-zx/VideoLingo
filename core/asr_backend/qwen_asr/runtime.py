"""Lazy MLX and PyTorch/Transformers model sessions for Qwen3-ASR and ForcedAligner."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
import gc
import time
from typing import Any

import numpy as np

from core.asr_backend.qwen_asr.config import _facade


def _free(engine: str) -> None:
    gc.collect()
    try:
        if engine == "mlx":
            import mlx.core as mx  # type: ignore[import-not-found]

            mx.clear_cache()
        else:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except ImportError:
        pass  # the engine never loaded; keep the original error visible


def _torch_device() -> tuple[str, Any]:
    import torch  # type: ignore[import-not-found]

    if torch.cuda.is_available():
        dtype = (
            getattr(torch, "bfloat16")
            if torch.cuda.is_bf16_supported()
            else getattr(torch, "float16")
        )
        return "cuda:0", dtype
    return "cpu", getattr(torch, "float32")


def _torch_attn_implementation(device: str) -> str | None:
    """Select the fastest supported attention backend for CUDA GPUs (e.g. 'sdpa' on RTX 2070 SM 7.5)."""
    if not str(device).startswith("cuda"):
        return None
    f = _facade()
    try:
        import torch  # type: ignore[import-not-found]

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if f._installed("flash_attn") and hasattr(torch.cuda, "get_device_capability"):
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 8:
                return "flash_attention_2"
    except (ImportError, RuntimeError, AssertionError, AttributeError):
        pass
    return "sdpa"


def _torch_quantization_config(device: str) -> Any:
    f = _facade()
    if not str(device).startswith("cuda") or not f._installed("bitsandbytes"):
        return None
    from transformers.utils.quantization_config import BitsAndBytesConfig  # type: ignore[import-not-found]

    return BitsAndBytesConfig(load_in_8bit=True)


@contextmanager
def asr_session(
    engine: str, repo_id: str
) -> Iterator[Callable[..., tuple[str | None, str]]]:
    """Load Qwen3-ASR once and yield run(clip, qwen_language or None, max_tokens=None) -> (language, text)."""
    f = _facade()
    source = f._model_source(repo_id)
    model: Any = None
    run: Any = None
    try:
        if engine == "mlx":
            from mlx_audio.stt.utils import load_model  # type: ignore[import-not-found]

            model = load_model(source)

            def _run_mlx(
                clip: np.ndarray,
                language: str | None,
                max_tokens: int | None = None,
            ) -> tuple[str | None, str]:
                f.check_cancel()
                out = model.generate(
                    clip,
                    language=language,
                    max_tokens=max_tokens or f.token_budget(len(clip) / f.SAMPLE_RATE),
                    chunk_duration=f.WINDOW_SECONDS + 1,
                    min_chunk_duration=f.MIN_WINDOW_SECONDS,
                )
                detected = language or ",".join(
                    dict.fromkeys(
                        lang_item for lang_item in (out.language or []) if lang_item
                    )
                )
                return detected, out.text.strip()

            run = _run_mlx
        else:
            from qwen_asr import Qwen3ASRModel  # type: ignore[import-not-found]
            from transformers.utils import logging as hf_logging  # type: ignore[import-not-found]

            hf_logging.set_verbosity_error()
            device, dtype = f._torch_device()
            attn = f._torch_attn_implementation(device)
            quant_cfg = f._torch_quantization_config(device)
            load_kwargs: dict[str, Any] = {
                "dtype": dtype,
                "device_map": device,
                "max_inference_batch_size": 1,
                "max_new_tokens": f.MAX_NEW_TOKENS,
            }
            if attn:
                load_kwargs["attn_implementation"] = attn
            if quant_cfg is not None:
                load_kwargs["quantization_config"] = quant_cfg
            precision_label = f"{dtype} (8bit)" if quant_cfg is not None else str(dtype)
            f.log_event(
                f"Qwen3-ASR 推理设备: {device}, 精度: {precision_label}, 注意力: {attn or 'default'}",
                tag="asr",
                label="ASR",
            )
            model = Qwen3ASRModel.from_pretrained(source, **load_kwargs)

            def _run_tf(
                clip: np.ndarray,
                language: str | None,
                max_tokens: int | None = None,
            ) -> tuple[str | None, str]:
                f.check_cancel()
                # qwen-asr reads this attribute for every generate() call.
                model.max_new_tokens = max_tokens or f.token_budget(
                    len(clip) / f.SAMPLE_RATE
                )
                out = model.transcribe(audio=(clip, f.SAMPLE_RATE), language=language)[
                    0
                ]
                return language or out.language, out.text.strip()

            run = _run_tf
        yield run
    finally:
        # Release the model even when a window fails, so a retry does not start out of memory.
        run = None
        model = None
        f._free(engine)


def _transcribe(
    engine: str,
    repo_id: str,
    clips: Sequence[np.ndarray],
    language: str | None,
) -> list[tuple[str | None, str]]:
    """Return [(qwen_language_name, text)] per clip."""
    f = _facade()
    with f.asr_session(engine, repo_id) as run:
        return [run(clip, language) for clip in clips]


def _align(
    engine: str,
    repo_id: str,
    jobs: Sequence[tuple[np.ndarray, str, str]],
) -> list[list[tuple[str, float, float]]]:
    """jobs: [(clip, text, qwen_language)] -> [[(token, start, end)]] relative to each clip."""
    f = _facade()
    source = f._model_source(repo_id)
    aligner: Any = None
    align: Any = None
    results: list[list[tuple[str, float, float]]] = []
    try:
        if engine == "mlx":
            from mlx_audio.stt.utils import load_model  # type: ignore[import-not-found]

            aligner = load_model(source)

            def _align_mlx(clip: np.ndarray, text: str, lang: str) -> Any:
                return aligner.generate(clip, text=text, language=lang)

            align = _align_mlx
        else:
            from qwen_asr import Qwen3ForcedAligner  # type: ignore[import-not-found]
            from transformers.utils import logging as hf_logging  # type: ignore[import-not-found]

            hf_logging.set_verbosity_error()
            device, dtype = f._torch_device()
            attn = f._torch_attn_implementation(device)
            quant_cfg = f._torch_quantization_config(device)
            load_kwargs: dict[str, Any] = {"dtype": dtype, "device_map": device}
            if attn:
                load_kwargs["attn_implementation"] = attn
            if quant_cfg is not None:
                load_kwargs["quantization_config"] = quant_cfg
            aligner = Qwen3ForcedAligner.from_pretrained(source, **load_kwargs)

            def _align_tf(clip: np.ndarray, text: str, lang: str) -> Any:
                return aligner.align(
                    audio=(clip, f.SAMPLE_RATE), text=text, language=lang
                )[0]

            align = _align_tf
        total_jobs = max(1, len(jobs))
        for j_idx, (clip, text, lang) in enumerate(jobs, 1):
            f.check_cancel()
            f.report_progress(
                0.75 + 0.23 * ((j_idx - 1) / total_jobs),
                f"时间戳对齐 {j_idx}/{total_jobs}",
                f"🔗 词级时间戳对齐 [{j_idx}/{total_jobs}] ({len(clip) / f.SAMPLE_RATE:.1f}s)...",
            )
            t_align = time.time()
            result = align(clip, text, lang)
            items = [(item.text, item.start_time, item.end_time) for item in result]
            f.log_event(
                f"完成时间戳对齐窗口 [{j_idx}/{total_jobs}] ({len(items)} 词, {time.time() - t_align:.1f}s)",
                tag="asr",
                label="ASR",
            )
            results.append(items)
    finally:
        align = None  # the closure holds a reference to the aligner
        aligner = None
        f._free(engine)
    return results
