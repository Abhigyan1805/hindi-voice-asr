"""Inference API: ``transcribe(path) -> text`` with the LoRA adapter applied to whisper-small.

    from hindi_asr.inference import transcribe
    print(transcribe("question.wav"))

The adapter defaults to ``artifacts/adapter`` when that directory exists, and
otherwise to the published Hugging Face adapter ``Dratinia/whisper-small-hindi-lora``.
Override it with the ``HINDI_ASR_ADAPTER`` env var or the ``adapter`` argument:
either a local directory or a Hub model id. Pass ``adapter=None`` to get the
zero-shot base model for comparison. Audio longer
than 30 s is decoded in consecutive 30 s windows.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .data import SAMPLE_RATE, load_file
from .model import BASE_MODEL, decode, load_finetuned, load_model, load_processor

LOCAL_ADAPTER = Path(__file__).resolve().parents[2] / "artifacts" / "adapter"
HUB_ADAPTER = "Dratinia/whisper-small-hindi-lora"
_WINDOW = 30 * SAMPLE_RATE
_UNSET = object()


def default_adapter() -> str:
    """``$HINDI_ASR_ADAPTER``, else the local ``artifacts/adapter``, else the Hub adapter."""
    if env := os.environ.get("HINDI_ASR_ADAPTER"):
        return env
    return str(LOCAL_ADAPTER) if LOCAL_ADAPTER.is_dir() else HUB_ADAPTER


def _device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


@lru_cache(maxsize=4)
def get_transcriber(adapter: str | None, base: str = BASE_MODEL) -> "Transcriber":
    return Transcriber(adapter, base)


class Transcriber:
    def __init__(self, adapter: str | None, base: str = BASE_MODEL):
        device, dtype = _device()
        self.processor = load_processor(base)
        if adapter:
            # A path that does not exist locally is treated as a Hugging Face Hub model id.
            if Path(adapter).exists() and not Path(adapter, "adapter_config.json").exists():
                raise FileNotFoundError(f"no LoRA adapter at {adapter} (expected adapter_config.json); see README 'Get the adapter'")
            self.model = load_finetuned(adapter, base, dtype=dtype, device=device)
        else:
            self.model = load_model(base, dtype=dtype, device=device).eval()
        self.adapter = adapter

    def __call__(self, audio: np.ndarray, num_beams: int = 1) -> str:
        chunks = [audio[i : i + _WINDOW] for i in range(0, max(len(audio), 1), _WINDOW)]
        chunks = [c for c in chunks if len(c) > SAMPLE_RATE // 10] or [audio]
        return " ".join(t for t in decode(self.model, self.processor, chunks, num_beams) if t)


def transcribe(path: str | os.PathLike, adapter=_UNSET, num_beams: int = 1) -> str:
    """Transcribe an audio file (wav/flac/mp3/...) to Hindi text.

    ``adapter`` is a local adapter directory or a Hub id such as
    ``"Dratinia/whisper-small-hindi-lora"``; ``None`` means the zero-shot base model.
    """
    if adapter is _UNSET:
        adapter = default_adapter()
    return get_transcriber(str(adapter) if adapter else None)(load_file(str(path)), num_beams=num_beams)
