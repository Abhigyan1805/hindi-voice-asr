"""Model loading, LoRA wrapping and batched decoding."""

from __future__ import annotations

import time

import numpy as np
import torch

from .data import SAMPLE_RATE

BASE_MODEL = "openai/whisper-small"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]
# Whisper decodes at most 448 positions; 4 are the forced prompt tokens.
MAX_NEW_TOKENS = 440


def load_processor(base: str = BASE_MODEL):
    from transformers import WhisperProcessor

    return WhisperProcessor.from_pretrained(base, language="hindi", task="transcribe")


def load_model(base: str = BASE_MODEL, dtype: torch.dtype = torch.float32, device: str = "cuda"):
    from transformers import WhisperForConditionalGeneration

    # Cast after loading: the dtype keyword was renamed between transformers 4.x and 5.x.
    model = WhisperForConditionalGeneration.from_pretrained(base).to(dtype)
    # Language and task are passed to generate(); the legacy forced ids would clash.
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    return model.to(device)


def add_lora(model, rank: int, alpha: int | None = None, dropout: float = 0.05):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha if alpha is not None else 2 * rank,
        lora_dropout=dropout,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    return get_peft_model(model, cfg)


def load_finetuned(adapter_dir: str, base: str = BASE_MODEL, dtype=torch.float16, device: str = "cuda", merge=True):
    """Base Whisper with a LoRA adapter applied (merged by default, so decoding costs the same as the base)."""
    from peft import PeftModel

    model = load_model(base, dtype=dtype, device=device)
    model = PeftModel.from_pretrained(model, adapter_dir)
    if merge:
        model = model.merge_and_unload()
    model.eval()
    return model


def features(processor, audios: list[np.ndarray], device: str, dtype: torch.dtype) -> torch.Tensor:
    feats = processor.feature_extractor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
    return feats.to(device=device, dtype=dtype)


@torch.inference_mode()
def decode(model, processor, audios: list[np.ndarray], num_beams: int = 1) -> list[str]:
    """Transcribe a batch of 16 kHz arrays (each at most 30 s) as Hindi."""
    param = next(model.parameters())
    feats = features(processor, audios, param.device.type, param.dtype)
    ids = model.generate(
        input_features=feats,
        language="hi",
        task="transcribe",
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=num_beams,
    )
    return [t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True)]


def transcribe_all(model, processor, audios: list[np.ndarray], batch_size: int = 16, num_beams: int = 1):
    """Batched decoding in length order; returns (texts in input order, wall seconds)."""
    order = sorted(range(len(audios)), key=lambda i: len(audios[i]), reverse=True)
    out: list[str] = [""] * len(audios)
    t0 = time.perf_counter()
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        for i, text in zip(idx, decode(model, processor, [audios[i] for i in idx], num_beams), strict=True):
            out[i] = text
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def measure_rtf(model, processor, audios: list[np.ndarray], warmup: int = 2) -> dict[str, float]:
    """Single-stream latency: one utterance at a time, features + decoding, greedy."""
    from .metrics import RTFMeter

    for a in audios[:warmup]:
        decode(model, processor, [a])
    meter = RTFMeter()
    lat = []
    for a in audios:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        decode(model, processor, [a])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        lat.append(dt)
        meter.add(dt, len(a) / SAMPLE_RATE)
    return {
        "rtf": round(meter.rtf, 4),
        "mean_latency_s": round(float(np.mean(lat)), 3),
        "p90_latency_s": round(float(np.percentile(lat, 90)), 3),
        "mean_audio_s": round(meter.audio_s / len(audios), 2),
        "n": len(audios),
    }
