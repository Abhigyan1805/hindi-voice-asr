"""FLEURS Hindi loading, audio decoding and light augmentation.

The data comes from the ungated ``google/fleurs`` parquet export
(``parquet-data/hi_in/*.parquet``), so no dataset loading script is needed.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
FLEURS_REPO = "google/fleurs"
FLEURS_FILES = {
    "train": "parquet-data/hi_in/train-00000-of-00001.parquet",
    "validation": "parquet-data/hi_in/validation-00000-of-00001.parquet",
    "test": "parquet-data/hi_in/test-00000-of-00001.parquet",
}


@dataclass
class Utterance:
    uid: str
    sentence_id: int
    audio: np.ndarray  # float32 mono at SAMPLE_RATE
    text: str  # raw transcription (with punctuation)
    text_norm: str  # FLEURS normalised transcription

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


def resample(audio: np.ndarray, sr: int, target: int = SAMPLE_RATE) -> np.ndarray:
    if sr == target:
        return audio.astype(np.float32, copy=False)
    import librosa

    return librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target)


def decode_audio(data: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample(audio, sr)


def load_file(path: str) -> np.ndarray:
    """Load any audio file soundfile (or librosa as a fallback) can read, as 16 kHz mono."""
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return resample(audio, sr)
    except Exception:  # noqa: BLE001 - e.g. mp3/m4a from a browser recorder
        import librosa

        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return audio.astype(np.float32)


def fleurs_path(split: str, cache_dir: str | None = None) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(FLEURS_REPO, FLEURS_FILES[split], repo_type="dataset", cache_dir=cache_dir)


def load_fleurs(split: str, limit: int | None = None, path: str | None = None) -> list[Utterance]:
    """Load one FLEURS hi_in split, sorted by file name for a stable order."""
    import pyarrow.parquet as pq

    table = pq.read_table(path or fleurs_path(split))
    rows = table.to_pylist()
    rows.sort(key=lambda r: r["audio"]["path"])
    if limit:
        rows = rows[:limit]
    return [
        Utterance(
            uid=r["audio"]["path"].removesuffix(".wav"),
            sentence_id=int(r["id"]),
            audio=decode_audio(r["audio"]["bytes"]),
            text=r["raw_transcription"].strip(),
            text_norm=r["transcription"].strip(),
        )
        for r in rows
    ]


def speed_perturb(audio: np.ndarray, factor: float) -> np.ndarray:
    """Kaldi-style speed perturbation: resample so tempo and pitch both change by ``factor``."""
    if factor == 1.0:
        return audio
    n_out = int(round(len(audio) / factor))
    x_old = np.arange(len(audio), dtype=np.float64)
    x_new = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = float(np.mean(audio**2)) + 1e-10
    noise = rng.standard_normal(len(audio)).astype(np.float32)
    noise *= np.sqrt(power / (10 ** (snr_db / 10)))
    return audio + noise


@dataclass
class Augment:
    """Cheap on-the-fly augmentation: 3-way speed perturbation plus occasional white noise."""

    speeds: tuple[float, ...] = (0.9, 1.0, 1.1)
    noise_prob: float = 0.3
    snr_range: tuple[float, float] = (10.0, 30.0)

    def __call__(self, audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        audio = speed_perturb(audio, float(rng.choice(self.speeds)))
        if rng.random() < self.noise_prob:
            audio = add_noise(audio, float(rng.uniform(*self.snr_range)), rng)
        return audio[: 30 * SAMPLE_RATE]
