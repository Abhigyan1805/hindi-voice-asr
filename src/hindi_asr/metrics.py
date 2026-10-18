"""WER / CER over normalised text, plus real-time factor bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass

import jiwer

from .text import normalize


def wer_cer(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Corpus-level WER and CER (in percent) after :func:`normalize`.

    Empty normalised references are dropped; an empty hypothesis counts as all
    deletions. CER is computed with spaces removed so it measures characters,
    not word segmentation.
    """
    refs, hyps = [], []
    for ref, hyp in zip(references, hypotheses, strict=True):
        ref_n = normalize(ref)
        if not ref_n:
            continue
        refs.append(ref_n)
        hyps.append(normalize(hyp))
    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer([r.replace(" ", "") for r in refs], [h.replace(" ", "") for h in hyps])
    return {"wer": round(100 * wer, 2), "cer": round(100 * cer, 2), "n": len(refs)}


@dataclass
class RTFMeter:
    """Accumulates compute time against audio time. RTF < 1 is faster than real time."""

    compute_s: float = 0.0
    audio_s: float = 0.0

    def add(self, compute_s: float, audio_s: float) -> None:
        self.compute_s += compute_s
        self.audio_s += audio_s

    @property
    def rtf(self) -> float:
        return self.compute_s / self.audio_s if self.audio_s else float("nan")
