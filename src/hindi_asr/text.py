"""Text normalisation used for training targets and for WER/CER scoring.

Whisper's own ``BasicTextNormalizer`` deletes Unicode combining marks, which
destroys Devanagari (every matra and virama is a combining mark), so scoring
Hindi with it under-reports errors. This module keeps the script intact and
only removes what should not count as an error: punctuation, case and spacing.
"""

from __future__ import annotations

import re
import unicodedata

# Devanagari danda / double danda are category Po and are removed with the rest
# of the punctuation; listed here for readers.
_DANDAS = "।॥"
_SPACE_RE = re.compile(r"\s+")
# Zero-width joiner / non-joiner change rendering, not words.
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)


def normalize(text: str) -> str:
    """Normalise a transcript for scoring.

    NFC-compose, drop zero-width characters, replace punctuation and symbols
    (Unicode categories P* and S*) with spaces, lowercase Latin text, and
    collapse whitespace. Devanagari letters, matras, nukta and digits survive.
    """
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S") or ch in _DANDAS:
            out.append(" ")
        else:
            out.append(ch)
    text = "".join(out).lower()
    return _SPACE_RE.sub(" ", text).strip()


def has_latin(text: str) -> bool:
    """True when the text contains Latin letters (a cheap code-switching signal)."""
    return any("a" <= ch.lower() <= "z" for ch in text)
