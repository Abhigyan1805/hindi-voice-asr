import numpy as np

from hindi_asr.data import Augment, speed_perturb
from hindi_asr.metrics import wer_cer
from hindi_asr.text import has_latin, normalize


def test_normalize_keeps_matras_and_strips_punctuation():
    assert normalize("नमस्ते, आप कैसे हैं?") == "नमस्ते आप कैसे हैं"
    assert normalize("यह ठीक है। अगला॥") == "यह ठीक है अगला"
    assert normalize("  Hello   WORLD!  ") == "hello world"


def test_normalize_composes_nukta():
    decomposed = "ड़"  # DA + NUKTA
    assert normalize(decomposed) == normalize("ड़")


def test_wer_cer_identity_and_punctuation_free():
    m = wer_cer(["नमस्ते, दुनिया।"], ["नमस्ते दुनिया"])
    assert m["wer"] == 0.0 and m["cer"] == 0.0


def test_wer_counts_matra_errors():
    m = wer_cer(["कुछ अणुओं में"], ["कुछ अणुओ में"])
    assert m["wer"] > 0 and 0 < m["cer"] < m["wer"]


def test_has_latin():
    assert has_latin("मेरा laptop")
    assert not has_latin("मेरा लैपटॉप 2024")


def test_speed_perturb_length_and_augment_deterministic():
    x = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    assert abs(len(speed_perturb(x, 1.1)) - 16000 / 1.1) <= 1
    a = Augment()
    y1 = a(x, np.random.default_rng((1, 2, 3)))
    y2 = a(x, np.random.default_rng((1, 2, 3)))
    assert np.array_equal(y1, y2)
