"""Quantised-path checks. Heavy ones are skipped unless ``python -m hindi_asr.quant convert`` has been run."""

import numpy as np
import pytest

from hindi_asr import quant


def test_variants_and_sizes(tmp_path):
    assert quant.VARIANTS == ("fp32", "torch-int8", "ct2-int8")
    (tmp_path / "a.bin").write_bytes(b"x" * 2_000_000)
    assert quant.dir_size_mb(tmp_path) == pytest.approx(2.0)
    assert quant.dir_size_mb(tmp_path / "a.bin") == pytest.approx(2.0)


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        quant.make_runner("int4", 1)


def test_dynamic_int8_replaces_linear_layers():
    torch = pytest.importorskip("torch")
    net = torch.nn.Sequential(torch.nn.Linear(16, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
    q = quant.quantize_torch_int8(net)
    assert all("quantized" in type(m).__module__ for m in (q[0], q[2]))
    assert q(torch.randn(2, 16)).shape == (2, 4)


@pytest.mark.skipif(not (quant.CT2_DIR / "model.bin").exists(), reason="run `python -m hindi_asr.quant convert` first")
@pytest.mark.parametrize("variant", ["ct2-int8", "torch-int8"])
def test_quantised_model_transcribes_a_clip(variant):
    if variant == "torch-int8" and not (quant.MERGED_DIR / "model.safetensors").exists():
        pytest.skip("merged model not available")
    from hindi_asr.data import load_fleurs

    try:
        clip = load_fleurs("test", limit=1)[0]
    except Exception as e:  # noqa: BLE001 - offline / dataset unavailable
        pytest.skip(f"FLEURS unavailable: {e}")
    text = quant.make_runner(variant, threads=2)(clip.audio[: 8 * 16_000].astype(np.float32))
    assert isinstance(text, str) and text.strip()
