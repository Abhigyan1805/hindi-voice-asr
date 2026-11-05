"""CPU int8 variants of the merged model, and a benchmark that compares them with fp32.

Variants (all greedy, one utterance at a time, on CPU):

* ``fp32``      - the project's own path: whisper-small + merged LoRA adapter, PyTorch fp32.
* ``torch-int8`` - the same model with ``torch.ao.quantization.quantize_dynamic`` on every
  ``nn.Linear`` (int8 weights, activations quantised on the fly).
* ``ct2-int8``  - the merged model converted to CTranslate2 with int8 weights, run with faster-whisper.

Convert (writes under ``artifacts/quant/``, which is git-ignored)::

    python -m hindi_asr.quant convert

Benchmark one variant per process, so peak RSS is that variant's own::

    python -m hindi_asr.quant bench fp32 --limit 150 --threads 4
    python -m hindi_asr.quant report            # merge the per-variant JSON into a table
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import time
from pathlib import Path

import numpy as np

from .metrics import wer_cer

ROOT = Path(__file__).resolve().parents[2]
QUANT_DIR = ROOT / "artifacts" / "quant"
MERGED_DIR = QUANT_DIR / "merged-fp32"
CT2_DIR = QUANT_DIR / "ct2-int8"
TORCH_INT8_FILE = QUANT_DIR / "torch-int8.pt"
OUT_DIR = ROOT / "results" / "quant"
VARIANTS = ("fp32", "torch-int8", "ct2-int8")
MAX_NEW_TOKENS = 440
MEM_CLIPS = QUANT_DIR / "mem_clips.npz"


def dir_size_mb(path: Path) -> float:
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files) / 1e6


def peak_rss_mb() -> float:
    """Peak resident set size of this process (Linux reports KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_merged(adapter: str | None = None):
    """The project's fp32 CPU model: base + LoRA adapter merged into the weights."""
    import torch

    from .inference import default_adapter
    from .model import load_finetuned

    return load_finetuned(adapter or default_adapter(), dtype=torch.float32, device="cpu")


def quantize_torch_int8(model):
    """Dynamic int8 quantisation of all linear layers (CPU, fbgemm/qnnpack)."""
    import torch

    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def convert(adapter: str | None = None, force: bool = False) -> dict[str, float]:
    """Write the three on-disk artifacts and return their sizes in MB."""
    import torch

    from .model import BASE_MODEL, load_processor

    QUANT_DIR.mkdir(parents=True, exist_ok=True)
    if force or not (MERGED_DIR / "model.safetensors").exists():
        model = load_merged(adapter)
        model.save_pretrained(MERGED_DIR)
        load_processor(BASE_MODEL).save_pretrained(MERGED_DIR)
    else:
        from transformers import WhisperForConditionalGeneration

        model = WhisperForConditionalGeneration.from_pretrained(MERGED_DIR).eval()
    if force or not TORCH_INT8_FILE.exists():
        torch.save(quantize_torch_int8(model).state_dict(), TORCH_INT8_FILE)
    del model
    if force or not (CT2_DIR / "model.bin").exists():
        from ctranslate2.converters import TransformersConverter

        shutil.rmtree(CT2_DIR, ignore_errors=True)
        TransformersConverter(str(MERGED_DIR), copy_files=["tokenizer.json"]).convert(str(CT2_DIR), quantization="int8")
        # transformers 5 saves a combined processor_config.json; faster-whisper wants the feature-extractor file.
        load_processor(BASE_MODEL).feature_extractor.save_pretrained(CT2_DIR)
    return {
        "fp32": dir_size_mb(MERGED_DIR / "model.safetensors"),
        "torch-int8": dir_size_mb(TORCH_INT8_FILE),
        "ct2-int8": dir_size_mb(CT2_DIR),
    }


class _TorchRunner:
    def __init__(self, variant: str, threads: int):
        import torch
        from transformers import WhisperForConditionalGeneration

        from .model import BASE_MODEL, decode, load_processor

        torch.set_num_threads(threads)
        self._decode = decode
        self.processor = load_processor(BASE_MODEL)
        model = WhisperForConditionalGeneration.from_pretrained(MERGED_DIR).eval()
        model.generation_config.forced_decoder_ids = None
        model.config.forced_decoder_ids = None
        if variant == "torch-int8":
            model = quantize_torch_int8(model)
        self.model = model

    def __call__(self, audio: np.ndarray) -> str:
        return self._decode(self.model, self.processor, [audio])[0]


class _CT2Runner:
    def __init__(self, threads: int):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(str(CT2_DIR), device="cpu", compute_type="int8", cpu_threads=threads)

    def __call__(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language="hi",
            task="transcribe",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        return " ".join(s.text.strip() for s in segments).strip()


def make_runner(variant: str, threads: int):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    return _CT2Runner(threads) if variant == "ct2-int8" else _TorchRunner(variant, threads)


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("torch", "transformers", "peft", "ctranslate2", "faster-whisper", "jiwer", "numpy"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "n/a"
    return out


def bench(variant: str, limit: int | None, threads: int, warmup: int = 2) -> dict:
    from .data import load_fleurs

    utts = load_fleurs("test", limit=limit)  # sorted by file name: a fixed, documented subset
    runner = make_runner(variant, threads)
    for u in utts[:warmup]:
        runner(u.audio)
    hyps, lat, audio_s = [], [], 0.0
    for i, u in enumerate(utts):
        t0 = time.perf_counter()
        hyps.append(runner(u.audio))
        lat.append(time.perf_counter() - t0)
        audio_s += u.duration
        if (i + 1) % 25 == 0:
            print(f"[{variant}] {i + 1}/{len(utts)}  rtf so far {sum(lat) / audio_s:.3f}", flush=True)
    size = {
        "fp32": lambda: dir_size_mb(MERGED_DIR / "model.safetensors"),
        "torch-int8": lambda: dir_size_mb(TORCH_INT8_FILE),
        "ct2-int8": lambda: dir_size_mb(CT2_DIR),
    }[variant]()
    res = {
        "variant": variant,
        **wer_cer([u.text for u in utts], hyps),
        "size_mb": round(size, 1),
        "rtf": round(sum(lat) / audio_s, 3),
        "mean_latency_s": round(float(np.mean(lat)), 3),
        "p90_latency_s": round(float(np.percentile(lat, 90)), 3),
        "mean_audio_s": round(audio_s / len(utts), 2),
        "peak_rss_mb": round(peak_rss_mb()),
        "threads": threads,
        "cpu": cpu_model(),
        "versions": versions(),
        "subset": f"first {len(utts)} test clips by file name" if limit else "full test split",
        "decoding": "greedy (beam 1), language=hi, max 440 new tokens, one clip at a time",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{variant}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    with open(OUT_DIR / f"{variant}.jsonl", "w") as f:
        for u, h in zip(utts, hyps, strict=True):
            f.write(json.dumps({"id": u.uid, "ref": u.text, "hyp": h}, ensure_ascii=False) + "\n")
    return res


def peak_memory(variant: str, threads: int, n: int = 5) -> dict:
    """Peak RSS of model + a few decoded clips, without the FLEURS parquet in memory.

    ``bench`` loads the whole test parquet, which inflates its own ``ru_maxrss``, so
    memory is measured here in a fresh process from clips cached by ``mem-clips``.
    """
    clips = np.load(MEM_CLIPS)
    runner = make_runner(variant, threads)
    for k in sorted(clips.files)[:n]:
        runner(clips[k])
    res = {"variant": variant, "peak_rss_mb": round(peak_rss_mb()), "clips": n, "threads": threads}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{variant}.mem.json").write_text(json.dumps(res, indent=2))
    return res


def report() -> str:
    rows = [json.loads((OUT_DIR / f"{v}.json").read_text()) for v in VARIANTS if (OUT_DIR / f"{v}.json").exists()]
    for r in rows:  # prefer the clean memory measurement when present
        mem = OUT_DIR / f"{r['variant']}.mem.json"
        if mem.exists():
            r["peak_rss_mb"] = json.loads(mem.read_text())["peak_rss_mb"]
    base = next((r for r in rows if r["variant"] == "fp32"), None)
    lines = [
        "| Variant | Size (MB) | WER % | CER % | RTF | Mean latency (p90) | Peak RSS (MB) | Speed-up vs fp32 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        speed = f"{base['rtf'] / r['rtf']:.2f}x" if base else "-"
        lines.append(
            f"| {r['variant']} | {r['size_mb']:.0f} | {r['wer']:.2f} | {r['cer']:.2f} | {r['rtf']:.3f} | "
            f"{r['mean_latency_s']:.2f} s ({r['p90_latency_s']:.2f} s) | {r['peak_rss_mb']} | {speed} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="write merged fp32, torch int8 and CTranslate2 int8 artifacts")
    c.add_argument("--adapter", default=None)
    c.add_argument("--force", action="store_true")
    b = sub.add_parser("bench", help="benchmark one variant on FLEURS Hindi test")
    b.add_argument("variant", choices=VARIANTS)
    b.add_argument("--limit", type=int, default=150, help="first N test clips by file name (0 = all 418)")
    b.add_argument("--threads", type=int, default=4)
    m = sub.add_parser("mem", help="peak RSS of one variant in a fresh process (run `mem-clips` once first)")
    m.add_argument("variant", choices=VARIANTS)
    m.add_argument("--threads", type=int, default=4)
    sub.add_parser("mem-clips", help="cache the first 5 test clips for `mem`")
    sub.add_parser("report", help="print the markdown table from results/quant/*.json")
    args = p.parse_args(argv)
    if args.cmd == "convert":
        print(json.dumps({k: round(v, 1) for k, v in convert(args.adapter, args.force).items()}, indent=2))
    elif args.cmd == "bench":
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        print(json.dumps(bench(args.variant, args.limit or None, args.threads), indent=2, ensure_ascii=False))
    elif args.cmd == "mem-clips":
        from .data import load_fleurs

        np.savez(MEM_CLIPS, **{f"c{i}": u.audio for i, u in enumerate(load_fleurs("test", limit=5))})
    elif args.cmd == "mem":
        print(json.dumps(peak_memory(args.variant, args.threads)))
    else:
        print(report())


if __name__ == "__main__":
    main()
