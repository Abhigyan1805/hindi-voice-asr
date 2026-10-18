"""Turn a Kaggle run's ``results.json`` + predictions into Markdown tables for RESULTS.md.

    python kaggle/report.py kaggle/output/full
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ORDER = ["whisper-small-zeroshot", "lora-r32-aug", "lora-r8-aug", "lora-r32-noaug", "whisper-medium-zeroshot"]
PARAMS = {"whisper-small-zeroshot": "242M", "whisper-medium-zeroshot": "769M"}


def main(out_dir: str) -> None:
    out = Path(out_dir)
    res = json.loads((out / "results.json").read_text())
    stages = res["stages"]
    names = [n for n in ORDER if n in stages]
    print("| Model | WER % | CER % | Latin-script subset WER % | Single-stream RTF (T4) | Mean latency s | Batched RTF (bs16) | Runaway outputs |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for n in names:
        s = stages[n]
        lat = s["latin_subset"]["wer"] if s.get("latin_subset") else "-"
        print(f"| {n} | {s['wer']:.2f} | {s['cer']:.2f} | {lat} | {s['single_stream']['rtf']:.3f} | {s['single_stream']['mean_latency_s']:.2f} | {s['batched_rtf']:.4f} | {s.get('runaway_outputs', '-')} |")
    print()
    tr = res.get("training", {})
    if tr:
        print("| Run | LoRA rank | Aug | Trainable params | Steps | Train min | Best val WER % (epoch) |")
        print("|---|---:|---|---:|---:|---:|---:|")
        for n, t in tr.items():
            c = t["config"]
            b = t.get("best", {})
            print(f"| {n} | {c['rank']} | {'yes' if c['augment'] else 'no'} | {t['trainable_params']:,} | {t['total_steps']} | {t['train_seconds'] / 60:.0f} | {b.get('wer')} ({b.get('epoch')}) |")
        print()
    print("```json")
    print(json.dumps({"data": res["data"], "versions": res["versions"], "skipped": res["skipped"], "elapsed_s": res["elapsed_s"]}, indent=1, ensure_ascii=False))
    print("```")


if __name__ == "__main__":
    main(sys.argv[1])
