"""End-to-end Kaggle run: baselines, LoRA training (+ ablation), test evaluation, RTF.

Runs inside the Kaggle kernel produced by ``kaggle/build_kernel.py``. Every stage
writes ``results.json`` before the next starts, and a wall-clock guard skips
stages that would not finish, because a script kernel only keeps its outputs if
it exits cleanly.

    python pipeline.py --mode full|smoke --out /kaggle/working
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import replace
from importlib import metadata

import torch

from hindi_asr.data import load_fleurs
from hindi_asr.metrics import wer_cer
from hindi_asr.model import load_finetuned, load_model, load_processor, measure_rtf, transcribe_all
from hindi_asr.text import has_latin
from hindi_asr.train import TrainConfig, seed_everything, train

START = time.time()


def versions() -> dict:
    pkgs = ["torch", "transformers", "peft", "jiwer", "numpy", "soundfile", "huggingface_hub", "pyarrow"]
    out = {"python": platform.python_version()}
    for p in pkgs:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    out["cuda"] = torch.version.cuda
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return out


class Run:
    def __init__(self, out: str, mode: str, budget_s: float):
        self.out, self.mode, self.deadline = out, mode, START + budget_s
        self.results = {"mode": mode, "versions": versions(), "stages": {}, "skipped": []}
        os.makedirs(os.path.join(out, "predictions"), exist_ok=True)
        os.makedirs(os.path.join(out, "adapters"), exist_ok=True)

    def remaining(self) -> float:
        return self.deadline - time.time()

    def save(self) -> None:
        self.results["elapsed_s"] = round(time.time() - START)
        with open(os.path.join(self.out, "results.json"), "w") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

    def fits(self, name: str, need_s: float) -> bool:
        if self.remaining() < need_s:
            print(f"SKIP {name}: {self.remaining():.0f}s left, needs ~{need_s:.0f}s", flush=True)
            self.results["skipped"].append(name)
            self.save()
            return False
        return True


def score(run: Run, name: str, model, processor, test, rtf_n: int, extra: dict | None = None) -> None:
    model.eval()
    hyps, secs = transcribe_all(model, processor, [u.audio for u in test], batch_size=16)
    m = wer_cer([u.text for u in test], hyps)
    latin = [i for i, u in enumerate(test) if has_latin(u.text)]
    rtf = measure_rtf(model, processor, [u.audio for u in test[:rtf_n]])
    audio_s = sum(u.duration for u in test)
    stage = {
        **m,
        "batched_seconds": round(secs, 1),
        "batched_rtf": round(secs / audio_s, 4),
        "single_stream": rtf,
        # Hallucination / repetition loops: hypothesis more than twice the reference length.
        "runaway_outputs": sum(len(h) > 2 * len(u.text) for u, h in zip(test, hyps, strict=True)),
        "latin_subset": wer_cer([test[i].text for i in latin], [hyps[i] for i in latin]) if latin else None,
        **(extra or {}),
    }
    run.results["stages"][name] = stage
    with open(os.path.join(run.out, "predictions", f"{name}.jsonl"), "w") as f:
        for u, h in zip(test, hyps, strict=True):
            f.write(json.dumps({"uid": u.uid, "sentence_id": u.sentence_id, "duration": round(u.duration, 2), "ref": u.text, "hyp": h}, ensure_ascii=False) + "\n")
    print(f"== {name}: {stage}", flush=True)
    run.save()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "smoke"], default="full")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--budget-hours", type=float, default=10.0)
    args = ap.parse_args()
    smoke = args.mode == "smoke"
    import transformers

    transformers.logging.set_verbosity_error()
    run = Run(args.out, args.mode, args.budget_hours * 3600)
    seed_everything(1234)

    lim = (lambda n: n) if smoke else (lambda n: None)
    t = time.time()
    train_u = load_fleurs("train", limit=lim(64))
    val_u = load_fleurs("validation", limit=lim(16))
    test_u = load_fleurs("test", limit=lim(24))
    run.results["data"] = {
        split: {"utterances": len(us), "hours": round(sum(u.duration for u in us) / 3600, 2), "sentences": len({u.sentence_id for u in us})}
        for split, us in (("train", train_u), ("validation", val_u), ("test", test_u))
    }
    run.results["data"]["test_latin_utterances"] = sum(has_latin(u.text) for u in test_u)
    run.results["data"]["load_seconds"] = round(time.time() - t)
    run.save()
    print("data", run.results["data"], flush=True)

    rtf_n = 4 if smoke else 60
    processor = load_processor()

    # 1. Zero-shot whisper-small.
    model = load_model(dtype=torch.float16)
    score(run, "whisper-small-zeroshot", model, processor, test_u, rtf_n)
    del model
    torch.cuda.empty_cache()

    base_cfg = TrainConfig(max_steps=6 if smoke else None, val_size=8 if smoke else 200, epochs=5)
    plans = [
        replace(base_cfg, name="lora-r32-aug", rank=32, augment=True),
        "whisper-medium-zeroshot",
        replace(base_cfg, name="lora-r8-aug", rank=8, augment=True),
        replace(base_cfg, name="lora-r32-noaug", rank=32, augment=False),
    ]
    for plan in plans:
        if plan == "whisper-medium-zeroshot":
            if smoke or not run.fits(plan, 40 * 60):
                continue
            model = load_model("openai/whisper-medium", dtype=torch.float16)
            score(run, plan, model, load_processor("openai/whisper-medium"), test_u, rtf_n)
            del model
            torch.cuda.empty_cache()
            continue
        if not run.fits(plan.name, 60 * 60):
            continue
        out = os.path.join(args.out, "runs", plan.name)
        log = train(plan, train_u, val_u, processor, out, deadline=run.deadline - 25 * 60)
        run.results.setdefault("training", {})[plan.name] = {k: v for k, v in log.items() if k != "history"}
        run.save()
        shutil.copytree(os.path.join(out, "best"), os.path.join(args.out, "adapters", plan.name), dirs_exist_ok=True)
        shutil.copy(os.path.join(out, "train_log.json"), os.path.join(args.out, "adapters", plan.name, "train_log.json"))
        model = load_finetuned(os.path.join(out, "best"))
        score(run, plan.name, model, processor, test_u, rtf_n, extra={"best_val": log.get("best")})
        del model
        torch.cuda.empty_cache()
        shutil.rmtree(os.path.join(out, "last"), ignore_errors=True)

    run.results["done"] = True
    run.save()
    print("DONE", json.dumps(run.results, indent=1, ensure_ascii=False)[:4000], flush=True)


if __name__ == "__main__":
    sys.exit(main())
