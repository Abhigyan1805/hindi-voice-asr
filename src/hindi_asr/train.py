"""LoRA fine-tuning loop for Whisper (plain PyTorch, fp16 autocast)."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import Augment, Utterance
from .metrics import wer_cer
from .model import BASE_MODEL, add_lora, load_model, transcribe_all


@dataclass
class TrainConfig:
    name: str = "r32-aug"
    base: str = BASE_MODEL
    rank: int = 32
    alpha: int | None = None  # defaults to 2 * rank
    dropout: float = 0.05
    augment: bool = True
    epochs: int = 5
    batch_size: int = 8  # micro-batch; 16 does not fit a T4 (15 GB) in fp16 autocast
    grad_accum: int = 2  # effective batch 16
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1234
    val_size: int = 200
    num_workers: int = 4
    max_steps: int | None = None  # smoke runs only
    extra: dict = field(default_factory=dict)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SpeechDataset(Dataset):
    def __init__(self, utts: list[Utterance], labels: list[list[int]], augment: Augment | None, seed: int):
        self.utts, self.labels, self.augment, self.seed = utts, labels, augment, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.utts)

    def __getitem__(self, i: int):
        audio = self.utts[i].audio
        if self.augment is not None:
            # Seeded per (epoch, item) so the augmentation is reproducible whatever the worker count.
            rng = np.random.default_rng((self.seed, self.epoch, i))
            audio = self.augment(audio, rng)
        return audio, self.labels[i]


class Collator:
    def __init__(self, processor, decoder_start_id: int):
        self.fe = processor.feature_extractor
        self.start = decoder_start_id

    def __call__(self, batch):
        audios, labels = zip(*batch, strict=True)
        feats = self.fe(list(audios), sampling_rate=16_000, return_tensors="pt").input_features
        # The model prepends decoder_start itself when shifting labels right.
        labels = [lab[1:] if lab and lab[0] == self.start else lab for lab in labels]
        width = max(len(lab) for lab in labels)
        out = torch.full((len(labels), width), -100, dtype=torch.long)
        for row, lab in enumerate(labels):
            out[row, : len(lab)] = torch.tensor(lab)
        return feats, out


def tokenize_labels(processor, utts: list[Utterance], max_len: int = 448):
    keep_u, keep_l, dropped = [], [], 0
    for u in utts:
        ids = processor.tokenizer(u.text).input_ids
        if len(ids) > max_len or u.duration > 30.0:
            dropped += 1
            continue
        keep_u.append(u)
        keep_l.append(ids)
    return keep_u, keep_l, dropped


def evaluate_val(model, processor, val: list[Utterance]) -> dict:
    model.eval()
    with torch.autocast("cuda", dtype=torch.float16):
        hyps, secs = transcribe_all(model, processor, [u.audio for u in val], batch_size=16)
    model.train()
    return {**wer_cer([u.text for u in val], hyps), "seconds": round(secs, 1)}


def train(cfg: TrainConfig, train_utts, val_utts, processor, out_dir: str, deadline: float | None = None) -> dict:
    """Train one LoRA adapter; saves ``best/`` (lowest validation WER) and ``last/`` under ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(cfg.seed)
    device = "cuda"
    model = load_model(cfg.base, dtype=torch.float32, device=device)
    model.config.use_cache = False
    model = add_lora(model, cfg.rank, cfg.alpha, cfg.dropout)
    trainable, total = model.get_nb_trainable_parameters()

    utts, labels, dropped = tokenize_labels(processor, train_utts)
    ds = SpeechDataset(utts, labels, Augment() if cfg.augment else None, cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=gen,
        num_workers=cfg.num_workers,
        collate_fn=Collator(processor, model.config.decoder_start_token_id),
        persistent_workers=False,
        drop_last=True,
    )
    val = val_utts[: cfg.val_size]
    steps_per_epoch = len(loader) // cfg.grad_accum
    total_steps = cfg.max_steps or steps_per_epoch * cfg.epochs
    warmup = max(1, int(cfg.warmup_frac * total_steps))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup))
    )
    scaler = torch.amp.GradScaler("cuda")

    log = {
        "config": asdict(cfg),
        "trainable_params": trainable,
        "total_params": total,
        "train_utterances": len(utts),
        "dropped_utterances": dropped,
        "train_audio_hours": round(sum(u.duration for u in utts) / 3600, 2),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "history": [],
        "val": [],
    }
    print(f"[{cfg.name}] trainable {trainable:,}/{total:,} params, {len(utts)} utts, {total_steps} steps", flush=True)

    best_wer = math.inf
    step, t0, stopped_early = 0, time.time(), False
    model.train()
    for epoch in range(math.ceil(total_steps / steps_per_epoch)):
        ds.epoch = epoch
        running, n = 0.0, 0
        for micro, (feats, labs) in enumerate(loader):
            feats, labs = feats.to(device, non_blocking=True), labs.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_features=feats, labels=labs).loss
            scaler.scale(loss / cfg.grad_accum).backward()
            running += loss.item() / cfg.grad_accum
            if (micro + 1) % cfg.grad_accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            n += 1
            if step % 25 == 0 or step == total_steps:
                rec = {"step": step, "epoch": epoch, "loss": round(running / n, 4), "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0)}
                log["history"].append(rec)
                print(f"[{cfg.name}] {rec}", flush=True)
                running, n = 0.0, 0
            if step >= total_steps:
                break
            if deadline and time.time() > deadline:
                stopped_early = True
                break
        model.config.use_cache = True
        v = {"epoch": epoch, "step": step, **evaluate_val(model, processor, val)}
        model.config.use_cache = False
        log["val"].append(v)
        print(f"[{cfg.name}] val {v}", flush=True)
        model.save_pretrained(os.path.join(out_dir, "last"))
        if v["wer"] < best_wer:
            best_wer = v["wer"]
            model.save_pretrained(os.path.join(out_dir, "best"))
            log["best"] = v
        with open(os.path.join(out_dir, "train_log.json"), "w") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        if step >= total_steps or stopped_early:
            break
    log["train_seconds"] = round(time.time() - t0)
    log["stopped_early"] = stopped_early
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    del model, opt
    torch.cuda.empty_cache()
    return log
