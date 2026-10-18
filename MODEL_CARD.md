# Model card: whisper-small Hindi LoRA (`lora-r32-aug`)

A LoRA adapter for [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) that turns it into a usable Hindi transcriber. It is stored locally in `artifacts/adapter/` and has not been published.

## Summary

| | |
|---|---|
| Base model | `openai/whisper-small` (242 M params, multilingual) |
| Adapter | LoRA, rank 32, alpha 64, dropout 0.05, on `q_proj`, `k_proj`, `v_proj`, `out_proj` of all encoder self-attention and decoder self-/cross-attention blocks: 7.08 M trainable params (2.8 %), 28 MB safetensors |
| Task | Hindi speech → Devanagari text (`language="hi"`, `task="transcribe"`), with punctuation as in FLEURS raw transcripts |
| Inference | merge into the base weights (`hindi_asr.inference.transcribe`); same speed and memory as whisper-small |
| Test WER / CER | **24.53 % / 10.11 %** on FLEURS `hi_in` test (base whisper-small zero-shot: 79.19 % / 49.44 %; whisper-medium zero-shot: 42.95 % / 21.40 %) |
| Latency | single-stream RTF 0.13 on a T4 (1.5 s mean for an 11.8 s clip), fp16, greedy |

## Training data

[Google FLEURS](https://huggingface.co/datasets/google/fleurs), `hi_in` config (CC-BY-4.0): read speech of FLoRes-101 Wikipedia sentences by native speakers, at 16 kHz.

- Train: 2,115 utterances / 6.6 h (5 dropped: over 30 s or over 448 label tokens). Validation: first 200 of 239 utterances, used for epoch selection. Test: all 418 utterances, held out.
- Targets are the `raw_transcription` field: cased, punctuated text, with English loanwords mostly in Devanagari and some acronyms in Latin script.
- Augmentation: speed perturbation ×{0.9, 1.0, 1.1} plus white noise at 10–30 dB SNR with p = 0.3, seeded per (epoch, item).

## Training procedure

Plain PyTorch loop (`src/hindi_asr/train.py`) on one Kaggle T4:

- fp16 autocast with GradScaler; frozen fp32 base model;
- AdamW, lr 1e-3, weight decay 0.01, 10 % linear warmup then linear decay, grad-norm clip 1.0;
- micro-batch 8 × 2 accumulation (effective 16), 5 epochs (660 optimizer steps), about 30 min;
- seed 1234;
- best epoch by validation WER (epoch 5 of 5).

Kernel `abhigyan1881/hva-whisper-lora` v1; the full log is in `results/train_log-lora-r32-aug.json`.

## Evaluation

See [RESULTS.md](RESULTS.md) for the full tables, the loop-free re-scoring, the ablation (rank 8 vs 32, with vs without augmentation) and qualitative examples. Scoring uses `hindi_asr.text.normalize`: NFC, punctuation and symbols removed, Latin lowercased. Whisper's English-centric normaliser is deliberately not used because it strips Devanagari vowel signs.

## Intended use

- Transcribing Hindi speech (questions, dictation, voice commands, read text) in a demo, prototype or research setting.
- A baseline or starting point for further Indian-language ASR fine-tuning.

## Limitations and risks

- **Domain:** trained and tested only on clean read speech. Expect higher error rates on conversational, spontaneous, telephone (8 kHz), far-field or noisy audio, and on strong regional accents not covered by FLEURS speakers.
- **Code-switching:** English words are usually written in Devanagari (`पीएच`, `एपल`), following the FLEURS convention. Applications that want Latin-script English (true Hinglish rendering) need extra data or post-processing.
- **Numbers and named entities:** these are inconsistent (digits vs words) and proper nouns are often misspelled.
- **Hallucination:** much rarer than in the base model (1 runaway output in 418 vs 35), but very short, fast or non-speech clips can still produce fluent text that was never said. Do not use it for legal, medical or other high-stakes transcription without human review.
- **Evaluation:** single seed and a 1.3 h test set; small differences between variants are not significant.
- **Languages:** decoding is forced to Hindi; other languages will be transliterated or translated badly.

## Licence

- Adapter weights: derived from `openai/whisper-small` (Apache-2.0) and trained on FLEURS (CC-BY-4.0). They are released under Apache-2.0 with attribution to Google FLEURS (Conneau et al., 2022, *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*).
- Code in this repository: MIT.

## Citation

Radford et al., 2022, *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper). Hu et al., 2021, *LoRA: Low-Rank Adaptation of Large Language Models*. Conneau et al., 2022, *FLEURS*.
