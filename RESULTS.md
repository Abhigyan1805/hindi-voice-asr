# Results

All numbers below come from one Kaggle run and are reproduced by `python kaggle/report.py`. The raw outputs are kept in [`results/`](results/): `results.json`, per-utterance predictions for every model, and the per-run training logs.

| | |
|---|---|
| Kaggle kernel | `abhigyan1881/hva-whisper-lora`, version 1 (private), script `kaggle/kernels/full/hva-whisper-lora.py` from commit `f6e83a8` |
| Smoke check | `abhigyan1881/hva-whisper-lora-smoke`, version 3 (64/16/24 utterances, 6 steps per run, passed; built just before the runaway-output counter was added) |
| Hardware | 1× Tesla T4 (15 GB), wall time 6,531 s (1.8 h) for everything below |
| Software | Python 3.13.15, torch 2.11.0+cu128, transformers 5.16.1, peft 0.21.2, jiwer 4.0.0, numpy 2.1.3, huggingface_hub 1.29.0, pyarrow 23.0.1 |
| Seed | 1234 (data order, LoRA init, augmentation per (epoch, item)) |
| Data | FLEURS `hi_in`: train 2,120 utt / 6.66 h (2,115 used; 5 dropped over 30 s or 448 tokens), validation 239 utt / 0.71 h, test **418 utt / 1.34 h** (265 distinct sentences) |
| Decoding | greedy, `language="hi"`, `task="transcribe"`, max 440 new tokens, fp16; LoRA merged into the base weights |
| Scoring | corpus WER/CER after `hindi_asr.text.normalize` (NFC; punctuation, symbols and danda removed; Latin lowercased); CER ignores spaces |

## FLEURS Hindi test (418 utterances)

| Model | Trainable params | WER % | CER % | Runaway outputs¹ | Single-stream RTF² | Mean latency (p90) | Batched RTF³ |
|---|---:|---:|---:|---:|---:|---:|---:|
| whisper-small, zero-shot | – | 79.19 | 49.44 | 35 | 0.173 | 2.04 s (4.93 s) | 0.0290 |
| whisper-medium, zero-shot | – | 42.95 | 21.40 | 6 | 0.241 | 2.84 s (4.08 s) | 0.0397 |
| **whisper-small + LoRA r32 + aug** (shipped) | 7.08 M (2.8 %) | **24.53** | **10.11** | 1 | 0.130 | 1.53 s (2.26 s) | 0.0166 |
| whisper-small + LoRA r32, no aug | 7.08 M | 25.67 | 10.25 | 0 | 0.130 | 1.53 s (2.25 s) | 0.0157 |
| whisper-small + LoRA r8 + aug | 1.77 M | 26.98 | 11.34 | 0 | 0.129 | 1.52 s (2.27 s) | 0.0158 |

1. A hypothesis more than twice as long as its reference: a repetition or hallucination loop (e.g. `पल पल पल …`, `आप आप आप …`).
2. One utterance at a time on a T4 (feature extraction plus decoding), over the first 60 test clips (mean 11.8 s of audio). RTF = compute time / audio time. Lower is faster.
3. Whole test set at batch size 16, sorted by length.

**Fine-tuning cuts WER from 79.2 % to 24.5 % (−69 % relative) and CER from 49.4 % to 10.1 %.** The 3× smaller model also beats zero-shot whisper-medium by 18 WER points. Latency is *lower* than the base model, even though the merged LoRA adds no compute. The base model spends time generating runaway loops; the fine-tuned model stops at the end of the sentence.

### How much of the base model's gap is just loops?

Zero-shot whisper-small loops on 35 of 418 clips, and those inflate its WER. Here every model is re-scored on the 380 clips where *no* model produced a runaway output:

| Model (380-clip loop-free subset) | WER % | CER % |
|---|---:|---:|
| whisper-small zero-shot | 55.91 | 29.80 |
| whisper-medium zero-shot | 39.07 | 17.17 |
| LoRA r32 + aug | **23.42** | **9.20** |
| LoRA r32, no aug | 25.07 | 9.79 |
| LoRA r8 + aug | 26.37 | 10.94 |

So the real like-for-like gain on clean outputs is 55.9 → 23.4 WER (−58 % relative). The rest of the headline gain comes from no longer looping.

### Ablation (single seed: read the differences as indicative)

| Run | LoRA rank | Augmentation | Best val WER % (epoch) | Test WER % | Test CER % | Train time |
|---|---:|---|---:|---:|---:|---:|
| lora-r32-aug | 32 | speed 0.9/1.0/1.1 + noise p=0.3 | 26.27 (5/5) | 24.53 | 10.11 | 30 min |
| lora-r32-noaug | 32 | none | 26.64 (5/5) | 25.67 | 10.25 | 29 min |
| lora-r8-aug | 8 | speed + noise | 28.64 (4/5) | 26.98 | 11.34 | 29 min |

- **Rank:** r32 beats r8 by 2.4 WER and 1.2 CER at the same LR and alpha/r ratio, for 4× the adapter size (28 MB vs 7 MB). The r8 run peaked at epoch 4 and slightly regressed afterwards.
- **Augmentation:** it helps by 1.1 WER, but CER is almost identical (10.11 vs 10.25), and this is one seed on a 1.3 h test set. The honest conclusion is "small, plausibly positive". FLEURS is clean read speech, so noise/speed robustness is not what this test measures. The benefit should show up more on real microphone input.
- Validation WER was still falling at epoch 5 for both r32 runs, so a longer schedule would probably help a little.

### Code-switching (Latin-script) subset

16 test clips have Latin-script words in the reference (acronyms and brand names such as `pH`, `MS`, `METI`, `Apple`, `TMZ`, `GMT`, `35mm`).

| Model | WER % | CER % |
|---|---:|---:|
| whisper-small zero-shot | 105.10 | 63.79 |
| whisper-medium zero-shot | 51.53 | 38.01 |
| LoRA r32 + aug | 32.65 | 16.40 |
| LoRA r8 + aug | 29.59 | 13.57 |

With n = 16 this is anecdotal. The main pattern is in the examples below: FLEURS transcribers are inconsistent (`METI` vs `मेटी`), and the fine-tuned model usually writes English acronyms in Devanagari (`पीएच`, `एमईटीआई`, `सीईपी`). Under strict WER that counts as an error even when it is a faithful transcription.

## CPU int8 variants (laptop CPU, 150-clip subset)

| Variant | Size on disk | WER % | CER % | RTF | Mean latency (p90) per clip | Peak RSS | Speed-up vs fp32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 PyTorch (baseline) | 967 MB | 23.51 | 9.10 | 0.654 | 7.93 s (10.61 s) | 1,560 MB | 1.00x |
| PyTorch dynamic int8 (`quantize_dynamic`, Linear layers) | 413 MB | 26.97 | 10.63 | 0.413 | 5.01 s (7.13 s) | 2,563 MB* | 1.58x |
| CTranslate2 int8 (faster-whisper) | 253 MB | 26.12 | 10.83 | 0.320 | 3.88 s (4.95 s) | 653 MB | 2.04x |

Measured on a laptop CPU (11th Gen Intel Core i5-11300H, 4 threads, WSL2, 5.9 GB RAM shared with other work), no GPU. The model is the project's own: whisper-small plus the `Dratinia/whisper-small-hindi-lora` adapter merged into the base weights. Scoring uses the project's `hindi_asr.text.normalize` and corpus WER/CER.

- **Subset:** the first **150** of the 418 FLEURS Hindi test clips by file name (mean 12.1 s of audio each), the same clips for every variant, including the fp32 baseline. Full-set numbers are not measured for these variants, so the fp32 23.51 % WER here is not comparable with the 24.53 % on the full 418 clips above; compare rows within this table only. One run each, no confidence intervals; with 150 clips a difference of a point or so is within noise.
- **Decoding:** greedy (beam 1), `language="hi"`, max 440 new tokens, one clip at a time, no batching, no VAD, no temperature fallback. RTF = compute time / audio time (feature extraction plus decoding), after 2 warm-up clips.
- **Accuracy cost:** int8 costs about **+2.6 WER / +1.7 CER points** (CTranslate2) and **+3.5 WER / +1.5 CER points** (PyTorch dynamic) over fp32. This is a larger cost than int8 usually shows on English Whisper; I did not investigate why, and did not try quantisation-aware calibration or keeping sensitive layers in higher precision. No runaway outputs (hypothesis more than twice the reference length) occurred in any variant on this subset.
- **Size:** PyTorch dynamic quantisation only converts `nn.Linear`; the tied token embedding / output projection (about 160 MB in fp32) stays fp32, so it ends at 413 MB. CTranslate2 quantises it as well (253 MB). Sizes are the weights file(s) only (fp32 `model.safetensors`; the `.pt` state dict; the CTranslate2 directory including tokenizer).
- **Peak RSS:** measured in a fresh process that loads the model and decodes 5 clips (`python -m hindi_asr.quant mem <variant>`). \*The PyTorch int8 figure is dominated by the fp32 model being loaded before it is quantised in memory; a deployment would load the saved int8 state, which I did not measure.
- **Not claimed:** this is a laptop CPU, not an embedded or on-device target. ONNX Runtime was not tried.
- **Versions:** Python 3.12.14, torch 2.15.0+cpu, transformers 5.22.1, peft 0.21.2, ctranslate2 4.8.5, faster-whisper 1.2.1, jiwer 4.0.0, numpy 2.5.4. Raw per-variant results and predictions: [`results/quant/`](results/quant/).

Reproduce (artifacts go to the git-ignored `artifacts/quant/`):

```bash
pip install -e '.[quant]'
python -m hindi_asr.quant convert
for v in fp32 torch-int8 ct2-int8; do python -m hindi_asr.quant bench $v --limit 150 --threads 4; done
python -m hindi_asr.quant mem-clips && for v in fp32 torch-int8 ct2-int8; do python -m hindi_asr.quant mem $v; done
python -m hindi_asr.quant report
```

## Qualitative examples (test set)

| | Text |
|---|---|
| REF | पिछले हफ्ते, METI ने घोषणा की कि Apple ने 34 ओवरहीटिंग के मामले सामने आये है, जिसे कंपनी ने गंभीरता से नहीं लिया। |
| small zero-shot | पिच्छले हवते में आम्टी आई ने गूशना की की आपल ने चोथ तिस अवर हीटिंके मामले सामने आई हैं जिसे कमपनी ने गंभीरता से नहीं लिया |
| medium zero-shot | पिछले हफते METI ने गहुशन की कि अपल ने 34 वर्हेटिंग के मामले सामने आये हैं, जिससे कम्पनी ने गमभीरता से नहीं लिया. |
| **LoRA r32** | पिछले हफ्ते, METI ने घोषणा की कि एपल ने 34 ओवरहीटिंग के मामले सामने आए हैं जिससे कंपनी ने गंभीरता से नहीं लिया. |

| | Text |
|---|---|
| REF | वाहन को उसी दिन तकरीबन 1200 GMT के समय पर दुर्घटनास्थल से दूर ले जाया गया. |
| small zero-shot | वाहन को उसी दिन तक्रीबन बारासो जीम्टी की समये पर दूर गतना स्थल से दूर लेजाया गया |
| medium zero-shot | वाहन को उसी दिन तकरीबन बारासो जी अम्टी के समय पर दुर्गटना स्थल से दूर ले जाया घया. |
| **LoRA r32** | वाहन को उसी दिन तकरीबन 1200 GMT के समय पर दुर्घटना स्थल से दूर ले जाया गया। |

| | Text |
|---|---|
| REF | इसे केमिकल का pH कहा जाता है. आप लाल गोभी के जूस का इस्तेमाल करके एक संकेतक बना सकते हैं. |
| small zero-shot | इसे केमिकल का पीज कहा जाता है आप लाल गोभी के जूस को इस्तमाल करके एक संके तक बना सकते हैं |
| **LoRA r32** | इसे केमिकल का पीएच कहा जाता है. आप लाल गोभी के जूस को इस्तेमाल करके एक संकेत तक बना सकते हैं. |

A failure that remains (a short, fast clip):

| | Text |
|---|---|
| REF | रोलैंडो मेंडोज़ा ने अपनी M16 राइफल से पर्यटकों के ऊपर फायर किया। |
| small zero-shot | अगरोलेंगा मेरोजानाती आँ शिक्टेंगाईखा अगर पल पड़िप़गा अखर पल पल पल पल पल … (loop) |
| medium zero-shot | अपने पाइट्रे पाइट्रे पाइट्रे पाइट्रे … (loop) |
| **LoRA r32** | क्रोलेंड और मेडों जाना है कि एक शिक्स दूवाई पर परिपरितुक का अपका सायच है. |

The typical remaining errors are spelling variants that Hindi tolerates (गोबी/गोभी, शुरुवाती/शुरुआती, ए/ये endings), word-boundary splits (दुर्घटना स्थल vs दुर्घटनास्थल), numbers read as words vs digits, and proper nouns.

## Caveats

- One seed per configuration, no confidence intervals. The ablation differences (1–2.5 WER) are within what seed variance can produce at this data size.
- FLEURS train and test share no sentences, but they do share the domain (read Wikipedia-style sentences). These numbers say little about conversational, telephone or noisy speech.
- RTF is measured on a T4 with fp16 Hugging Face `generate`, so it is not an optimised serving stack. CTranslate2 or faster-whisper would be several times faster.
- The base model's RTF is inflated by its loops (p90 latency 4.9 s vs 2.3 s).
